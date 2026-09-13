"""Strict Phase 4E checkpoint envelopes and atomic JSON persistence.

A checkpoint with exclusive ``checkpointCutoff = c`` contains exactly the
dynamic state produced by complete timestamp batches with event timestamps
strictly less than ``c``. Events at ``c`` have not been applied. Checkpoints
can therefore only be created after a complete batch and before the next one.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from src.data.schemas import RATING_VALUES
from src.features.catalog import (
    CatalogLookups,
    build_catalog_lookups,
    catalog_snapshot_id,
)
from src.features.expanding import _resolve_timestamp_batch_features
from src.state.moments import (
    HistoricalRatingState,
    MAX_ENTITY_ID,
    MOVIE_ACTIVITY_WINDOW,
    _TRUSTED_HISTORY_TOKEN,
    _timestamp,
    persisted_timestamp,
    timestamp_to_string,
)


CHECKPOINT_SCHEMA_VERSION = 1
_ENVELOPE_FIELDS = {
    "schemaVersion",
    "checkpointCutoff",
    "catalogSnapshotId",
    "historySourceId",
    "historyEventCount",
    "processedBatchCount",
    "processedEventCount",
    "state",
}
_CURRENT_STATE_FIELDS = {
    "global",
    "users",
    "movies",
    "userGenres",
    "lastUserRatings",
    "movieRatingBatches30d",
    "movieRatingCounts30d",
    "latestAppliedTimestamp",
    "timestampTrackingComplete",
    "genreTrackingComplete",
    "activityExpirationWatermark",
    "activityExpiredCount",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HISTORY_ID_VERSION = "canonical-rating-history-v1"
_CANONICAL_BATCH_TOKEN = object()
_HISTORY_SESSION_TOKEN = object()
_SESSION_CHECKPOINT_TOKEN = object()
MAX_EVENT_ID = 2**64 - 1


def _catalog_snapshot_id(value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("catalogSnapshotId must be a lowercase SHA-256 digest")
    return value


def _history_source_id(value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("historySourceId must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _cutoff(value: Any) -> pd.Timestamp:
    try:
        return persisted_timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpointCutoff: {value!r}") from exc


def _checkpoint_cutoff_value(value: Any) -> pd.Timestamp:
    if isinstance(value, str):
        return _cutoff(value)
    if not isinstance(value, pd.Timestamp):
        raise TypeError("checkpointCutoff must be a pandas Timestamp or canonical string")
    return _timestamp(value)


@dataclass(frozen=True, slots=True, init=False)
class CanonicalTimestampBatch:
    """One source-bound, indivisible timestamp group."""

    history_source_id: str
    ordinal: int
    timestamp: pd.Timestamp
    ratings: tuple[tuple[int, int, float], ...]
    rating_event_ids: tuple[int, ...]

    def __init__(
        self,
        history_source_id: str,
        ordinal: int,
        timestamp: pd.Timestamp,
        ratings: tuple[tuple[int, int, float], ...],
        rating_event_ids: tuple[int, ...] = (),
        *,
        _token: object,
    ) -> None:
        if _token is not _CANONICAL_BATCH_TOKEN:
            raise TypeError("canonical batches are created by their history source")
        object.__setattr__(self, "history_source_id", history_source_id)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "ratings", ratings)
        object.__setattr__(self, "rating_event_ids", rating_event_ids)


def _canonical_batches_and_source_id(
    rating_events: pd.DataFrame,
) -> tuple[tuple[CanonicalTimestampBatch, ...], str]:
    """Validate and canonically identify one complete rating-history source."""
    if not isinstance(rating_events, pd.DataFrame):
        raise TypeError("rating_events must be a pandas DataFrame")
    required = ("ratingEventId", "userId", "movieId", "rating", "timestamp")
    if missing := set(required) - set(rating_events.columns):
        raise ValueError(f"rating_events: missing required columns: {sorted(missing)}")
    if rating_events.loc[:, list(required)].isna().any().any():
        raise ValueError("rating_events: required columns must not contain missing values")
    for column in ("ratingEventId", "userId", "movieId"):
        if not is_integer_dtype(rating_events[column].dtype):
            raise TypeError(f"rating_events: {column} must have an integer dtype")
        maximum = MAX_EVENT_ID if column == "ratingEventId" else MAX_ENTITY_ID
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
            for value in rating_events[column].astype(object)
        ):
            raise ValueError(f"rating_events: {column} is outside its integral domain")
    if rating_events["ratingEventId"].duplicated().any():
        raise ValueError("rating_events: ratingEventId must be unique")
    rating_values = rating_events["rating"].astype(object)
    if is_bool_dtype(rating_events["rating"].dtype) or any(
        isinstance(value, (bool, np.bool_)) for value in rating_values
    ):
        raise TypeError(
            "rating_events: rating must not contain booleans or have a boolean dtype"
        )
    if not is_numeric_dtype(rating_events["rating"].dtype):
        raise TypeError("rating_events: rating must have a numeric dtype")
    ratings = rating_events["rating"].astype(float)
    invalid_ratings = set(ratings.unique()) - RATING_VALUES
    if invalid_ratings:
        raise ValueError(
            f"rating_events: invalid rating values: {sorted(invalid_ratings)}"
        )
    timestamps = rating_events["timestamp"]
    if not is_datetime64_any_dtype(timestamps.dtype):
        raise TypeError("rating_events: timestamp must have a pandas datetime64 dtype")
    if timestamps.dt.tz is not None:
        raise ValueError("rating_events: timestamp must be timezone-naive")

    ordered = rating_events.loc[:, list(required)].sort_values(
        ["timestamp", "ratingEventId"], kind="stable"
    )
    source_rows = [
        {
            "ratingEventId": int(row.ratingEventId),
            "userId": int(row.userId),
            "movieId": int(row.movieId),
            "rating": float(row.rating),
            "timestamp": timestamp_to_string(pd.Timestamp(row.timestamp)),
        }
        for row in ordered.itertuples(index=False)
    ]
    source_content = json.dumps(
        {"version": _HISTORY_ID_VERSION, "events": source_rows},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    source_id = hashlib.sha256(source_content).hexdigest()

    batches: list[CanonicalTimestampBatch] = []
    for ordinal, (timestamp, group) in enumerate(
        ordered.groupby("timestamp", sort=False)
    ):
        batch_ratings = tuple(
            (int(row.userId), int(row.movieId), float(row.rating))
            for row in group[["userId", "movieId", "rating"]].itertuples(
                index=False
            )
        )
        batch_event_ids = tuple(int(value) for value in group["ratingEventId"])
        batches.append(
            CanonicalTimestampBatch(
                history_source_id=source_id,
                ordinal=ordinal,
                timestamp=pd.Timestamp(timestamp),
                ratings=batch_ratings,
                rating_event_ids=batch_event_ids,
                _token=_CANONICAL_BATCH_TOKEN,
            )
        )
    return tuple(batches), source_id


def _batch_event_frame(batch: CanonicalTimestampBatch) -> pd.DataFrame:
    """Reconstruct one canonical batch at event grain for feature resolution."""
    return pd.DataFrame(
        {
            "ratingEventId": pd.Series(batch.rating_event_ids, dtype="UInt64"),
            "userId": pd.Series(
                (rating[0] for rating in batch.ratings), dtype="uint32"
            ),
            "movieId": pd.Series(
                (rating[1] for rating in batch.ratings), dtype="uint32"
            ),
            "timestamp": pd.Series(
                [batch.timestamp] * len(batch.ratings), dtype="datetime64[ns]"
            ),
        }
    )


def _validate_checkpoint_state(
    state: HistoricalRatingState,
    checkpoint_cutoff: pd.Timestamp,
    *,
    require_canonical_rolling: bool = True,
) -> None:
    if not state.timestamp_tracking_complete:
        raise ValueError(
            "checkpoint state is lifecycle-ambiguous because observations were "
            "applied without timestamps"
        )
    if not state.genre_tracking_complete:
        raise ValueError(
            "checkpoint state is missing trusted Phase 4C genre processing"
        )
    if (
        state.activity_expiration_watermark is not None
        and state.activity_expiration_watermark > checkpoint_cutoff
    ):
        raise ValueError(
            "rolling activity was already expired beyond checkpointCutoff"
        )
    latest = state.latest_applied_timestamp
    if latest is not None and latest >= checkpoint_cutoff:
        raise ValueError(
            "checkpointCutoff must be strictly after the latest complete timestamp "
            "batch; nothing at the cutoff may have been applied"
        )
    if any(timestamp >= checkpoint_cutoff for timestamp in state.last_user_rating.values()):
        raise ValueError("recency state contains an event at or after checkpointCutoff")
    if require_canonical_rolling:
        if state.activity_expiration_watermark != checkpoint_cutoff:
            raise ValueError(
                "activityExpirationWatermark must equal checkpointCutoff"
            )
        rolling_boundary = checkpoint_cutoff - MOVIE_ACTIVITY_WINDOW
        if any(
            timestamp < rolling_boundary or timestamp >= checkpoint_cutoff
            for timestamp, _ in state._movie_rating_batches_30d
        ):
            raise ValueError(
                "rolling state must contain exactly retained batches in "
                "[checkpointCutoff - 30 days, checkpointCutoff)"
            )


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Validated provenance returned with restored dynamic state."""

    schema_version: int
    checkpoint_cutoff: pd.Timestamp
    catalog_snapshot_id: str
    history_source_id: str
    history_event_count: int
    processed_batch_count: int
    processed_event_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
        ):
            raise TypeError("schemaVersion must be an integer")
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schemaVersion: {self.schema_version}"
            )
        cutoff = _checkpoint_cutoff_value(self.checkpoint_cutoff)
        object.__setattr__(self, "checkpoint_cutoff", cutoff)
        _catalog_snapshot_id(self.catalog_snapshot_id)
        _history_source_id(self.history_source_id)
        history_count = _nonnegative_integer(
            self.history_event_count, "historyEventCount"
        )
        batch_count = _nonnegative_integer(
            self.processed_batch_count, "processedBatchCount"
        )
        processed_count = _nonnegative_integer(
            self.processed_event_count, "processedEventCount"
        )
        if processed_count > history_count:
            raise ValueError("processedEventCount cannot exceed historyEventCount")
        if (batch_count == 0) != (processed_count == 0):
            raise ValueError(
                "processed batch and event counts must both be zero or both be positive"
            )
        if batch_count > processed_count:
            raise ValueError(
                "processedBatchCount cannot exceed processedEventCount"
            )


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """An immutable checkpoint snapshot around a validated state payload."""

    metadata: CheckpointMetadata
    _state_payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, CheckpointMetadata):
            raise TypeError("metadata must be CheckpointMetadata")
        if not isinstance(self._state_payload, Mapping):
            raise TypeError("state must be an object")
        if set(self._state_payload) != _CURRENT_STATE_FIELDS:
            raise ValueError(
                "checkpoint state must use the complete current state schema; "
                "legacy raw state payloads are not checkpoint artifacts"
            )
        state = HistoricalRatingState.from_dict(self._state_payload)
        _validate_checkpoint_state(state, self.metadata.checkpoint_cutoff)
        if state.global_moments.count != self.metadata.processed_event_count:
            raise ValueError(
                "processedEventCount must equal the checkpoint state event count"
            )
        object.__setattr__(self, "_state_payload", state.to_dict())

    @classmethod
    def from_state(
        cls,
        *,
        state: HistoricalRatingState,
        checkpoint_cutoff: Any,
        catalog_snapshot_id: str,
    ) -> Checkpoint:
        """Reject detached state that cannot prove its canonical source prefix."""
        raise ValueError(
            "strict checkpoints require CheckpointHistorySession checkpoint "
            "publication; a bare state cannot prove canonical-prefix completeness"
        )

    @classmethod
    def _from_history_session(
        cls,
        *,
        state: HistoricalRatingState,
        checkpoint_cutoff: pd.Timestamp,
        catalog_snapshot_id: str,
        history_source_id: str,
        history_event_count: int,
        processed_batch_count: int,
        processed_event_count: int,
        _token: object,
    ) -> Checkpoint:
        """Snapshot state after the owning session proves the cutoff boundary."""
        if _token is not _SESSION_CHECKPOINT_TOKEN:
            raise TypeError("strict checkpoint publication is controlled by its session")
        cutoff = _checkpoint_cutoff_value(checkpoint_cutoff)
        snapshot_id = _catalog_snapshot_id(catalog_snapshot_id)
        snapshot = HistoricalRatingState.from_dict(state.to_dict())
        _validate_checkpoint_state(
            snapshot, cutoff, require_canonical_rolling=False
        )
        snapshot.expire_movie_activity(cutoff)
        _validate_checkpoint_state(snapshot, cutoff)
        return cls(
            metadata=CheckpointMetadata(
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                checkpoint_cutoff=cutoff,
                catalog_snapshot_id=snapshot_id,
                history_source_id=history_source_id,
                history_event_count=history_event_count,
                processed_batch_count=processed_batch_count,
                processed_event_count=processed_event_count,
            ),
            _state_payload=snapshot.to_dict(),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Checkpoint:
        """Validate an untrusted checkpoint envelope before restoring state."""
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint payload must be an object")
        if set(payload) != _ENVELOPE_FIELDS:
            raise ValueError("checkpoint payload has unexpected or missing fields")
        version = payload["schemaVersion"]
        cutoff = _cutoff(payload["checkpointCutoff"])
        snapshot_id = _catalog_snapshot_id(payload["catalogSnapshotId"])
        source_id = _history_source_id(payload["historySourceId"])
        state_payload = payload["state"]
        if not isinstance(state_payload, Mapping):
            raise TypeError("state must be an object")
        if set(state_payload) != _CURRENT_STATE_FIELDS:
            raise ValueError(
                "checkpoint state must use the complete current state schema; "
                "legacy raw state payloads are not checkpoint artifacts"
            )
        return cls(
            metadata=CheckpointMetadata(
                version,
                cutoff,
                snapshot_id,
                source_id,
                payload["historyEventCount"],
                payload["processedBatchCount"],
                payload["processedEventCount"],
            ),
            _state_payload=dict(state_payload),
        )

    @classmethod
    def from_json(cls, content: str | bytes | bytearray) -> Checkpoint:
        """Decode and validate canonical checkpoint JSON content."""
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("checkpoint is not valid UTF-8 JSON") from exc
        return cls.from_dict(payload)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Checkpoint:
        """Read and validate a checkpoint JSON file."""
        return cls.from_json(Path(path).read_bytes())

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-compatible checkpoint envelope."""
        state = HistoricalRatingState.from_dict(self._state_payload)
        _validate_checkpoint_state(state, self.metadata.checkpoint_cutoff)
        return {
            "schemaVersion": self.metadata.schema_version,
            "checkpointCutoff": timestamp_to_string(
                self.metadata.checkpoint_cutoff
            ),
            "catalogSnapshotId": self.metadata.catalog_snapshot_id,
            "historySourceId": self.metadata.history_source_id,
            "historyEventCount": self.metadata.history_event_count,
            "processedBatchCount": self.metadata.processed_batch_count,
            "processedEventCount": self.metadata.processed_event_count,
            "state": state.to_dict(),
        }

    def to_json(self) -> str:
        """Return deterministic UTF-8 JSON with normalized keys and spacing."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def restore(self) -> tuple[CheckpointMetadata, HistoricalRatingState]:
        """Return validated metadata and an independent live mutable state."""
        state = HistoricalRatingState.from_dict(self._state_payload)
        _validate_checkpoint_state(state, self.metadata.checkpoint_cutoff)
        return self.metadata, state

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically publish checkpoint JSON without exposing partial files."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(self.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


@dataclass(slots=True, init=False)
class CheckpointHistorySession:
    """Own canonical history, ordered batch progress, and checkpoint authority."""

    _batches: tuple[CanonicalTimestampBatch, ...]
    _history_source_id: str
    _history_event_count: int
    _state: HistoricalRatingState
    _cursor: int = 0
    _processed_event_count: int = 0
    _applied_event_ids: set[int]
    _catalog: CatalogLookups | None
    _catalog_snapshot_id: str | None

    def __init__(
        self,
        *,
        batches: tuple[CanonicalTimestampBatch, ...],
        history_source_id: str,
        history_event_count: int,
        _token: object,
    ) -> None:
        if _token is not _HISTORY_SESSION_TOKEN:
            raise TypeError(
                "history sessions are created from validated canonical events"
            )
        self._batches = batches
        self._history_source_id = history_source_id
        self._history_event_count = history_event_count
        self._state = HistoricalRatingState._for_trusted_history(
            _token=_TRUSTED_HISTORY_TOKEN
        )
        self._cursor = 0
        self._processed_event_count = 0
        self._applied_event_ids = set()
        self._catalog = None
        self._catalog_snapshot_id = None

    @classmethod
    def from_canonical_events(
        cls,
        rating_events: pd.DataFrame,
        *,
        expected_history_source_id: str | None = None,
    ) -> CheckpointHistorySession:
        """Validate and take ownership of one complete canonical event sequence."""
        batches, source_id = _canonical_batches_and_source_id(rating_events)
        if (
            expected_history_source_id is not None
            and _history_source_id(expected_history_source_id) != source_id
        ):
            raise ValueError(
                "canonical rating history does not match expected historySourceId"
            )
        return cls(
            batches=batches,
            history_source_id=source_id,
            history_event_count=sum(len(batch.ratings) for batch in batches),
            _token=_HISTORY_SESSION_TOKEN,
        )

    @classmethod
    def resume_from_checkpoint(
        cls,
        *,
        checkpoint: Checkpoint,
        rating_events: pd.DataFrame,
        canonical_movies: pd.DataFrame,
        movie_genres: pd.DataFrame,
    ) -> CheckpointHistorySession:
        """Restore a checkpoint against its exact history and catalog sources.

        All source, catalog, cutoff, and progress checks finish before the new
        session takes ownership of mutable restored state.
        """
        if not isinstance(checkpoint, Checkpoint):
            raise TypeError("checkpoint must be a validated Checkpoint")

        metadata = checkpoint.metadata
        batches, source_id = _canonical_batches_and_source_id(rating_events)
        if source_id != metadata.history_source_id:
            raise ValueError(
                "canonical rating history does not match checkpoint historySourceId"
            )

        actual_catalog_id = catalog_snapshot_id(canonical_movies, movie_genres)
        if actual_catalog_id != metadata.catalog_snapshot_id:
            raise ValueError(
                "canonical catalog does not match checkpoint catalogSnapshotId"
            )
        catalog = build_catalog_lookups(
            canonical_movies,
            movie_genres,
            required_movie_ids=rating_events["movieId"],
        )

        _, restored_state = checkpoint.restore()

        timestamps = tuple(batch.timestamp for batch in batches)
        required_cursor = bisect_left(timestamps, metadata.checkpoint_cutoff)
        expected_event_count = sum(
            len(batch.rating_event_ids) for batch in batches[:required_cursor]
        )
        if metadata.history_event_count != len(rating_events):
            raise ValueError(
                "checkpoint historyEventCount does not match canonical history"
            )
        if metadata.processed_batch_count != required_cursor:
            raise ValueError(
                "checkpoint processedBatchCount does not match its exclusive cutoff"
            )
        if metadata.processed_event_count != expected_event_count:
            raise ValueError(
                "checkpoint processedEventCount does not match its canonical prefix"
            )
        expected_latest = (
            batches[required_cursor - 1].timestamp if required_cursor else None
        )
        if restored_state.latest_applied_timestamp != expected_latest:
            raise ValueError(
                "checkpoint state timestamp does not match its canonical prefix"
            )

        session = cls(
            batches=batches,
            history_source_id=source_id,
            history_event_count=len(rating_events),
            _token=_HISTORY_SESSION_TOKEN,
        )
        session._state = restored_state
        session._cursor = required_cursor
        session._processed_event_count = expected_event_count
        session._applied_event_ids = {
            event_id
            for batch in batches[:required_cursor]
            for event_id in batch.rating_event_ids
        }
        session._catalog = catalog
        session._catalog_snapshot_id = actual_catalog_id
        return session

    @property
    def history_source_id(self) -> str:
        return self._history_source_id

    @property
    def batches(self) -> tuple[CanonicalTimestampBatch, ...]:
        """Expose immutable source-bound batches for inspection and guarded use."""
        return self._batches

    @property
    def processed_batch_count(self) -> int:
        return self._cursor

    @property
    def processed_event_count(self) -> int:
        return self._processed_event_count

    @property
    def applied_event_ids(self) -> frozenset[int]:
        """Return immutable canonical event provenance for the applied prefix."""
        return frozenset(self._applied_event_ids)

    def snapshot_state(self) -> HistoricalRatingState:
        """Return an independent state copy without surrendering session ownership."""
        return HistoricalRatingState.from_dict(self._state.to_dict())

    def process_next_batch(
        self,
        *,
        movie_genres: Mapping[int, tuple[str, ...]] | None = None,
    ) -> CanonicalTimestampBatch:
        """Consume exactly the next complete timestamp group."""
        if self._cursor >= len(self._batches):
            raise ValueError("canonical history has no unprocessed timestamp batch")
        return self.process_batch(
            self._batches[self._cursor], movie_genres=movie_genres
        )

    def process_batch(
        self,
        batch: CanonicalTimestampBatch,
        *,
        movie_genres: Mapping[int, tuple[str, ...]] | None = None,
    ) -> CanonicalTimestampBatch:
        """Consume the next source batch atomically with the session catalog."""
        consumed, _ = self._consume_batch(
            batch,
            movie_genres=movie_genres,
            emit_features=False,
        )
        return consumed

    def _consume_batch(
        self,
        batch: CanonicalTimestampBatch,
        *,
        movie_genres: Mapping[int, tuple[str, ...]] | None,
        emit_features: bool,
    ) -> tuple[CanonicalTimestampBatch, pd.DataFrame | None]:
        """Validate, prepare on a copy, then atomically commit one batch."""
        if not isinstance(batch, CanonicalTimestampBatch):
            raise TypeError("batch must come from a canonical-history session")
        if batch.history_source_id != self._history_source_id:
            raise ValueError("batch belongs to a different canonical history source")
        if self._cursor >= len(self._batches):
            raise ValueError("canonical history has no unprocessed timestamp batch")
        expected = self._batches[self._cursor]
        if batch != expected:
            raise ValueError(
                f"canonical batches must be consumed in order; expected ordinal "
                f"{self._cursor}"
            )
        duplicate_ids = self._applied_event_ids.intersection(batch.rating_event_ids)
        if duplicate_ids:
            raise ValueError(
                "canonical batch contains already-applied ratingEventId values"
            )

        if self._catalog is not None:
            if movie_genres is not None:
                raise ValueError(
                    "a resumed session uses its bound canonical catalog; "
                    "movie_genres cannot be overridden"
                )
            resolved_genres = self._catalog.genres_by_movie
        else:
            if movie_genres is None:
                raise ValueError(
                    "new history sessions require an explicit movie_genres bridge"
                )
            resolved_genres = movie_genres
        if emit_features and self._catalog is None:
            raise ValueError(
                "replay-time features require a session restored with a catalog"
            )

        prepared_state = HistoricalRatingState.from_dict(self._state.to_dict())
        prepared_state.expire_movie_activity(batch.timestamp)
        features = (
            _resolve_timestamp_batch_features(
                prepared_state,
                _batch_event_frame(batch),
                self._catalog,
            )
            if emit_features and self._catalog is not None
            else None
        )
        prepared_state._apply_trusted_timestamp_batch(
            batch.ratings,
            timestamp=batch.timestamp,
            movie_genres=resolved_genres,
            _token=_TRUSTED_HISTORY_TOKEN,
        )

        self._state = prepared_state
        self._cursor += 1
        self._processed_event_count += len(batch.ratings)
        self._applied_event_ids.update(batch.rating_event_ids)
        return batch, features

    def replay_batch(self, batch: CanonicalTimestampBatch) -> CanonicalTimestampBatch:
        """Replay exactly the next source batch using the validated catalog."""
        if self._catalog is None:
            raise ValueError("replay requires a session restored from a checkpoint")
        consumed, _ = self._consume_batch(
            batch, movie_genres=None, emit_features=False
        )
        return consumed

    def replay_next_batch(self) -> CanonicalTimestampBatch:
        """Replay the next complete timestamp batch at or after the cutoff."""
        if self._cursor >= len(self._batches):
            raise ValueError("canonical history has no unprocessed timestamp batch")
        return self.replay_batch(self._batches[self._cursor])

    def replay_next_batch_features(self) -> pd.DataFrame:
        """Emit all 17 pre-batch features, then atomically apply the next batch."""
        if self._catalog is None:
            raise ValueError("replay requires a session restored from a checkpoint")
        if self._cursor >= len(self._batches):
            raise ValueError("canonical history has no unprocessed timestamp batch")
        _, features = self._consume_batch(
            self._batches[self._cursor],
            movie_genres=None,
            emit_features=True,
        )
        assert features is not None
        return features

    def replay_batches(
        self, batches: Iterable[CanonicalTimestampBatch]
    ) -> tuple[CanonicalTimestampBatch, ...]:
        """Replay one contiguous range as a single atomic session transaction."""
        if self._catalog is None:
            raise ValueError("replay requires a session restored from a checkpoint")
        candidates = tuple(batches)
        expected = self._batches[self._cursor : self._cursor + len(candidates)]
        if candidates != expected:
            raise ValueError(
                "replay batches must be the next contiguous canonical source range"
            )
        candidate_ids = tuple(
            event_id
            for batch in candidates
            for event_id in batch.rating_event_ids
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("replay range contains duplicate ratingEventId values")
        if self._applied_event_ids.intersection(candidate_ids):
            raise ValueError("replay range contains already-applied ratingEventId values")

        working = CheckpointHistorySession(
            batches=self._batches,
            history_source_id=self._history_source_id,
            history_event_count=self._history_event_count,
            _token=_HISTORY_SESSION_TOKEN,
        )
        working._state = HistoricalRatingState.from_dict(self._state.to_dict())
        working._cursor = self._cursor
        working._processed_event_count = self._processed_event_count
        working._applied_event_ids = set(self._applied_event_ids)
        working._catalog = self._catalog
        working._catalog_snapshot_id = self._catalog_snapshot_id

        consumed = tuple(working.replay_batch(batch) for batch in candidates)
        self._state = working._state
        self._cursor = working._cursor
        self._processed_event_count = working._processed_event_count
        self._applied_event_ids = working._applied_event_ids
        return consumed

    def replay_remaining(self) -> tuple[CanonicalTimestampBatch, ...]:
        """Replay the complete canonical suffix from the checkpoint cutoff."""
        return self.replay_batches(self._batches[self._cursor :])

    def checkpoint(
        self,
        *,
        checkpoint_cutoff: Any,
        catalog_snapshot_id: str,
    ) -> Checkpoint:
        """Publish only when progress is exactly the source prefix before cutoff."""
        cutoff = _checkpoint_cutoff_value(checkpoint_cutoff)
        snapshot_id = _catalog_snapshot_id(catalog_snapshot_id)
        if (
            self._catalog_snapshot_id is not None
            and snapshot_id != self._catalog_snapshot_id
        ):
            raise ValueError(
                "catalogSnapshotId does not match the catalog bound to this session"
            )
        timestamps = tuple(batch.timestamp for batch in self._batches)
        required_cursor = bisect_left(timestamps, cutoff)
        if self._cursor < required_cursor:
            raise ValueError(
                "checkpoint prefix is incomplete: canonical events before the "
                "checkpointCutoff remain unprocessed"
            )
        if self._cursor > required_cursor:
            raise ValueError(
                "checkpoint prefix is invalid: canonical events at or after the "
                "checkpointCutoff were already processed"
            )
        expected_event_count = sum(
            len(batch.ratings) for batch in self._batches[: self._cursor]
        )
        if (
            self._processed_event_count != expected_event_count
            or self._state.global_moments.count != expected_event_count
        ):
            raise ValueError("history session state no longer matches source progress")
        expected_latest = (
            self._batches[self._cursor - 1].timestamp if self._cursor else None
        )
        if self._state.latest_applied_timestamp != expected_latest:
            raise ValueError("history session timestamp no longer matches source progress")
        return Checkpoint._from_history_session(
            state=self._state,
            checkpoint_cutoff=cutoff,
            catalog_snapshot_id=snapshot_id,
            history_source_id=self._history_source_id,
            history_event_count=self._history_event_count,
            processed_batch_count=self._cursor,
            processed_event_count=self._processed_event_count,
            _token=_SESSION_CHECKPOINT_TOKEN,
        )
