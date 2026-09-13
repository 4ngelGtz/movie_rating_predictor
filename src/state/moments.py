"""Serializable sufficient statistics for expanding rating histories.

``M2`` is the sum of squared deviations from ``mean``.  For ``count > 0``,
population variance is exactly ``M2 / count`` (never ``M2 / (count - 1)``).
The empty-state mean is intentionally internal; feature fallbacks are resolved
by the feature builder rather than encoded as observations in this state.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
import re
from typing import Any

import pandas as pd


MOVIE_ACTIVITY_WINDOW = pd.Timedelta(days=30)
MAX_ENTITY_ID = 2**32 - 1
_PERSISTED_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}\Z"
)
_TRUSTED_HISTORY_TOKEN = object()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    """Validate an untrusted JSON integer without coercing fractional values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if positive and not 1 <= value <= MAX_ENTITY_ID:
        raise ValueError(f"{name} must be in the uint32 entity domain")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _timestamp(value: Any) -> pd.Timestamp:
    if not isinstance(value, (str, pd.Timestamp)):
        raise TypeError("timestamp must be an ISO-8601 string or pandas Timestamp")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("timestamp must not be missing")
    if timestamp.tz is not None:
        raise ValueError("timestamp must be timezone-naive")
    return timestamp


def persisted_timestamp(value: Any) -> pd.Timestamp:
    """Parse exactly the deterministic naïve format emitted by checkpoints."""
    if not isinstance(value, str) or _PERSISTED_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(
            "persisted timestamp must match YYYY-MM-DDTHH:MM:SS.fffffffff"
        )
    timestamp = _timestamp(value)
    if timestamp_to_string(timestamp) != value:
        raise ValueError("persisted timestamp is not canonical")
    return timestamp


def timestamp_to_string(timestamp: pd.Timestamp) -> str:
    """Serialize a naïve timestamp without losing pandas nanosecond precision."""
    return timestamp.isoformat(timespec="nanoseconds")


@dataclass(slots=True)
class RunningMoments:
    """Mergeable Welford-style ``count``, ``mean``, and ``M2`` state."""

    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("count must be an integer")
        if self.count < 0:
            raise ValueError("count must be nonnegative")
        if not math.isfinite(self.mean) or not math.isfinite(self.M2):
            raise ValueError("mean and M2 must be finite")
        if self.M2 < 0.0:
            raise ValueError("M2 must be nonnegative")
        if self.count == 0 and (self.mean != 0.0 or self.M2 != 0.0):
            raise ValueError("empty moments must have mean=0.0 and M2=0.0")

    @classmethod
    def from_values(cls, values: Iterable[float]) -> RunningMoments:
        """Build deterministic batch moments from a multiset of values.

        Numeric sorting is a reduction detail, not a temporal ordering rule.
        It makes merging a simultaneous batch independent of its physical row
        order.  MovieLens half-star ratings are represented exactly in binary.
        """
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return cls()
        if not all(math.isfinite(value) for value in ordered):
            raise ValueError("moment values must be finite")
        count = len(ordered)
        mean = math.fsum(ordered) / count
        M2 = math.fsum((value - mean) ** 2 for value in ordered)
        return cls(count=count, mean=mean, M2=M2)

    def merge(self, other: RunningMoments) -> None:
        """Merge disjoint observations using the parallel Welford recurrence."""
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.M2 = other.M2
            return

        combined_count = self.count + other.count
        delta = other.mean - self.mean
        self.M2 += other.M2 + delta * delta * self.count * other.count / combined_count
        self.mean += delta * other.count / combined_count
        self.count = combined_count

    def add(self, value: float) -> None:
        """Add one finite observation without allocating batch state."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("moment values must be finite")
        if self.count == 0:
            self.count = 1
            self.mean = value
            return
        combined_count = self.count + 1
        delta = value - self.mean
        self.M2 += delta * delta * self.count / combined_count
        self.mean += delta / combined_count
        self.count = combined_count

    @property
    def population_variance(self) -> float:
        """Return ``M2 / N``; empty and singleton states resolve to zero."""
        if self.count < 2:
            return 0.0
        # Guard only against a tiny negative caused by floating-point roundoff.
        return max(self.M2 / self.count, 0.0)

    @property
    def population_std(self) -> float:
        return math.sqrt(self.population_variance)

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible representation with stable field order."""
        return {"count": self.count, "mean": self.mean, "M2": self.M2}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunningMoments:
        """Restore state produced by :meth:`to_dict`."""
        payload = _mapping(payload, "moments payload")
        if set(payload) != {"count", "mean", "M2"}:
            raise ValueError("moments payload must contain exactly count, mean, and M2")
        return cls(
            count=_integer(payload["count"], "count"),
            mean=_finite_float(payload["mean"], "mean"),
            M2=_finite_float(payload["M2"], "M2"),
        )


@dataclass(slots=True)
class HistoricalRatingState:
    """Sparse rating moments and temporal state for complete timestamp batches."""

    global_moments: RunningMoments = field(default_factory=RunningMoments)
    user_moments: dict[int, RunningMoments] = field(default_factory=dict)
    movie_moments: dict[int, RunningMoments] = field(default_factory=dict)
    user_genre_moments: dict[tuple[int, str], RunningMoments] = field(
        default_factory=dict
    )
    last_user_rating: dict[int, pd.Timestamp] = field(default_factory=dict)
    movie_rating_count_30d: dict[int, int] = field(default_factory=dict)
    _movie_rating_batches_30d: deque[
        tuple[pd.Timestamp, tuple[tuple[int, int], ...]]
    ] = field(default_factory=deque)
    _latest_applied_timestamp: pd.Timestamp | None = None
    _timestamp_tracking_complete: bool = False
    _genre_tracking_complete: bool = False
    _activity_expiration_watermark: pd.Timestamp | None = None
    _activity_expired_count: int = 0

    @classmethod
    def _for_trusted_history(cls, *, _token: object) -> HistoricalRatingState:
        """Create state owned by a canonical-history processing session."""
        if _token is not _TRUSTED_HISTORY_TOKEN:
            raise TypeError("trusted history state is created by its owning session")
        return cls(
            _timestamp_tracking_complete=True,
            _genre_tracking_complete=True,
        )

    @property
    def latest_applied_timestamp(self) -> pd.Timestamp | None:
        """Timestamp of the latest complete batch, if timestamped state exists."""
        return self._latest_applied_timestamp

    @property
    def timestamp_tracking_complete(self) -> bool:
        """Whether every observation was applied through a timestamped batch."""
        return self._timestamp_tracking_complete

    @property
    def genre_tracking_complete(self) -> bool:
        """Whether every trusted batch received an explicit genre bridge."""
        return self._genre_tracking_complete

    @property
    def activity_expiration_watermark(self) -> pd.Timestamp | None:
        """Latest time through which rolling expiration has been requested."""
        return self._activity_expiration_watermark

    @property
    def activity_expired_count(self) -> int:
        """Canonical events irreversibly removed from the rolling queue."""
        return self._activity_expired_count

    def expire_movie_activity(self, prediction_timestamp: Any) -> None:
        """Expire events strictly older than ``t - 30 days``.

        Events exactly on the left boundary remain eligible. Timestamp batches
        are stored once, so each batch enters and leaves the queue once.
        """
        boundary = _timestamp(prediction_timestamp) - MOVIE_ACTIVITY_WINDOW
        expiration_timestamp = boundary + MOVIE_ACTIVITY_WINDOW
        if (
            self._activity_expiration_watermark is None
            or expiration_timestamp > self._activity_expiration_watermark
        ):
            self._activity_expiration_watermark = expiration_timestamp
        while (
            self._movie_rating_batches_30d
            and self._movie_rating_batches_30d[0][0] < boundary
        ):
            _, movie_counts = self._movie_rating_batches_30d.popleft()
            for movie_id, count in movie_counts:
                self._activity_expired_count += count
                remaining = self.movie_rating_count_30d[movie_id] - count
                if remaining:
                    self.movie_rating_count_30d[movie_id] = remaining
                else:
                    del self.movie_rating_count_30d[movie_id]

    def apply_timestamp_batch(
        self,
        ratings: Iterable[tuple[int, int, float]],
        *,
        timestamp: Any | None = None,
        movie_genres: Mapping[int, tuple[str, ...]] | None = None,
    ) -> None:
        """Apply one low-level batch atomically and invalidate checkpoint trust.

        Each tuple is ``(userId, movieId, rating)`` for one canonical event.
        The caller is responsible for passing each event exactly once. Batch
        moments are reduced independently of input order before being merged.
        When ``timestamp`` is supplied, recency and rolling state are updated
        for the complete batch as well. When ``movie_genres`` is supplied,
        each canonical event additionally updates one sparse user-genre state
        per distinct genre associated with its movie. These relationship
        updates do not alter canonical global, user, or movie counts. Because
        this API cannot prove the iterable is a complete canonical timestamp
        group, any call makes the state ineligible for a strict checkpoint.
        """
        self._apply_timestamp_batch(
            ratings,
            timestamp=timestamp,
            movie_genres=movie_genres,
        )
        self._timestamp_tracking_complete = False
        self._genre_tracking_complete = False

    def _apply_trusted_timestamp_batch(
        self,
        ratings: Iterable[tuple[int, int, float]],
        *,
        timestamp: Any,
        movie_genres: Mapping[int, tuple[str, ...]],
        _token: object,
    ) -> None:
        """Apply the next batch selected by the owning history session."""
        if _token is not _TRUSTED_HISTORY_TOKEN:
            raise TypeError("trusted updates are controlled by the history session")
        if not self._timestamp_tracking_complete or not self._genre_tracking_complete:
            raise ValueError(
                "trusted complete-batch updates require checkpoint-compatible state"
            )
        if movie_genres is None:
            raise ValueError(
                "trusted complete-batch updates require an explicit movie_genres bridge"
            )
        self._apply_timestamp_batch(
            ratings,
            timestamp=timestamp,
            movie_genres=movie_genres,
        )

    def _apply_timestamp_batch(
        self,
        ratings: Iterable[tuple[int, int, float]],
        *,
        timestamp: Any | None,
        movie_genres: Mapping[int, tuple[str, ...]] | None,
    ) -> None:
        batch = [
            (int(user_id), int(movie_id), float(rating))
            for user_id, movie_id, rating in ratings
        ]
        if not batch:
            return
        batch_timestamp = _timestamp(timestamp) if timestamp is not None else None
        if (
            batch_timestamp is not None
            and self._latest_applied_timestamp is not None
            and batch_timestamp <= self._latest_applied_timestamp
        ):
            raise ValueError(
                "timestamp batches must be complete and strictly increasing; "
                "a timestamp cannot be applied in multiple partial batches"
            )
        if movie_genres is not None:
            for movie_id in {movie_id for _, movie_id, _ in batch}:
                genre_ids = movie_genres.get(movie_id, ())
                if len(genre_ids) != len(set(genre_ids)):
                    raise ValueError(
                        "movie_genres memberships must be distinct for each movie; "
                        f"duplicate found for movieId {movie_id}"
                    )

        if len(batch) == 1:
            user_id, movie_id, rating = batch[0]
            genres = movie_genres.get(movie_id, ()) if movie_genres is not None else ()
            self.global_moments.add(rating)
            self.user_moments.setdefault(user_id, RunningMoments()).add(rating)
            self.movie_moments.setdefault(movie_id, RunningMoments()).add(rating)
            for genre_id in genres:
                self.user_genre_moments.setdefault(
                    (user_id, genre_id), RunningMoments()
                ).add(rating)
            if batch_timestamp is not None:
                self.last_user_rating[user_id] = batch_timestamp
                self._movie_rating_batches_30d.append(
                    (batch_timestamp, ((movie_id, 1),))
                )
                self.movie_rating_count_30d[movie_id] = (
                    self.movie_rating_count_30d.get(movie_id, 0) + 1
                )
                self._latest_applied_timestamp = batch_timestamp
            return

        self.global_moments.merge(
            RunningMoments.from_values(rating for _, _, rating in batch)
        )

        users: dict[int, list[float]] = {}
        movies: dict[int, list[float]] = {}
        user_genres: dict[tuple[int, str], list[float]] = {}
        for user_id, movie_id, rating in batch:
            users.setdefault(user_id, []).append(rating)
            movies.setdefault(movie_id, []).append(rating)
            if movie_genres is not None:
                for genre_id in movie_genres.get(movie_id, ()):
                    user_genres.setdefault((user_id, genre_id), []).append(rating)

        for user_id in sorted(users):
            self.user_moments.setdefault(user_id, RunningMoments()).merge(
                RunningMoments.from_values(users[user_id])
            )
        for movie_id in sorted(movies):
            self.movie_moments.setdefault(movie_id, RunningMoments()).merge(
                RunningMoments.from_values(movies[movie_id])
            )
        for key in sorted(user_genres):
            self.user_genre_moments.setdefault(key, RunningMoments()).merge(
                RunningMoments.from_values(user_genres[key])
            )

        if batch_timestamp is not None:
            for user_id in sorted(users):
                self.last_user_rating[user_id] = batch_timestamp
            movie_counts = tuple(
                (movie_id, len(movies[movie_id])) for movie_id in sorted(movies)
            )
            self._movie_rating_batches_30d.append((batch_timestamp, movie_counts))
            for movie_id, count in movie_counts:
                self.movie_rating_count_30d[movie_id] = (
                    self.movie_rating_count_30d.get(movie_id, 0) + count
                )
            self._latest_applied_timestamp = batch_timestamp

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, JSON-compatible checkpoint content."""
        return {
            "global": self.global_moments.to_dict(),
            "users": [
                {"userId": key, **self.user_moments[key].to_dict()}
                for key in sorted(self.user_moments)
            ],
            "movies": [
                {"movieId": key, **self.movie_moments[key].to_dict()}
                for key in sorted(self.movie_moments)
            ],
            "userGenres": [
                {
                    "userId": user_id,
                    "genreId": genre_id,
                    **self.user_genre_moments[(user_id, genre_id)].to_dict(),
                }
                for user_id, genre_id in sorted(self.user_genre_moments)
            ],
            "lastUserRatings": [
                {
                    "userId": key,
                    "timestamp": timestamp_to_string(self.last_user_rating[key]),
                }
                for key in sorted(self.last_user_rating)
            ],
            "movieRatingBatches30d": [
                {
                    "timestamp": timestamp_to_string(timestamp),
                    "movies": [
                        {"movieId": movie_id, "count": count}
                        for movie_id, count in movie_counts
                    ],
                }
                for timestamp, movie_counts in self._movie_rating_batches_30d
            ],
            "movieRatingCounts30d": [
                {"movieId": movie_id, "count": self.movie_rating_count_30d[movie_id]}
                for movie_id in sorted(self.movie_rating_count_30d)
            ],
            "latestAppliedTimestamp": (
                timestamp_to_string(self._latest_applied_timestamp)
                if self._latest_applied_timestamp is not None
                else None
            ),
            "timestampTrackingComplete": self._timestamp_tracking_complete,
            "genreTrackingComplete": self._genre_tracking_complete,
            "activityExpirationWatermark": (
                timestamp_to_string(self._activity_expiration_watermark)
                if self._activity_expiration_watermark is not None
                else None
            ),
            "activityExpiredCount": self._activity_expired_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HistoricalRatingState:
        """Restore deterministic state, retaining legacy Phase 4A-C support.

        Unlike the earlier implementation, this method validates persisted
        numeric types exactly and never truncates fractional counts or IDs.
        """
        payload = _mapping(payload, "state payload")
        phase_4a_keys = {"global", "users", "movies"}
        phase_4b_keys = phase_4a_keys | {
            "lastUserRatings",
            "movieRatingBatches30d",
        }
        phase_4c_keys = phase_4b_keys | {"userGenres"}
        current_keys = phase_4c_keys | {
            "movieRatingCounts30d",
            "latestAppliedTimestamp",
            "timestampTrackingComplete",
            "genreTrackingComplete",
            "activityExpirationWatermark",
            "activityExpiredCount",
        }
        if set(payload) not in (
            phase_4a_keys,
            phase_4b_keys,
            phase_4c_keys,
            current_keys,
        ):
            raise ValueError("state payload has unexpected or missing fields")

        state = cls(global_moments=RunningMoments.from_dict(payload["global"]))
        for raw_row in _list(payload["users"], "users"):
            row = _mapping(raw_row, "user state row")
            if set(row) != {"userId", "count", "mean", "M2"}:
                raise ValueError("user state row has unexpected or missing fields")
            user_id = _integer(row["userId"], "userId", positive=True)
            if user_id in state.user_moments:
                raise ValueError(f"duplicate userId in state payload: {user_id}")
            state.user_moments[user_id] = RunningMoments.from_dict(
                {key: row[key] for key in ("count", "mean", "M2")}
            )
        for raw_row in _list(payload["movies"], "movies"):
            row = _mapping(raw_row, "movie state row")
            if set(row) != {"movieId", "count", "mean", "M2"}:
                raise ValueError("movie state row has unexpected or missing fields")
            movie_id = _integer(row["movieId"], "movieId", positive=True)
            if movie_id in state.movie_moments:
                raise ValueError(f"duplicate movieId in state payload: {movie_id}")
            state.movie_moments[movie_id] = RunningMoments.from_dict(
                {key: row[key] for key in ("count", "mean", "M2")}
            )

        for raw_row in _list(payload.get("userGenres", []), "userGenres"):
            row = _mapping(raw_row, "user-genre state row")
            if set(row) != {"userId", "genreId", "count", "mean", "M2"}:
                raise ValueError("user-genre state row has unexpected or missing fields")
            user_id = _integer(row["userId"], "userId", positive=True)
            genre_id = row["genreId"]
            if not isinstance(genre_id, str):
                raise TypeError("genreId must be a string")
            key = (user_id, genre_id)
            if (
                not genre_id
                or genre_id.strip() != genre_id
                or "|" in genre_id
                or genre_id == "(no genres listed)"
                or key in state.user_genre_moments
            ):
                raise ValueError("user-genre state keys must be canonical and unique")
            state.user_genre_moments[key] = RunningMoments.from_dict(
                {name: row[name] for name in ("count", "mean", "M2")}
            )

        for raw_row in _list(payload.get("lastUserRatings", []), "lastUserRatings"):
            row = _mapping(raw_row, "recency state row")
            if set(row) != {"userId", "timestamp"}:
                raise ValueError("recency state row has unexpected or missing fields")
            user_id = _integer(row["userId"], "userId", positive=True)
            if user_id in state.last_user_rating:
                raise ValueError(f"duplicate userId in recency payload: {user_id}")
            timestamp_parser = (
                persisted_timestamp if set(payload) == current_keys else _timestamp
            )
            state.last_user_rating[user_id] = timestamp_parser(row["timestamp"])

        previous_timestamp: pd.Timestamp | None = None
        calculated_counts: dict[int, int] = {}
        for raw_row in _list(
            payload.get("movieRatingBatches30d", []), "movieRatingBatches30d"
        ):
            row = _mapping(raw_row, "rolling batch row")
            if set(row) != {"timestamp", "movies"}:
                raise ValueError("rolling batch row has unexpected or missing fields")
            timestamp_parser = (
                persisted_timestamp if set(payload) == current_keys else _timestamp
            )
            timestamp = timestamp_parser(row["timestamp"])
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("rolling timestamp batches must be strictly increasing")
            previous_timestamp = timestamp
            seen_movies: set[int] = set()
            movie_counts: list[tuple[int, int]] = []
            raw_movies = _list(row["movies"], "rolling batch movies")
            if not raw_movies:
                raise ValueError("rolling timestamp batch must contain movie counts")
            for raw_movie in raw_movies:
                movie = _mapping(raw_movie, "rolling movie row")
                if set(movie) != {"movieId", "count"}:
                    raise ValueError("rolling movie row has unexpected or missing fields")
                movie_id = _integer(movie["movieId"], "movieId", positive=True)
                count = _integer(movie["count"], "rolling count")
                if movie_id in seen_movies or count <= 0:
                    raise ValueError("rolling batch movie keys must be unique and positive")
                seen_movies.add(movie_id)
                movie_counts.append((movie_id, count))
                calculated_counts[movie_id] = calculated_counts.get(movie_id, 0) + count
            state._movie_rating_batches_30d.append(
                (timestamp, tuple(sorted(movie_counts)))
            )

        if "movieRatingCounts30d" in payload:
            stored_counts: dict[int, int] = {}
            for raw_row in _list(
                payload["movieRatingCounts30d"], "movieRatingCounts30d"
            ):
                row = _mapping(raw_row, "rolling summary row")
                if set(row) != {"movieId", "count"}:
                    raise ValueError("rolling summary row has unexpected or missing fields")
                movie_id = _integer(row["movieId"], "movieId", positive=True)
                count = _integer(row["count"], "rolling count")
                if movie_id in stored_counts or count <= 0:
                    raise ValueError("rolling summary keys must be unique and positive")
                stored_counts[movie_id] = count
            if stored_counts != calculated_counts:
                raise ValueError("rolling count summary does not match timestamp batches")
        state.movie_rating_count_30d = calculated_counts

        if "latestAppliedTimestamp" in payload:
            latest = payload["latestAppliedTimestamp"]
            state._latest_applied_timestamp = (
                None if latest is None else persisted_timestamp(latest)
            )
            tracking = payload["timestampTrackingComplete"]
            if not isinstance(tracking, bool):
                raise TypeError("timestampTrackingComplete must be a boolean")
            state._timestamp_tracking_complete = tracking
            genre_tracking = payload["genreTrackingComplete"]
            if not isinstance(genre_tracking, bool):
                raise TypeError("genreTrackingComplete must be a boolean")
            state._genre_tracking_complete = genre_tracking
            watermark = payload["activityExpirationWatermark"]
            state._activity_expiration_watermark = (
                None if watermark is None else persisted_timestamp(watermark)
            )
            state._activity_expired_count = _integer(
                payload["activityExpiredCount"], "activityExpiredCount"
            )
            if state._activity_expired_count < 0:
                raise ValueError("activityExpiredCount must be nonnegative")
        else:
            # Legacy Phase 4A-C payloads remain loadable as raw state, but lack
            # sufficient lifecycle provenance to become strict checkpoints.
            state._timestamp_tracking_complete = False
            state._genre_tracking_complete = False

        state._validate_invariants()
        return state

    def _validate_invariants(self) -> None:
        """Reject impossible cross-component state before it becomes live."""
        user_total = sum(moments.count for moments in self.user_moments.values())
        movie_total = sum(moments.count for moments in self.movie_moments.values())
        if user_total != self.global_moments.count or movie_total != user_total:
            raise ValueError("global, user, and movie moment counts are inconsistent")
        if any(moments.count == 0 for moments in self.user_moments.values()):
            raise ValueError("user moment entries must contain observations")
        if any(moments.count == 0 for moments in self.movie_moments.values()):
            raise ValueError("movie moment entries must contain observations")
        if any(moments.count == 0 for moments in self.user_genre_moments.values()):
            raise ValueError("user-genre moment entries must contain observations")
        if any(
            user_id not in self.user_moments
            for user_id, _ in self.user_genre_moments
        ):
            raise ValueError("user-genre state contains an unknown user")
        if any(
            moments.count > self.user_moments[user_id].count
            for (user_id, _), moments in self.user_genre_moments.items()
        ):
            raise ValueError("user-genre count exceeds complete user history")
        if set(self.last_user_rating) - set(self.user_moments):
            raise ValueError("recency state contains a user without rating moments")
        if set(self.movie_rating_count_30d) - set(self.movie_moments):
            raise ValueError("rolling state contains a movie without rating moments")
        if any(
            count > self.movie_moments[movie_id].count
            for movie_id, count in self.movie_rating_count_30d.items()
        ):
            raise ValueError("rolling count exceeds complete movie history")
        if self._latest_applied_timestamp is not None:
            if any(
                timestamp > self._latest_applied_timestamp
                for timestamp in self.last_user_rating.values()
            ):
                raise ValueError("recency timestamp exceeds latest applied batch")
            if any(
                timestamp > self._latest_applied_timestamp
                for timestamp, _ in self._movie_rating_batches_30d
            ):
                raise ValueError("rolling timestamp exceeds latest applied batch")
            if (
                self._timestamp_tracking_complete
                and max(self.last_user_rating.values(), default=None)
                != self._latest_applied_timestamp
            ):
                raise ValueError("latest applied batch and recency state are inconsistent")
        elif self.global_moments.count and self._timestamp_tracking_complete:
            raise ValueError("timestamp-complete nonempty state needs a latest batch timestamp")
        if self._timestamp_tracking_complete and set(self.last_user_rating) != set(
            self.user_moments
        ):
            raise ValueError("timestamp-complete state needs recency for every user")
        if self._genre_tracking_complete and not self._timestamp_tracking_complete:
            raise ValueError("genre-complete state also requires complete timestamp batches")
        if self._activity_expired_count and self._activity_expiration_watermark is None:
            raise ValueError("expired rolling events require an expiration watermark")
        retained_count = sum(self.movie_rating_count_30d.values())
        if self._timestamp_tracking_complete and (
            retained_count + self._activity_expired_count
            != self.global_moments.count
        ):
            raise ValueError(
                "rolling lifecycle counts do not conserve canonical events"
            )
