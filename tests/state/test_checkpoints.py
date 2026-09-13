from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.features.expanding import _resolved_features
from src.state.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    CanonicalTimestampBatch,
    Checkpoint,
    CheckpointHistorySession,
    CheckpointMetadata,
)
from src.state.moments import HistoricalRatingState


T = pd.Timestamp("2020-01-01 10:00:00.123456789")
CATALOG_ID = "a" * 64


def event_frame(
    rows: list[tuple[int, int, int, float, pd.Timestamp]],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ratingEventId": pd.Series([row[0] for row in rows], dtype="UInt64"),
            "userId": pd.Series([row[1] for row in rows], dtype="uint32"),
            "movieId": pd.Series([row[2] for row in rows], dtype="uint32"),
            "rating": pd.Series([row[3] for row in rows], dtype="float32"),
            "timestamp": pd.Series(
                [row[4] for row in rows], dtype="datetime64[ns]"
            ),
        }
    )


def populated_session(*, reverse: bool = False) -> CheckpointHistorySession:
    first = [(1, 2, 20, 2.0, T), (2, 1, 10, 4.0, T)]
    second_timestamp = T + pd.Timedelta(seconds=1)
    second = [
        (3, 1, 20, 5.0, second_timestamp),
        (4, 2, 10, 3.0, second_timestamp),
    ]
    if reverse:
        first.reverse()
        second.reverse()
    genres = {20: ("Drama",), 10: ("Comedy", "Drama")}
    session = CheckpointHistorySession.from_canonical_events(
        event_frame(first + second)
    )
    session.process_next_batch(movie_genres=genres)
    session.process_next_batch(movie_genres=genres)
    return session


def populated_state(*, reverse: bool = False) -> HistoricalRatingState:
    return populated_session(reverse=reverse).snapshot_state()


def populated_checkpoint() -> Checkpoint:
    return populated_session().checkpoint(
        checkpoint_cutoff=T + pd.Timedelta(seconds=2),
        catalog_snapshot_id=CATALOG_ID,
    )


def checkpoint_payload() -> dict[str, object]:
    return populated_checkpoint().to_dict()


def test_checkpoint_json_round_trip_restores_exact_metadata_and_state() -> None:
    original = populated_checkpoint()

    restored_checkpoint = Checkpoint.from_json(original.to_json())
    metadata, restored_state = restored_checkpoint.restore()

    assert metadata.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert metadata.checkpoint_cutoff == T + pd.Timedelta(seconds=2)
    assert metadata.catalog_snapshot_id == CATALOG_ID
    assert metadata.history_source_id == populated_session().history_source_id
    assert metadata.history_event_count == 4
    assert metadata.processed_batch_count == 2
    assert metadata.processed_event_count == 4
    assert restored_state.to_dict() == original.restore()[1].to_dict()
    assert restored_checkpoint.to_json() == original.to_json()


def test_exclusive_cutoff_state_contains_only_events_strictly_before_cutoff() -> None:
    cutoff = T + pd.Timedelta(seconds=1)
    session = CheckpointHistorySession.from_canonical_events(
        event_frame(
            [
                (1, 1, 10, 2.0, T),
                (2, 1, 10, 4.0, cutoff),
                (3, 1, 10, 5.0, cutoff + pd.Timedelta(seconds=1)),
            ]
        )
    )
    session.process_next_batch(movie_genres={})

    _, restored = session.checkpoint(
        checkpoint_cutoff=cutoff,
        catalog_snapshot_id=CATALOG_ID,
    ).restore()

    assert restored.global_moments.count == 1
    assert restored.global_moments.mean == 2.0
    assert restored.latest_applied_timestamp == T


def test_checkpoint_cannot_split_or_include_a_same_timestamp_batch() -> None:
    state = HistoricalRatingState()
    state.apply_timestamp_batch([(1, 10, 2.0)], timestamp=T)

    with pytest.raises(ValueError, match="bare state cannot prove"):
        Checkpoint.from_state(
            state=state,
            checkpoint_cutoff=T + pd.Timedelta(seconds=1),
            catalog_snapshot_id=CATALOG_ID,
        )

    session = CheckpointHistorySession.from_canonical_events(
        event_frame([(1, 1, 10, 2.0, T), (2, 2, 20, 4.0, T)])
    )
    session.process_next_batch(movie_genres={})
    with pytest.raises(ValueError, match="at or after"):
        session.checkpoint(
            checkpoint_cutoff=T,
            catalog_snapshot_id=CATALOG_ID,
        )


def test_truncated_tied_timestamp_cannot_claim_original_history_source() -> None:
    full_events = event_frame(
        [(1, 1, 10, 2.0, T), (2, 2, 20, 4.0, T)]
    )
    full_session = CheckpointHistorySession.from_canonical_events(full_events)

    with pytest.raises(ValueError, match="does not match expected historySourceId"):
        CheckpointHistorySession.from_canonical_events(
            full_events.iloc[:1].copy(),
            expected_history_source_id=full_session.history_source_id,
        )


def test_history_source_id_ignores_physical_row_and_index_order() -> None:
    events = event_frame(
        [
            (1, 1, 10, 2.0, T),
            (2, 2, 20, 4.0, T),
            (3, 3, 30, 3.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    expected = CheckpointHistorySession.from_canonical_events(
        events
    ).history_source_id
    reversed_rows = events.iloc[::-1].copy()
    reversed_rows.index = [19, 7, 41]

    assert (
        CheckpointHistorySession.from_canonical_events(
            reversed_rows
        ).history_source_id
        == expected
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("ratingEventId", 99),
        ("userId", 99),
        ("movieId", 99),
        ("rating", 4.5),
        ("timestamp", T + pd.Timedelta(nanoseconds=1)),
    ],
)
def test_history_source_id_changes_with_canonical_event_content(
    column: str, value: object
) -> None:
    events = event_frame(
        [
            (1, 1, 10, 2.0, T),
            (2, 2, 20, 4.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    original = CheckpointHistorySession.from_canonical_events(
        events
    ).history_source_id
    changed = events.copy()
    changed.loc[0, column] = value

    assert (
        CheckpointHistorySession.from_canonical_events(changed).history_source_id
        != original
    )


def test_history_source_id_changes_when_event_is_added_or_removed() -> None:
    events = event_frame(
        [
            (1, 1, 10, 2.0, T),
            (2, 2, 20, 4.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    original = CheckpointHistorySession.from_canonical_events(
        events
    ).history_source_id
    removed = events.iloc[:1].copy()
    added = event_frame(
        [
            (1, 1, 10, 2.0, T),
            (2, 2, 20, 4.0, T + pd.Timedelta(seconds=1)),
            (3, 3, 30, 3.0, T + pd.Timedelta(seconds=2)),
        ]
    )

    assert (
        CheckpointHistorySession.from_canonical_events(removed).history_source_id
        != original
    )
    assert (
        CheckpointHistorySession.from_canonical_events(added).history_source_id
        != original
    )


def test_trusted_batch_and_session_constructors_are_not_public_bypasses() -> None:
    with pytest.raises(TypeError, match="created by their history source"):
        CanonicalTimestampBatch(
            CATALOG_ID, 0, T, ((1, 10, 4.0),), _token=object()
        )
    with pytest.raises(TypeError, match="validated canonical events"):
        CheckpointHistorySession(
            batches=(),
            history_source_id=CATALOG_ID,
            history_event_count=0,
            _token=object(),
        )


def test_foreign_session_batch_is_rejected() -> None:
    first = CheckpointHistorySession.from_canonical_events(
        event_frame([(1, 1, 10, 2.0, T)])
    )
    second = CheckpointHistorySession.from_canonical_events(
        event_frame([(2, 2, 20, 4.0, T)])
    )

    with pytest.raises(ValueError, match="different canonical history source"):
        second.process_batch(first.batches[0], movie_genres={})


def test_already_consumed_batch_is_rejected() -> None:
    session = CheckpointHistorySession.from_canonical_events(
        event_frame(
            [
                (1, 1, 10, 2.0, T),
                (2, 2, 20, 4.0, T + pd.Timedelta(seconds=1)),
            ]
        )
    )
    consumed = session.process_next_batch(movie_genres={})

    with pytest.raises(ValueError, match="expected ordinal 1"):
        session.process_batch(consumed, movie_genres={})


def test_skipped_prefix_batch_is_rejected() -> None:
    session = CheckpointHistorySession.from_canonical_events(
        event_frame(
            [
                (1, 1, 10, 2.0, T),
                (2, 2, 20, 4.0, T + pd.Timedelta(seconds=1)),
            ]
        )
    )

    with pytest.raises(ValueError, match="expected ordinal 0"):
        session.process_batch(session.batches[1], movie_genres={})


def test_skipped_middle_batch_is_rejected() -> None:
    session = CheckpointHistorySession.from_canonical_events(
        event_frame(
            [
                (1, 1, 10, 2.0, T),
                (2, 2, 20, 3.0, T + pd.Timedelta(seconds=1)),
                (3, 3, 30, 4.0, T + pd.Timedelta(seconds=2)),
            ]
        )
    )
    session.process_next_batch(movie_genres={})

    with pytest.raises(ValueError, match="expected ordinal 1"):
        session.process_batch(session.batches[2], movie_genres={})


def test_valid_prefix_checkpoints_between_source_timestamps() -> None:
    session = CheckpointHistorySession.from_canonical_events(
        event_frame(
            [
                (1, 1, 10, 2.0, T),
                (2, 2, 20, 3.0, T + pd.Timedelta(seconds=1)),
                (3, 3, 30, 4.0, T + pd.Timedelta(seconds=3)),
            ]
        )
    )
    session.process_next_batch(movie_genres={})
    session.process_next_batch(movie_genres={})

    checkpoint = session.checkpoint(
        checkpoint_cutoff=T + pd.Timedelta(seconds=2),
        catalog_snapshot_id=CATALOG_ID,
    )

    assert checkpoint.metadata.processed_batch_count == 2
    assert checkpoint.metadata.processed_event_count == 2
    assert checkpoint.restore()[1].global_moments.count == 2


def test_full_tied_batch_is_processed_atomically() -> None:
    session = CheckpointHistorySession.from_canonical_events(
        event_frame(
            [(1, 1, 10, 2.0, T), (2, 2, 20, 4.0, T)]
        )
    )

    batch = session.process_next_batch(movie_genres={})
    checkpoint = session.checkpoint(
        checkpoint_cutoff=T + pd.Timedelta(nanoseconds=1),
        catalog_snapshot_id=CATALOG_ID,
    )

    assert len(batch.ratings) == 2
    assert checkpoint.metadata.processed_batch_count == 1
    assert checkpoint.metadata.processed_event_count == 2
    assert checkpoint.restore()[1].global_moments.count == 2


def test_empty_prefix_before_first_batch_is_checkpointable() -> None:
    session = CheckpointHistorySession.from_canonical_events(
        event_frame([(1, 1, 10, 2.0, T)])
    )

    checkpoint = session.checkpoint(
        checkpoint_cutoff=T,
        catalog_snapshot_id=CATALOG_ID,
    )

    assert checkpoint.metadata.processed_batch_count == 0
    assert checkpoint.metadata.processed_event_count == 0
    assert checkpoint.restore()[1].global_moments.count == 0


def test_complete_source_is_checkpointable_after_final_batch() -> None:
    session = CheckpointHistorySession.from_canonical_events(
        event_frame(
            [
                (1, 1, 10, 2.0, T),
                (2, 2, 20, 4.0, T + pd.Timedelta(seconds=1)),
            ]
        )
    )
    session.process_next_batch(movie_genres={})
    session.process_next_batch(movie_genres={})

    checkpoint = session.checkpoint(
        checkpoint_cutoff=T + pd.Timedelta(seconds=2),
        catalog_snapshot_id=CATALOG_ID,
    )

    assert checkpoint.metadata.processed_event_count == 2


def test_checkpoint_rejects_incomplete_prefix_for_cutoff() -> None:
    session = CheckpointHistorySession.from_canonical_events(
        event_frame(
            [
                (1, 1, 10, 2.0, T),
                (2, 2, 20, 4.0, T + pd.Timedelta(seconds=1)),
            ]
        )
    )
    session.process_next_batch(movie_genres={})

    with pytest.raises(ValueError, match="events before.*remain unprocessed"):
        session.checkpoint(
            checkpoint_cutoff=T + pd.Timedelta(seconds=2),
            catalog_snapshot_id=CATALOG_ID,
        )


def test_trusted_history_accepts_every_movielens_half_star_rating() -> None:
    rows = [
        (index, index, index, index / 2, T)
        for index in range(1, 11)
    ]
    session = CheckpointHistorySession.from_canonical_events(event_frame(rows))
    session.process_next_batch(movie_genres={})

    assert session.processed_event_count == 10


@pytest.mark.parametrize("rating", [3.7, 99.0, 0.0, 5.5, np.nan, np.inf])
def test_trusted_history_rejects_invalid_rating_domain(rating: float) -> None:
    with pytest.raises(ValueError, match="missing|invalid rating"):
        CheckpointHistorySession.from_canonical_events(
            event_frame([(1, 1, 10, rating, T)])
        )


@pytest.mark.parametrize("rating", [True, False, np.bool_(True), np.bool_(False)])
def test_trusted_history_rejects_boolean_rating_values(rating: object) -> None:
    events = event_frame([(1, 1, 10, 1.0, T)])
    events["rating"] = pd.Series([rating], dtype=object)

    with pytest.raises(TypeError, match="must not contain booleans"):
        CheckpointHistorySession.from_canonical_events(events)


def test_trusted_history_rejects_boolean_rating_series() -> None:
    events = event_frame(
        [(1, 1, 10, 1.0, T), (2, 2, 20, 1.0, T)]
    )
    events["rating"] = pd.Series([True, False], dtype=bool)

    with pytest.raises(TypeError, match="boolean dtype"):
        CheckpointHistorySession.from_canonical_events(events)


def test_untimestamped_state_cannot_be_checkpointed() -> None:
    state = HistoricalRatingState()
    state.apply_timestamp_batch([(1, 10, 4.0)])
    with pytest.raises(ValueError, match="bare state cannot prove"):
        Checkpoint.from_state(
            state=state,
            checkpoint_cutoff=T,
            catalog_snapshot_id=CATALOG_ID,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("schemaVersion"), "unexpected or missing"),
        (lambda payload: payload.__setitem__("schemaVersion", 2), "unsupported"),
    ],
)
def test_checkpoint_schema_version_is_required_and_strict(mutation, message: str) -> None:
    payload = checkpoint_payload()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        Checkpoint.from_dict(payload)


@pytest.mark.parametrize("snapshot_id", [None, "", "A" * 64, "a" * 63, "z" * 64])
def test_catalog_snapshot_id_must_be_lowercase_sha256(snapshot_id: object) -> None:
    payload = checkpoint_payload()
    payload["catalogSnapshotId"] = snapshot_id
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        Checkpoint.from_dict(payload)


def test_missing_catalog_snapshot_id_is_rejected() -> None:
    payload = checkpoint_payload()
    del payload["catalogSnapshotId"]
    with pytest.raises(ValueError, match="unexpected or missing"):
        Checkpoint.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("historySourceId", "bad", "lowercase SHA-256"),
        ("historyEventCount", -1, "nonnegative"),
        ("historyEventCount", 3, "cannot exceed"),
        ("processedBatchCount", 0, "both be zero or both be positive"),
        ("processedEventCount", 3, "must equal.*state event count"),
    ],
)
def test_history_source_progress_metadata_is_strict(
    field: str, value: object, message: str
) -> None:
    payload = checkpoint_payload()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        Checkpoint.from_dict(payload)


def test_processed_batch_count_cannot_exceed_processed_event_count() -> None:
    payload = checkpoint_payload()
    payload["processedBatchCount"] = payload["processedEventCount"] + 1

    with pytest.raises(
        ValueError, match="processedBatchCount cannot exceed processedEventCount"
    ):
        Checkpoint.from_dict(payload)


@pytest.mark.parametrize("field", ["processedBatchCount", "processedEventCount"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_processed_progress_counts_reject_invalid_numeric_types(
    field: str, value: object
) -> None:
    payload = checkpoint_payload()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match="integer|nonnegative"):
        Checkpoint.from_dict(payload)


@pytest.mark.parametrize("cutoff", [None, "not-a-timestamp", "2020-01-01T00:00:00Z"])
def test_checkpoint_cutoff_must_be_valid_and_timezone_naive(cutoff: object) -> None:
    payload = checkpoint_payload()
    payload["checkpointCutoff"] = cutoff
    with pytest.raises((TypeError, ValueError)):
        Checkpoint.from_dict(payload)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("global", "count", 1.9),
        ("global", "count", True),
        ("global", "mean", float("nan")),
        ("global", "M2", float("inf")),
        ("users", "userId", 1.9),
        ("movies", "movieId", "1.5"),
        ("users", "count", -1),
    ],
)
def test_malformed_numeric_payloads_are_never_coerced(
    section: str, field: str, value: object
) -> None:
    payload = checkpoint_payload()
    state = payload["state"]
    assert isinstance(state, dict)
    target = state[section] if section == "global" else state[section][0]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        Checkpoint.from_dict(payload)


def test_rolling_summary_must_match_queue_contents() -> None:
    payload = checkpoint_payload()
    state = payload["state"]
    assert isinstance(state, dict)
    state["movieRatingCounts30d"][0]["count"] += 1
    with pytest.raises(ValueError, match="summary does not match"):
        Checkpoint.from_dict(payload)


def test_invalid_persisted_recency_timestamp_is_rejected() -> None:
    payload = checkpoint_payload()
    payload["state"]["lastUserRatings"][0]["timestamp"] = "not-a-timestamp"
    with pytest.raises(ValueError):
        Checkpoint.from_dict(payload)


def test_user_genre_keys_must_be_canonical_and_unique() -> None:
    payload = checkpoint_payload()
    payload["state"]["userGenres"][0]["genreId"] = " Drama"
    with pytest.raises(ValueError, match="canonical and unique"):
        Checkpoint.from_dict(payload)

    duplicate = checkpoint_payload()
    duplicate["state"]["userGenres"].append(
        deepcopy(duplicate["state"]["userGenres"][0])
    )
    with pytest.raises(ValueError, match="canonical and unique"):
        Checkpoint.from_dict(duplicate)


def test_invalid_moments_and_cross_component_counts_are_rejected() -> None:
    negative_m2 = checkpoint_payload()
    negative_m2["state"]["users"][0]["M2"] = -1.0
    with pytest.raises(ValueError, match="M2 must be nonnegative"):
        Checkpoint.from_dict(negative_m2)

    inconsistent = checkpoint_payload()
    inconsistent["state"]["global"]["count"] += 1
    with pytest.raises(ValueError, match="moment counts are inconsistent"):
        Checkpoint.from_dict(inconsistent)


def test_equivalent_insertion_and_batch_orders_have_identical_json() -> None:
    first = populated_session().checkpoint(
        checkpoint_cutoff=T + pd.Timedelta(seconds=2),
        catalog_snapshot_id=CATALOG_ID,
    )
    second = populated_session(reverse=True).checkpoint(
        checkpoint_cutoff=T + pd.Timedelta(seconds=2),
        catalog_snapshot_id=CATALOG_ID,
    )
    assert first.to_json() == second.to_json()


def test_file_round_trip_preserves_checkpoint(tmp_path) -> None:
    checkpoint = Checkpoint.from_dict(checkpoint_payload())
    path = tmp_path / "state.json"
    checkpoint.save(path)
    assert Checkpoint.load(path).to_json() == checkpoint.to_json()
    assert path.read_text(encoding="utf-8") == checkpoint.to_json() + "\n"


def test_failed_atomic_publication_does_not_replace_valid_checkpoint(
    tmp_path, monkeypatch
) -> None:
    checkpoint = Checkpoint.from_dict(checkpoint_payload())
    path = tmp_path / "state.json"
    checkpoint.save(path)
    original = path.read_bytes()

    def fail_replace(source, destination) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr("src.state.checkpoints.os.replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        checkpoint.save(path)

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_legacy_raw_state_payload_is_not_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="unexpected or missing"):
        Checkpoint.from_dict(deepcopy(populated_state().to_dict()))


def test_cutoff_serialization_preserves_nanoseconds() -> None:
    checkpoint = Checkpoint.from_dict(checkpoint_payload())
    encoded = json.loads(checkpoint.to_json())
    assert encoded["checkpointCutoff"].endswith("123456789")
    assert Checkpoint.from_json(checkpoint.to_json()).metadata.checkpoint_cutoff == (
        T + pd.Timedelta(seconds=2)
    )


def test_over_expiration_blocks_checkpoint_at_earlier_cutoff() -> None:
    cutoff = T + pd.Timedelta(days=1)
    payload = checkpoint_payload()
    payload["checkpointCutoff"] = cutoff.isoformat(timespec="nanoseconds")
    payload["state"]["activityExpirationWatermark"] = (
        cutoff + pd.Timedelta(days=31)
    ).isoformat(timespec="nanoseconds")

    with pytest.raises(ValueError, match="expired beyond checkpointCutoff"):
        Checkpoint.from_dict(payload)


def test_expiration_exactly_at_checkpoint_cutoff_is_valid() -> None:
    cutoff = T + pd.Timedelta(days=31)
    _, restored = populated_session().checkpoint(
        checkpoint_cutoff=cutoff,
        catalog_snapshot_id=CATALOG_ID,
    ).restore()

    assert restored.activity_expiration_watermark == cutoff
    assert restored.activity_expired_count == restored.global_moments.count
    assert restored.movie_rating_count_30d == {}


def test_coordinated_rolling_deletion_is_rejected_by_event_conservation() -> None:
    payload = checkpoint_payload()
    payload["state"]["movieRatingBatches30d"] = []
    payload["state"]["movieRatingCounts30d"] = []

    with pytest.raises(ValueError, match="do not conserve canonical events"):
        Checkpoint.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestampTrackingComplete", 1, "must be a boolean"),
        ("timestampTrackingComplete", False, "complete timestamp batches"),
        ("genreTrackingComplete", 1, "must be a boolean"),
        ("genreTrackingComplete", False, "missing trusted Phase 4C"),
        ("activityExpirationWatermark", None, "must equal checkpointCutoff"),
        ("activityExpiredCount", 1, "do not conserve canonical events"),
    ],
)
def test_checkpoint_lifecycle_metadata_is_strict(
    field: str, value: object, message: str
) -> None:
    payload = checkpoint_payload()
    payload["state"][field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        Checkpoint.from_dict(payload)


def test_omitted_genre_processing_is_checkpoint_ineligible() -> None:
    session = CheckpointHistorySession.from_canonical_events(
        event_frame([(1, 1, 10, 4.0, T)])
    )
    with pytest.raises(ValueError, match="explicit movie_genres bridge"):
        session.process_next_batch(movie_genres=None)
    assert session.processed_batch_count == 0

    state = HistoricalRatingState()
    state.apply_timestamp_batch([(1, 10, 4.0)], timestamp=T)
    with pytest.raises(ValueError, match="bare state cannot prove"):
        Checkpoint.from_state(
            state=state,
            checkpoint_cutoff=T + pd.Timedelta(seconds=1),
            catalog_snapshot_id=CATALOG_ID,
        )


@pytest.mark.parametrize(
    ("version", "digest"),
    [(999, CATALOG_ID), (CHECKPOINT_SCHEMA_VERSION, "bad")],
)
def test_direct_checkpoint_construction_cannot_bypass_metadata_validation(
    version: int, digest: str
) -> None:
    with pytest.raises(ValueError):
        Checkpoint(
            CheckpointMetadata(
                version,
                T + pd.Timedelta(seconds=2),
                digest,
                "b" * 64,
                4,
                2,
                4,
            ),
            checkpoint_payload()["state"],
        )


def test_direct_checkpoint_construction_validates_state() -> None:
    metadata = CheckpointMetadata(
        CHECKPOINT_SCHEMA_VERSION,
        T + pd.Timedelta(seconds=2),
        CATALOG_ID,
        "b" * 64,
        4,
        2,
        4,
    )
    with pytest.raises(TypeError, match="state must be an object"):
        Checkpoint(metadata, [])


def test_rolling_movie_order_is_canonicalized() -> None:
    original = checkpoint_payload()
    reordered = deepcopy(original)
    reordered["state"]["movieRatingBatches30d"][0]["movies"].reverse()

    assert Checkpoint.from_dict(reordered).to_json() == Checkpoint.from_dict(
        original
    ).to_json()


@pytest.mark.parametrize(
    "value",
    [
        "now",
        "today",
        "January 1, 2020",
        "2020-01-01T10:00:02",
        "2020-01-01 10:00:02.123456789",
    ],
)
def test_persisted_timestamps_require_exact_canonical_format(value: str) -> None:
    payload = checkpoint_payload()
    payload["checkpointCutoff"] = value
    with pytest.raises(ValueError, match="invalid checkpointCutoff"):
        Checkpoint.from_dict(payload)

    payload = checkpoint_payload()
    payload["state"]["lastUserRatings"][0]["timestamp"] = value
    with pytest.raises(ValueError, match="must match"):
        Checkpoint.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    ["latestAppliedTimestamp", "activityExpirationWatermark"],
)
def test_lifecycle_timestamps_use_exact_canonical_format(field: str) -> None:
    payload = checkpoint_payload()
    payload["state"][field] = "now"
    with pytest.raises(ValueError, match="must match"):
        Checkpoint.from_dict(payload)


def test_rolling_timestamps_use_exact_canonical_format() -> None:
    payload = checkpoint_payload()
    payload["state"]["movieRatingBatches30d"][0]["timestamp"] = "today"
    with pytest.raises(ValueError, match="must match"):
        Checkpoint.from_dict(payload)


def test_checkpoint_restore_then_continue_matches_uninterrupted_state() -> None:
    cutoff = T + pd.Timedelta(days=30)
    genres = {
        10: ("Comedy", "Drama"),
        20: ("Drama",),
        30: ("Thriller",),
    }
    before = (
        (T - pd.Timedelta(seconds=1), [(1, 10, 1.0), (2, 20, 5.0)]),
        (T, [(1, 20, 2.0), (3, 10, 4.0)]),
    )
    after = (
        (cutoff, [(1, 10, 5.0), (2, 30, 3.0)]),
        (cutoff + pd.Timedelta(seconds=1), [(3, 20, 2.0)]),
    )
    rows: list[tuple[int, int, int, float, pd.Timestamp]] = []
    event_id = 1
    for timestamp, batch in (*before, *after):
        for user_id, movie_id, rating in batch:
            rows.append((event_id, user_id, movie_id, rating, timestamp))
            event_id += 1
    session = CheckpointHistorySession.from_canonical_events(event_frame(rows))
    session.process_next_batch(movie_genres=genres)
    session.process_next_batch(movie_genres=genres)
    restored = Checkpoint.from_json(
        session.checkpoint(
            checkpoint_cutoff=cutoff,
            catalog_snapshot_id=CATALOG_ID,
        ).to_json()
    ).restore()[1]
    uninterrupted = session.snapshot_state()
    uninterrupted.expire_movie_activity(cutoff)
    assert restored.to_dict() == uninterrupted.to_dict()

    for timestamp, batch in after:
        for state in (uninterrupted, restored):
            state.expire_movie_activity(timestamp)
        for user_id, movie_id, _ in batch:
            expected = _resolved_features(
                uninterrupted, user_id, movie_id, timestamp, genres
            )
            actual = _resolved_features(restored, user_id, movie_id, timestamp, genres)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
        for state in (uninterrupted, restored):
            state.apply_timestamp_batch(
                batch, timestamp=timestamp, movie_genres=genres
            )

    assert restored.to_dict() == uninterrupted.to_dict()
