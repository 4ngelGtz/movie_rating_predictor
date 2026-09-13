from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.catalog import catalog_snapshot_id
from src.features.expanding import _resolved_features, build_expanding_rating_features
from src.state.checkpoints import Checkpoint, CheckpointHistorySession
from src.state.moments import HistoricalRatingState


T = pd.Timestamp("2020-02-01 12:00:00")


def events(
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


def catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.DataFrame(
        {
            "movieId": pd.Series([10, 20, 30], dtype="uint32"),
            "title": pd.Series(["Ten", "Twenty", "Thirty"], dtype="string"),
            "releaseYear": pd.Series([2000, 2001, 2002], dtype="UInt16"),
            "genreStatus": pd.Series(["listed"] * 3, dtype="string"),
            "imdbId": pd.Series([100, 200, 300], dtype="UInt32"),
            "tmdbId": pd.Series([1000, 2000, 3000], dtype="UInt32"),
        }
    )
    genres = pd.DataFrame(
        {
            "movieId": pd.Series([10, 10, 20, 30], dtype="uint32"),
            "genreId": pd.Series(
                ["Comedy", "Drama", "Drama", "Thriller"], dtype="string"
            ),
        }
    )
    return movies, genres


def source() -> pd.DataFrame:
    return events(
        [
            (1, 1, 10, 2.0, T - pd.Timedelta(days=30)),
            (2, 2, 20, 4.0, T - pd.Timedelta(seconds=1)),
            (3, 1, 20, 5.0, T),
            (4, 2, 10, 3.0, T),
            (5, 1, 30, 4.5, T + pd.Timedelta(seconds=1)),
        ]
    )


def checkpoint_and_uninterrupted():
    rating_events = source()
    movies, genre_rows = catalog()
    genres = {10: ("Comedy", "Drama"), 20: ("Drama",), 30: ("Thriller",)}
    uninterrupted = CheckpointHistorySession.from_canonical_events(rating_events)
    prefix = CheckpointHistorySession.from_canonical_events(rating_events)
    for _ in range(2):
        uninterrupted.process_next_batch(movie_genres=genres)
        prefix.process_next_batch(movie_genres=genres)
    checkpoint = prefix.checkpoint(
        checkpoint_cutoff=T,
        catalog_snapshot_id=catalog_snapshot_id(movies, genre_rows),
    )
    while uninterrupted.processed_event_count < len(rating_events):
        uninterrupted.process_next_batch(movie_genres=genres)
    return rating_events, movies, genre_rows, checkpoint, uninterrupted


def resume(rating_events, movies, genres, checkpoint):
    return CheckpointHistorySession.resume_from_checkpoint(
        checkpoint=checkpoint,
        rating_events=rating_events,
        canonical_movies=movies,
        movie_genres=genres,
    )


def test_happy_path_resume_matches_uninterrupted_processing() -> None:
    rating_events, movies, genres, checkpoint, uninterrupted = (
        checkpoint_and_uninterrupted()
    )

    resumed = resume(rating_events, movies, genres, checkpoint)
    resumed.replay_remaining()

    assert resumed.snapshot_state().to_dict() == uninterrupted.snapshot_state().to_dict()
    assert resumed.processed_event_count == len(rating_events)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("ratingEventId", 99),
        ("userId", 99),
        ("movieId", 10),
        ("rating", 1.5),
        ("timestamp", T + pd.Timedelta(seconds=5)),
    ],
)
def test_changed_canonical_history_is_rejected_before_replay(
    column: str, value: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    rating_events, movies, genres, checkpoint, _ = checkpoint_and_uninterrupted()
    changed = rating_events.copy()
    changed.loc[4, column] = value
    applied = False

    def unexpected_apply(*args, **kwargs) -> None:
        nonlocal applied
        applied = True

    monkeypatch.setattr(
        "src.state.moments.HistoricalRatingState._apply_trusted_timestamp_batch",
        unexpected_apply,
    )
    with pytest.raises(ValueError, match="historySourceId"):
        resume(changed, movies, genres, checkpoint)
    assert not applied


@pytest.mark.parametrize("change", ["metadata", "membership"])
def test_changed_catalog_is_rejected_before_replay(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rating_events, movies, genres, checkpoint, _ = checkpoint_and_uninterrupted()
    movies = movies.copy()
    genres = genres.copy()
    if change == "metadata":
        movies.loc[movies["movieId"] == 30, "title"] = "Changed"
    else:
        genres.loc[len(genres)] = (30, "Drama")
    applied = False

    def unexpected_apply(*args, **kwargs) -> None:
        nonlocal applied
        applied = True

    monkeypatch.setattr(
        "src.state.moments.HistoricalRatingState._apply_trusted_timestamp_batch",
        unexpected_apply,
    )
    with pytest.raises(ValueError, match="catalogSnapshotId"):
        resume(rating_events, movies, genres, checkpoint)
    assert not applied


def test_retrying_same_replay_batch_is_rejected_without_double_counting() -> None:
    rating_events, movies, genres, checkpoint, _ = checkpoint_and_uninterrupted()
    resumed = resume(rating_events, movies, genres, checkpoint)
    batch = resumed.replay_next_batch()
    after_first = resumed.snapshot_state().to_dict()

    with pytest.raises(ValueError, match="expected ordinal"):
        resumed.replay_batch(batch)

    assert resumed.snapshot_state().to_dict() == after_first


def test_overlapping_replay_range_is_rejected_before_any_state_change() -> None:
    rating_events, movies, genres, checkpoint, _ = checkpoint_and_uninterrupted()
    resumed = resume(rating_events, movies, genres, checkpoint)
    first = resumed.replay_next_batch()
    before_overlap = resumed.snapshot_state().to_dict()

    with pytest.raises(ValueError, match="next contiguous"):
        resumed.replay_batches((first, resumed.batches[first.ordinal + 1]))

    assert resumed.snapshot_state().to_dict() == before_overlap


def test_tied_replay_batch_has_one_strict_pre_batch_snapshot() -> None:
    rating_events, movies, genre_rows, checkpoint, _ = checkpoint_and_uninterrupted()
    resumed = resume(rating_events, movies, genre_rows, checkpoint)
    batch = resumed.batches[resumed.processed_batch_count]
    before = resumed.snapshot_state()
    genres = {10: ("Comedy", "Drama"), 20: ("Drama",), 30: ("Thriller",)}

    observed_counts = [
        _resolved_features(before, user_id, movie_id, batch.timestamp, genres)[0]
        for user_id, movie_id, _ in batch.ratings
    ]
    assert observed_counts == [2, 2]

    resumed.replay_batch(batch)
    assert resumed.snapshot_state().global_moments.count == 4


def test_resume_preserves_exact_rolling_left_boundary() -> None:
    rating_events = events(
        [
            (1, 1, 10, 2.0, T - pd.Timedelta(days=30, nanoseconds=1)),
            (2, 2, 10, 3.0, T - pd.Timedelta(days=30)),
            (3, 3, 10, 4.0, T),
        ]
    )
    movies, genres = catalog()
    genre_map = {10: ("Comedy", "Drama"), 20: ("Drama",), 30: ("Thriller",)}
    prefix = CheckpointHistorySession.from_canonical_events(rating_events)
    prefix.process_next_batch(movie_genres=genre_map)
    prefix.process_next_batch(movie_genres=genre_map)
    checkpoint = prefix.checkpoint(
        checkpoint_cutoff=T,
        catalog_snapshot_id=catalog_snapshot_id(movies, genres),
    )

    resumed = resume(rating_events, movies, genres, checkpoint)
    assert resumed.snapshot_state().movie_rating_count_30d == {10: 1}
    resumed.replay_next_batch()
    assert resumed.snapshot_state().movie_rating_count_30d == {10: 2}


def test_resumed_multi_genre_state_matches_uninterrupted_processing() -> None:
    rating_events, movies, genres, checkpoint, uninterrupted = (
        checkpoint_and_uninterrupted()
    )
    resumed = resume(rating_events, movies, genres, checkpoint)
    resumed.replay_remaining()

    expected = uninterrupted.snapshot_state().user_genre_moments
    actual = resumed.snapshot_state().user_genre_moments
    assert {key: value.to_dict() for key, value in actual.items()} == {
        key: value.to_dict() for key, value in expected.items()
    }


def session_snapshot(session: CheckpointHistorySession) -> tuple[object, ...]:
    return (
        session.snapshot_state().to_dict(),
        session.processed_batch_count,
        session.processed_event_count,
        session.applied_event_ids,
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda session, batch: session.process_next_batch(movie_genres={}),
        lambda session, batch: session.process_batch(batch, movie_genres={}),
        lambda session, batch: session.replay_batch(batch, movie_genres={}),
        lambda session, batch: session.replay_next_batch(movie_genres={}),
        lambda session, batch: session.replay_batches((batch,), movie_genres={}),
        lambda session, batch: session.replay_remaining(movie_genres={}),
        lambda session, batch: session.replay_next_batch_features(movie_genres={}),
    ],
)
def test_resumed_processing_paths_cannot_override_bound_catalog(operation) -> None:
    rating_events, movies, genres, checkpoint, _ = checkpoint_and_uninterrupted()
    session = resume(rating_events, movies, genres, checkpoint)
    before = session_snapshot(session)
    batch = session.batches[session.processed_batch_count]

    with pytest.raises((TypeError, ValueError), match="overrid|unexpected keyword"):
        operation(session, batch)

    assert session_snapshot(session) == before


def test_invalid_new_session_processing_is_fully_atomic() -> None:
    rating_events = events(
        [
            (1, 1, 10, 2.0, T - pd.Timedelta(days=31)),
            (2, 2, 20, 4.0, T),
        ]
    )
    session = CheckpointHistorySession.from_canonical_events(rating_events)
    session.process_next_batch(movie_genres={10: ("Drama",)})
    before = session_snapshot(session)

    with pytest.raises(ValueError, match="explicit movie_genres"):
        session.process_next_batch(movie_genres=None)

    assert session_snapshot(session) == before


@pytest.mark.parametrize("method", ["process_next_batch", "process_batch"])
def test_resumed_processing_uses_bound_catalog_without_an_argument(method: str) -> None:
    rating_events, movies, genres, checkpoint, _ = checkpoint_and_uninterrupted()
    processed = resume(rating_events, movies, genres, checkpoint)
    replayed = resume(rating_events, movies, genres, checkpoint)

    if method == "process_next_batch":
        processed.process_next_batch()
    else:
        processed.process_batch(
            processed.batches[processed.processed_batch_count]
        )
    replayed.replay_next_batch()

    assert session_snapshot(processed) == session_snapshot(replayed)


def test_failure_after_expiration_preparation_does_not_mutate_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rating_events, movies, genres, checkpoint, _ = checkpoint_and_uninterrupted()
    session = resume(rating_events, movies, genres, checkpoint)
    before = session_snapshot(session)

    def fail_feature_resolution(*args, **kwargs):
        raise RuntimeError("simulated feature failure")

    monkeypatch.setattr(
        "src.state.checkpoints._resolve_timestamp_batch_features",
        fail_feature_resolution,
    )
    with pytest.raises(RuntimeError, match="simulated feature failure"):
        session.replay_next_batch_features()

    assert session_snapshot(session) == before


def replay_feature_source() -> pd.DataFrame:
    return events(
        [
            (1, 1, 10, 1.0, T - pd.Timedelta(days=30, nanoseconds=1)),
            (2, 1, 10, 2.0, T - pd.Timedelta(days=30)),
            (3, 1, 20, 4.0, T - pd.Timedelta(seconds=1)),
            (4, 1, 10, 5.0, T),
            (5, 3, 30, 3.5, T),
            (6, 1, 20, 4.5, T + pd.Timedelta(seconds=2)),
            (7, 4, 10, 2.5, T + pd.Timedelta(days=31)),
        ]
    )


def test_json_resume_emits_complete_features_equal_to_uninterrupted_builder() -> None:
    rating_events = replay_feature_source()
    movies, genre_rows = catalog()
    genre_map = {10: ("Comedy", "Drama"), 20: ("Drama",), 30: ("Thriller",)}
    prefix = CheckpointHistorySession.from_canonical_events(rating_events)
    for batch in prefix.batches:
        if batch.timestamp >= T:
            break
        prefix.process_batch(batch, movie_genres=genre_map)
    checkpoint = Checkpoint.from_json(
        prefix.checkpoint(
            checkpoint_cutoff=T,
            catalog_snapshot_id=catalog_snapshot_id(movies, genre_rows),
        ).to_json()
    )
    session = resume(rating_events, movies, genre_rows, checkpoint)
    replayed = []

    while session.processed_event_count < len(rating_events):
        replayed.append(session.replay_next_batch_features())

    actual = pd.concat(replayed, ignore_index=True).sort_values("ratingEventId")
    expected = build_expanding_rating_features(
        rating_events, genre_rows, movies
    ).query("timestamp >= @T").sort_values("ratingEventId")
    pdt.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )
    assert actual.loc[actual["timestamp"].eq(T), "global_rating_count"].tolist() == [
        3,
        3,
    ]
    at_cutoff = actual.loc[actual["timestamp"].eq(T)].set_index("ratingEventId")
    assert at_cutoff.loc[4, "movie_rating_count_30d"] == 1
    assert at_cutoff.loc[4, "user_target_genre_mean_delta"] != 0


def test_replay_feature_emission_advances_rolling_expiration_automatically() -> None:
    rating_events = replay_feature_source()
    movies, genre_rows = catalog()
    genre_map = {10: ("Comedy", "Drama"), 20: ("Drama",), 30: ("Thriller",)}
    prefix = CheckpointHistorySession.from_canonical_events(rating_events)
    for batch in prefix.batches:
        if batch.timestamp >= T:
            break
        prefix.process_batch(batch, movie_genres=genre_map)
    checkpoint = prefix.checkpoint(
        checkpoint_cutoff=T,
        catalog_snapshot_id=catalog_snapshot_id(movies, genre_rows),
    )
    session = resume(rating_events, movies, genre_rows, checkpoint)

    first = session.replay_next_batch_features()
    second = session.replay_next_batch_features()
    third = session.replay_next_batch_features()

    assert first.loc[first["ratingEventId"].eq(4), "movie_rating_count_30d"].item() == 1
    assert second.loc[0, "movie_rating_count_30d"] == 1
    assert third.loc[0, "movie_rating_count_30d"] == 0


@pytest.mark.parametrize("method", ["replay_batches", "replay_remaining"])
@pytest.mark.parametrize("failure_call", [2, 3])
def test_multi_batch_replay_failure_is_atomic_for_the_whole_call(
    method: str,
    failure_call: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rating_events = replay_feature_source()
    movies, genre_rows = catalog()
    checkpoint_source = CheckpointHistorySession.from_canonical_events(rating_events)
    checkpoint = checkpoint_source.checkpoint(
        checkpoint_cutoff=rating_events["timestamp"].min(),
        catalog_snapshot_id=catalog_snapshot_id(movies, genre_rows),
    )
    session = resume(rating_events, movies, genre_rows, checkpoint)
    before = session_snapshot(session)
    original = HistoricalRatingState._apply_trusted_timestamp_batch
    calls = 0

    def fail_after_prepared_update(state, *args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        original(state, *args, **kwargs)
        if calls == failure_call:
            raise RuntimeError("simulated range failure")

    monkeypatch.setattr(
        HistoricalRatingState,
        "_apply_trusted_timestamp_batch",
        fail_after_prepared_update,
    )
    with pytest.raises(RuntimeError, match="simulated range failure"):
        if method == "replay_batches":
            session.replay_batches(
                session.batches[session.processed_batch_count :]
            )
        else:
            session.replay_remaining()

    assert calls == failure_call
    assert session_snapshot(session) == before


@pytest.mark.parametrize("method", ["replay_batches", "replay_remaining"])
def test_multi_batch_replay_success_matches_individual_replay(method: str) -> None:
    rating_events = replay_feature_source()
    movies, genre_rows = catalog()
    checkpoint_source = CheckpointHistorySession.from_canonical_events(rating_events)
    checkpoint = checkpoint_source.checkpoint(
        checkpoint_cutoff=rating_events["timestamp"].min(),
        catalog_snapshot_id=catalog_snapshot_id(movies, genre_rows),
    )
    ranged = resume(rating_events, movies, genre_rows, checkpoint)
    individual = resume(rating_events, movies, genre_rows, checkpoint)

    if method == "replay_batches":
        ranged.replay_batches(ranged.batches[ranged.processed_batch_count :])
    else:
        ranged.replay_remaining()
    while individual.processed_event_count < len(rating_events):
        individual.replay_next_batch()

    assert session_snapshot(ranged) == session_snapshot(individual)
