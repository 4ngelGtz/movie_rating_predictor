from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.catalog import build_catalog_lookups, catalog_snapshot_id
from src.features.expanding import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    build_expanding_rating_features,
)
from src.state.checkpoints import Checkpoint, CheckpointHistorySession


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
            "movieId": pd.Series([10, 20, 30, 40, 50, 60, 70], dtype="uint32"),
            "title": pd.Series(
                ["Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy"],
                dtype="string",
            ),
            "releaseYear": pd.Series(
                [2000, 2025, pd.NA, 1990, 2010, 2015, 2005], dtype="UInt16"
            ),
            "genreStatus": pd.Series(
                [
                    "listed",
                    "listed",
                    "listed",
                    "missing_in_source",
                    "listed",
                    "listed",
                    "listed",
                ],
                dtype="string",
            ),
            "imdbId": pd.Series([100, 200, 300, 400, 500, 600, 700], dtype="UInt32"),
            "tmdbId": pd.Series(
                [1000, 2000, 3000, 4000, 5000, 6000, 7000], dtype="UInt32"
            ),
        }
    )
    genres = pd.DataFrame(
        {
            "movieId": pd.Series(
                [10, 20, 20, 30, 50, 50, 60, 70, 70], dtype="uint32"
            ),
            "genreId": pd.Series(
                [
                    "Drama",
                    "Comedy",
                    "Drama",
                    "Romance",
                    "Drama",
                    "Thriller",
                    "Comedy",
                    "Romance",
                    "Sci-Fi",
                ],
                dtype="string",
            ),
        }
    )
    return movies, genres


def comprehensive_history() -> pd.DataFrame:
    return events(
        [
            (1, 1, 10, 1.0, T - pd.Timedelta(days=30, nanoseconds=1)),
            (2, 1, 10, 2.0, T - pd.Timedelta(days=30)),
            (3, 2, 10, 3.0, T - pd.Timedelta(days=30)),
            (4, 1, 50, 5.0, T - pd.Timedelta(days=15)),
            (5, 1, 20, 4.0, T - pd.Timedelta(seconds=1)),
            (6, 1, 30, 4.5, T),
            (7, 3, 40, 3.5, T),
            (8, 3, 40, 3.5, T),
            (14, 6, 10, 2.0, T),
            (9, 1, 20, 2.5, T + pd.Timedelta(nanoseconds=1)),
            (10, 4, 60, 5.0, T + pd.Timedelta(days=10)),
            (11, 1, 70, 1.5, T + pd.Timedelta(days=30)),
            (12, 5, 10, 4.0, T + pd.Timedelta(days=30)),
            (13, 2, 50, 3.0, T + pd.Timedelta(days=30, nanoseconds=1)),
        ]
    )


def aligned(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("ratingEventId").reset_index(drop=True)


def shuffled(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.sample(frac=1, random_state=seed).copy()
    result.index = np.arange(seed * 100 + 7, seed * 100 + 7 + len(result))
    return result


def assert_equivalent_at_cutoff(
    rating_events: pd.DataFrame,
    movies: pd.DataFrame,
    movie_genres: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, CheckpointHistorySession, Checkpoint]:
    """Exercise checkpoint -> JSON -> resume -> feature replay at one cutoff."""
    catalog_lookups = build_catalog_lookups(
        movies, movie_genres, required_movie_ids=rating_events["movieId"]
    )
    snapshot_id = catalog_snapshot_id(movies, movie_genres)
    uninterrupted_features = build_expanding_rating_features(
        rating_events, movie_genres, movies
    )

    prefix = CheckpointHistorySession.from_canonical_events(rating_events)
    for batch in prefix.batches:
        if batch.timestamp >= cutoff:
            break
        prefix.process_batch(
            batch, movie_genres=catalog_lookups.genres_by_movie
        )
    checkpoint = Checkpoint.from_json(
        prefix.checkpoint(
            checkpoint_cutoff=cutoff,
            catalog_snapshot_id=snapshot_id,
        ).to_json()
    )
    resumed = CheckpointHistorySession.resume_from_checkpoint(
        checkpoint=checkpoint,
        rating_events=rating_events,
        canonical_movies=movies,
        movie_genres=movie_genres,
    )

    expected_prefix_ids = frozenset(
        int(value)
        for value in rating_events.loc[
            rating_events["timestamp"].lt(cutoff), "ratingEventId"
        ]
    )
    expected_prefix_batches = rating_events.loc[
        rating_events["timestamp"].lt(cutoff), "timestamp"
    ].nunique()
    assert checkpoint.metadata.history_source_id == prefix.history_source_id
    assert checkpoint.metadata.catalog_snapshot_id == snapshot_id
    assert checkpoint.metadata.checkpoint_cutoff == cutoff
    assert checkpoint.metadata.history_event_count == len(rating_events)
    assert checkpoint.metadata.processed_event_count == len(expected_prefix_ids)
    assert checkpoint.metadata.processed_batch_count == expected_prefix_batches
    assert resumed.history_source_id == checkpoint.metadata.history_source_id
    assert resumed.processed_event_count == len(expected_prefix_ids)
    assert resumed.processed_batch_count == expected_prefix_batches
    assert resumed.applied_event_ids == expected_prefix_ids

    replayed_batches = []
    while resumed.processed_event_count < len(rating_events):
        replayed_batches.append(resumed.replay_next_batch_features())
    expected = uninterrupted_features.loc[
        uninterrupted_features["timestamp"].ge(cutoff)
    ]
    actual = (
        pd.concat(replayed_batches, ignore_index=True)
        if replayed_batches
        else expected.iloc[0:0].copy()
    )
    pdt.assert_frame_equal(aligned(actual), aligned(expected), check_exact=True)

    expected_suffix_ids = rating_events.loc[
        rating_events["timestamp"].ge(cutoff), "ratingEventId"
    ].astype(int)
    assert len(actual) == len(expected_suffix_ids)
    assert actual["ratingEventId"].is_unique
    assert set(actual["ratingEventId"].astype(int)) == set(expected_suffix_ids)
    assert tuple(actual.columns) == (*CONTEXT_COLUMNS, *FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 17

    uninterrupted_state = CheckpointHistorySession.from_canonical_events(
        rating_events
    )
    while uninterrupted_state.processed_event_count < len(rating_events):
        uninterrupted_state.process_next_batch(
            movie_genres=catalog_lookups.genres_by_movie
        )
    common_state_cutoff = max(cutoff, rating_events["timestamp"].max())
    expected_state = uninterrupted_state.snapshot_state()
    expected_state.expire_movie_activity(common_state_cutoff)
    actual_state = resumed.snapshot_state()
    actual_state.expire_movie_activity(common_state_cutoff)
    assert actual_state.to_dict() == expected_state.to_dict()
    assert resumed.processed_event_count == len(rating_events)
    assert resumed.processed_batch_count == rating_events["timestamp"].nunique()
    assert resumed.applied_event_ids == frozenset(
        rating_events["ratingEventId"].astype(int)
    )
    return actual, resumed, checkpoint


CUTOFFS = (
    pytest.param(T - pd.Timedelta(days=30, nanoseconds=2), id="before-first"),
    pytest.param(
        T - pd.Timedelta(days=30, nanoseconds=1), id="exactly-first-timestamp"
    ),
    pytest.param(T - pd.Timedelta(days=30), id="timestamp-boundary"),
    pytest.param(
        T - pd.Timedelta(days=30) + pd.Timedelta(nanoseconds=1),
        id="between-timestamps",
    ),
    pytest.param(T - pd.Timedelta(nanoseconds=1), id="before-expiration-point"),
    pytest.param(T, id="tied-batch-and-window-boundary"),
    pytest.param(T + pd.Timedelta(nanoseconds=1), id="after-tied-batch"),
    pytest.param(T + pd.Timedelta(days=30, nanoseconds=1), id="near-final"),
    pytest.param(T + pd.Timedelta(days=30, nanoseconds=2), id="after-all"),
)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_all_features_and_final_state_match_across_cutoff_classes(
    cutoff: pd.Timestamp,
) -> None:
    movies, movie_genres = catalog()
    assert_equivalent_at_cutoff(
        comprehensive_history(), movies, movie_genres, cutoff
    )


def test_strictly_before_first_cutoff_replays_complete_history() -> None:
    rating_events = comprehensive_history()
    movies, movie_genres = catalog()
    cutoff = rating_events["timestamp"].min() - pd.Timedelta(nanoseconds=1)

    replayed, resumed, checkpoint = assert_equivalent_at_cutoff(
        rating_events, movies, movie_genres, cutoff
    )

    assert checkpoint.metadata.checkpoint_cutoff == cutoff
    assert checkpoint.metadata.processed_event_count == 0
    assert checkpoint.metadata.processed_batch_count == 0
    assert len(replayed) == len(rating_events)
    assert resumed.processed_event_count == len(rating_events)
    assert resumed.processed_batch_count == rating_events["timestamp"].nunique()
    assert resumed.applied_event_ids == frozenset(
        rating_events["ratingEventId"].astype(int)
    )


def test_json_resume_replay_preserves_subsecond_recency_precision() -> None:
    rating_events = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 1, 10, 4.0, T + pd.Timedelta(nanoseconds=1)),
            (3, 1, 10, 4.0, T + pd.Timedelta(nanoseconds=1_001)),
            (4, 1, 10, 4.0, T + pd.Timedelta(nanoseconds=1_001_001)),
            (5, 1, 10, 4.0, T + pd.Timedelta(nanoseconds=1_001_001_001)),
        ]
    )
    movies, movie_genres = catalog()

    replayed, _, _ = assert_equivalent_at_cutoff(
        rating_events, movies, movie_genres, T + pd.Timedelta(nanoseconds=1)
    )

    recency_by_event = replayed.set_index("ratingEventId")[
        "user_seconds_since_last_rating"
    ]
    assert recency_by_event.loc[[2, 3, 4, 5]].tolist() == pytest.approx(
        [1e-9, 1e-6, 1e-3, 1.0]
    )
    assert recency_by_event.loc[2] == pytest.approx(1e-9)
    assert str(replayed["user_seconds_since_last_rating"].dtype) == "float64"


def test_boundary_cold_start_multigenre_and_static_values_are_hand_checkable() -> None:
    rating_events = comprehensive_history()
    movies, movie_genres = catalog()
    replayed, _, _ = assert_equivalent_at_cutoff(
        rating_events, movies, movie_genres, T
    )
    rows = replayed.set_index("ratingEventId")

    # The event one nanosecond before the left edge has expired; both events
    # exactly at t - 30 days remain in [t - 30 days, t).
    assert rows.loc[6, "movie_rating_count_30d"] == 0
    assert rows.loc[14, "movie_rating_count_30d"] == 2
    uninterrupted = build_expanding_rating_features(
        rating_events, movie_genres, movies
    ).set_index("ratingEventId")
    assert uninterrupted.loc[1, "global_rating_count"] == 0
    assert uninterrupted.loc[1, "global_mean_rating"] == 3.5
    assert uninterrupted.loc[1, "user_mean_rating"] == 3.5
    assert uninterrupted.loc[1, "movie_mean_rating"] == 3.5
    assert uninterrupted.loc[6, "global_rating_count"] == 5
    tied = rows.loc[[6, 7, 8, 14]]
    assert tied["global_rating_count"].tolist() == [5, 5, 5, 5]

    # User 3, movie 40, and its missing genre are simultaneous cold starts.
    assert rows.loc[7, "user_rating_count"] == 0
    assert rows.loc[7, "movie_rating_count"] == 0
    assert rows.loc[7, "movie_genre_count"] == 0
    assert rows.loc[7, "user_target_genre_rating_count"] == 0
    assert rows.loc[7, "user_target_genre_mean_delta"] == 0.0

    # Before event 5, movie 20 has Drama support but no Comedy support for user
    # 1. Only the three supported Drama associations contribute. After event 5,
    # its rating contributes once to each of Drama and Comedy, giving five
    # association observations rather than four canonical events.
    assert uninterrupted.loc[5, "user_target_genre_rating_count"] == 3
    assert uninterrupted.loc[
        5, "user_target_genre_mean_rating"
    ] == pytest.approx(8 / 3)
    assert rows.loc[9, "user_target_genre_rating_count"] == 5
    assert rows.loc[9, "user_target_genre_mean_rating"] == pytest.approx(16 / 5)
    assert rows.loc[9, "movie_genre_count"] == 2
    assert rows.loc[9, "movie_release_year"] == 2025
    assert rows.loc[9, "movie_age_years"] < 0

    assert pd.isna(rows.loc[6, "movie_release_year"])
    assert bool(rows.loc[6, "movie_release_year_missing"])
    # Distinct canonical IDs conserve duplicate-looking events.
    assert (
        rows.loc[7, "user_rating_count"]
        == rows.loc[8, "user_rating_count"]
        == 0
    )


def test_physical_event_index_and_catalog_order_do_not_change_equivalence() -> None:
    rating_events = comprehensive_history()
    movies, movie_genres = catalog()
    expected_features = aligned(
        build_expanding_rating_features(rating_events, movie_genres, movies)
    )
    expected_source_id = CheckpointHistorySession.from_canonical_events(
        rating_events
    ).history_source_id
    expected_catalog_id = catalog_snapshot_id(movies, movie_genres)

    for seed in range(4):
        ordered_events = shuffled(rating_events, seed)
        ordered_movies = shuffled(movies, seed + 10)
        ordered_genres = shuffled(movie_genres, seed + 20)
        pdt.assert_frame_equal(
            aligned(
                build_expanding_rating_features(
                    ordered_events, ordered_genres, ordered_movies
                )
            ),
            expected_features,
            check_exact=True,
        )
        _, resumed, checkpoint = assert_equivalent_at_cutoff(
            ordered_events, ordered_movies, ordered_genres, T
        )
        assert resumed.history_source_id == expected_source_id
        assert checkpoint.metadata.catalog_snapshot_id == expected_catalog_id


def randomized_case(seed: int, event_count: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    offsets = np.array(
        [
            -2_592_000_000_000_001,
            -2_592_000_000_000_000,
            -7,
            0,
            0,
            1,
            86_400_000_000_000,
        ],
        dtype=np.int64,
    )
    rows = []
    for event_id in range(1, event_count + 1):
        rows.append(
            (
                event_id,
                int(rng.integers(1, 7)),
                int(rng.choice([10, 20, 30, 40, 50, 60, 70])),
                float(rng.integers(1, 11) / 2),
                T + pd.Timedelta(nanoseconds=int(rng.choice(offsets))),
            )
        )
    return events(rows)


@pytest.mark.parametrize("seed", [11, 29, 47, 83, 101])
def test_randomized_uninterrupted_vs_json_resume_replay_matrix(seed: int) -> None:
    rating_events = shuffled(randomized_case(seed), seed)
    movies, movie_genres = catalog()
    movies = shuffled(movies, seed + 1)
    movie_genres = shuffled(movie_genres, seed + 2)
    cutoffs = (
        rating_events["timestamp"].min(),
        T,
        rating_events["timestamp"].max() + pd.Timedelta(nanoseconds=1),
    )
    for cutoff in cutoffs:
        assert_equivalent_at_cutoff(
            rating_events, movies, movie_genres, pd.Timestamp(cutoff)
        )
