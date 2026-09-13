import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from tests.features.catalog_fixtures import build_features as build_expanding_rating_features


T = pd.Timestamp("2020-01-01 10:00:00")
EMPTY_MOVIE_GENRES = pd.DataFrame(
    {
        "movieId": pd.Series(dtype="uint32"),
        "genreId": pd.Series(dtype="string"),
    }
)


def events(rows: list[tuple[int, int, int, float, pd.Timestamp]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ratingEventId": pd.Series([row[0] for row in rows], dtype="UInt64"),
            "userId": pd.Series([row[1] for row in rows], dtype="uint32"),
            "movieId": pd.Series([row[2] for row in rows], dtype="uint32"),
            "rating": pd.Series([row[3] for row in rows], dtype="float32"),
            "timestamp": pd.to_datetime([row[4] for row in rows]),
        }
    )


def aligned(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("ratingEventId").reset_index(drop=True)


def test_user_recency_basic_sequence_cold_start_and_isolation() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 2, 10, 3.0, T + pd.Timedelta(seconds=10)),
            (3, 1, 20, 5.0, T + pd.Timedelta(seconds=90)),
            (4, 1, 30, 2.0, T + pd.Timedelta(days=40)),
        ]
    )
    result = build_expanding_rating_features(source, EMPTY_MOVIE_GENRES)

    assert np.isnan(result.loc[0, "user_seconds_since_last_rating"])
    assert np.isnan(result.loc[1, "user_seconds_since_last_rating"])
    assert result.loc[2, "user_seconds_since_last_rating"] == 90.0
    assert result.loc[3, "user_seconds_since_last_rating"] == 40 * 86_400 - 90


def test_same_timestamp_user_events_share_previous_rating_time() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 1, 20, 3.0, T + pd.Timedelta(seconds=60)),
            (3, 1, 30, 5.0, T + pd.Timedelta(seconds=60)),
            (4, 1, 40, 2.0, T + pd.Timedelta(seconds=75)),
        ]
    )
    result = build_expanding_rating_features(source, EMPTY_MOVIE_GENRES)

    assert result["user_seconds_since_last_rating"].tolist() == pytest.approx(
        [np.nan, 60.0, 60.0, 15.0], nan_ok=True
    )


def test_movie_activity_boundaries_are_exact() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 2, 20, 4.0, T),
            (3, 3, 10, 4.0, T + pd.Timedelta(days=30)),
            (4, 4, 20, 4.0, T + pd.Timedelta(days=30, seconds=1)),
        ]
    )
    result = build_expanding_rating_features(source, EMPTY_MOVIE_GENRES)

    assert result["movie_rating_count_30d"].tolist() == [0, 0, 1, 0]


def test_tied_movie_events_do_not_count_each_other_then_all_enter_window() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 2, 10, 3.0, T + pd.Timedelta(seconds=1)),
            (3, 3, 10, 5.0, T + pd.Timedelta(seconds=1)),
            (4, 4, 10, 2.0, T + pd.Timedelta(seconds=2)),
            (5, 5, 20, 4.0, T + pd.Timedelta(seconds=2)),
        ]
    )
    result = build_expanding_rating_features(source, EMPTY_MOVIE_GENRES)

    assert result["movie_rating_count_30d"].tolist() == [0, 1, 1, 3, 0]


def test_duplicate_canonical_movie_events_both_enter_future_activity() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 1, 10, 4.0, T),
            (3, 2, 10, 3.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    result = build_expanding_rating_features(source, EMPTY_MOVIE_GENRES)

    assert result["movie_rating_count_30d"].tolist() == [0, 0, 2]


def test_tied_input_order_does_not_change_recency_or_activity() -> None:
    source = events(
        [
            (1, 1, 10, 2.0, T),
            (2, 1, 10, 4.0, T + pd.Timedelta(seconds=1)),
            (3, 1, 10, 5.0, T + pd.Timedelta(seconds=1)),
            (4, 2, 20, 3.0, T + pd.Timedelta(seconds=1)),
            (5, 1, 10, 3.0, T + pd.Timedelta(seconds=2)),
        ]
    )
    expected = aligned(build_expanding_rating_features(source, EMPTY_MOVIE_GENRES))

    for seed in range(5):
        shuffled = source.sample(frac=1, random_state=seed).reset_index(drop=True)
        pdt.assert_frame_equal(
            aligned(build_expanding_rating_features(shuffled, EMPTY_MOVIE_GENRES)),
            expected,
        )


def test_expanding_and_temporal_features_share_one_pre_batch_snapshot() -> None:
    source = events(
        [
            (1, 1, 10, 1.0, T),
            (2, 1, 10, 3.0, T + pd.Timedelta(seconds=10)),
            (3, 1, 10, 5.0, T + pd.Timedelta(seconds=10)),
            (4, 1, 10, 4.0, T + pd.Timedelta(seconds=20)),
        ]
    )
    result = build_expanding_rating_features(source, EMPTY_MOVIE_GENRES)
    tied = result.iloc[1:3]
    future = result.iloc[3]

    assert tied["user_rating_count"].tolist() == [1, 1]
    assert tied["movie_rating_count"].tolist() == [1, 1]
    assert tied["user_seconds_since_last_rating"].tolist() == [10.0, 10.0]
    assert tied["movie_rating_count_30d"].tolist() == [1, 1]
    assert future["global_rating_count"] == 3
    assert future["user_rating_count"] == 3
    assert future["user_mean_rating"] == 3.0
    assert future["user_rating_std_pop"] == pytest.approx(np.sqrt(8.0 / 3.0))
    assert future["movie_rating_count"] == 3
    assert future["movie_rating_count_30d"] == 3
    assert future["user_seconds_since_last_rating"] == 10.0


def test_phase_4b_feature_dtypes_match_dictionary() -> None:
    result = build_expanding_rating_features(
        events([(1, 1, 10, 4.0, T)]), EMPTY_MOVIE_GENRES
    )
    assert str(result["user_seconds_since_last_rating"].dtype) == "float64"
    assert str(result["movie_rating_count_30d"].dtype) == "uint64"


def test_temporal_state_is_sparse_for_large_entity_ids() -> None:
    largest_uint32 = 2**32 - 1
    source = events(
        [
            (1, largest_uint32, largest_uint32, 4.0, T),
            (2, largest_uint32, largest_uint32, 5.0, T + pd.Timedelta(seconds=7)),
        ]
    )
    result = build_expanding_rating_features(source, EMPTY_MOVIE_GENRES)

    assert result.loc[1, "user_seconds_since_last_rating"] == 7.0
    assert result.loc[1, "movie_rating_count_30d"] == 1
