import json
import math

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.expanding import (
    DYNAMIC_FEATURE_COLUMNS as FEATURE_COLUMNS,
    _resolved_features,
)
from src.state.moments import HistoricalRatingState
from tests.features.catalog_fixtures import build_features as build_expanding_rating_features


T = pd.Timestamp("2020-01-01 10:00:00")
GENRE_FEATURES = [
    "user_target_genre_rating_count",
    "user_target_genre_mean_rating",
    "user_target_genre_mean_delta",
]


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


def bridge(rows: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.Series([row[0] for row in rows], dtype="uint32"),
            "genreId": pd.Series([row[1] for row in rows], dtype="string"),
        }
    )


def aligned(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("ratingEventId").reset_index(drop=True)


def test_omitted_movie_genres_is_rejected() -> None:
    with pytest.raises(ValueError, match="movie_genres must be explicitly supplied"):
        build_expanding_rating_features(events([(1, 1, 10, 4.0, T)]))


def test_single_genre_count_and_mean_are_strictly_historical() -> None:
    source = events(
        [
            (1, 1, 10, 2.0, T),
            (2, 1, 20, 4.0, T + pd.Timedelta(seconds=1)),
            (3, 1, 10, 5.0, T + pd.Timedelta(seconds=2)),
        ]
    )
    result = build_expanding_rating_features(
        source, bridge([(10, "Drama"), (20, "Drama")])
    )

    assert result["user_target_genre_rating_count"].tolist() == [0, 1, 2]
    assert result["user_target_genre_mean_rating"].tolist() == pytest.approx(
        [3.5, 2.0, 3.0]
    )
    assert result["user_target_genre_mean_delta"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0]
    )


def test_multi_genre_updates_associations_but_conserves_canonical_events() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 1, 10, 2.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    movie_genres = bridge([(10, "Drama"), (10, "Romance")])
    result = build_expanding_rating_features(source, movie_genres)

    assert len(result) == len(source) == 2
    assert result["ratingEventId"].tolist() == [1, 2]
    second = result.iloc[1]
    assert second["user_target_genre_rating_count"] == 2
    assert second["user_target_genre_mean_rating"] == 4.0
    assert second["global_rating_count"] == 1
    assert second["user_rating_count"] == 1
    assert second["movie_rating_count"] == 1
    assert second["movie_rating_count_30d"] == 1


def test_target_genres_use_association_weighted_mean_and_are_order_invariant() -> None:
    source = events(
        [
            (1, 1, 10, 1.0, T),
            (2, 1, 20, 3.0, T + pd.Timedelta(seconds=1)),
            (3, 1, 30, 5.0, T + pd.Timedelta(seconds=2)),
            (4, 1, 40, 1.0, T + pd.Timedelta(seconds=3)),
            (5, 1, 50, 4.0, T + pd.Timedelta(seconds=4)),
        ]
    )
    movie_genres = bridge(
        [
            (10, "Drama"),
            (20, "Drama"),
            (30, "Romance"),
            (40, "Comedy"),
            (50, "Drama"),
            (50, "Romance"),
        ]
    )
    result = build_expanding_rating_features(source, movie_genres)
    target = result.iloc[-1]

    # Drama has ratings [1, 3], Romance has [5]: association-weighted mean is
    # 3.0, not the unweighted mean of per-genre means (2.0 + 5.0) / 2 = 3.5.
    assert target["user_target_genre_rating_count"] == 3
    assert target["user_target_genre_mean_rating"] == 3.0
    assert target["user_mean_rating"] == 2.5
    assert target["user_target_genre_mean_delta"] == 0.5

    reversed_bridge = movie_genres.iloc[::-1].reset_index(drop=True)
    pdt.assert_frame_equal(
        aligned(result),
        aligned(build_expanding_rating_features(source, reversed_bridge)),
    )


def test_same_timestamp_genre_events_share_one_pre_batch_snapshot() -> None:
    source = events(
        [
            (1, 1, 10, 2.0, T),
            (2, 1, 20, 4.0, T + pd.Timedelta(seconds=1)),
            (3, 1, 30, 5.0, T + pd.Timedelta(seconds=1)),
            (4, 1, 10, 3.0, T + pd.Timedelta(seconds=2)),
        ]
    )
    result = build_expanding_rating_features(
        source,
        bridge([(10, "Drama"), (20, "Drama"), (30, "Drama")]),
    )

    tied = result.iloc[1:3]
    assert tied["user_target_genre_rating_count"].tolist() == [1, 1]
    assert tied["user_target_genre_mean_rating"].tolist() == [2.0, 2.0]
    assert result.loc[3, "user_target_genre_rating_count"] == 3
    assert result.loc[3, "user_target_genre_mean_rating"] == pytest.approx(11 / 3)


def test_cold_starts_unrelated_genres_and_missing_genres_follow_user_fallback() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 1, 20, 2.0, T + pd.Timedelta(seconds=1)),
            (3, 2, 20, 5.0, T + pd.Timedelta(seconds=2)),
            (4, 1, 30, 3.0, T + pd.Timedelta(seconds=3)),
            (5, 1, 10, 1.0, T + pd.Timedelta(seconds=4)),
        ]
    )
    result = build_expanding_rating_features(
        source, bridge([(10, "Comedy"), (20, "Drama")])
    )

    # Known user, unseen Drama: fallback is that user's historical mean (4.0).
    assert result.loc[1, GENRE_FEATURES].tolist() == [0, 4.0, 0.0]
    # Unseen user: user mean and genre mean both use the pre-batch global mean.
    assert result.loc[2, "user_target_genre_rating_count"] == 0
    assert result.loc[2, "user_target_genre_mean_rating"] == 3.0
    assert result.loc[2, "user_target_genre_mean_delta"] == 0.0
    # Movie 30 has no bridge row: no updates and the same user-mean fallback.
    assert result.loc[3, "user_target_genre_rating_count"] == 0
    assert result.loc[3, "user_target_genre_mean_rating"] == 3.0
    assert result.loc[3, "user_target_genre_mean_delta"] == 0.0
    # Drama history never contaminates the independent Comedy state.
    assert result.loc[4, "user_target_genre_rating_count"] == 1
    assert result.loc[4, "user_target_genre_mean_rating"] == 4.0


def test_mixed_target_support_ignores_unsupported_genre_fallback() -> None:
    source = events(
        [
            (1, 1, 10, 5.0, T),
            (2, 1, 20, 1.0, T + pd.Timedelta(seconds=1)),
            (3, 1, 30, 3.0, T + pd.Timedelta(seconds=2)),
        ]
    )
    result = build_expanding_rating_features(
        source,
        bridge([(10, "Drama"), (20, "Comedy"), (30, "Drama"), (30, "Romance")]),
    )
    target = result.iloc[-1]

    assert target["user_mean_rating"] == 3.0
    assert target["user_target_genre_rating_count"] == 1
    assert target["user_target_genre_mean_rating"] == 5.0
    assert target["user_target_genre_mean_delta"] == 2.0


def test_distinct_duplicate_events_each_update_every_genre_association() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 1, 10, 4.0, T),
            (3, 1, 20, 2.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    result = build_expanding_rating_features(
        source,
        bridge(
            [(10, "Drama"), (10, "Romance"), (20, "Drama"), (20, "Romance")]
        ),
    )

    assert result.loc[2, "global_rating_count"] == 2
    assert result.loc[2, "user_rating_count"] == 2
    assert result.loc[2, "user_target_genre_rating_count"] == 4
    assert result.loc[2, "user_target_genre_mean_rating"] == 4.0


def test_all_dynamic_families_resolve_from_same_snapshot_and_shuffle_invariant() -> None:
    source = events(
        [
            (1, 1, 10, 2.0, T),
            (2, 1, 20, 4.0, T + pd.Timedelta(seconds=10)),
            (3, 1, 30, 5.0, T + pd.Timedelta(seconds=10)),
            (4, 1, 10, 3.0, T + pd.Timedelta(seconds=20)),
        ]
    )
    movie_genres = bridge(
        [
            (10, "Drama"),
            (20, "Drama"),
            (20, "Romance"),
            (30, "Drama"),
        ]
    )
    expected = aligned(build_expanding_rating_features(source, movie_genres))
    tied = expected.loc[expected["ratingEventId"].isin([2, 3])]

    assert tied["global_rating_count"].tolist() == [1, 1]
    assert tied["user_rating_count"].tolist() == [1, 1]
    assert tied["user_seconds_since_last_rating"].tolist() == [10.0, 10.0]
    assert tied["movie_rating_count_30d"].tolist() == [0, 0]
    assert tied["user_target_genre_rating_count"].tolist() == [1, 1]
    assert tied["user_target_genre_mean_rating"].tolist() == [2.0, 2.0]

    for seed in range(5):
        shuffled_events = source.sample(frac=1, random_state=seed).reset_index(drop=True)
        shuffled_genres = movie_genres.sample(
            frac=1, random_state=seed + 20
        ).reset_index(drop=True)
        pdt.assert_frame_equal(
            aligned(build_expanding_rating_features(shuffled_events, shuffled_genres)),
            expected,
        )


def test_randomized_genre_features_match_direct_strict_history_reference() -> None:
    rng = np.random.default_rng(20260912)
    movie_genres = bridge(
        [
            (10, "Drama"),
            (10, "Romance"),
            (20, "Comedy"),
            (30, "Drama"),
            (30, "Thriller"),
            (40, "Romance"),
        ]
    )
    genre_sets = {
        movie_id: set(group["genreId"].astype(str))
        for movie_id, group in movie_genres.groupby("movieId")
    }
    rows = []
    for event_id in range(1, 51):
        rows.append(
            (
                event_id,
                int(rng.integers(1, 5)),
                int(rng.choice([10, 20, 30, 40, 50])),
                float(rng.choice(np.arange(0.5, 5.5, 0.5))),
                T + pd.Timedelta(seconds=int(rng.integers(0, 12))),
            )
        )
    source = events(rows)
    result = build_expanding_rating_features(source, movie_genres)

    for position, event in source.iterrows():
        prior = source.loc[source["timestamp"].lt(event["timestamp"])]
        prior_user = prior.loc[prior["userId"].eq(event["userId"])]
        prior_ratings = [float(value) for value in prior["rating"]]
        user_ratings = [float(value) for value in prior_user["rating"]]
        global_mean = (
            math.fsum(prior_ratings) / len(prior_ratings) if prior_ratings else 3.5
        )
        user_mean = (
            math.fsum(user_ratings) / len(user_ratings)
            if user_ratings
            else global_mean
        )
        target_genres = genre_sets.get(int(event["movieId"]), set())
        association_ratings = []
        for historical in prior_user.itertuples(index=False):
            overlap = target_genres.intersection(
                genre_sets.get(int(historical.movieId), set())
            )
            association_ratings.extend([float(historical.rating)] * len(overlap))
        expected_count = len(association_ratings)
        expected_mean = (
            math.fsum(association_ratings) / expected_count
            if association_ratings
            else user_mean
        )

        assert (
            result.loc[position, "user_target_genre_rating_count"]
            == expected_count
        )
        assert result.loc[
            position, "user_target_genre_mean_rating"
        ] == pytest.approx(expected_mean, abs=1e-6)
        assert result.loc[position, "user_target_genre_mean_delta"] == pytest.approx(
            expected_mean - user_mean, abs=1e-6
        )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(10, "")], "must not be empty"),
        ([(10, "(no genres listed)")], "sentinel is not a genre"),
        ([(10, "Drama"), (10, "Drama")], "must be unique"),
    ],
)
def test_noncanonical_genre_bridge_rows_are_rejected(
    rows: list[tuple[int, str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_expanding_rating_features(events([(1, 1, 10, 4.0, T)]), bridge(rows))


def test_genre_feature_dtypes_match_dictionary() -> None:
    result = build_expanding_rating_features(
        events([(1, 1, 10, 4.0, T)]), bridge([(10, "Drama")])
    )
    assert str(result["user_target_genre_rating_count"].dtype) == "uint64"
    assert str(result["user_target_genre_mean_rating"].dtype) == "float32"
    assert str(result["user_target_genre_mean_delta"].dtype) == "float32"


def test_json_restore_then_continue_preserves_user_genre_features() -> None:
    genres_by_movie = {
        10: ("Drama", "Romance"),
        20: ("Drama",),
        30: ("Drama", "Thriller"),
    }
    uninterrupted = HistoricalRatingState()
    uninterrupted.apply_timestamp_batch(
        [(1, 10, 5.0), (2, 20, 2.0)],
        timestamp=T,
        movie_genres=genres_by_movie,
    )
    restored = HistoricalRatingState.from_dict(
        json.loads(json.dumps(uninterrupted.to_dict()))
    )

    next_timestamp = T + pd.Timedelta(seconds=1)
    expected = _resolved_features(
        uninterrupted, 1, 30, next_timestamp, genres_by_movie
    )
    actual = _resolved_features(restored, 1, 30, next_timestamp, genres_by_movie)
    assert dict(zip(FEATURE_COLUMNS, actual, strict=True)) == dict(
        zip(FEATURE_COLUMNS, expected, strict=True)
    )
    assert actual[-3:] == pytest.approx((1, 5.0, 0.0))

    tied_batch = [(1, 20, 1.0), (1, 30, 3.0)]
    uninterrupted.apply_timestamp_batch(
        tied_batch,
        timestamp=next_timestamp,
        movie_genres=genres_by_movie,
    )
    restored.apply_timestamp_batch(
        reversed(tied_batch),
        timestamp=next_timestamp,
        movie_genres=genres_by_movie,
    )
    assert restored.to_dict() == uninterrupted.to_dict()

    final_timestamp = T + pd.Timedelta(seconds=2)
    assert _resolved_features(
        restored, 1, 10, final_timestamp, genres_by_movie
    ) == _resolved_features(
        uninterrupted, 1, 10, final_timestamp, genres_by_movie
    )
