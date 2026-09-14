from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.genome import GENOME_FEATURE_COLUMNS, build_genome_features


T = pd.Timestamp("2020-01-01 10:00:00")


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


def scores(vectors: dict[int, list[float]]) -> pd.DataFrame:
    vectors = {
        movie_id: [*vector, *([0.0] * (10 - len(vector)))]
        for movie_id, vector in vectors.items()
    }
    rows = [
        (movie_id, tag_id, relevance)
        for movie_id, vector in vectors.items()
        for tag_id, relevance in enumerate(vector, start=1)
    ]
    return pd.DataFrame(
        {
            "movieId": pd.Series([row[0] for row in rows], dtype="uint32"),
            "tagId": pd.Series([row[1] for row in rows], dtype="uint16"),
            "relevance": pd.Series([row[2] for row in rows], dtype="float32"),
        }
    )


def by_id(frame: pd.DataFrame, event_id: int) -> pd.Series:
    return frame.loc[frame["ratingEventId"].eq(event_id)].iloc[0]


def test_strict_prior_centroids_margin_and_liked_similarities() -> None:
    source = events(
        [
            (99, 1, 10, 5.0, T),
            (1, 1, 20, 2.0, T),
            (50, 1, 30, 4.0, T + pd.Timedelta(seconds=1)),
            (2, 1, 10, 3.0, T + pd.Timedelta(seconds=2)),
        ]
    )
    result = build_genome_features(source, scores({10: [1, 0], 20: [0, 1], 30: [1, 1]}))

    # Both events at T see empty history regardless of ratingEventId ordering.
    assert by_id(result, 99)[list(GENOME_FEATURE_COLUMNS[:5])].isna().all()
    assert by_id(result, 1)[list(GENOME_FEATURE_COLUMNS[:5])].isna().all()

    target = by_id(result, 50)
    root_half = 1 / np.sqrt(2)
    assert target["genome_user_positive_cosine"] == pytest.approx(root_half)
    assert target["genome_user_negative_cosine"] == pytest.approx(root_half)
    assert target["genome_preference_margin"] == pytest.approx(0.0)
    assert target["genome_nearest_liked_similarity"] == pytest.approx(root_half)
    assert target["genome_top5_liked_similarity"] == pytest.approx(root_half)

    later = by_id(result, 2)
    assert later["genome_user_positive_cosine"] == pytest.approx(2 / np.sqrt(5))
    assert later["genome_user_negative_cosine"] == pytest.approx(0.0)
    assert later["genome_preference_margin"] == pytest.approx(2 / np.sqrt(5))
    assert later["genome_nearest_liked_similarity"] == pytest.approx(1.0)
    assert later["genome_top5_liked_similarity"] == pytest.approx((1 + root_half) / 2)


def test_top5_uses_five_largest_and_fewer_than_five_uses_all() -> None:
    vectors = {1: [1.0, 0.0]}
    for movie_id, second in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], start=10):
        vectors[movie_id] = [1.0, second]
    rows = [
        (movie_id, 1, movie_id, 5.0, T + pd.Timedelta(seconds=movie_id - 10))
        for movie_id in range(10, 16)
    ]
    rows.append((100, 1, 1, 3.0, T + pd.Timedelta(seconds=10)))
    result = build_genome_features(events(rows), scores(vectors))
    target = by_id(result, 100)
    similarities = np.array([1 / np.sqrt(1 + second**2) for second in [0, .2, .4, .6, .8, 1]])
    assert target["genome_nearest_liked_similarity"] == pytest.approx(similarities.max())
    assert target["genome_top5_liked_similarity"] == pytest.approx(
        np.sort(similarities)[-5:].mean()
    )


def test_static_statistics_are_population_values_and_top10_mean() -> None:
    vector = [value / 11 for value in range(12)]
    result = build_genome_features(events([(1, 1, 10, 3.0, T)]), scores({10: vector})).iloc[0]
    assert result["genome_movie_relevance_mean"] == pytest.approx(np.mean(vector))
    assert result["genome_movie_relevance_std"] == pytest.approx(np.std(vector, ddof=0))
    assert result["genome_movie_top10_mean"] == pytest.approx(np.mean(sorted(vector)[-10:]))


def test_user_isolation_and_missing_positive_or_negative_history() -> None:
    source = events(
        [
            (1, 1, 10, 5.0, T),
            (2, 2, 20, 2.0, T),
            (3, 1, 30, 3.0, T + pd.Timedelta(seconds=1)),
            (4, 2, 30, 3.0, T + pd.Timedelta(seconds=1)),
            (5, 3, 30, 3.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    result = build_genome_features(source, scores({10: [1, 0], 20: [0, 1], 30: [1, 1]}))
    user1, user2, user3 = (by_id(result, value) for value in (3, 4, 5))
    assert np.isfinite(user1["genome_user_positive_cosine"])
    assert pd.isna(user1["genome_user_negative_cosine"])
    assert pd.isna(user1["genome_preference_margin"])
    assert pd.isna(user2["genome_user_positive_cosine"])
    assert np.isfinite(user2["genome_user_negative_cosine"])
    assert user2[["genome_nearest_liked_similarity", "genome_top5_liked_similarity"]].isna().all()
    assert user3[list(GENOME_FEATURE_COLUMNS[:5])].isna().all()


def test_missing_or_zero_norm_genome_vectors_remain_missing_and_do_not_update_history() -> None:
    source = events(
        [
            (1, 1, 99, 5.0, T),
            (2, 1, 40, 5.0, T + pd.Timedelta(seconds=1)),
            (3, 1, 10, 3.0, T + pd.Timedelta(seconds=2)),
        ]
    )
    result = build_genome_features(source, scores({10: [1, 0], 40: [0, 0]}))
    assert by_id(result, 1)[list(GENOME_FEATURE_COLUMNS)].isna().all()
    assert by_id(result, 2)[list(GENOME_FEATURE_COLUMNS)].isna().all()
    target = by_id(result, 3)
    assert target[list(GENOME_FEATURE_COLUMNS[:5])].isna().all()
    assert target[list(GENOME_FEATURE_COLUMNS[5:])].notna().all()


def test_results_are_deterministic_under_row_permutation_and_ignore_event_id_order() -> None:
    source = events(
        [
            (90, 1, 10, 5.0, T),
            (2, 1, 20, 1.0, T),
            (80, 1, 30, 3.0, T + pd.Timedelta(seconds=1)),
            (1, 2, 10, 5.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    genome = scores({10: [1, 0], 20: [0, 1], 30: [1, 1]})
    expected = (
        build_genome_features(source, genome)
        .sort_values("ratingEventId")
        .reset_index(drop=True)
    )
    for seed in range(5):
        actual = build_genome_features(source.sample(frac=1, random_state=seed), genome)
        actual = actual.sort_values("ratingEventId").reset_index(drop=True)
        pdt.assert_frame_equal(actual, expected, check_exact=True)
