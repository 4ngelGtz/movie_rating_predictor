import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.expanding import FEATURE_COLUMNS, build_expanding_rating_features


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


def aligned(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("ratingEventId").reset_index(drop=True)


def test_basic_expanding_history_has_exact_global_user_and_movie_features() -> None:
    source = events(
        [
            (1, 1, 10, 1.0, T),
            (2, 1, 20, 3.0, T + pd.Timedelta(seconds=1)),
            (3, 2, 10, 5.0, T + pd.Timedelta(seconds=2)),
            (4, 1, 10, 5.0, T + pd.Timedelta(seconds=3)),
        ]
    )
    result = build_expanding_rating_features(source)

    assert result["global_rating_count"].tolist() == [0, 1, 2, 3]
    assert result["global_mean_rating"].tolist() == pytest.approx([3.5, 1.0, 2.0, 3.0])
    assert result["user_rating_count"].tolist() == [0, 1, 0, 2]
    assert result["user_mean_rating"].tolist() == pytest.approx([3.5, 1.0, 2.0, 2.0])
    assert result["user_rating_std_pop"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 1.0]
    )
    assert result["movie_rating_count"].tolist() == [0, 0, 1, 2]
    assert result["movie_mean_rating"].tolist() == pytest.approx([3.5, 1.0, 1.0, 3.0])
    assert result["movie_rating_std_pop"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 2.0]
    )


def test_cold_start_fallbacks_use_only_pre_batch_global_state() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 2, 20, 1.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    result = build_expanding_rating_features(source)

    first = result.iloc[0]
    assert first["global_rating_count"] == 0
    assert first["global_mean_rating"] == 3.5
    assert first["user_mean_rating"] == 3.5
    assert first["movie_mean_rating"] == 3.5
    assert first["user_rating_std_pop"] == 0.0
    assert first["movie_rating_std_pop"] == 0.0
    second = result.iloc[1]
    assert second["global_mean_rating"] == 4.0
    assert second["user_mean_rating"] == 4.0
    assert second["movie_mean_rating"] == 4.0


def test_simultaneous_events_share_pre_batch_state_under_permutation() -> None:
    source = events(
        [
            (1, 1, 10, 2.0, T),
            (2, 1, 20, 4.0, T + pd.Timedelta(seconds=1)),
            (3, 2, 10, 5.0, T + pd.Timedelta(seconds=1)),
            (4, 1, 10, 3.0, T + pd.Timedelta(seconds=2)),
        ]
    )
    expected = aligned(build_expanding_rating_features(source))
    simultaneous = expected.loc[expected["ratingEventId"].isin([2, 3])]
    assert simultaneous["global_rating_count"].tolist() == [1, 1]
    assert simultaneous["global_mean_rating"].tolist() == [2.0, 2.0]
    assert (
        simultaneous.loc[
            simultaneous["ratingEventId"].eq(2), "user_rating_count"
        ].item()
        == 1
    )
    assert (
        simultaneous.loc[
            simultaneous["ratingEventId"].eq(3), "movie_rating_count"
        ].item()
        == 1
    )

    for seed in range(5):
        shuffled = source.sample(frac=1, random_state=seed).reset_index(drop=True)
        pdt.assert_frame_equal(aligned(build_expanding_rating_features(shuffled)), expected)


def test_entity_isolation_and_global_event_conservation() -> None:
    source = events(
        [
            (1, 1, 10, 1.0, T),
            (2, 2, 20, 5.0, T + pd.Timedelta(seconds=1)),
            (3, 1, 20, 3.0, T + pd.Timedelta(seconds=2)),
        ]
    )
    result = build_expanding_rating_features(source)
    last = result.iloc[-1]

    assert last["global_rating_count"] == 2
    assert last["user_rating_count"] == 1
    assert last["user_mean_rating"] == 1.0
    assert last["movie_rating_count"] == 1
    assert last["movie_mean_rating"] == 5.0


def test_population_std_differs_from_sample_std() -> None:
    source = events(
        [
            (1, 1, 10, 1.0, T),
            (2, 1, 10, 3.0, T + pd.Timedelta(seconds=1)),
            (3, 1, 10, 5.0, T + pd.Timedelta(seconds=2)),
        ]
    )
    last = build_expanding_rating_features(source).iloc[-1]

    assert last["user_rating_std_pop"] == pytest.approx(1.0)
    assert last["movie_rating_std_pop"] == pytest.approx(1.0)
    assert last["user_rating_std_pop"] != pytest.approx(np.sqrt(2.0))


def test_distinct_ids_for_duplicate_source_rows_remain_distinct_events() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 1, 10, 4.0, T),
            (3, 1, 10, 2.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    result = build_expanding_rating_features(source)

    assert result.loc[:1, "global_rating_count"].tolist() == [0, 0]
    assert result.loc[2, "global_rating_count"] == 2
    assert result.loc[2, "user_rating_count"] == 2
    assert result.loc[2, "movie_rating_count"] == 2


def test_output_conserves_event_identity_and_contract_dtypes() -> None:
    source = events(
        [(9, 1, 10, 4.0, T), (3, 2, 20, 2.0, T + pd.Timedelta(seconds=1))]
    )
    result = build_expanding_rating_features(source)

    assert len(result) == len(source)
    assert result["ratingEventId"].tolist() == source["ratingEventId"].tolist()
    assert result["ratingEventId"].is_unique
    for column in ("global_rating_count", "user_rating_count", "movie_rating_count"):
        assert str(result[column].dtype) == "uint64"
    for column in set(FEATURE_COLUMNS) - {
        "global_rating_count",
        "user_rating_count",
        "user_seconds_since_last_rating",
        "movie_rating_count",
        "movie_rating_count_30d",
    }:
        assert str(result[column].dtype) == "float32"


def test_duplicate_rating_event_identity_is_rejected() -> None:
    source = events([(1, 1, 10, 4.0, T), (1, 1, 10, 4.0, T)])
    with pytest.raises(ValueError, match="ratingEventId must be unique"):
        build_expanding_rating_features(source)
