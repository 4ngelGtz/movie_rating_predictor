from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.movie_rolling import (
    MOVIE_ROLLING_FEATURE_COLUMNS,
    resolve_movie_windows,
    safe_ratio,
)
from src.features.expanding import build_expanding_rating_features
from src.features.expanding import FEATURE_COLUMNS, iter_expanding_rating_features
from src.state.moments import HistoricalRatingState
from tests.features.catalog_fixtures import canonical_movies_for
from tests.state.test_checkpoints import populated_checkpoint


def build_movie_rolling_features(source):
    bridge = pd.DataFrame({
        "movieId": pd.Series(dtype="uint32"),
        "genreId": pd.Series(dtype="string"),
    })
    return build_expanding_rating_features(
        source, bridge, canonical_movies_for(source, bridge), movie_windows=True,
    )


T = pd.Timestamp("2020-03-31 12:00:00")


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


def indexed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("ratingEventId").sort_index()


def test_exact_boundaries_movie_isolation_values_ratios_and_cold_start() -> None:
    source = events(
        [
            (1, 1, 10, 1.0, T - pd.Timedelta(days=60)),
            (2, 2, 10, 2.0, T - pd.Timedelta(days=30)),
            (3, 3, 10, 4.0, T - pd.Timedelta(days=29)),
            (4, 4, 10, 5.0, T - pd.Timedelta(days=60, nanoseconds=1)),
            (5, 5, 20, 5.0, T - pd.Timedelta(days=1)),
            (6, 6, 10, 3.0, T),
        ]
    )
    result = indexed(build_movie_rolling_features(source))
    target = result.loc[6]

    assert target["movie_rating_count_30d"] == 2
    assert target["movie_rating_count_60d"] == 3
    assert target["movie_rating_avg_30d"] == pytest.approx(3.0)
    assert target["movie_rating_avg_60d"] == pytest.approx(7.0 / 3.0)
    assert target["movie_rating_count_30d_over_60d"] == pytest.approx(2.0 / 3.0)
    assert target["movie_rating_avg_30d_over_60d"] == pytest.approx(9.0 / 7.0)
    assert result.loc[4, "movie_rating_count_30d"] == 0
    assert result.loc[4, "movie_rating_count_60d"] == 0
    for column in MOVIE_ROLLING_FEATURE_COLUMNS[2:]:
        assert np.isnan(result.loc[4, column])


def test_current_and_simultaneous_events_are_excluded_and_order_invariant() -> None:
    source = events(
        [
            (1, 1, 10, 2.0, T - pd.Timedelta(days=1)),
            (2, 2, 10, 4.0, T),
            (3, 3, 10, 5.0, T),
            (4, 4, 10, 3.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    expected = indexed(build_movie_rolling_features(source))
    assert expected.loc[[2, 3], "movie_rating_count_30d"].tolist() == [1, 1]
    assert expected.loc[[2, 3], "movie_rating_avg_30d"].tolist() == [2.0, 2.0]
    assert expected.loc[4, "movie_rating_count_30d"] == 3

    for seed in range(5):
        shuffled = source.sample(frac=1, random_state=seed).reset_index(drop=True)
        pdt.assert_frame_equal(
            indexed(build_movie_rolling_features(shuffled)), expected
        )


def test_30_day_history_is_always_a_subset_of_60_day_history() -> None:
    source = events(
        [
            (1, 1, 10, 1.0, T - pd.Timedelta(days=61)),
            (2, 2, 10, 2.0, T - pd.Timedelta(days=45)),
            (3, 3, 10, 3.0, T - pd.Timedelta(days=30)),
            (4, 4, 10, 4.0, T - pd.Timedelta(days=1)),
            (5, 5, 10, 5.0, T),
        ]
    )
    result = build_movie_rolling_features(source)
    assert (
        result["movie_rating_count_30d"] <= result["movie_rating_count_60d"]
    ).all()


def test_output_dtypes_and_unique_names_are_contracted() -> None:
    result = build_movie_rolling_features(events([(1, 1, 10, 4.0, T)]))
    assert len(MOVIE_ROLLING_FEATURE_COLUMNS) == len(
        set(MOVIE_ROLLING_FEATURE_COLUMNS)
    )
    assert str(result["movie_rating_count_30d"].dtype) == "uint64"
    assert str(result["movie_rating_count_60d"].dtype) == "uint64"
    for column in MOVIE_ROLLING_FEATURE_COLUMNS[2:]:
        assert str(result[column].dtype) == "float32"


def test_medium_history_without_recent_history() -> None:
    source = events([
        (1, 1, 10, 4.0, T - pd.Timedelta(days=45)),
        (2, 2, 10, 2.0, T),
    ])
    target = build_movie_rolling_features(source).iloc[-1]
    assert target["movie_rating_count_30d"] == 0
    assert target["movie_rating_count_60d"] == 1
    assert np.isnan(target["movie_rating_avg_30d"])
    assert target["movie_rating_avg_60d"] == 4.0
    assert target["movie_rating_count_30d_over_60d"] == 0.0
    assert np.isnan(target["movie_rating_avg_30d_over_60d"])


@pytest.mark.parametrize("denominator", [0.0, np.nan])
def test_undefined_ratio_stays_missing(denominator) -> None:
    assert np.isnan(safe_ratio(3.0, denominator))


def test_one_state_update_per_timestamp_and_legacy_output_parity(monkeypatch) -> None:
    source = events([
        (1, 1, 10, 2.0, T - pd.Timedelta(days=60)),
        (2, 2, 10, 4.0, T - pd.Timedelta(days=30)),
        (3, 3, 10, 5.0, T),
        (4, 4, 10, 1.0, T),
        (5, 5, 10, 3.0, T + pd.Timedelta(nanoseconds=1)),
    ])
    bridge = pd.DataFrame({
        "movieId": pd.Series(dtype="uint32"),
        "genreId": pd.Series(dtype="string"),
    })
    movies = canonical_movies_for(source, bridge)
    legacy = build_expanding_rating_features(source, bridge, movies)
    calls = []
    original = HistoricalRatingState._record_movie_activity

    def tracked(self, timestamp, batch):
        calls.append(timestamp)
        return original(self, timestamp, batch)

    monkeypatch.setattr(HistoricalRatingState, "_record_movie_activity", tracked)
    current = pd.concat(iter_expanding_rating_features(
        source, bridge, movies, chunk_size=1, movie_windows=True,
    ), ignore_index=True)
    assert calls == sorted(source.timestamp.unique())
    pdt.assert_frame_equal(current.loc[:, legacy.columns], legacy)
    assert tuple(current.columns[4:21]) == FEATURE_COLUMNS
    assert current["movie_rating_count_30d"].tolist() == [0, 1, 1, 1, 2]


def test_extended_state_cannot_silently_lose_history_in_legacy_checkpoint() -> None:
    state = HistoricalRatingState.for_movie_window_features()
    with pytest.raises(ValueError, match="requires full replay"):
        state.to_dict()


def test_restored_legacy_checkpoint_is_historical_only() -> None:
    checkpoint = populated_checkpoint()
    restored_checkpoint = type(checkpoint).from_json(checkpoint.to_json())
    _, state = restored_checkpoint.restore()

    assert state.supports_movie_window_features is False
    assert state.to_dict() == checkpoint.restore()[1].to_dict()
    assert state.movie_moments[10].count == 2
    assert state.movie_moments[10].mean == pytest.approx(3.5)
    assert state.movie_rating_count_30d[10] == 2

    message = (
        "initialized or restored without PRD30 movie-window tracking.*"
        "cannot be upgraded in place.*full replay"
    )
    with pytest.raises(ValueError, match=message):
        resolve_movie_windows(state, 10)

    with pytest.raises(AttributeError):
        state.movie_windows = True
    with pytest.raises(AttributeError):
        state.supports_movie_window_features = True
    with pytest.raises(AttributeError):
        del state.supports_movie_window_features
    with pytest.raises(AttributeError, match="tracking mode is immutable"):
        del state._movie_window_capability
    with pytest.raises(AttributeError, match="tracking mode is immutable"):
        state._movie_window_capability = (
            HistoricalRatingState.for_movie_window_features()._movie_window_capability
        )
    assert state.supports_movie_window_features is False
    with pytest.raises(ValueError, match=message):
        resolve_movie_windows(state, 10)


def test_fresh_prd30_state_resolves_complete_movie_window_history() -> None:
    state = HistoricalRatingState.for_movie_window_features()
    assert state.supports_movie_window_features is True
    with pytest.raises(AttributeError, match="tracking mode is immutable"):
        del state._movie_window_capability
    with pytest.raises(AttributeError, match="tracking mode is immutable"):
        state._movie_window_capability = HistoricalRatingState()._movie_window_capability
    with pytest.raises(AttributeError):
        del state.supports_movie_window_features
    with pytest.raises(AttributeError):
        state.supports_movie_window_features = False
    assert state.supports_movie_window_features is True
    empty = resolve_movie_windows(state, 10)
    assert empty[0] == 0
    assert all(np.isnan(value) for value in empty[1:])

    history_timestamp = T - pd.Timedelta(days=1)
    state.expire_movie_activity(history_timestamp)
    state.apply_timestamp_batch(
        [(1, 10, 4.0), (2, 10, 3.0)], timestamp=history_timestamp
    )
    state.expire_movie_activity(T)

    assert state.supports_movie_window_features is True
    assert state.movie_rating_count_30d[10] == 2
    assert resolve_movie_windows(state, 10) == pytest.approx(
        (2, 3.5, 3.5, 1.0, 1.0)
    )
