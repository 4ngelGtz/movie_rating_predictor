from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.temporal import historical_events, is_strictly_prior, rolling_window_mask


T = pd.Timestamp("2020-01-31 10:00:00")


def frame(times: list[pd.Timestamp | str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event": range(len(times)),
            "userId": [1, 1, 2, 1, 2][: len(times)],
            "movieId": [10, 20, 10, 30, 20][: len(times)],
            "timestamp": pd.to_datetime(times),
        }
    )


def test_strict_cutoff_excludes_prediction_time_and_future() -> None:
    events = frame([T - pd.Timedelta(minutes=1), T, T + pd.Timedelta(minutes=1)])
    assert historical_events(events, T)["event"].tolist() == [0]


def test_same_timestamp_targets_cannot_see_one_another() -> None:
    events = frame([T - pd.Timedelta(seconds=1), T, T, T])
    available = historical_events(events, T)
    assert available["event"].tolist() == [0]
    assert not is_strictly_prior(events["timestamp"], T).iloc[1:].any()


def test_rolling_window_boundaries_are_left_inclusive_right_exclusive() -> None:
    events = frame(
        [
            T - pd.Timedelta(days=30, seconds=1),
            T - pd.Timedelta(days=30),
            T - pd.Timedelta(seconds=1),
            T,
        ]
    )
    assert rolling_window_mask(events["timestamp"], T, "30 days").tolist() == (
        [False, True, True, False]
    )


def test_first_chronological_event_has_empty_history() -> None:
    events = frame([T, T + pd.Timedelta(seconds=1)])
    assert historical_events(events, T).empty


def test_entity_filter_does_not_weaken_temporal_cutoff() -> None:
    events = pd.DataFrame(
        {
            "event": list("ABCDEFG"),
            "userId": [1, 2, 1, 2, 1, 1, 2],
            "movieId": [10, 10, 20, 20, 10, 30, 10],
            "timestamp": pd.to_datetime(
                [
                    T - pd.Timedelta(minutes=1),
                    T - pd.Timedelta(minutes=1),
                    T - pd.Timedelta(minutes=1),
                    T - pd.Timedelta(minutes=1),
                    T,
                    T,
                    T,
                ]
            ),
        }
    )

    user_history = historical_events(events, T, entity_filters={"userId": 1})
    movie_history = historical_events(events, T, entity_filters={"movieId": 10})
    pair_history = historical_events(
        events, T, entity_filters={"userId": 1, "movieId": 10}
    )

    assert set(user_history["event"]) == {"A", "C"}
    assert set(movie_history["event"]) == {"A", "B"}
    assert set(pair_history["event"]) == {"A"}


def test_same_timestamp_batch_becomes_visible_only_after_timestamp() -> None:
    events = pd.DataFrame(
        {
            "event": list("ABCD"),
            "userId": [1, 1, 1, 1],
            "movieId": [1, 2, 3, 4],
            "timestamp": pd.to_datetime(
                [
                    T - pd.Timedelta(minutes=1),
                    T,
                    T,
                    T + pd.Timedelta(minutes=1),
                ]
            ),
        }
    )

    assert set(historical_events(events, T)["event"]) == {"A"}
    assert set(
        historical_events(events, T + pd.Timedelta(minutes=1))["event"]
    ) == {"A", "B", "C"}


def test_repeated_interaction_obeys_same_strict_cutoff() -> None:
    events = pd.DataFrame(
        {
            "event": list("ABC"),
            "userId": [1, 1, 1],
            "movieId": [10, 10, 10],
            "timestamp": pd.to_datetime([T - pd.Timedelta(seconds=1), T, T]),
        }
    )

    history = historical_events(
        events, T, entity_filters={"userId": 1, "movieId": 10}
    )
    assert set(history["event"]) == {"A"}


def test_eligibility_is_independent_of_input_order() -> None:
    events = frame([T, T - pd.Timedelta(seconds=1), T, T + pd.Timedelta(seconds=1), T])
    for seed in range(5):
        shuffled = events.sample(frac=1, random_state=seed)
        assert set(historical_events(shuffled, T)["event"]) == {1}


def test_many_duplicate_timestamps_are_deterministically_all_excluded() -> None:
    events = frame([T] * 5)
    for seed in range(5):
        shuffled = events.sample(frac=1, random_state=seed)
        assert historical_events(shuffled, T).empty


def test_equal_timestamp_exclusion_is_a_leakage_regression_invariant() -> None:
    timestamps = pd.Series(pd.to_datetime([T, T, T]))
    assert is_strictly_prior(timestamps, T).sum() == 0


def test_invalid_temporal_inputs_fail_clearly() -> None:
    with pytest.raises(TypeError, match="datetime64"):
        is_strictly_prior(pd.Series(["2020-01-01"]), T)
    with pytest.raises(ValueError, match="positive"):
        rolling_window_mask(pd.Series(pd.to_datetime([T])), T, pd.Timedelta(0))
    with pytest.raises(ValueError, match="timezone-naive"):
        is_strictly_prior(pd.Series(pd.to_datetime([T])), T.tz_localize("UTC"))


@pytest.mark.parametrize("prediction_timestamp", [1_700_000_000, 1.5])
def test_numeric_prediction_timestamps_are_rejected(prediction_timestamp: object) -> None:
    with pytest.raises(TypeError, match="numeric prediction timestamps"):
        is_strictly_prior(pd.Series(pd.to_datetime([T])), prediction_timestamp)


@pytest.mark.parametrize("prediction_timestamp", ["now", "TODAY", " yesterday ", "tomorrow"])
def test_relative_prediction_timestamp_strings_are_rejected(
    prediction_timestamp: str,
) -> None:
    with pytest.raises(ValueError, match="deterministic"):
        is_strictly_prior(pd.Series(pd.to_datetime([T])), prediction_timestamp)


def test_explicit_prediction_timestamp_types_are_accepted() -> None:
    timestamps = pd.Series(pd.to_datetime([T - pd.Timedelta(seconds=1), T]))
    accepted = [
        T,
        datetime(2020, 1, 31, 10, 0),
        np.datetime64("2020-01-31T10:00:00"),
        "2020-01-31 10:00:00",
    ]
    for prediction_timestamp in accepted:
        assert is_strictly_prior(timestamps, prediction_timestamp).tolist() == [True, False]


@pytest.mark.parametrize("window", [30, 30.0, "30", " 30 "])
def test_numeric_or_unitless_windows_are_rejected(window: object) -> None:
    with pytest.raises(TypeError, match="numeric windows|string windows"):
        rolling_window_mask(pd.Series(pd.to_datetime([T])), T, window)


def test_explicit_window_types_are_accepted() -> None:
    timestamps = pd.Series(pd.to_datetime([T - pd.Timedelta(days=30), T]))
    accepted = [
        pd.Timedelta(days=30),
        timedelta(days=30),
        np.timedelta64(30, "D"),
        "30 days",
        "12 hours",
    ]
    for window in accepted:
        mask = rolling_window_mask(timestamps, T, window)
        assert not mask.iloc[-1]
