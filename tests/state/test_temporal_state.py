import json

import pandas as pd

from src.state.moments import HistoricalRatingState


T = pd.Timestamp("2020-01-01 10:00:00")


def test_temporal_state_is_sparse_and_expires_only_before_left_boundary() -> None:
    state = HistoricalRatingState()
    state.apply_timestamp_batch([(1, 10, 4.0), (2, 10, 3.0)], timestamp=T)
    state.apply_timestamp_batch(
        [(1, 20, 5.0)], timestamp=T + pd.Timedelta(seconds=1)
    )

    state.expire_movie_activity(T + pd.Timedelta(days=30))
    assert state.movie_rating_count_30d == {10: 2, 20: 1}
    state.expire_movie_activity(T + pd.Timedelta(days=30, seconds=1))
    assert state.movie_rating_count_30d == {20: 1}
    assert state.last_user_rating == {
        1: T + pd.Timedelta(seconds=1),
        2: T,
    }


def test_temporal_state_json_round_trip_is_deterministic() -> None:
    state = HistoricalRatingState()
    batch = [(2, 20, 4.0), (1, 10, 2.0), (3, 20, 5.0)]
    state.apply_timestamp_batch(batch, timestamp=T)
    reversed_state = HistoricalRatingState()
    reversed_state.apply_timestamp_batch(reversed(batch), timestamp=T)
    assert reversed_state.to_dict() == state.to_dict()
    encoded = json.dumps(state.to_dict(), separators=(",", ":"))
    restored = HistoricalRatingState.from_dict(json.loads(encoded))

    assert json.dumps(restored.to_dict(), separators=(",", ":")) == encoded
    assert restored.movie_rating_count_30d == {10: 1, 20: 2}
    assert restored.last_user_rating == {1: T, 2: T, 3: T}

    next_timestamp = T + pd.Timedelta(days=31)
    state.expire_movie_activity(next_timestamp)
    restored.expire_movie_activity(next_timestamp)
    state.apply_timestamp_batch([(1, 10, 3.0)], timestamp=next_timestamp)
    restored.apply_timestamp_batch([(1, 10, 3.0)], timestamp=next_timestamp)
    assert restored.to_dict() == state.to_dict()
