import json

import pytest

from src.state.moments import HistoricalRatingState, RunningMoments


def test_running_moments_use_population_variance() -> None:
    moments = RunningMoments.from_values([1.0, 2.0, 3.0])
    assert moments.count == 3
    assert moments.mean == 2.0
    assert moments.M2 == 2.0
    assert moments.population_variance == pytest.approx(2.0 / 3.0)
    assert moments.population_std == pytest.approx((2.0 / 3.0) ** 0.5)
    assert moments.population_std != pytest.approx(1.0)  # sample ddof=1


def test_empty_and_singleton_population_std_are_zero() -> None:
    assert RunningMoments().population_std == 0.0
    assert RunningMoments.from_values([4.5]).population_std == 0.0


def test_batch_updates_are_order_independent_and_count_every_event() -> None:
    first = HistoricalRatingState()
    second = HistoricalRatingState()
    batch = [(1, 10, 1.0), (1, 20, 5.0), (2, 10, 3.0)]
    first.apply_timestamp_batch(batch)
    second.apply_timestamp_batch(reversed(batch))

    assert first.to_dict() == second.to_dict()
    assert first.global_moments.count == 3
    assert first.user_moments[1].count == 2
    assert first.user_moments[2].count == 1
    assert first.movie_moments[10].count == 2
    assert first.movie_moments[20].count == 1


def test_state_checkpoint_representation_is_deterministic_and_round_trips() -> None:
    state = HistoricalRatingState()
    state.apply_timestamp_batch(
        [(2, 20, 4.0), (1, 10, 2.0)],
        movie_genres={10: ("Drama", "Romance"), 20: ("Comedy",)},
    )
    payload = state.to_dict()

    assert [row["userId"] for row in payload["users"]] == [1, 2]
    assert [row["movieId"] for row in payload["movies"]] == [10, 20]
    assert [(row["userId"], row["genreId"]) for row in payload["userGenres"]] == [
        (1, "Drama"),
        (1, "Romance"),
        (2, "Comedy"),
    ]
    assert json.dumps(payload, separators=(",", ":")) == json.dumps(
        HistoricalRatingState.from_dict(payload).to_dict(), separators=(",", ":")
    )


def test_duplicate_movie_genre_memberships_are_rejected_before_state_updates() -> None:
    state = HistoricalRatingState()
    before = state.to_dict()

    with pytest.raises(ValueError, match="memberships must be distinct.*movieId 10"):
        state.apply_timestamp_batch(
            [(1, 10, 4.0)],
            movie_genres={10: ("Drama", "Drama")},
        )

    assert state.to_dict() == before
