import pandas as pd
import pytest

from src.entities.builders import build_genres_and_movie_genres, build_rating_events
from src.state.contracts import STATE_CONTRACTS
from src.state.updates import relationship_state_inputs_as_of, state_input_events_as_of


T = pd.Timestamp("2020-01-02 10:00:00")


def movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.Series([10, 20, 30], dtype="uint32"),
            "title": pd.Series(["A", "B", "C"], dtype="string"),
            "genres": pd.Series(
                ["Action|Comedy|Drama", "Comedy", "(no genres listed)"], dtype="string"
            ),
        }
    )


def events() -> pd.DataFrame:
    ratings = pd.DataFrame(
        {
            "userId": pd.Series([1, 1, 2, 3], dtype="uint32"),
            "movieId": pd.Series([10, 20, 10, 30], dtype="uint32"),
            "rating": pd.Series([4.0, 3.0, 5.0, 2.0], dtype="float32"),
            "timestamp": pd.to_datetime(
                [T - pd.Timedelta(seconds=1), T, T, T + pd.Timedelta(seconds=1)]
            ),
        }
    )
    ratings.insert(0, "ratingEventId", pd.Series(range(1, len(ratings) + 1), dtype="UInt64"))
    return build_rating_events(ratings)


def test_current_and_same_timestamp_events_are_excluded_from_state() -> None:
    eligible = state_input_events_as_of(events(), T)
    assert eligible["ratingEventId"].tolist() == [1]


def test_complete_timestamp_batch_is_visible_only_after_timestamp() -> None:
    eligible = state_input_events_as_of(events(), T + pd.Timedelta(seconds=1))
    assert eligible["ratingEventId"].tolist() == [1, 2, 3]


def test_cold_start_has_no_state_input_rows() -> None:
    eligible = state_input_events_as_of(
        events(), T, entity_filters={"userId": 3}
    )
    assert eligible.empty

    assert state_input_events_as_of(events(), T, entity_filters={"movieId": 30}).empty

    genres, bridge = build_genres_and_movie_genres(movies())
    updates = relationship_state_inputs_as_of(
        events(),
        bridge,
        T,
        related_id_column="genreId",
        related_entities=genres,
        movies=movies()[["movieId"]],
    )
    assert updates.loc[
        updates["userId"].eq(2) & updates["genreId"].eq("Comedy")
    ].empty


def test_one_multi_genre_event_yields_linked_updates_not_extra_events() -> None:
    genres, bridge = build_genres_and_movie_genres(movies())
    updates = relationship_state_inputs_as_of(
        events(),
        bridge,
        T,
        related_id_column="genreId",
        related_entities=genres,
        movies=movies()[["movieId"]],
    )
    assert len(updates) == 3
    assert updates["ratingEventId"].nunique() == 1
    assert set(updates["genreId"]) == {"Action", "Comedy", "Drama"}
    assert not updates.duplicated(["ratingEventId", "genreId"]).any()


def test_multi_genre_expansion_conserves_canonical_event_identity() -> None:
    genres, bridge = build_genres_and_movie_genres(movies())
    canonical = events().iloc[[0, 3]].copy()
    updates = relationship_state_inputs_as_of(
        canonical,
        bridge,
        T + pd.Timedelta(seconds=2),
        related_id_column="genreId",
        related_entities=genres,
        movies=movies()[["movieId"]],
    )
    assert len(canonical) == 2
    assert canonical["ratingEventId"].nunique() == 2
    assert len(updates) == 3
    assert updates["ratingEventId"].nunique() == 1


def test_same_timestamp_duplicates_are_excluded_independent_of_order_and_index() -> None:
    rating_events = events()
    duplicate = rating_events.iloc[[1]].copy()
    duplicate.loc[:, "ratingEventId"] = 5
    rating_events = pd.concat([rating_events, duplicate], ignore_index=True)
    rating_events.index = [7, 7, 2, 2, 7]

    for seed in range(3):
        shuffled = rating_events.sample(frac=1, random_state=seed)
        assert set(state_input_events_as_of(shuffled, T)["ratingEventId"]) == {1}
        assert set(
            state_input_events_as_of(shuffled, T + pd.Timedelta(seconds=1))[
                "ratingEventId"
            ]
        ) == {1, 2, 3, 5}


def test_missing_genre_movie_produces_no_genre_state_update() -> None:
    genres, bridge = build_genres_and_movie_genres(movies())
    updates = relationship_state_inputs_as_of(
        events(),
        bridge,
        T + pd.Timedelta(seconds=2),
        related_id_column="genreId",
        related_entities=genres,
        movies=movies()[["movieId"]],
    )
    assert 4 not in set(updates["ratingEventId"])


def test_invalid_cross_entity_row_cannot_create_state_input() -> None:
    genres, bridge = build_genres_and_movie_genres(movies())
    invalid = pd.concat(
        [bridge, pd.DataFrame({"movieId": [999], "genreId": ["Action"]})],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="do not resolve to movies"):
        relationship_state_inputs_as_of(
            events(),
            invalid,
            T,
            related_id_column="genreId",
            related_entities=genres,
            movies=movies()[["movieId"]],
        )


def test_rating_event_with_unknown_movie_cannot_be_silently_dropped() -> None:
    genres, bridge = build_genres_and_movie_genres(movies())
    invalid_events = events().copy()
    invalid_events.loc[0, "movieId"] = 999
    with pytest.raises(ValueError, match="do not resolve to movies"):
        relationship_state_inputs_as_of(
            invalid_events,
            bridge,
            T,
            related_id_column="genreId",
            related_entities=genres,
            movies=movies()[["movieId"]],
        )


def test_state_contract_keys_include_exclusive_as_of_timestamp() -> None:
    assert STATE_CONTRACTS["user_state"].primary_key == ("userId", "asOfTimestamp")
    assert STATE_CONTRACTS["user_genre_state"].primary_key == (
        "userId",
        "genreId",
        "asOfTimestamp",
    )
    assert "deferred" in STATE_CONTRACTS["actor_state"].materialization
    assert STATE_CONTRACTS["genre_state"].input_semantics == (
        "cumulative H(t), not an incremental delta"
    )
