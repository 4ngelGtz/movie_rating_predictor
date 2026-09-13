import pandas as pd
import pytest

from src.entities.builders import (
    build_genres_and_movie_genres,
    build_movies,
    build_rating_events,
    build_users,
    read_rating_events,
    validate_movie_genres,
)
from src.entities.contracts import RELATIONSHIP_CONTRACTS


def source_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.Series([10, 20, 30], dtype="uint32"),
            "title": pd.Series(
                ["Mixed Movie (2001)", "Unknown Year", "No Genre (1999) "],
                dtype="string",
            ),
            "genres": pd.Series(
                ["Action|Comedy|Drama", "Comedy", "(no genres listed)"],
                dtype="string",
            ),
        }
    )


def source_links() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.Series([10, 20], dtype="uint32"),
            "imdbId": pd.Series([100, pd.NA], dtype="UInt32"),
            "tmdbId": pd.Series([pd.NA, 200], dtype="UInt32"),
        }
    )


def source_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": pd.Series([1, 1, 2], dtype="uint32"),
            "movieId": pd.Series([10, 20, 30], dtype="uint32"),
            "rating": pd.Series([4.0, 3.5, 5.0], dtype="float32"),
            "timestamp": pd.to_datetime(
                ["2020-01-02 10:00:00", "2020-01-01 10:00:00", "2020-01-03 10:00:00"]
            ),
        }
    )


def source_tags() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": pd.Series([1], dtype="uint32"),
            "timestamp": pd.to_datetime(["2019-12-31 10:00:00"]),
        }
    )


def identified_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    result = ratings.copy()
    result.insert(0, "ratingEventId", pd.Series(range(1, len(result) + 1), dtype="UInt64"))
    return result


def test_user_entity_has_unique_key_and_first_observed_activity() -> None:
    users = build_users(source_ratings(), source_tags())
    assert users["userId"].tolist() == [1, 2]
    assert users["userId"].is_unique
    assert users.loc[users["userId"].eq(1), "firstObservedTimestamp"].item() == pd.Timestamp(
        "2019-12-31 10:00:00"
    )
    assert users.loc[users["userId"].eq(1), "firstRatingTimestamp"].item() == pd.Timestamp(
        "2020-01-01 10:00:00"
    )


def test_user_first_observed_activity_includes_earlier_tag_event() -> None:
    users = build_users(source_ratings(), source_tags())
    user = users.loc[users["userId"].eq(1)].iloc[0]
    assert user["firstObservedTimestamp"] == pd.Timestamp("2019-12-31 10:00:00")
    assert user["firstRatingTimestamp"] == pd.Timestamp("2020-01-01 10:00:00")


def test_canonical_user_builder_requires_tags() -> None:
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        build_users(source_ratings())


def test_movie_entity_preserves_title_and_nullable_external_ids() -> None:
    movies = build_movies(source_movies(), source_links())
    assert movies["movieId"].is_unique
    assert movies.loc[2, "title"] == "No Genre (1999) "
    assert movies["releaseYear"].tolist() == [2001, pd.NA, 1999]
    assert str(movies["imdbId"].dtype) == "UInt32"
    assert str(movies["tmdbId"].dtype) == "UInt32"
    assert pd.isna(movies.loc[movies["movieId"].eq(30), "imdbId"]).item()
    assert movies.loc[movies["movieId"].eq(30), "genreStatus"].item() == "missing_in_source"


def test_release_year_boundaries_and_year_zero_policy() -> None:
    source = pd.DataFrame(
        {
            "movieId": pd.Series([1, 2, 3], dtype="uint32"),
            "title": pd.Series(
                ["Early (0001)", "Invalid (0000)", "Late (9999)"],
                dtype="string",
            ),
            "genres": pd.Series(["Drama", "Drama", "Drama"], dtype="string"),
        }
    )
    canonical = build_movies(source, source_links().iloc[:0])

    assert canonical["releaseYear"].tolist() == [1, pd.NA, 9999]
    assert str(canonical["releaseYear"].dtype) == "UInt16"


def test_multi_genre_movie_produces_one_unique_bridge_row_per_genre() -> None:
    genres, bridge = build_genres_and_movie_genres(source_movies())
    movie_ten = bridge.loc[bridge["movieId"].eq(10)]
    assert movie_ten["genreId"].tolist() == ["Action", "Comedy", "Drama"]
    assert not bridge.duplicated(["movieId", "genreId"]).any()
    assert set(genres["genreId"]) == {"Action", "Comedy", "Drama"}


def test_no_genres_sentinel_is_explicit_and_not_an_entity_or_bridge_row() -> None:
    canonical_movies = build_movies(source_movies(), source_links())
    genres, bridge = build_genres_and_movie_genres(source_movies())
    assert "(no genres listed)" not in set(genres["genreId"])
    assert bridge.loc[bridge["movieId"].eq(30)].empty
    assert canonical_movies.loc[
        canonical_movies["movieId"].eq(30), "genreStatus"
    ].item() == "missing_in_source"


def test_rating_event_grain_preserves_source_rows_and_assigns_identity() -> None:
    ratings = source_ratings()
    ratings = pd.concat([ratings, ratings.iloc[[0]]], ignore_index=True)
    users = build_users(ratings, source_tags())
    movies = build_movies(source_movies(), source_links())
    events = build_rating_events(identified_ratings(ratings), users=users, movies=movies)
    assert len(events) == len(ratings)
    assert events["ratingEventId"].tolist() == [1, 2, 3, 4]
    assert events["ratingEventId"].is_unique
    assert events["highRating"].tolist() == [True, False, True, True]
    assert events.duplicated(["userId", "movieId", "rating", "timestamp"]).sum() == 1


def test_rating_event_rejects_unresolved_relationship() -> None:
    users = build_users(source_ratings(), source_tags())
    movies = build_movies(source_movies(), source_links())
    ratings = source_ratings().copy()
    ratings.loc[0, "movieId"] = 999
    with pytest.raises(ValueError, match="do not resolve to movies"):
        build_rating_events(identified_ratings(ratings), users=users, movies=movies)


def test_event_ids_are_stable_after_sort_filter_and_partition(tmp_path) -> None:
    ratings = source_ratings()
    ratings = pd.concat([ratings, ratings.iloc[[0]]], ignore_index=True)
    path = tmp_path / "ratings.parquet"
    ratings.to_parquet(path, index=False)

    first = read_rating_events(path)
    rebuilt = read_rating_events(path)
    transformed = first.sort_values("timestamp", ascending=False)
    partitions = [
        build_rating_events(transformed.iloc[:2]),
        build_rating_events(transformed.iloc[2:]),
    ]

    assert first["ratingEventId"].tolist() == rebuilt["ratingEventId"].tolist()
    assert pd.concat(partitions)["ratingEventId"].tolist() == transformed[
        "ratingEventId"
    ].tolist()
    assert first.loc[first["movieId"].eq(10), "ratingEventId"].tolist() == [1, 4]


def test_optional_bridge_participation_does_not_mean_nullable_keys() -> None:
    contract = RELATIONSHIP_CONTRACTS["movie_genre"]
    assert contract.optional_participation is True
    genres, bridge = build_genres_and_movie_genres(source_movies())
    assert not bridge[["movieId", "genreId"]].isna().any().any()
    validate_movie_genres(bridge, source_movies()[["movieId"]], genres)


def test_build_rating_events_requires_preassigned_provenance() -> None:
    with pytest.raises(ValueError, match="ratingEventId"):
        build_rating_events(source_ratings())


@pytest.mark.parametrize(
    "genres",
    ["Action|", "Action||Comedy", "Action|(no genres listed)"],
)
def test_malformed_genre_lists_are_rejected(genres: str) -> None:
    movies = source_movies().iloc[[0]].copy()
    movies.loc[:, "genres"] = genres
    with pytest.raises(ValueError, match="empty tokens|cannot be mixed"):
        build_genres_and_movie_genres(movies)


def test_movie_genre_bridge_rejects_duplicate_and_orphan_rows() -> None:
    genres, bridge = build_genres_and_movie_genres(source_movies())
    duplicate = pd.concat([bridge, bridge.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique key"):
        validate_movie_genres(duplicate, source_movies()[["movieId"]], genres)

    orphan = bridge.copy()
    orphan.loc[0, "genreId"] = "Unknown"
    with pytest.raises(ValueError, match="do not resolve to genres"):
        validate_movie_genres(orphan, source_movies()[["movieId"]], genres)


def test_movie_genre_bridge_rejects_compound_genre_ids() -> None:
    compound_bridge = pd.DataFrame(
        {
            "movieId": pd.Series([10], dtype="uint32"),
            "genreId": pd.Series(["Drama|Action"], dtype="string"),
        }
    )
    compound_genres = pd.DataFrame(
        {
            "genreId": pd.Series(["Drama|Action"], dtype="string"),
            "genreName": pd.Series(["Drama|Action"], dtype="string"),
        }
    )
    with pytest.raises(ValueError, match="one normalized genre token"):
        validate_movie_genres(
            compound_bridge,
            source_movies()[["movieId"]],
            compound_genres,
        )
