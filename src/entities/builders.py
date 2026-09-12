"""Build and validate the canonical Phase 2 entity tables.

These functions reshape validated Phase 0 inputs. They do not write files or
compute predictive aggregates.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.data.schemas import RATING_VALUES


NO_GENRES_LISTED = "(no genres listed)"
RATING_EVENT_COLUMNS = ("userId", "movieId", "rating", "timestamp")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{table}: missing required columns: {sorted(missing)}")


def _require_unique(frame: pd.DataFrame, columns: list[str], table: str) -> None:
    if duplicate_rows := int(frame.duplicated(columns, keep=False).sum()):
        raise ValueError(f"{table}: {duplicate_rows} rows violate unique key {tuple(columns)}")


def _require_non_null(frame: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    violations = {
        column: int(frame[column].isna().sum())
        for column in columns
        if frame[column].isna().any()
    }
    if violations:
        raise ValueError(f"{table}: nulls in non-nullable columns: {violations}")


def _validate_timestamps(values: pd.Series, table: str) -> None:
    if not is_datetime64_any_dtype(values.dtype):
        raise ValueError(f"{table}: timestamp must have a pandas datetime64 dtype")
    if values.isna().any():
        raise ValueError(f"{table}: timestamp must not contain missing values")
    if values.dt.tz is not None:
        raise ValueError(f"{table}: timestamp must be timezone-naive")


def _validate_references(
    values: pd.Series, valid_values: pd.Series, table: str, referenced_table: str
) -> None:
    unknown = pd.Index(values.dropna().unique()).difference(valid_values.dropna().unique())
    if not unknown.empty:
        examples = unknown[:5].tolist()
        raise ValueError(
            f"{table}: {len(unknown)} keys do not resolve to {referenced_table}; "
            f"examples: {examples}"
        )


def _validated_genre_rows(movies: pd.DataFrame) -> pd.DataFrame:
    """Return exploded genres after validating the source string grammar."""
    exploded = movies[["movieId", "genres"]].assign(
        genreId=movies["genres"].str.split("|")
    ).explode("genreId")
    if exploded["genreId"].eq("").any():
        raise ValueError("movies: genre lists must not contain empty tokens")
    if exploded["genreId"].str.strip().ne(exploded["genreId"]).any():
        raise ValueError("movies: genre labels contain surrounding whitespace")
    sentinel_counts = exploded["genreId"].eq(NO_GENRES_LISTED).groupby(exploded["movieId"]).sum()
    token_counts = exploded.groupby("movieId").size()
    if ((sentinel_counts > 0) & (token_counts > 1)).any():
        raise ValueError("movies: the no-genres sentinel cannot be mixed with real genres")
    return exploded


def build_users(ratings: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Return one user row with first observed and first rating timestamps.

    Lifecycle timestamps are derived from the complete extract, not static
    predictors that may be attached to historical rows.
    """
    _require_columns(ratings, ("userId", "timestamp"), "ratings")
    _require_non_null(ratings, ("userId",), "ratings")
    _validate_timestamps(ratings["timestamp"], "ratings")
    first_ratings = (
        ratings.groupby("userId", sort=True, observed=True)["timestamp"]
        .min()
        .rename("firstRatingTimestamp")
        .reset_index()
    )
    _require_columns(tags, ("userId", "timestamp"), "tags")
    _require_non_null(tags, ("userId",), "tags")
    _validate_timestamps(tags["timestamp"], "tags")
    activity = [ratings[["userId", "timestamp"]], tags[["userId", "timestamp"]]]
    users = (
        pd.concat(activity, ignore_index=True)
        .groupby("userId", sort=True, observed=True)["timestamp"]
        .min()
        .rename("firstObservedTimestamp")
        .reset_index()
        .merge(first_ratings, on="userId", how="left", validate="one_to_one")
    )
    _require_unique(users, ["userId"], "users")
    return users


def build_movies(movies: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Return one canonical movie row with nullable external identifiers."""
    _require_columns(movies, ("movieId", "title", "genres"), "movies")
    _require_columns(links, ("movieId", "imdbId", "tmdbId"), "links")
    _require_non_null(movies, ("movieId", "title", "genres"), "movies")
    _require_non_null(links, ("movieId",), "links")
    _require_unique(movies, ["movieId"], "movies")
    _require_unique(links, ["movieId"], "links")
    _validate_references(links["movieId"], movies["movieId"], "links", "movies")
    _validated_genre_rows(movies)

    result = movies[["movieId", "title", "genres"]].copy()
    parsed_year = result["title"].str.strip().str.extract(r"\((\d{4})\)$", expand=False)
    result["releaseYear"] = pd.to_numeric(parsed_year, errors="coerce").astype("UInt16")
    result["genreStatus"] = pd.Series(
        np.where(result["genres"].eq(NO_GENRES_LISTED), "missing_in_source", "listed"),
        index=result.index,
        dtype="string",
    )
    result = result.merge(
        links[["movieId", "imdbId", "tmdbId"]],
        on="movieId",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    result["imdbId"] = result["imdbId"].astype("UInt32")
    result["tmdbId"] = result["tmdbId"].astype("UInt32")
    result = result.drop(columns="genres")
    _require_unique(result, ["movieId"], "canonical_movies")
    return result


def build_genres_and_movie_genres(
    movies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize the source genre list into an entity and an M:N bridge."""
    _require_columns(movies, ("movieId", "genres"), "movies")
    _require_non_null(movies, ("movieId", "genres"), "movies")
    _require_unique(movies, ["movieId"], "movies")
    exploded = _validated_genre_rows(movies)
    bridge = exploded.loc[
        exploded["genreId"].ne(NO_GENRES_LISTED), ["movieId", "genreId"]
    ].reset_index(drop=True)
    bridge["genreId"] = bridge["genreId"].astype("string")
    genres = (
        bridge[["genreId"]]
        .drop_duplicates()
        .sort_values("genreId", kind="stable")
        .reset_index(drop=True)
    )
    genres["genreName"] = genres["genreId"].astype("string")
    validate_movie_genres(bridge, movies[["movieId"]], genres)
    return genres, bridge


def validate_movie_genres(
    movie_genres: pd.DataFrame, movies: pd.DataFrame, genres: pd.DataFrame
) -> None:
    """Validate uniqueness and referential integrity of the movie-genre bridge."""
    _require_columns(movie_genres, ("movieId", "genreId"), "movie_genres")
    _require_columns(movies, ("movieId",), "movies")
    _require_columns(genres, ("genreId",), "genres")
    _require_non_null(movie_genres, ("movieId", "genreId"), "movie_genres")
    _require_unique(movie_genres, ["movieId", "genreId"], "movie_genres")
    _require_unique(movies, ["movieId"], "movies")
    _require_unique(genres, ["genreId"], "genres")
    _validate_references(movie_genres["movieId"], movies["movieId"], "movie_genres", "movies")
    _validate_references(movie_genres["genreId"], genres["genreId"], "movie_genres", "genres")


def read_rating_events(
    parquet_path: Path,
    *,
    users: pd.DataFrame | None = None,
    movies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Read the canonical ratings snapshot and assign source-row provenance.

    IDs are scoped to this immutable source snapshot. They are assigned before
    any sorting, filtering, or partitioning and must be preserved thereafter.
    """
    ratings = pd.read_parquet(parquet_path, columns=list(RATING_EVENT_COLUMNS), engine="pyarrow")
    ratings.insert(
        0,
        "ratingEventId",
        pd.Series(np.arange(1, len(ratings) + 1, dtype=np.uint64), dtype="UInt64"),
    )
    return build_rating_events(ratings, users=users, movies=movies)


def build_rating_events(
    ratings: pd.DataFrame,
    *,
    users: pd.DataFrame | None = None,
    movies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate already-identified rating rows and derive the target.

    ``ratingEventId`` must have been assigned at the canonical source-read
    boundary. This function preserves it and never uses it as a temporal
    tie-breaker.
    """
    required = ("ratingEventId", *RATING_EVENT_COLUMNS)
    _require_columns(ratings, required, "ratings")
    _require_non_null(ratings, required, "ratings")
    _validate_timestamps(ratings["timestamp"], "ratings")
    _require_unique(ratings, ["ratingEventId"], "ratings")
    invalid_ratings = set(ratings["rating"].unique()) - RATING_VALUES
    if invalid_ratings:
        raise ValueError(f"ratings: invalid rating values: {sorted(invalid_ratings)}")
    if users is not None:
        _require_columns(users, ("userId",), "users")
        _require_unique(users, ["userId"], "users")
        _validate_references(ratings["userId"], users["userId"], "ratings", "users")
    if movies is not None:
        _require_columns(movies, ("movieId",), "movies")
        _require_unique(movies, ["movieId"], "movies")
        _validate_references(ratings["movieId"], movies["movieId"], "ratings", "movies")

    result = ratings.loc[:, required].copy()
    result["highRating"] = result["rating"].ge(4.0)
    _require_unique(result, ["ratingEventId"], "rating_events")
    return result
