"""Validated frozen-catalog lookups for static movie context features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype

from src.entities.builders import NO_GENRES_LISTED


CANONICAL_MOVIE_COLUMNS = (
    "movieId",
    "title",
    "releaseYear",
    "genreStatus",
    "imdbId",
    "tmdbId",
)
GENRE_STATUSES = frozenset({"listed", "missing_in_source"})


@dataclass(frozen=True)
class CatalogLookups:
    """Read-only lookup maps built from one validated canonical snapshot."""

    genres_by_movie: dict[int, tuple[str, ...]]
    release_year_by_movie: dict[int, int | None]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], table: str) -> None:
    if missing := set(columns) - set(frame.columns):
        raise ValueError(f"{table}: missing required columns: {sorted(missing)}")


def _validate_movies(canonical_movies: pd.DataFrame | None) -> None:
    if canonical_movies is None:
        raise ValueError(
            "canonical_movies must be explicitly supplied when building Phase 4D features"
        )
    _require_columns(canonical_movies, CANONICAL_MOVIE_COLUMNS, "canonical_movies")
    if canonical_movies["movieId"].isna().any():
        raise ValueError("canonical_movies: movieId must not contain missing values")
    if not is_integer_dtype(canonical_movies["movieId"].dtype):
        raise TypeError("canonical_movies: movieId must have an integer dtype")
    if canonical_movies["movieId"].duplicated().any():
        raise ValueError("canonical_movies: movieId must be unique")
    if canonical_movies["title"].isna().any():
        raise ValueError("canonical_movies: title must not contain missing values")
    if not is_string_dtype(canonical_movies["title"].dtype):
        raise TypeError("canonical_movies: title must have a string dtype")
    if not is_integer_dtype(canonical_movies["releaseYear"].dtype):
        raise TypeError("canonical_movies: releaseYear must have a nullable integer dtype")
    nonmissing_years = canonical_movies["releaseYear"].dropna()
    if ((nonmissing_years < 1) | (nonmissing_years > 9999)).any():
        raise ValueError("canonical_movies: releaseYear must be a four-digit parsed value")
    if canonical_movies["genreStatus"].isna().any():
        raise ValueError("canonical_movies: genreStatus must not contain missing values")
    if not is_string_dtype(canonical_movies["genreStatus"].dtype):
        raise TypeError("canonical_movies: genreStatus must have a string dtype")
    invalid_statuses = set(canonical_movies["genreStatus"].unique()) - GENRE_STATUSES
    if invalid_statuses:
        raise ValueError(
            "canonical_movies: invalid genreStatus values: "
            f"{sorted(invalid_statuses)}"
        )
    for column in ("imdbId", "tmdbId"):
        if not is_integer_dtype(canonical_movies[column].dtype):
            raise TypeError(f"canonical_movies: {column} must have a nullable integer dtype")


def _validate_and_index_bridge(
    movie_genres: pd.DataFrame | None,
    canonical_movies: pd.DataFrame,
) -> dict[int, tuple[str, ...]]:
    if movie_genres is None:
        raise ValueError(
            "movie_genres must be explicitly supplied when building Phase 4C-D features"
        )
    required = ("movieId", "genreId")
    _require_columns(movie_genres, required, "movie_genres")
    if movie_genres[list(required)].isna().any().any():
        raise ValueError("movie_genres: required columns must not contain missing values")
    if not is_integer_dtype(movie_genres["movieId"].dtype):
        raise TypeError("movie_genres: movieId must have an integer dtype")
    if not is_string_dtype(movie_genres["genreId"].dtype):
        raise TypeError("movie_genres: genreId must have a string dtype")
    if movie_genres.duplicated(list(required)).any():
        raise ValueError("movie_genres: (movieId, genreId) must be unique")

    genre_ids = movie_genres["genreId"]
    if genre_ids.eq("").any():
        raise ValueError("movie_genres: genreId must not be empty")
    if genre_ids.str.strip().ne(genre_ids).any():
        raise ValueError("movie_genres: genreId must not have surrounding whitespace")
    if genre_ids.str.contains("|", regex=False).any():
        raise ValueError("movie_genres: genreId must be one normalized genre token")
    if genre_ids.eq(NO_GENRES_LISTED).any():
        raise ValueError("movie_genres: no-genres sentinel is not a genre entity")

    known_movies = set(canonical_movies["movieId"].astype(int))
    unknown_movies = sorted(set(movie_genres["movieId"].astype(int)) - known_movies)
    if unknown_movies:
        raise ValueError(
            "movie_genres: movieId values do not resolve to canonical_movies; "
            f"examples: {unknown_movies[:5]}"
        )

    mutable: dict[int, list[str]] = {}
    for row in movie_genres[list(required)].itertuples(index=False):
        mutable.setdefault(int(row.movieId), []).append(str(row.genreId))
    result = {
        movie_id: tuple(sorted(genres)) for movie_id, genres in mutable.items()
    }
    if any(len(genres) > 255 for genres in result.values()):
        raise ValueError("movie_genres: a movie cannot exceed uint8 genre-count capacity")

    statuses = canonical_movies.set_index("movieId")["genreStatus"]
    for raw_movie_id, status in statuses.items():
        movie_id = int(raw_movie_id)
        has_memberships = bool(result.get(movie_id))
        if status == "missing_in_source" and has_memberships:
            raise ValueError(
                "canonical catalog: missing_in_source movie must have no genre memberships"
            )
        if status == "listed" and not has_memberships:
            raise ValueError(
                "canonical catalog: listed movie must have at least one genre membership"
            )
    return result


def build_catalog_lookups(
    canonical_movies: pd.DataFrame | None,
    movie_genres: pd.DataFrame | None,
    *,
    required_movie_ids: pd.Series | None = None,
) -> CatalogLookups:
    """Validate one canonical catalog snapshot and build deterministic lookups."""
    _validate_movies(canonical_movies)
    assert canonical_movies is not None
    genres_by_movie = _validate_and_index_bridge(movie_genres, canonical_movies)

    known_movies = set(canonical_movies["movieId"].astype(int))
    if required_movie_ids is not None:
        missing_movies = sorted(set(required_movie_ids.astype(int)) - known_movies)
        if missing_movies:
            raise ValueError(
                "rating_events: movieId values do not resolve to canonical_movies; "
                f"examples: {missing_movies[:5]}"
            )

    release_year_by_movie = {
        int(row.movieId): None if pd.isna(row.releaseYear) else int(row.releaseYear)
        for row in canonical_movies[["movieId", "releaseYear"]].itertuples(index=False)
    }
    return CatalogLookups(genres_by_movie, release_year_by_movie)


def _json_value(value: object) -> str | int | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value
    return int(value)


def catalog_snapshot_id(
    canonical_movies: pd.DataFrame,
    movie_genres: pd.DataFrame,
) -> str:
    """Return an order-invariant SHA-256 identity for canonical catalog content.

    The digest deliberately covers every Phase 2 canonical movie attribute,
    not only the attributes consumed by Phase 4D, plus every bridge membership.
    """
    build_catalog_lookups(canonical_movies, movie_genres)
    movies = canonical_movies.loc[:, CANONICAL_MOVIE_COLUMNS].sort_values(
        "movieId", kind="stable"
    )
    bridge = movie_genres.loc[:, ["movieId", "genreId"]].sort_values(
        ["movieId", "genreId"], kind="stable"
    )
    payload = {
        "schema": "movie-rating-predictor/catalog-snapshot/v1",
        "canonical_movies": [
            [_json_value(value) for value in row]
            for row in movies.itertuples(index=False, name=None)
        ],
        "movie_genre": [
            [_json_value(value) for value in row]
            for row in bridge.itertuples(index=False, name=None)
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
