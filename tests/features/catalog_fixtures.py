"""Canonical catalog fixtures shared by feature-family regression tests."""

from __future__ import annotations

import pandas as pd

from src.features.expanding import build_expanding_rating_features as _build_features


def canonical_movies_for(
    rating_events: pd.DataFrame,
    movie_genres: pd.DataFrame | None,
) -> pd.DataFrame:
    movie_ids = set(rating_events["movieId"].astype(int))
    if movie_genres is not None and "movieId" in movie_genres:
        movie_ids.update(movie_genres["movieId"].dropna().astype(int))
    listed = (
        set(movie_genres["movieId"].dropna().astype(int))
        if movie_genres is not None and "movieId" in movie_genres
        else set()
    )
    ordered = sorted(movie_ids)
    return pd.DataFrame(
        {
            "movieId": pd.Series(ordered, dtype="uint32"),
            "title": pd.Series(
                [f"Movie {movie_id} (2000)" for movie_id in ordered],
                dtype="string",
            ),
            "releaseYear": pd.Series([2000] * len(ordered), dtype="UInt16"),
            "genreStatus": pd.Series(
                [
                    "listed" if movie_id in listed else "missing_in_source"
                    for movie_id in ordered
                ],
                dtype="string",
            ),
            "imdbId": pd.Series([pd.NA] * len(ordered), dtype="UInt32"),
            "tmdbId": pd.Series([pd.NA] * len(ordered), dtype="UInt32"),
        }
    )


def build_features(
    rating_events: pd.DataFrame,
    movie_genres: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return _build_features(
        rating_events,
        movie_genres,
        canonical_movies_for(rating_events, movie_genres),
    )
