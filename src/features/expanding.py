"""Leakage-safe expanding rating and user-target-genre features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from src.data.schemas import RATING_VALUES
from src.entities.builders import NO_GENRES_LISTED
from src.state.moments import HistoricalRatingState, RunningMoments


GLOBAL_MEAN_COLD_START = 3.5
CONTEXT_COLUMNS = ("ratingEventId", "userId", "movieId", "timestamp")
FEATURE_COLUMNS = (
    "global_rating_count",
    "global_mean_rating",
    "user_rating_count",
    "user_mean_rating",
    "user_rating_std_pop",
    "user_seconds_since_last_rating",
    "movie_rating_count",
    "movie_mean_rating",
    "movie_rating_std_pop",
    "movie_rating_count_30d",
    "user_target_genre_rating_count",
    "user_target_genre_mean_rating",
    "user_target_genre_mean_delta",
)


def _validate_rating_events(rating_events: pd.DataFrame) -> None:
    required = {*CONTEXT_COLUMNS, "rating"}
    if missing := required - set(rating_events.columns):
        raise ValueError(f"rating_events: missing required columns: {sorted(missing)}")
    if rating_events[list(required)].isna().any().any():
        raise ValueError("rating_events: required columns must not contain missing values")
    for column in ("ratingEventId", "userId", "movieId"):
        if not is_integer_dtype(rating_events[column].dtype):
            raise TypeError(f"rating_events: {column} must have an integer dtype")
    if rating_events["ratingEventId"].duplicated().any():
        raise ValueError("rating_events: ratingEventId must be unique")
    if not is_numeric_dtype(rating_events["rating"].dtype):
        raise TypeError("rating_events: rating must have a numeric dtype")
    invalid_ratings = set(rating_events["rating"].astype(float).unique()) - RATING_VALUES
    if invalid_ratings:
        raise ValueError(f"rating_events: invalid rating values: {sorted(invalid_ratings)}")
    timestamps = rating_events["timestamp"]
    if not is_datetime64_any_dtype(timestamps.dtype):
        raise TypeError("rating_events: timestamp must have a pandas datetime64 dtype")
    if timestamps.dt.tz is not None:
        raise ValueError("rating_events: timestamp must be timezone-naive")


def _movie_genres_by_movie(
    movie_genres: pd.DataFrame | None,
) -> dict[int, tuple[str, ...]]:
    """Validate and index the canonical Phase 2 movie-genre bridge."""
    if movie_genres is None:
        raise ValueError(
            "movie_genres must be explicitly supplied when building Phase 4C features"
        )
    required = ["movieId", "genreId"]
    if missing := set(required) - set(movie_genres.columns):
        raise ValueError(f"movie_genres: missing required columns: {sorted(missing)}")
    if movie_genres[required].isna().any().any():
        raise ValueError("movie_genres: required columns must not contain missing values")
    if not is_integer_dtype(movie_genres["movieId"].dtype):
        raise TypeError("movie_genres: movieId must have an integer dtype")
    if not is_string_dtype(movie_genres["genreId"].dtype):
        raise TypeError("movie_genres: genreId must have a string dtype")
    if movie_genres.duplicated(required).any():
        raise ValueError("movie_genres: (movieId, genreId) must be unique")

    genre_ids = movie_genres["genreId"]
    if genre_ids.eq("").any():
        raise ValueError("movie_genres: genreId must not be empty")
    if genre_ids.str.strip().ne(genre_ids).any():
        raise ValueError("movie_genres: genreId must not have surrounding whitespace")
    if genre_ids.eq(NO_GENRES_LISTED).any():
        raise ValueError("movie_genres: no-genres sentinel is not a genre entity")

    result: dict[int, list[str]] = {}
    for row in movie_genres[required].itertuples(index=False):
        result.setdefault(int(row.movieId), []).append(str(row.genreId))
    return {
        movie_id: tuple(sorted(genres)) for movie_id, genres in result.items()
    }


def _resolved_features(
    state: HistoricalRatingState,
    user_id: int,
    movie_id: int,
    prediction_timestamp: pd.Timestamp,
    movie_genres: dict[int, tuple[str, ...]],
) -> tuple[
    int,
    float,
    int,
    float,
    float,
    float,
    int,
    float,
    float,
    int,
    int,
    float,
    float,
]:
    global_moments = state.global_moments
    global_mean = (
        global_moments.mean if global_moments.count else GLOBAL_MEAN_COLD_START
    )
    user_moments = state.user_moments.get(user_id, RunningMoments())
    movie_moments = state.movie_moments.get(movie_id, RunningMoments())
    previous_user_rating = state.last_user_rating.get(user_id)
    user_seconds_since_last_rating = (
        (prediction_timestamp - previous_user_rating).total_seconds()
        if previous_user_rating is not None
        else np.nan
    )
    user_mean = user_moments.mean if user_moments.count else global_mean
    target_genre_moments = [
        state.user_genre_moments.get((user_id, genre_id))
        for genre_id in movie_genres.get(movie_id, ())
    ]
    target_genre_moments = [
        moments for moments in target_genre_moments if moments is not None
    ]
    target_genre_count = sum(moments.count for moments in target_genre_moments)
    target_genre_mean = (
        math.fsum(
            moments.mean * moments.count for moments in target_genre_moments
        )
        / target_genre_count
        if target_genre_count
        else user_mean
    )
    return (
        global_moments.count,
        global_mean,
        user_moments.count,
        user_mean,
        user_moments.population_std,
        user_seconds_since_last_rating,
        movie_moments.count,
        movie_moments.mean if movie_moments.count else global_mean,
        movie_moments.population_std,
        state.movie_rating_count_30d.get(movie_id, 0),
        target_genre_count,
        target_genre_mean,
        target_genre_mean - user_mean,
    )


def build_expanding_rating_features(
    rating_events: pd.DataFrame,
    movie_genres: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Phase 4A-C features at one row per canonical rating event.

    Events are ordered only by timestamp.  For each timestamp, every row is
    scored against the same pre-batch state; the complete batch is applied only
    after all its features have been emitted.  ``ratingEventId`` is copied as
    provenance and is never used for temporal ordering or tie-breaking.
    Output rows retain the physical input order for join convenience.
    ``movie_genres`` must be supplied explicitly, including when the valid
    canonical bridge is empty.
    """
    _validate_rating_events(rating_events)
    genres_by_movie = _movie_genres_by_movie(movie_genres)
    event_count = len(rating_events)
    values: dict[str, np.ndarray] = {
        "global_rating_count": np.empty(event_count, dtype=np.uint64),
        "global_mean_rating": np.empty(event_count, dtype=np.float32),
        "user_rating_count": np.empty(event_count, dtype=np.uint64),
        "user_mean_rating": np.empty(event_count, dtype=np.float32),
        "user_rating_std_pop": np.empty(event_count, dtype=np.float32),
        "user_seconds_since_last_rating": np.empty(event_count, dtype=np.float64),
        "movie_rating_count": np.empty(event_count, dtype=np.uint64),
        "movie_mean_rating": np.empty(event_count, dtype=np.float32),
        "movie_rating_std_pop": np.empty(event_count, dtype=np.float32),
        "movie_rating_count_30d": np.empty(event_count, dtype=np.uint64),
        "user_target_genre_rating_count": np.empty(event_count, dtype=np.uint64),
        "user_target_genre_mean_rating": np.empty(event_count, dtype=np.float32),
        "user_target_genre_mean_delta": np.empty(event_count, dtype=np.float32),
    }
    if event_count == 0:
        return pd.concat(
            [
                rating_events.loc[:, CONTEXT_COLUMNS].reset_index(drop=True),
                pd.DataFrame(values),
            ],
            axis=1,
        )

    work = rating_events.loc[:, (*CONTEXT_COLUMNS, "rating")].copy()
    work["inputPosition"] = np.arange(event_count)
    work = work.sort_values("timestamp", kind="stable")
    state = HistoricalRatingState()

    for timestamp, batch in work.groupby("timestamp", sort=False):
        state.expire_movie_activity(timestamp)
        for row in batch.itertuples(index=False):
            position = row.inputPosition
            resolved = _resolved_features(
                state,
                int(row.userId),
                int(row.movieId),
                timestamp,
                genres_by_movie,
            )
            for column, value in zip(FEATURE_COLUMNS, resolved, strict=True):
                values[column][position] = value

        state.apply_timestamp_batch(
            (
                (int(row.userId), int(row.movieId), float(row.rating))
                for row in batch.itertuples(index=False)
            ),
            timestamp=timestamp,
            movie_genres=genres_by_movie,
        )

    return pd.concat(
        [
            rating_events.loc[:, CONTEXT_COLUMNS].reset_index(drop=True),
            pd.DataFrame(values),
        ],
        axis=1,
    )
