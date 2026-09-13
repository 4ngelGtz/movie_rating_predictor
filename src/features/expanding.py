"""Leakage-safe expanding rating and user-target-genre features."""

from __future__ import annotations

from datetime import datetime
import math

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

from src.data.schemas import RATING_VALUES
from src.features.catalog import build_catalog_lookups
from src.state.moments import HistoricalRatingState, RunningMoments


GLOBAL_MEAN_COLD_START = 3.5
CONTEXT_COLUMNS = ("ratingEventId", "userId", "movieId", "timestamp")
DYNAMIC_FEATURE_COLUMNS = (
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
STATIC_FEATURE_COLUMNS = (
    "movie_genre_count",
    "movie_release_year",
    "movie_release_year_missing",
    "movie_age_years",
)
FEATURE_COLUMNS = (*DYNAMIC_FEATURE_COLUMNS, *STATIC_FEATURE_COLUMNS)


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


def _movie_age_years(timestamp: pd.Timestamp, release_year: int) -> float:
    """Return elapsed Gregorian years without pandas timestamp range limits."""
    event_datetime = datetime(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
        timestamp.microsecond,
    )
    elapsed = event_datetime - datetime(release_year, 1, 1)
    elapsed_seconds = (
        elapsed.days * 86_400
        + elapsed.seconds
        + elapsed.microseconds / 1_000_000
        + timestamp.nanosecond / 1_000_000_000
    )
    return elapsed_seconds / (365.2425 * 86_400)


def build_expanding_rating_features(
    rating_events: pd.DataFrame,
    movie_genres: pd.DataFrame | None = None,
    canonical_movies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Phase 4A-D features at one row per canonical rating event.

    Events are ordered only by timestamp.  For each timestamp, every row is
    scored against the same pre-batch state; the complete batch is applied only
    after all its features have been emitted.  ``ratingEventId`` is copied as
    provenance and is never used for temporal ordering or tie-breaking.
    Output rows retain the physical input order for join convenience.
    ``canonical_movies`` and ``movie_genres`` are one explicit, versioned
    canonical catalog snapshot and must both be supplied, including when the
    valid canonical bridge is empty.
    """
    _validate_rating_events(rating_events)
    catalog = build_catalog_lookups(
        canonical_movies,
        movie_genres,
        required_movie_ids=rating_events["movieId"],
    )
    genres_by_movie = catalog.genres_by_movie
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
        "movie_genre_count": np.empty(event_count, dtype=np.uint8),
        "movie_release_year": np.empty(event_count, dtype=np.float64),
        "movie_release_year_missing": np.empty(event_count, dtype=np.bool_),
        "movie_age_years": np.empty(event_count, dtype=np.float32),
    }
    if event_count == 0:
        values["movie_release_year"] = pd.array([], dtype="Int16")
        return pd.concat(
            [
                rating_events.loc[:, CONTEXT_COLUMNS].reset_index(drop=True),
                pd.DataFrame(values),
            ],
            axis=1,
        )

    # Resolve every static value before dynamic state exists. Besides keeping
    # catalog failures atomic, this makes static work independent of batch order.
    for position, row in enumerate(
        rating_events[["movieId", "timestamp"]].itertuples(index=False)
    ):
        movie_id = int(row.movieId)
        release_year = catalog.release_year_by_movie[movie_id]
        values["movie_genre_count"][position] = len(
            genres_by_movie.get(movie_id, ())
        )
        values["movie_release_year_missing"][position] = release_year is None
        if release_year is None:
            values["movie_release_year"][position] = np.nan
            values["movie_age_years"][position] = np.nan
        else:
            values["movie_release_year"][position] = release_year
            values["movie_age_years"][position] = _movie_age_years(
                pd.Timestamp(row.timestamp), release_year
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
            for column, value in zip(DYNAMIC_FEATURE_COLUMNS, resolved, strict=True):
                values[column][position] = value

        state.apply_timestamp_batch(
            (
                (int(row.userId), int(row.movieId), float(row.rating))
                for row in batch.itertuples(index=False)
            ),
            timestamp=timestamp,
            movie_genres=genres_by_movie,
        )

    values["movie_release_year"] = pd.array(values["movie_release_year"], dtype="Int16")
    return pd.concat(
        [
            rating_events.loc[:, CONTEXT_COLUMNS].reset_index(drop=True),
            pd.DataFrame(values),
        ],
        axis=1,
    )
