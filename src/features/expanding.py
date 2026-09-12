"""Leakage-safe expanding global, user, and movie rating features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from src.data.schemas import RATING_VALUES
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


def _resolved_features(
    state: HistoricalRatingState,
    user_id: int,
    movie_id: int,
    prediction_timestamp: pd.Timestamp,
) -> tuple[int, float, int, float, float, float, int, float, float, int]:
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
    return (
        global_moments.count,
        global_mean,
        user_moments.count,
        user_moments.mean if user_moments.count else global_mean,
        user_moments.population_std,
        user_seconds_since_last_rating,
        movie_moments.count,
        movie_moments.mean if movie_moments.count else global_mean,
        movie_moments.population_std,
        state.movie_rating_count_30d.get(movie_id, 0),
    )


def build_expanding_rating_features(rating_events: pd.DataFrame) -> pd.DataFrame:
    """Build Phase 4A features at one output row per canonical rating event.

    Events are ordered only by timestamp.  For each timestamp, every row is
    scored against the same pre-batch state; the complete batch is applied only
    after all its features have been emitted.  ``ratingEventId`` is copied as
    provenance and is never used for temporal ordering or tie-breaking.
    Output rows retain the physical input order for join convenience.
    """
    _validate_rating_events(rating_events)
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
            )
            for column, value in zip(FEATURE_COLUMNS, resolved, strict=True):
                values[column][position] = value

        state.apply_timestamp_batch(
            (
                (int(row.userId), int(row.movieId), float(row.rating))
                for row in batch.itertuples(index=False)
            ),
            timestamp=timestamp,
        )

    return pd.concat(
        [
            rating_events.loc[:, CONTEXT_COLUMNS].reset_index(drop=True),
            pd.DataFrame(values),
        ],
        axis=1,
    )
