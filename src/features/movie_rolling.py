"""Movie-window output definitions; history lives in HistoricalRatingState."""

from __future__ import annotations

import math
import numpy as np

from src.state.moments import HistoricalRatingState


MOVIE_ROLLING_FEATURE_COLUMNS = (
    "movie_rating_count_30d",
    "movie_rating_count_60d",
    "movie_rating_avg_30d",
    "movie_rating_avg_60d",
    "movie_rating_avg_30d_over_60d",
    "movie_rating_count_30d_over_60d",
)
MOVIE_ROLLING_ADDITIONAL_FEATURE_COLUMNS = MOVIE_ROLLING_FEATURE_COLUMNS[1:]


def safe_ratio(numerator: float, denominator: float) -> float:
    """Undefined division stays missing, including a zero denominator."""
    if denominator == 0 or not math.isfinite(denominator):
        return np.nan
    return numerator / denominator


def resolve_movie_windows(
    state: HistoricalRatingState, movie_id: int,
) -> tuple[int, float, float, float, float]:
    """Read the five additional outputs from the same pre-batch movie state."""
    if not state.supports_movie_window_features:
        raise ValueError(
            "HistoricalRatingState was initialized or restored without PRD30 "
            "movie-window tracking; historical state cannot be upgraded in place. "
            "A full replay with movie-window tracking enabled is required."
        )
    count_30 = state.movie_rating_count_30d.get(movie_id, 0)
    count_60 = state.movie_rating_count_60d.get(movie_id, 0)
    avg_30 = np.float32(safe_ratio(
        state.movie_rating_sum_30d.get(movie_id, 0), count_30
    ))
    avg_60 = np.float32(safe_ratio(
        state.movie_rating_sum_60d.get(movie_id, 0), count_60
    ))
    return (
        count_60, avg_30, avg_60,
        safe_ratio(avg_30, avg_60), safe_ratio(count_30, count_60),
    )
