"""Authoritative point-in-time eligibility rules for event features.

All functions use an exclusive prediction-time boundary. They deliberately do
not use input row order to break ties between equal timestamps.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from numbers import Number
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


ROLLING_INTERVAL = "[t - window, t)"
_RELATIVE_TIMESTAMP_STRINGS = frozenset({"now", "today", "yesterday", "tomorrow"})


def _validated_timestamps(timestamps: pd.Series) -> pd.Series:
    if not isinstance(timestamps, pd.Series):
        raise TypeError("timestamps must be a pandas Series")
    if not is_datetime64_any_dtype(timestamps.dtype):
        raise TypeError("timestamps must have a pandas datetime64 dtype")
    if timestamps.isna().any():
        raise ValueError("timestamps must not contain missing values")
    if timestamps.dt.tz is not None:
        raise ValueError("timestamps must be timezone-naive")
    return timestamps


def _validated_prediction_timestamp(prediction_timestamp: Any) -> pd.Timestamp:
    if prediction_timestamp is None or prediction_timestamp is pd.NaT:
        raise ValueError("prediction_timestamp must not be missing")
    if isinstance(prediction_timestamp, Number):
        raise TypeError("numeric prediction timestamps require an explicit-unit API")
    if isinstance(prediction_timestamp, str):
        if prediction_timestamp.strip().casefold() in _RELATIVE_TIMESTAMP_STRINGS:
            raise ValueError("prediction_timestamp must be deterministic")
    elif not isinstance(prediction_timestamp, (pd.Timestamp, datetime, np.datetime64)):
        raise TypeError(
            "prediction_timestamp must be a Timestamp, datetime, datetime64, "
            "or deterministic datetime string"
        )
    timestamp = pd.Timestamp(prediction_timestamp)
    if pd.isna(timestamp):
        raise ValueError("prediction_timestamp must not be missing")
    if timestamp.tz is not None:
        raise ValueError("prediction_timestamp must be timezone-naive")
    return timestamp


def _validated_window(window: Any) -> pd.Timedelta:
    if window is None or window is pd.NaT:
        raise ValueError("window must not be missing")
    if isinstance(window, (pd.Timedelta, timedelta, np.timedelta64)):
        pass
    elif isinstance(window, Number):
        raise TypeError("numeric windows are ambiguous; use an explicit duration unit")
    elif isinstance(window, str):
        if not re.search(r"[A-Za-z]", window):
            raise TypeError("string windows must include an explicit duration unit")
    else:
        raise TypeError(
            "window must be a Timedelta, timedelta, timedelta64, or explicit-unit string"
        )
    duration = pd.Timedelta(window)
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError("window must be positive")
    return duration


def is_strictly_prior(timestamps: pd.Series, prediction_timestamp: Any) -> pd.Series:
    """Return a mask for events available strictly before prediction time."""
    values = _validated_timestamps(timestamps)
    cutoff = _validated_prediction_timestamp(prediction_timestamp)
    return values < cutoff


def rolling_window_mask(
    timestamps: pd.Series, prediction_timestamp: Any, window: Any
) -> pd.Series:
    """Return the mask for the left-closed, right-open interval ``[t-W, t)``."""
    values = _validated_timestamps(timestamps)
    cutoff = _validated_prediction_timestamp(prediction_timestamp)
    duration = _validated_window(window)
    return values.ge(cutoff - duration) & values.lt(cutoff)


def historical_events(
    events: pd.DataFrame,
    prediction_timestamp: Any,
    *,
    timestamp_column: str = "timestamp",
    entity_filters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Select strict prior history, optionally scoped to exact entity values.

    The selected rows retain their original order. Ordering is irrelevant to
    eligibility: every row at the cutoff is excluded before entity filtering.
    """
    if timestamp_column not in events:
        raise KeyError(f"missing timestamp column: {timestamp_column}")
    mask = is_strictly_prior(events[timestamp_column], prediction_timestamp)
    for column, value in (entity_filters or {}).items():
        if column not in events:
            raise KeyError(f"missing entity column: {column}")
        mask &= events[column].eq(value)
    return events.loc[mask].copy()
