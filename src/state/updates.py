"""Select point-in-time state inputs while retaining source-event provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.entities.builders import validate_movie_genres
from src.features.temporal import historical_events


def _validate_rating_event_identity(rating_events: pd.DataFrame) -> None:
    required = {"ratingEventId", "userId", "movieId", "rating", "timestamp"}
    if missing := required - set(rating_events.columns):
        raise ValueError(f"rating_events: missing required columns: {sorted(missing)}")
    if rating_events["ratingEventId"].isna().any():
        raise ValueError("rating_events: ratingEventId must not be missing")
    if rating_events["ratingEventId"].duplicated().any():
        raise ValueError("rating_events: ratingEventId must be unique")


def state_input_events_as_of(
    rating_events: pd.DataFrame,
    as_of_timestamp: Any,
    *,
    entity_filters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return cumulative ``H(t)`` inputs, not an incremental update delta."""
    _validate_rating_event_identity(rating_events)
    return historical_events(
        rating_events,
        as_of_timestamp,
        timestamp_column="timestamp",
        entity_filters=entity_filters,
    )


def relationship_state_inputs_as_of(
    rating_events: pd.DataFrame,
    relationship: pd.DataFrame,
    as_of_timestamp: Any,
    *,
    related_id_column: str,
    related_entities: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Expand cumulative eligible history to relationship-state inputs.

    Every output row retains ``ratingEventId``. A multi-valued movie
    relationship can therefore yield several state-update rows without
    creating or implying additional rating events. Repeated calls at later
    cutoffs overlap; callers must not apply each result as a fresh delta.
    """
    if related_id_column != "genreId":
        raise ValueError("only the available movie-genre relationship is supported")
    validate_movie_genres(relationship, movies, related_entities)
    unknown_movies = pd.Index(rating_events["movieId"].dropna().unique()).difference(
        movies["movieId"].dropna().unique()
    )
    if not unknown_movies.empty:
        raise ValueError(
            "rating_events: movieId values do not resolve to movies; "
            f"examples: {unknown_movies[:5].tolist()}"
        )
    eligible = state_input_events_as_of(rating_events, as_of_timestamp)
    return eligible.merge(
        relationship[["movieId", related_id_column]],
        on="movieId",
        how="inner",
        validate="many_to_many",
        sort=False,
    )
