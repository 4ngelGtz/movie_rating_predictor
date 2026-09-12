"""Definitions for sparse, point-in-time entity state tables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateTableContract:
    """The identity and update boundary of a future state table.

    ``asOfTimestamp`` is an exclusive query cutoff: a complete logical snapshot
    at ``t`` summarizes all rating events whose timestamp is strictly less than
    ``t``. It is not the timestamp of the last applied event batch.
    """

    entity_keys: tuple[str, ...]
    primary_key: tuple[str, ...]
    source_relationship: str
    cold_start: str = "absent key means zero events; undefined aggregates remain missing"
    materialization: str = "ready"
    input_semantics: str = "cumulative H(t), not an incremental delta"


def _state(
    *entity_keys: str,
    source_relationship: str = "rating_event",
    materialization: str = "ready",
) -> StateTableContract:
    return StateTableContract(
        entity_keys=entity_keys,
        primary_key=(*entity_keys, "asOfTimestamp"),
        source_relationship=source_relationship,
        materialization=materialization,
    )


STATE_CONTRACTS = {
    "user_state": _state("userId"),
    "movie_state": _state("movieId"),
    "genre_state": _state("genreId", source_relationship="movie_genre"),
    "director_state": _state(
        "directorId",
        source_relationship="movie_director",
        materialization="deferred until stable director mappings exist",
    ),
    "actor_state": _state(
        "actorId",
        source_relationship="movie_actor",
        materialization="deferred until stable actor mappings exist",
    ),
    "user_genre_state": _state("userId", "genreId", source_relationship="movie_genre"),
    "user_director_state": _state(
        "userId",
        "directorId",
        source_relationship="movie_director",
        materialization="deferred until stable director mappings exist",
    ),
    "user_actor_state": _state(
        "userId",
        "actorId",
        source_relationship="movie_actor",
        materialization="deferred until stable actor mappings exist",
    ),
}
