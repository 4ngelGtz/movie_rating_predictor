"""Machine-readable summaries of the Phase 2 entity contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityContract:
    """The identity, grain, and source of one canonical entity."""

    key: tuple[str, ...]
    grain: str
    source_tables: tuple[str, ...]
    static_attributes: tuple[str, ...] = ()
    temporal_attributes: tuple[str, ...] = ()
    materialization: str = "available"


@dataclass(frozen=True)
class RelationshipContract:
    """The key and cardinality of one canonical relationship."""

    key: tuple[str, ...]
    grain: str
    cardinality: str
    optional_participation: bool
    relationship_attributes: tuple[str, ...] = ()
    materialization: str = "available"


ENTITY_CONTRACTS = {
    "user": EntityContract(
        key=("userId",),
        grain="one MovieLens user observed in ratings or tags",
        source_tables=("ratings", "tags"),
        temporal_attributes=("firstObservedTimestamp", "firstRatingTimestamp"),
    ),
    "movie": EntityContract(
        key=("movieId",),
        grain="one MovieLens catalog movie",
        source_tables=("movies", "links"),
        static_attributes=(
            "title",
            "releaseYear",
            "imdbId",
            "tmdbId",
            "genreStatus",
        ),
    ),
    "genre": EntityContract(
        key=("genreId",),
        grain="one exact canonical MovieLens genre label",
        source_tables=("movies",),
        static_attributes=("genreName",),
    ),
    "director": EntityContract(
        key=("directorId",),
        grain="one externally identified person credited as a director",
        source_tables=(),
        static_attributes=("directorName",),
        materialization="blocked: no people metadata or stable person IDs in current sources",
    ),
    "actor": EntityContract(
        key=("actorId",),
        grain="one externally identified person credited as an actor",
        source_tables=(),
        static_attributes=("actorName",),
        materialization="blocked: no people metadata or stable person IDs in current sources",
    ),
    "rating_event": EntityContract(
        key=("ratingEventId",),
        grain="one source rating row for one user and one movie at one recorded timestamp",
        source_tables=("ratings",),
        temporal_attributes=("timestamp", "rating", "highRating"),
    ),
}


RELATIONSHIP_CONTRACTS = {
    "user_rating_event": RelationshipContract(
        key=("ratingEventId", "userId"),
        grain="one rating event linked to its user",
        cardinality="user 1:N rating_event",
        optional_participation=False,
    ),
    "movie_rating_event": RelationshipContract(
        key=("ratingEventId", "movieId"),
        grain="one rating event linked to its movie",
        cardinality="movie 1:N rating_event",
        optional_participation=False,
    ),
    "movie_genre": RelationshipContract(
        key=("movieId", "genreId"),
        grain="one listed genre membership for one movie",
        cardinality="movie M:N genre",
        optional_participation=True,
    ),
    "movie_director": RelationshipContract(
        key=("movieId", "directorId"),
        grain="one director credit for one movie",
        cardinality="movie M:N director",
        optional_participation=True,
        relationship_attributes=("creditOrder",),
        materialization="blocked pending stable external person identifiers",
    ),
    "movie_actor": RelationshipContract(
        key=("movieId", "actorId"),
        grain="one cast credit for one movie",
        cardinality="movie M:N actor",
        optional_participation=True,
        relationship_attributes=("billingOrder",),
        materialization="blocked pending stable external person identifiers",
    ),
}
