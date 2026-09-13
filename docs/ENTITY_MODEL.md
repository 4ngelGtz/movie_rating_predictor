# Phase 2 Entity and State Contract

This document is the authoritative entity and state contract. It defines
canonical entities and point-in-time state identity. It does not define
predictive aggregates, train models, or change the Phase 0 processed schemas.
Source column names are retained so IDs are not silently rewritten or
reinterpreted.

## Code mapping

Phase 2 code mirrors this document as follows:

```text
ENTITY_MODEL.md  (authoritative human contract)
        │
        ├── src/entities/contracts.py   # registry: entities and relationships
        ├── src/state/contracts.py      # registry: state-table identity
        │
        ├── src/entities/builders.py    # build and validate canonical tables
        │         │
        │         └── rating_events (+ movie_genres, …)
        │                   │
        └── src/state/updates.py        # select H(t) and expand to state inputs
```

- `src/entities/contracts.py` and `src/state/contracts.py` are machine-readable
  registries of entity, relationship, and state-table identity. They document
  keys, grains, and materialization status; they do not build tables or apply
  temporal cuts.
- `src/entities/builders.py` constructs and validates canonical entity and
  bridge tables from Phase 0 inputs, including one-time `ratingEventId`
  assignment at the ratings Parquet read boundary.
- `src/state/updates.py` selects cumulative point-in-time state inputs from
  those rating events (`timestamp < asOfTimestamp`) and expands multi-valued
  relationships while preserving `ratingEventId` provenance. It does not
  compute predictive aggregates.

Builders feed updates: only validated `rating_events` (and available bridges
such as movie–genre) become state-update inputs. Contracts are the shared
vocabulary both layers must stay aligned with.

## Temporal foundation

Phase 1 remains authoritative. At prediction time `t`, available event history
is exactly `{event | timestamp < t}`. Events at `t` are simultaneous: first
read state before `t`, then predict every event at `t`, and only then apply the
entire timestamp batch for later predictions. An `asOfTimestamp` is therefore
an exclusive cutoff, not the last included timestamp.

The source timestamps are timezone-naive, whole-second observations stored as
Parquet `timestamp[us]`. There is no ingestion timestamp or within-second
sequence. No entity or state key may be used to invent one.

## Entity Relantionship overview

```text
user 1 ─── N rating_event N ─── 1 movie
                                  │
                                  ├── M:N genre
                                  ├── M:N director  (mapping unavailable)
                                  └── M:N actor     (mapping unavailable)
```



## Canonical entities


| Entity       | Primary key     | One-row-per grain                                                               | Sources           | Attributes and time semantics                                                                                                                                                                                                                                                                                                                     |
| ------------ | --------------- | ------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| User         | `userId`        | MovieLens user observed in ratings or tags                                      | `ratings`, `tags` | There are no truly static user attributes. `firstObservedTimestamp` is the earliest rating or tag time, and nullable `firstRatingTimestamp` is the earliest rating time. Both are lifecycle metadata derived from the extract and must not be attached to earlier observations as predictors.                                                     |
| Movie        | `movieId`       | MovieLens catalog movie                                                         | `movies`, `links` | `title` is preserved verbatim. Nullable `releaseYear` is conservatively parsed only from a terminal `(YYYY)` after ignoring outer whitespace; `(0000)` is missing because the Gregorian calendar and movie-age definition have no year zero. `imdbId` and `tmdbId` use nullable types; neither is the canonical key. `genreStatus` is `listed` or `missing_in_source`. Feature Dictionary v1 treats these undated values as a frozen canonical catalog snapshot under its documented availability assumption. |
| Genre        | `genreId`       | Exact canonical MovieLens genre label                                           | `movies`          | `genreName` equals the already-normalized source token. Labels are not case-folded, slugged, or otherwise merged. `(no genres listed)` is not a genre entity.                                                                                                                                                                                     |
| Director     | `directorId`    | Externally identified person with a director credit                             | not available     | No current table contains directors or stable person IDs. Names must not be used as invented identifiers. Future cached enrichment must supply a stable, source-qualified person ID.                                                                                                                                                              |
| Actor        | `actorId`       | Externally identified person with a cast credit                                 | not available     | Same identity constraint as directors. A person may appear in any number of movies.                                                                                                                                                                                                                                                               |
| Rating event | `ratingEventId` | One preserved source rating row for one user and one movie at one recorded time | `ratings`         | Carries `userId`, `movieId`, `rating`, `timestamp`, and derived target `highRating = rating >= 4`. Rating and target are outcomes unavailable to their own prediction.                                                                                                                                                                            |


`ratingEventId` is a 1-based, source-snapshot-scoped surrogate assigned once by
`read_rating_events`, directly from canonical Parquet row order before sorting,
filtering, or partitioning. Phase 0 validates preservation of that row order.
Every downstream table must carry the assigned ID rather than recompute it.
The ID is identity/provenance, not chronology. The descriptive tuple
`(userId, movieId, timestamp)` happens to be unique in the current extract, but
is not a permanent primary key because future data may contain repeated
ratings. Exact duplicate rows are retained as distinct source events and
receive distinct IDs; no silent deduplication occurs. Mixing different source
snapshots requires an additional snapshot/version identity because their
numeric ID ranges are independent.

Allowed ratings are `0.5, 1.0, ..., 5.0`. User and movie IDs are required and
must resolve to their canonical entities. All current rating references resolve.

## Relationships


| Relationship       | Key and cardinality                                  | Nullable and relationship metadata                                                                                                                                                                                         |
| ------------------ | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| User–rating event  | `ratingEventId` → one `userId`; user `1:N` events    | Required                                                                                                                                                                                                                   |
| Movie–rating event | `ratingEventId` → one `movieId`; movie `1:N` events  | Required                                                                                                                                                                                                                   |
| Movie–genre        | unique `(movieId, genreId)`; movie `M:N` genre       | Participation is optional: a movie may have zero, one, or many rows. Keys in an existing row are always non-null. The 246 current `(no genres listed)` movies have zero rows and explicit `genreStatus=missing_in_source`. |
| Movie–director     | unique `(movieId, directorId)`; movie `M:N` director | Zero or many credits. If a reliable source provides ordering, preserve it as nullable `creditOrder` on this relationship. Not currently materializable.                                                                    |
| Movie–actor        | unique `(movieId, actorId)`; movie `M:N` actor       | Zero or many credits. Preserve source billing position as nullable `billingOrder` on this relationship, never as an actor attribute. Not currently materializable.                                                         |


External IDs are nullable because a catalog row or future enrichment match can
be absent. Duplicate non-null external IDs must not merge MovieLens movies:
the current links contain 35 rows involved in duplicate `tmdbId` values, while
`movieId` remains unique and authoritative.

## State tables

State is sparse: rows are created only for keys touched by eligible historical
events, never by constructing Cartesian products. A complete logical snapshot
has primary key `(<entity keys>, asOfTimestamp)`, where `asOfTimestamp` is the
exclusive query cutoff and the snapshot represents all of `H(asOfTimestamp)`.
It is not the timestamp of the last applied batch. This phase defines identity
and update eligibility only; counts, sums, means, windows, affinities, fallback
values, and feature dtypes belong to feature specification.


| State                 | Entity keys            | Source update relationship                 | Status                           |
| --------------------- | ---------------------- | ------------------------------------------ | -------------------------------- |
| `user_state`          | `userId`               | Rating event by the user                   | Ready                            |
| `movie_state`         | `movieId`              | Rating event for the movie                 | Ready                            |
| `genre_state`         | `genreId`              | Rating event joined through movie–genre    | Ready                            |
| `director_state`      | `directorId`           | Rating event joined through movie–director | Deferred pending stable mappings |
| `actor_state`         | `actorId`              | Rating event joined through movie–actor    | Deferred pending stable mappings |
| `user_genre_state`    | `(userId, genreId)`    | Rating event joined through movie–genre    | Ready                            |
| `user_director_state` | `(userId, directorId)` | Rating event joined through movie–director | Deferred pending stable mappings |
| `user_actor_state`    | `(userId, actorId)`    | Rating event joined through movie–actor    | Deferred pending stable mappings |


User–movie state is intentionally absent: the current extract contains no
repeated user–movie interactions, so it has no demonstrated incremental signal.
Genre–movie or people–movie state would restate static bridges rather than an
event-updated state. Other Cartesian pairs are not justified.

### Update provenance and multi-valued relationships

One rating is always one row in `rating_events`. If its movie has three genres,
that single event yields three genre update inputs and three user–genre update
inputs. Each update input retains the same `ratingEventId`. Consequently:

- canonical event count is the number of distinct `ratingEventId` values;
- state-update row count is the number of affected relationship keys;
- a duplicated bridge key is invalid because it would inflate updates;
- a movie with missing genre information produces no genre-derived updates;
- future director/actor expansion must follow the same provenance rule.

For a timestamp batch `t`, no update input from `t` is eligible for a snapshot
with `asOfTimestamp=t`. The complete batch becomes eligible only for cutoffs
strictly later than `t`.

The current state helpers return cumulative historical inputs `H(t)`, not an
incremental delta. Results for later cutoffs overlap earlier results and must
not be reapplied wholesale to an existing accumulator.

A future persisted checkpoint must represent a complete `H(checkpointCutoff)`
and must be written only between complete timestamp batches. To answer a later
query at `t`, restore the checkpoint and replay complete batches satisfying
`checkpointCutoff <= event.timestamp < t`; compute every prediction in a batch
before applying that batch. Operational metadata may separately record the
last fully applied batch timestamp, which is nullable for empty history and is
strictly less than the query cutoff. It is not an `asOfTimestamp` substitute.

Each source event may be applied at most once to each affected state key. The
idempotency/provenance identity is `(state table, entity key, ratingEventId)`.
For multi-valued relationships, one event legitimately has one such identity
per related key. Retry or replay logic must prevent a second application of the
same identity. Serializable state primitives now exist, but operational
duplicate-application enforcement, checkpoint/replay orchestration, persisted
cutoff metadata, and physical checkpoint cadence remain Phase 4E work.

## Cold start and invariants

An absent state key means no eligible prior rating events: support count is
conceptually `0`, while aggregates without observations are missing rather than
silently set to zero. Any later fallback must be explicit and computed under
the same observation-level `< t` rule.

Validation requires:

- unique canonical entity keys and unique bridge keys;
- optional bridge participation, with non-null keys in every existing row and
complete referential integrity;
- nullable external IDs without requiring uniqueness for identity;
- one canonical event per source row with a unique provenance ID;
- rating values in the Phase 0 domain and timezone-naive event timestamps;
- strict temporal selection through `src.features.temporal`, including equal
timestamp exclusion;
- no genre/person update without a valid movie relationship row.



## Source limitations and assumptions

- The catalog, links, and genome tables have no availability timestamp. Feature
  Dictionary v1 explicitly adopts a frozen canonical snapshot assumption for
  `movies` and `movie_genre`; historical availability cannot be proven and this
  limitation must be disclosed. Genome data remains deferred.
- Phase 4D provides a deterministic, order-invariant SHA-256
  `catalog_snapshot_id` for canonical `movies` plus `movie_genre`. Persisting and
  enforcing it on checkpoints and feature outputs remains Phase 4E work.
- Release year is embedded in most titles, not supplied as a dedicated field;
  parsing is conservative and missing when the pattern is ambiguous or is
  `(0000)`, which is not a valid Gregorian calendar year.
- `(no genres listed)` distinguishes explicit missing genre metadata from a
real genre but cannot tell whether the movie truly has no genres.
- Genre lists reject empty tokens and reject the no-genres sentinel when it is
mixed with real genre labels.
- There are no actor/director sources, IDs, names, credit order, or billing
order in this repository. People entities and all people-derived state remain
deferred until a versioned external snapshot is cached and its availability
policy is documented.
- Recorded event time is assumed to approximate availability time because the
source has no separate arrival timestamp.
- All 7,801 current tag users are also rating users, although 110 have a tag
earlier than their first rating. User lifecycle metadata therefore considers
both timestamp-bearing event tables.
