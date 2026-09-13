# Implementation Plan

This document is the current engineering sequence. `PROJECT_CONTEXT.md`
preserves the project vision and explains how the original enrichment-first
roadmap evolved into this compact-baseline sequence.

## Phase 0 — Data foundation — COMPLETE

All six immutable raw MovieLens sources have explicit schemas, memory-conscious
dtypes, domain and key validation, typed Parquet outputs, and exact read-back
verification.

## Phase 1 — Temporal contract — COMPLETE

All timestamp-bearing history follows these rules:

```text
event.timestamp < prediction timestamp t
fixed window = [t - W, t)
read state -> score every equal-timestamp event -> apply the complete batch
```

Physical row order and entity identifiers never break timestamp ties.

## Phase 2 — Entity model — COMPLETE

Canonical users, movies, genres, rating events, movie-genre relationships, and
sparse state identities are defined. Every source rating row receives stable
snapshot-scoped `ratingEventId` provenance that survives relationship expansion.
Director and actor contracts exist, but their mappings remain unavailable.

## Phase 3 — Feature Dictionary v1 — COMPLETE

The feature contract was defined before broad implementation. It specifies
exact formulas, sources, temporal windows, point-in-time rules, online update
behavior, cold-start fallbacks, and dtypes for 17 predictors: two global, four
user, four movie, three user × target-genre, and four static/context features.
IDs, timestamps, and provenance columns are not predictors.

## Phase 4 — Feature implementation — COMPLETE

Feature state transitions are shared foundations for historical construction
and eventual online updates. Equal-timestamp and cold-start behavior are tested
at each implemented step.

### 4A — Expanding global/user/movie history — COMPLETE

Implements expanding global, user, and movie counts, means, and population
standard deviations with leakage-safe pre-batch fallback resolution.

### 4B — Recency and rolling activity — COMPLETE

Implements user time since last rating and exact trailing `[t - 30 days, t)`
movie activity with an inclusive left boundary.

### 4C — User × target-genre history — COMPLETE

Implements sparse user-genre association state and target-movie genre support,
association-weighted mean rating, and delta from the resolved user mean.

### 4D — Static catalog/context — COMPLETE

Implements genre count, release year and missingness, and movie age at the event
timestamp. A deterministic, order-invariant SHA-256 `catalog_snapshot_id` can
identify the canonical `movies` plus `movie_genre` content. Phase 4E-1 persists
that identity, and Phase 4E-2 enforces it at resume.

### 4E — Checkpoint/replay and full feature materialization — COMPLETE

#### 4E-1 — Checkpoint schema, cutoff semantics, persistence/restore — COMPLETE

The v1 JSON checkpoint is a strict, explicitly versioned envelope containing
an exclusive `checkpointCutoff`, the required SHA-256 `catalogSnapshotId`, a
deterministic SHA-256 `historySourceId`, source/progress counts, and the complete
deterministic Phase 4A-C state payload. A cutoff `c` means exactly
`H(c) = {events | timestamp < c}`: events at `c` have not been applied.

`CheckpointHistorySession` validates and owns the complete canonical event
frame, groups every timestamp tie into one indivisible batch, and controls the
ordered processing cursor. Publication requires that cursor to equal the exact
source batch boundary before `c`; this proves every source event below `c` was
consumed and none at or above `c` was consumed, including when no event occurs
at `c`. Every update requires the explicit movie-genre bridge. Bare state and
the low-level mutable update API cannot establish checkpoint eligibility.
Trusted source validation reuses the canonical MovieLens rating domain:
`0.5, 1.0, ..., 5.0`.
Phase 4E-1 validates persisted provenance format and internal consistency.

The state records the rolling expiration watermark and cumulative number of
expired canonical events. At publication the watermark equals `c`, and expired
plus retained rolling events must equal global event support. Restore validates
those lifecycle invariants together with types, entity/key domains, moments,
recency timestamps, rolling order, and queue/summary consistency before
returning mutable state. Persisted timestamps use exactly
`YYYY-MM-DDTHH:MM:SS.fffffffff`; relative and timezone-aware forms are invalid.
File publication uses same-directory temporary files and atomic replacement.

Legacy raw Phase 4A-C state payloads remain distinct from checkpoint artifacts.

#### 4E-2 — Resume/replay, provenance enforcement, duplicates — COMPLETE

`CheckpointHistorySession.resume_from_checkpoint` rebuilds the canonical
timestamp batches and verifies the complete `historySourceId`, event count,
exclusive-cutoff batch cursor, prefix event count, and latest applied timestamp
before restored state becomes session-owned. It independently rebuilds the
Phase 4D catalog digest and lookups and requires an exact `catalogSnapshotId`
match before replay.

Replay consumes only the original contiguous source suffix beginning at
`checkpointCutoff`, retains complete timestamp batches, and uses the same
trusted Phase 4A-C update path and validated genre bridge. Canonical
`ratingEventId` values are retained on source-bound batches. The session tracks
the applied prefix and rejects retries, overlaps, skipped ranges, foreign
batches, and duplicate event IDs before applying a replay range. Rolling
activity is expired to each batch timestamp before that batch is applied, so
the retained interval remains exactly `[t - 30 days, t)`.

The validated Phase 4D catalog is owned by a resumed session. Its ordinary
processing methods use that catalog when no bridge argument is supplied and
reject any attempted override. All processing paths share one transactional
batch primitive: validate the source, order, duplicate status, and catalog
inputs; prepare expiration, feature reads, and the complete update on an
independent state copy; then commit state, cursor, counts, and applied IDs.
Failures leave every session component unchanged.

Multi-batch `replay_batches()` and `replay_remaining()` extend that guarantee to
the entire public call. They replay the validated range on an isolated session
copy and publish its final state and provenance only after every batch succeeds.
An error in any later batch discards all progress prepared by that range call.

`replay_next_batch_features()` resolves one row per canonical event with all 17
Phase 4A-D predictors from the shared pre-batch `H(t)`, applies the complete
timestamp batch only after every row is resolved, and advances the session.
It returns the in-memory rows used for immediate equivalence testing.

#### 4E-3 — Uninterrupted vs checkpoint/resume/replay equivalence — COMPLETE

The uninterrupted `build_expanding_rating_features(...)` result is the sole
feature reference path. Equivalence tests process the exact canonical prefix,
publish and JSON-round-trip a checkpoint, resume against the bound history and
catalog, feature-replay the suffix, and align both results by
`ratingEventId`. The context columns and all 17 Phase 4A-D predictors compare
exactly, including dtypes and missing values.

Coverage includes empty prefixes, timestamp boundaries, between-timestamp
cutoffs, tied batches, 30-day rolling boundaries at nanosecond precision, the
final batch, and cutoffs after all events. Fixtures exercise cold starts,
missing and multi-genre movies, partial genre support, association weighting,
duplicate-looking events with distinct IDs, missing and future release years,
and shuffled event, index, movie, and bridge ordering.

After complete replay, canonical serialized dynamic state is compared with an
uninterrupted history session at a common expiration cutoff. This covers every
moment map, recency, rolling queues and counts, expiration lifecycle,
user-genre associations, and timestamp lifecycle metadata. Source/catalog
identities, prefix and final progress counters, and reconstructed applied-event
provenance are also asserted. A bounded randomized matrix and independent
row/ID conservation plus hand-computed checks supplement direct equivalence.

#### 4E-4 — Full feature materialization — COMPLETE

`python -m src.features.materialize` validates the canonical processed ratings,
movies, and links Parquet inputs, derives the canonical Phase 2 rating-event,
movie, and movie-genre views, and writes one event-grain Parquet row containing
context/provenance plus exactly the 17 v1 predictors.

The production builder processes chronologically sorted timestamp batches in
bounded-memory chunks. Chunk boundaries never split a timestamp tie, and each
chunk is written as a Parquet row group. Publication stages and validates the
complete artifact before atomic replacement, so input or generation failures
leave existing outputs unchanged.

The deterministic metadata JSON records materialization and feature-contract
versions, row and predictor counts, predictor names, timestamp bounds, complete
output dtypes, canonical source paths and SHA-256 values, `historySourceId`,
`catalogSnapshotId`, derivation descriptions, and strict-prior temporal
semantics. The default regeneration command and ignored output paths are:

```text
.venv/bin/python -m src.features.materialize
data/features/rating_features_v1.parquet
data/features/rating_features_v1.metadata.json
```

The canonical MovieLens 20M run published and independently validated
20,000,263 unique event rows and 17 predictors.

Phase 4E does not include new predictor families, TMDb/person enrichment,
tags/genome features, embeddings, collaborative filtering, model training,
hyperparameter optimization, the final temporal modeling pipeline, FastAPI,
feature-store infrastructure, or deployment work.

## Phase 5 — Training dataset and temporal modeling — NOT STARTED

Use the materialized point-in-time features to build labeled event-level rows,
choose temporal train/validation/test boundaries, fit a simple baseline, and
evaluate it. The intended progression is:

```text
feature materialization
    -> labeled event-level dataset
    -> temporal train/validation/test split
    -> simple baseline model
    -> evaluation
```

## Phase 6 — Current online state — NOT STARTED

Materialize the latest state required for online prediction after the baseline
feature and model contracts are stable.

## Phase 7 — FastAPI `/predict` — NOT STARTED

Build the entity-based prediction interface from `userId`, `movieId`, and
`timestamp` using the same feature definitions as training.

## Phase 8 — FastAPI `/ratings` — NOT STARTED

Ingest new rating events and update the relevant feature state after validation.

## Phase 9 — End-to-end parity and temporal tests — NOT STARTED

Demonstrate offline/online feature parity and complete serving-level leakage,
boundary, cold-start, replay, and update tests. Foundational temporal tests are
already developed alongside the phases they protect.

## Deferred / future extensions

- TMDb director/actor enrichment with cached, normalized mappings and stable,
  source-qualified person IDs
- people-derived and other predictor families excluded from Feature Dictionary
  v1
- tags, genome features, embeddings, and collaborative filtering
- feature-store products and infrastructure beyond demonstrated needs
