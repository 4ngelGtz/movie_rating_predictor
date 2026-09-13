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

## Phase 4 — Feature implementation — IN PROGRESS

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
that identity; runtime enforcement belongs to later Phase 4E work.

### 4E — Checkpoint/replay and full feature materialization — IN PROGRESS

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
Phase 4E-1 validates persisted provenance format and internal consistency;
matching a restored checkpoint's identities against the actual canonical
sources remains Phase 4E-2 resume/replay work.

The state records the rolling expiration watermark and cumulative number of
expired canonical events. At publication the watermark equals `c`, and expired
plus retained rolling events must equal global event support. Restore validates
those lifecycle invariants together with types, entity/key domains, moments,
recency timestamps, rolling order, and queue/summary consistency before
returning mutable state. Persisted timestamps use exactly
`YYYY-MM-DDTHH:MM:SS.fffffffff`; relative and timezone-aware forms are invalid.
File publication uses same-directory temporary files and atomic replacement.

Legacy raw Phase 4A-C state payloads remain distinct from checkpoint artifacts.
Replay, idempotency, runtime catalog matching, and materialization remain later
4E substeps.

Remaining Phase 4E work must deliver:

1. Replay of complete batches in `[checkpointCutoff, t)`, scoring every event in
   a tied batch before applying that batch.
2. Duplicate-application protection at the canonical event/provenance grain,
   including relationship-state keys where one event has multiple legitimate
   updates.
3. Enforcement of the persisted `catalog_snapshot_id` during replay and on
   feature outputs.
4. Tests proving uninterrupted and checkpoint/restored replay produce equivalent
   features.
5. Event-conservation checks through replay and multi-valued genre expansion.
6. Full materialization of the 17-feature dataset from canonical Parquet inputs.
7. Provenance metadata sufficient to reproduce the materialized dataset,
   including the source snapshot, catalog identity, feature contract version,
   and temporal cutoff semantics.

The checkpoint cadence, physical serialization and feature-output formats,
storage layout, and command/API shape remain implementation choices.

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
