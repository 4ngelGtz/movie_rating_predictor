# Implementation Plan

The repository now has the target directory layout, immutable raw inputs, a
minimal Python dependency file, and explicit temporal rules. Implementation
should proceed in small, testable phases.

## Phase 0 — Parquet data layer

1. Define expected schemas for every raw CSV.
2. Read CSVs with explicit, memory-conscious pandas dtypes.
3. Validate required columns, nullability, keys where applicable, row counts,
   and allowed rating values.
4. Write one typed Parquet file per source to `data/processed/` with PyArrow.
5. Read each Parquet output back and verify its schema and row count against the
   source without changing anything in `data/raw/`.

## Phase 1 — Temporal contract

Completed in the temporal contract layer, tests, and reproducible data audit.
Equal timestamps are simultaneous; deterministic physical row order is not a
temporal tie-breaker.

## Phase 2 — Entity definitions

Define canonical users, movies, genres, people, rating events, relationship
bridges, and sparse point-in-time state identity. Preserve source-event
provenance across multi-valued relationships. Do not fabricate unavailable
actor/director identities or implement predictive aggregates.

## Phase 3 — Feature specification and enrichment

Create a feature dictionary before feature code. For each feature, record its
formula, source, window, strict point-in-time rule, online update rule, fallback,
and dtype. Cache normalized TMDb movie-director and movie-actor mappings using
stable IDs before enabling people-based state.

## Phase 4 — Feature builders

Implement small pandas functions whose state transitions are shared by
historical dataset construction and online updates. Test leakage prevention and
cold-start behavior before scaling to the full ratings file.

## Phase 5 — Training and serving

Build event-level labeled rows, define temporal train/validation/test cutoffs,
and train a simple baseline. Materialize current state and add entity-based
FastAPI `/predict` and `/ratings` endpoints only after offline/online parity can
be tested from the shared feature functions.

## Deliberately deferred

- Model selection and training
- Feature engineering implementation
- External metadata requests
- FastAPI dependencies and endpoints
- Infrastructure beyond pandas and Parquet
