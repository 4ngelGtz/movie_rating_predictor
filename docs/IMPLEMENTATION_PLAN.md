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

## Phase 1 — Temporal and entity definitions

Completed in the temporal contract layer, tests, and reproducible data audit.
Equal timestamps are simultaneous; deterministic physical row order is not a
temporal tie-breaker. Detailed entity/cross-entity feature schemas remain Phase
2 work because Phase 1 intentionally defines semantics without building them.

## Phase 2 — Feature specification

Create a feature dictionary before feature code. For each feature, record its
formula, source, window, strict point-in-time rule, online update rule, fallback,
and dtype. Start with user, movie, and user-genre state; defer people-based
features until cached metadata exists.

## Phase 3 — Enrichment and feature builders

Cache normalized TMDb movie-director and movie-actor mappings using stable IDs.
Then implement small pandas functions whose state transitions are shared by
historical dataset construction and online updates. Test leakage prevention and
cold-start behavior before scaling to the full ratings file.

## Phase 4 — Training and serving

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
