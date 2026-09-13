# Movie Rating Predictor

A pandas-first MovieLens 20M project for predicting whether a user will rate a
movie at least 4. Every training feature must use only information available
strictly before the rating event and must be reproducible by the future online
serving path.

No model or complete production feature pipeline has been implemented yet.
Phase 4A–4D feature logic is complete; Phase 4E checkpoint/replay and full
materialization is next. The existing notebooks contain exploratory analysis
only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

## Repository structure

```text
movie_rating_predictor/
├── data/
│   ├── raw/          # Immutable MovieLens CSV files; local only
│   ├── processed/    # Validated, typed Parquet datasets
│   └── features/     # Materialized point-in-time feature/state tables
├── docs/             # Contracts and implementation plan
├── notebooks/        # Exploration and validation, not production logic
├── src/
│   ├── data/         # Ingestion, schema validation, and Parquet conversion
│   ├── entities/     # Canonical entity contracts and relational builders
│   ├── features/     # Shared offline/online feature definitions
│   ├── state/        # Point-in-time state contracts and update provenance
│   ├── training/     # Dataset construction, temporal splits, and training
│   └── serving/      # Future FastAPI application and state updates
├── tests/
├── PROJECT_CONTEXT.md
├── requirements.txt
└── README.md
```

Generated data is ignored by Git. Keep the MovieLens source files in
`data/raw/` and never edit them in place.

## Current status

| Phase | Status |
|---|---|
| Phase 0 — Data foundation | COMPLETE |
| Phase 1 — Temporal contract | COMPLETE |
| Phase 2 — Entity model | COMPLETE |
| Phase 3 — Feature Dictionary v1 | COMPLETE |
| Phase 4A — Expanding global/user/movie history | COMPLETE |
| Phase 4B — Recency and rolling activity | COMPLETE |
| Phase 4C — User × target-genre history | COMPLETE |
| Phase 4D — Static catalog/context | COMPLETE |
| Phase 4E — Checkpoint/replay and full materialization | NEXT |
| Phase 5 — Training dataset and temporal modeling | NOT STARTED |
| Phase 6+ — Online state and serving | NOT STARTED |

The current engineering roadmap and Phase 4E contract are in
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## Phase 0: Parquet data layer

With the environment activated, validate and convert all six raw sources:

```bash
python -m src.data.build_parquet
```

The command reads the immutable CSV files in `data/raw/`, validates their
schemas and values, and writes one Snappy-compressed Parquet file per source to
`data/processed/`. These typed Parquet files are the canonical processed input
for later phases. Raw files are never modified. Source timestamps contain no
timezone metadata, so they are preserved as timezone-naive values with an
unknown source timezone. Genome relevance is stored as `float32`, which may
introduce the small approximation expected from that representation.

## Phase 1: Temporal contract

Phase 1 defines and tests the strict point-in-time rule used by all future
features: an event is available only when `event_timestamp < prediction_time`.
Equal-timestamp events are simultaneous, and rolling windows use `[t - W, t)`.
See [`docs/TEMPORAL_CONTRACT.md`](docs/TEMPORAL_CONTRACT.md) for the complete
contract and measured data audit. Reproduce the ratings audit with:

```bash
python -m src.data.audit_temporal
```

## Phase 2: Entity definitions

Phase 2 defines canonical users, movies, genres, rating events, relationship
bridges, and sparse point-in-time state identity. It does not compute predictive
aggregates. See [`docs/ENTITY_MODEL.md`](docs/ENTITY_MODEL.md) for entity grains,
keys, cardinalities, cold-start rules, source limitations, and how
`src/entities` / `src/state` contracts, builders, and updates interact.

## Phase 3: Feature Dictionary v1

Phase 3 specifies the compact first-baseline feature set, exact formulas,
strict point-in-time rules, online update behavior, dtypes, and leakage-safe
cold-start fallbacks. It intentionally adds no feature builders. See
[`docs/FEATURE_DICTIONARY_V1.md`](docs/FEATURE_DICTIONARY_V1.md) for the
authoritative contract.

## Phase 4A: Expanding historical features

Phase 4A implements the leakage-safe temporal engine for global, user, and
movie expanding rating count/mean/population-standard-deviation features in
`src/features/expanding.py`. Equal-timestamp events are scored from one shared
pre-batch state and applied only after the complete batch is emitted. The
underlying sparse `count`/`mean`/`M2` state in `src/state/moments.py` has a
deterministic, JSON-compatible checkpoint representation; checkpoint cutoffs,
replay orchestration, and full materialization remain deferred to Phase 4E.

## Phase 4B: Recency and rolling activity

Phase 4B extends the same timestamp-batch engine with user rating recency in
elapsed seconds and canonical movie rating activity over the exact trailing
`[t - 30 days, t)` interval. Sparse last-user timestamps and rolling movie
timestamp batches live alongside the expanding state, and current-batch events
become visible only after every tied row is emitted.

## Phase 4C: User-target-genre history

Phase 4C adds sparse `(userId, genreId)` association moments to the same
timestamp-batch engine. Each canonical event still updates global, user, and
movie state exactly once, while its distinct canonical movie-genre memberships
each receive one auxiliary relationship update. Target-movie genres are
resolved into association support, an association-weighted historical mean,
and its delta from the user's historical mean using the same strict pre-batch
snapshot.

## Phase 4D: Static catalog and event context

Phase 4D resolves target-movie genre count, canonical release year, explicit
release-year missingness, and age at the scored event timestamp from one
explicit canonical `movies` plus `movie_genre` snapshot. The builder validates
and indexes that frozen catalog once, so static resolution cannot duplicate or
drop canonical rating events and does not alter the timestamp-batch lifecycle.
`src/features/catalog.py` also provides an order-invariant SHA-256
`catalog_snapshot_id` covering both canonical tables. Deterministic content
identity is therefore available; persisting and enforcing it on feature outputs
and checkpoints remains Phase 4E work.

> A feature row is produced using one explicit frozen canonical catalog
> snapshot. At present, “versioned” means its content can be identified
> deterministically; persisted version enforcement remains Phase 4E work.

## Phase 4E: Checkpoint/replay and full materialization

Phase 4E will persist and restore complete state at explicit exclusive cutoffs,
replay complete timestamp batches with duplicate-application protection, attach
and enforce `catalog_snapshot_id`, prove uninterrupted/replay equivalence and
event conservation, and materialize all 17 v1 predictors from canonical Parquet
inputs with reproducible provenance metadata. Serialization formats, checkpoint
cadence, storage layout, and command shape remain implementation choices.

It does not add new predictors or external enrichment, and it does not include
modeling, final temporal splits, serving APIs, or feature-store infrastructure.
After 4E, Phase 5 constructs labeled event-level data, defines temporal splits,
fits a simple baseline, and evaluates it.

TMDb director/actor mappings remain a planned extension after the compact
leakage-safe baseline. They are not required for Feature Dictionary v1 or Phase
4E.

## Current notebooks

> **Exploratory analysis only:** temporal calculations in the notebooks do not
> necessarily satisfy the production temporal contract. In particular, do not
> reuse tie ordering, current-row rolling values, or full-history quantities as
> predictive features.

- `01_temporal_high_rate_analysis.ipynb` studies high-rating behavior over
  calendar time and within each user's rating history.
- `02_genre_rating_eda.ipynb` studies the multi-label genre taxonomy and rating
  outcomes by genre.

## Project contracts

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the phased
plan, [`docs/TEMPORAL_CONTRACT.md`](docs/TEMPORAL_CONTRACT.md) for the rules that
all future features must follow, [`docs/ENTITY_MODEL.md`](docs/ENTITY_MODEL.md)
for the canonical Phase 2 entity/state model, and
[`docs/FEATURE_DICTIONARY_V1.md`](docs/FEATURE_DICTIONARY_V1.md) for the Phase 3
feature contract.
