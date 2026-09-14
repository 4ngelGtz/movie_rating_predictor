# Implementation Plan

This is the historical engineering roadmap and current forward plan. It records
phase boundaries and major architectural decisions; it is not a feature,
temporal, or model-results specification. Follow the
[`feature dictionary`](FEATURE_DICTIONARY_V1.md),
[`temporal contract`](TEMPORAL_CONTRACT.md), and
[`promotion record`](MODEL_PROMOTION_GENOME_V1.md) for those details.

The earlier enrichment-first project brief is preserved as historical context
in [`history/PROJECT_CONTEXT.md`](history/PROJECT_CONTEXT.md). Its status and
phase numbering are superseded by this plan.

## Completed foundation

### Phase 0 — Data foundation

All six MovieLens inputs have explicit schemas, domain/key validation,
memory-conscious dtypes, typed Parquet outputs, and exact round-trip checks.
Raw inputs remain immutable and generated data stays outside Git.

### Phase 1 — Temporal contract

The project adopted strict-prior history (`event.timestamp < t`), half-open
rolling windows (`[t - W, t)`), and atomic equal-timestamp batches. Physical row
order and identifiers never manufacture chronology inside a tie.

### Phase 2 — Entity model

Canonical users, movies, genres, rating events, movie-genre relationships, and
sparse state identities were defined. Snapshot-scoped `ratingEventId` provides
stable event provenance. Director/actor contracts remain placeholders because
stable source mappings are unavailable.

### Phase 3 — Feature Dictionary v1

The original 17-predictor baseline contract was specified before broad
implementation. It fixed formulas, sources, temporal behavior, online update
rules, cold-start behavior, and dtypes. IDs, timestamps, and provenance fields
remain observation context rather than predictors.

### Phase 4 — Feature implementation

The feature and state layer was delivered incrementally:

- **4A:** expanding global, user, and movie history;
- **4B:** user recency and exact 30-day movie activity;
- **4C:** user × target-genre history;
- **4D:** frozen catalog features and deterministic catalog identity;
- **4E:** checkpoint/restore, source-bound replay, atomic timestamp batches,
  uninterrupted/replay equivalence, and bounded-memory materialization;
- **4F:** eight controlled Genome predictors and v2 materialization.

Important decisions from this phase are retained in code and dedicated
contracts: feature state is updated only after scoring a full timestamp batch;
checkpoint cutoffs are exclusive; persisted state is bound to canonical history
and catalog identities; and failed replay/materialization operations do not
partially publish state or artifacts.

The Genome snapshot is classified as undated `static_external_metadata`.
User-dependent Genome values still use strict-prior rating history. The feature
block does not introduce embeddings, dimensionality reduction, approximate
nearest-neighbor search, or raw Genome dimensions.

### Phase 5 — Temporal modeling

The numbered notebooks under `notebooks/model/` created the frozen 17-feature
baseline, fixed the calendar splits, trained XGBoost with validation early
stopping, and saved evaluation and cohort evidence under `models/`.

### Phase 5A — Genome comparison and PRD promotion

One 25-feature candidate was compared with the reproduced 17-feature baseline
under the same target, splits, parameters, seed, early-stopping rule, missing
handling, and evaluation protocol. The Genome candidate was promoted as
`xgboost_genome_prd_v1` after consistent pooled and quarterly improvements.

The resulting architecture is notebook-first: notebooks own training and
evaluation; `prd_config.py` holds the executable contract;
`build_prd_manifest.py` derives technical metadata from saved artifacts;
`prd.py` validates the generated pointer; and `train_prd.py` /
`compare_genome.py` remain optional historical reproduction CLIs. The original
baseline and controlled comparison evidence remain frozen.

## Planned work

### Phase 6 — Current online state

Materialize durable current state for the promoted 25-feature contract,
including Genome history, and define operational freshness/availability
monitoring.

### Phase 7 — Prediction API

Implement `/predict` from `userId`, `movieId`, and prediction timestamp using
the same feature contract and frozen metadata snapshots as training.

### Phase 8 — Rating ingestion

Implement `/ratings` validation and post-prediction state updates with explicit
idempotency and timestamp-batch behavior.

### Phase 9 — End-to-end parity

Demonstrate full offline/online parity and serving-level leakage, boundary,
cold-start, replay, failure, and update behavior.

## Deferred work

- Integrate Genome history into the durable checkpoint format.
- Improve exact nearest/top-five Genome similarity scaling if operational load
  requires it.
- Harden feature publication with an output digest and persistent rollback
  strategy for second-stage metadata publication failures.
- **Partially covered:** materialization validates `ratingEventId` row count and
  uniqueness; exact output-to-source ID set equality remains deferred.
- **Future hardening:** validate context-column dtypes explicitly in addition to
  the existing predictor dtype contract.
- **Future hardening:** optionally add golden provenance digest regressions for
  canonical fixtures and artifacts.
- Future notebook experiment runs may write a small `run_manifest.json` beside
  result artifacts (experiment name, feature artifact path/size, training
  runtimes, notes). Do not modify the immutable
  `models/experiments/genome_experiment_v1/run_manifest.json`.
- Add TMDb director/actor enrichment only after stable source-qualified person
  IDs and snapshot availability semantics exist.
- Evaluate tag features, raw Genome dimensions, embeddings, collaborative
  filtering, and feature-store infrastructure only as separately contracted
  extensions.
