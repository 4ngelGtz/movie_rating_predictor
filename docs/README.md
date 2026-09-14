# Documentation Index

## Canonical/current

- [`TEMPORAL_CONTRACT.md`](TEMPORAL_CONTRACT.md) — prediction-time information
  rules, timestamp ties, rolling windows, and the reproducible source audit.
- [`ENTITY_MODEL.md`](ENTITY_MODEL.md) — canonical entities, relationships,
  state grains, and source limitations.
- [`FEATURE_DICTIONARY_V1.md`](FEATURE_DICTIONARY_V1.md) — exact 25-feature PRD
  reference: definitions, dtypes, temporal/static classification, nullability,
  and missing behavior.
- [`../models/prd_model_manifest.json`](../models/prd_model_manifest.json) —
  machine-readable current model, artifact, feature, split, parameter, and
  evaluation contract.

## Decision and historical evidence

- [`MODEL_PROMOTION_GENOME_V1.md`](MODEL_PROMOTION_GENOME_V1.md) — why the Genome
  candidate was promoted, comparison evidence, artifact provenance, costs, and
  limitations.
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — completed phases, major
  architectural decisions, deferred work, and the serving roadmap.
- [`history/PROJECT_CONTEXT.md`](history/PROJECT_CONTEXT.md) — superseded early
  project brief retained to explain the original enrichment-first direction and
  historical phase numbering.

The numbered notebooks in `notebooks/model/` and original 17-feature artifacts
in `models/` are frozen Phase 5 evidence. The two top-level notebooks are
exploratory EDA and are not production feature specifications. The immutable
`models/genome_experiment_v1/` directory contains controlled promotion evidence.

A future whitepaper should synthesize these sources; it should not become a new
executable or feature-contract source of truth.
