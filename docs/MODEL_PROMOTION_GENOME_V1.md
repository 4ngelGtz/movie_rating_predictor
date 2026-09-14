# Genome PRD Model Promotion Record

## Decision

Promote the 25-feature Genome model as the project's canonical PRD model and
offline temporal production baseline under the name `xgboost_genome_prd_v1`.
The machine-readable default pointer is
`models/prd_model_manifest.json`. The original 17-feature model remains the
historical `phase5_baseline_17` benchmark; none of its artifacts are replaced.
The complete `models/genome_experiment_v1/` directory is immutable promotion
evidence and is not a reproduction output directory.

This promotion identifies the default project model. It does not claim
real-world online validation or business lift.

## Evidence

The decision uses the saved controlled 17-vs-25 experiment in
`models/genome_experiment_v1/comparison.json`. Model A reproduced the saved
Phase 5 baseline exactly. Model B changed only the predictor allow-list: it
added the eight contracted Genome features while preserving the target,
temporal partitions, random seed, XGBoost parameters, native missing-value
handling, early stopping, and evaluation methodology.

| Pooled test metric | 17-feature historical baseline | 25-feature Genome model | Absolute change |
|---|---:|---:|---:|
| PR-AUC | 0.7799453703 | 0.8099251412 | +0.0299797709 |
| ROC-AUC | 0.7952745701 | 0.8203564905 | +0.0250819203 |
| Log loss | 0.5489970446 | 0.5177754164 | -0.0312216282 |
| Brier score | 0.1852765083 | 0.1733776182 | -0.0118988901 |
| ECE, 10 uniform bins | 0.0277286448 | 0.0234892775 | -0.0042393673 |

Pooled PR-AUC improved by 3.8438% relative. All five out-of-time quarters from
2014Q1 through 2015Q1 improved on PR-AUC, ROC-AUC, log loss, and Brier score.
This temporal consistency, together with improved aggregate calibration, is
the basis for promotion rather than the pooled metric alone.

## Promoted contract

The PRD model consumes the exact 25 predictors and dtypes recorded in the
machine-readable manifest and `docs/FEATURE_DICTIONARY_V1.md`. The target
remains `highRating = (rating >= 4.0)`. Training uses events before 2012,
validation uses `[2012-01-01, 2014-01-01)`, and test uses events from
2014-01-01 onward.

Genome scores are `static_external_metadata`. That assumption applies to the
undated relevance vectors, not to user behavior: all user-dependent Genome
aggregations use only ratings with `timestamp < target_timestamp`. A complete
same-timestamp batch is scored before any event in that batch updates history.
Missing Genome inputs remain NaN and XGBoost handles them natively.

`src/training/prd_config.py` is the canonical executable contract for feature
order and dtypes, nullability, target threshold, temporal splits, XGBoost
parameters, and the ratings source SHA-256. Training and scoring accept a
feature frame only when its 25 columns match that exact order and dtype
contract. The saved booster predates this interface and does not embed feature
names or types; its original bytes remain unchanged, while
`src.training.prd.predict_prd` enforces identity, order, and dtypes before
prediction.

The manifest schema records source provenance precisely. Commit
`f5f9ff709a191c5e75647fb50f617217764ca77d` is the feature implementation base
commit. The experiment runner was uncommitted when the model artifact was
created, so the artifact cannot be cryptographically tied to one complete
committed source revision. Commit `87b25f6` later recorded the PRD promotion;
it is the promotion decision revision, not the artifact-producing revision.
The manifest retains `promotion_commit: null` as part of its historical
artifact-provenance record, and the later promotion commit does not
retroactively change that provenance.

## Computational tradeoff

- Model training increased from 2,550.19 seconds to 2,711.37 seconds in the
  controlled reproduction: approximately +6.32%.
- Full v2 materialization took approximately 40.05 minutes.
- The full v2 feature artifact is approximately 1.735 GiB.

## Known limitations

1. Genome is an undated static external metadata snapshot derived partly from
   user-generated information. Its historical availability cannot be proven.
2. Exact nearest/top-five liked-history computation scales linearly with each
   user's valid liked history.
3. Genome state is not integrated into the Phase 4E checkpoint format.
4. Phase 6 and future online-state design must add durable Genome state and
   offline/online parity validation before serving.
5. Promotion is based on offline temporal evaluation and is not evidence of
   online business lift.

## Reproduction

Generate the 25-feature input artifact:

```bash
.venv/bin/python -m src.features.materialize
```

Validate the canonical PRD manifest and local feature metadata without
training:

```bash
.venv/bin/python -m src.training.prd
.venv/bin/python -m src.training.build_prd_manifest --check
```

`--check` regenerates technical metadata in memory from the frozen artifacts
and canonical config. It does not choose a model or overwrite the pointer.
Promotion remains a human decision; `status = "PRD"` is emitted only when
explicitly requested.

Reproduce only the promoted model and its full evaluation, without first
training the historical baseline. This is an optional historical CLI;
notebooks remain the primary training interface:

```bash
.venv/bin/python -m src.training.train_prd
```

The controlled two-model comparison remains reproducible with:

```bash
.venv/bin/python -m src.training.compare_genome
```

That command writes to `models/genome_experiment_reproduction_v1/` by default.
It refuses to target `models/genome_experiment_v1/` or a path inside it unless
`--force-canonical-evidence-overwrite` is supplied explicitly. Both PRD training
and comparison validation hash the ratings source before positional label
reconstruction.

Generated feature Parquet remains ignored. Small model, result, validation,
and decision artifacts follow the existing `models/` convention.
