# Movie Rating Predictor

This project predicts whether a MovieLens 20M user will rate a movie at least
4 stars. It is an offline production-development (PRD) system built around
point-in-time-correct features that can later be reproduced by an online
serving path.

## Current PRD status

The canonical model is **`xgboost_genome_prd_v1`**, an XGBoost classifier with
25 predictors: the historical 17-feature rating/catalog baseline plus eight
Genome features. Its machine-readable source of truth is
[`models/prd_model_manifest.json`](models/prd_model_manifest.json); the
executable contract is [`src/training/prd_config.py`](src/training/prd_config.py).

The target is `highRating = (rating >= 4.0)`. Every history-dependent feature
uses only events with `event.timestamp < prediction_timestamp`. Events sharing
a timestamp are scored from the same pre-batch state and applied only after the
complete batch is scored. Movie catalog data and undated Genome vectors use
documented frozen-snapshot assumptions; user-dependent Genome aggregates remain
strict-prior.

The model is an offline temporal baseline, not evidence of online business
lift. Online state and serving work have not started.

## Data

The repository expects the six MovieLens 20M sources (`rating`, `movie`,
`link`, `tag`, `genome_scores`, and `genome_tags`) under `data/raw/`. Raw CSVs,
processed Parquet files, and materialized feature datasets are local generated
data and are ignored by Git.

The main pipeline converts validated source CSVs to typed Parquet, constructs
canonical entities, replays timestamp batches to materialize leakage-safe
features, and trains or validates the model against fixed calendar splits:

- train: before 2012-01-01;
- validation: 2012-01-01 through 2013-12-31;
- test: 2014-01-01 onward.

The complete temporal semantics are in
[`docs/TEMPORAL_CONTRACT.md`](docs/TEMPORAL_CONTRACT.md).

## Main result

On the 846,774-row 2014+ test partition, the PRD model achieved:

| Metric | Result |
|---|---:|
| PR-AUC | 0.809925 |
| ROC-AUC | 0.820356 |
| Log loss | 0.517775 |
| Brier score | 0.173378 |
| ECE, 10 uniform bins | 0.023489 |

It improved PR-AUC, ROC-AUC, log loss, and Brier score over the frozen
17-feature baseline in every test quarter from 2014Q1 through 2015Q1. See the
[`Genome promotion record`](docs/MODEL_PROMOTION_GENOME_V1.md) for the complete
comparison, provenance, tradeoffs, and limitations.

## Repository map

```text
data/                 Local raw, processed, and materialized feature data
docs/                 Canonical contracts, roadmap, and promotion record
models/               PRD manifest plus immutable current/historical evidence
notebooks/            Exploratory EDA and frozen Phase 5 baseline evidence
src/data/             Source schemas, validation, conversion, and temporal audit
src/entities/         Canonical entity contracts and relational builders
src/features/         Catalog, temporal, baseline, Genome, and materialization logic
src/state/            Point-in-time state, checkpoint, and replay implementation
src/training/         PRD contract, validation, training, and comparison entrypoints
src/serving/          Reserved for future online serving work
tests/                Unit, temporal, replay, materialization, and PRD contract tests
```

The training layer has deliberately narrow responsibilities:

- `prd_config.py`: canonical executable PRD contract;
- `modeling.py`: reusable training and evaluation utilities;
- `train_prd.py`: canonical 25-feature reproduction;
- `compare_genome.py`: historical 17-vs-25 controlled comparison;
- `prd.py`: manifest, artifact, and contract validation.

## Setup and reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Build typed Parquet inputs from data/raw/.
python -m src.data.build_parquet

# Materialize the canonical 25-feature dataset.
python -m src.features.materialize

# Validate the promoted artifact, manifest, and local feature metadata.
python -m src.training.prd

# Run the test suite.
pytest -q -ra -W default
```

`python -m src.training.train_prd` retrains the promoted model into a separate
reproduction directory and is intentionally not part of routine validation.
`python -m src.training.compare_genome` likewise reproduces the historical
two-model experiment outside the immutable evidence directory. Both are
computationally expensive.

## Documentation

Start with [`docs/README.md`](docs/README.md). The feature dictionary owns exact
feature definitions and missing-value behavior; the temporal contract owns
event-time semantics; the promotion record owns model-selection evidence; and
the implementation plan preserves the phased engineering history and future
work without duplicating those contracts.
