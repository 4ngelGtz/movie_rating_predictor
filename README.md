# Movie Rating Predictor

A pandas-first MovieLens 20M project for predicting whether a user will rate a
movie at least 4. Every training feature must use only information available
strictly before the rating event and must be reproducible by the future online
serving path.

No model or production feature pipeline has been implemented yet. The existing
notebooks contain exploratory analysis only.

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
│   ├── features/     # Shared offline/online feature definitions
│   ├── training/     # Dataset construction, temporal splits, and training
│   └── serving/      # Future FastAPI application and state updates
├── tests/
├── PROJECT_CONTEXT.md
├── requirements.txt
└── README.md
```

Generated data is ignored by Git. Keep the MovieLens source files in
`data/raw/` and never edit them in place.

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

## Current notebooks

> **Exploratory analysis only:** temporal calculations in the notebooks do not
> necessarily satisfy the production temporal contract. In particular, do not
> reuse tie ordering, current-row rolling values, or full-history quantities as
> predictive features.

- `01_temporal_high_rate_analysis.ipynb` studies high-rating behavior over
  calendar time and within each user's rating history.
- `02_genre_rating_eda.ipynb` studies the multi-label genre taxonomy and rating
  outcomes by genre.

## Next work

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the phased
plan and [`docs/TEMPORAL_CONTRACT.md`](docs/TEMPORAL_CONTRACT.md) for the rules
that all future features, training data, and serving code must follow.
