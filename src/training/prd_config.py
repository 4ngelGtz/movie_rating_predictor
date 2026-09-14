"""Executable source of truth for the canonical PRD model contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.materialize import FEATURE_COLUMNS, FEATURE_DTYPES


ROOT = Path(__file__).resolve().parents[2]
PRD_MODEL_NAME = "xgboost_genome_prd_v1"
FEATURE_IMPLEMENTATION_BASE_COMMIT = (
    "f5f9ff709a191c5e75647fb50f617217764ca77d"
)
PRD_FEATURES = tuple(FEATURE_COLUMNS)
PRD_FEATURE_DTYPES = dict(FEATURE_DTYPES)
PRD_NULLABLE_FEATURES = (
    "user_seconds_since_last_rating",
    "movie_release_year",
    "movie_age_years",
    "genome_user_positive_cosine",
    "genome_user_negative_cosine",
    "genome_preference_margin",
    "genome_nearest_liked_similarity",
    "genome_top5_liked_similarity",
    "genome_movie_relevance_mean",
    "genome_movie_relevance_std",
    "genome_movie_top10_mean",
)
PRD_FEATURE_CLASSES = {
    "strict_prior_historical": (
        *PRD_FEATURES[:13],
        *PRD_FEATURES[17:22],
    ),
    "static_catalog": PRD_FEATURES[13:16],
    "event_context_from_static_catalog": (PRD_FEATURES[16],),
    "static_external_metadata": PRD_FEATURES[22:25],
}

PRD_TARGET_THRESHOLD = 4.0
PRD_SPLITS = {
    "train": (None, pd.Timestamp("2012-01-01")),
    "validation": (
        pd.Timestamp("2012-01-01"),
        pd.Timestamp("2014-01-01"),
    ),
    "test": (pd.Timestamp("2014-01-01"), None),
}
PRD_MODEL_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "max_bin": 128,
    "tree_method": "hist",
    "eval_metric": ["logloss", "auc", "aucpr"],
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": 8,
    "missing": np.nan,
}

RATINGS_SOURCE_PATH = ROOT / "data/processed/ratings.parquet"
RATINGS_SOURCE_SHA256 = (
    "67d8f695ae2b729b0b754c285127af3a760ae93f4bb7d274e395936443cd58b2"
)
FEATURE_ARTIFACT_PATH = ROOT / "data/features/rating_features_v2.parquet"
FEATURE_METADATA_PATH = ROOT / "data/features/rating_features_v2.metadata.json"
PRD_ARTIFACT_PATH = ROOT / "models/genome_experiment_v1/model_b_genome_25.model.json"
IMMUTABLE_EXPERIMENT_EVIDENCE_DIR = ROOT / "models/genome_experiment_v1"
DEFAULT_COMPARISON_REPRODUCTION_DIR = (
    ROOT / "models/genome_experiment_reproduction_v1"
)
DEFAULT_PRD_REPRODUCTION_DIR = ROOT / "models/xgboost_genome_prd_v1_reproduction"

GENOME_METADATA_CLASSIFICATION = "static_external_metadata"
GENOME_TEMPORAL_RULE = "rating.timestamp < target_timestamp"
SAME_TIMESTAMP_RULE = (
    "score the complete timestamp batch before applying any event in that batch"
)
TARGET_LEAKAGE_RULE = "target rating and highRating are forbidden model inputs"
GENOME_MISSINGNESS_RULE = (
    "preserve NaN; ignore invalid historical Genome vectors; "
    "no fitted or future fallback"
)


def serialized_splits() -> dict[str, object]:
    """Return the manifest representation of the executable split contract."""
    return {
        "boundary_convention": "start inclusive, end exclusive",
        **{
            name: {
                "start": start.isoformat() if start is not None else None,
                "end": end.isoformat() if end is not None else None,
            }
            for name, (start, end) in PRD_SPLITS.items()
        },
    }


def target_contract() -> dict[str, str]:
    """Return the manifest target contract used by executable label creation."""
    return {
        "name": "highRating",
        "definition": f"rating >= {PRD_TARGET_THRESHOLD:.1f}",
        "source_alignment": "canonical rating joined by 1-based ratingEventId",
    }


def validate_split_contract() -> None:
    """Reject missing, internally invalid, overlapping, or reordered splits."""
    if tuple(PRD_SPLITS) != ("train", "validation", "test"):
        raise ValueError("PRD splits must be ordered train, validation, test")
    train_start, train_end = PRD_SPLITS["train"]
    validation_start, validation_end = PRD_SPLITS["validation"]
    test_start, test_end = PRD_SPLITS["test"]
    if train_start is not None or test_end is not None:
        raise ValueError("PRD train/test outer boundaries changed")
    if None in (train_end, validation_start, validation_end, test_start):
        raise ValueError("PRD internal split boundaries must be finite")
    if not train_end <= validation_start < validation_end <= test_start:
        raise ValueError("PRD temporal splits overlap or are out of order")


def serialized_model_params() -> dict[str, object]:
    """Return JSON-compatible parameters that actually control training."""
    return {
        key: value
        for key, value in PRD_MODEL_PARAMS.items()
        if key != "missing"
    }
