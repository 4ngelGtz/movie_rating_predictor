"""Executable source of truth for the canonical PRD model contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.training.feature_contracts import BASELINE_17, GENOME_25, PRD_30


ROOT = Path(__file__).resolve().parents[2]
PRD_MODEL_NAME = "xgboost_genome_prd_v1"
FEATURE_IMPLEMENTATION_BASE_COMMIT = (
    "f5f9ff709a191c5e75647fb50f617217764ca77d"
)
ARTIFACT_PROVENANCE_STATEMENT = (
    "The feature implementation is committed at the base commit, but the "
    "experiment runner was uncommitted when this artifact was created; the "
    "artifact cannot be cryptographically tied to one complete committed "
    "source revision."
)
PRD_PROMOTION_DATE = "2026-09-14"
PRD_VALIDATION_SCOPE = (
    "offline temporal production baseline; not online business validation"
)
FEATURE_CONTRACT_DOCUMENTATION = "docs/FEATURE_DICTIONARY_V1.md"
PRD_INPUT_CONTRACT = "src.training.prd_config.PRD_FEATURES"
PRD_SCORING_INTERFACE = "src.training.prd.predict_prd"
MISSING_VALUE_BEHAVIOR = "NaN handled natively by XGBoost"
FEATURE_CONTRACT_VERSION = PRD_30.version
CURRENT_EVALUATION_SCOPE = (
    "Current artifact metrics only; no improvement claim against the historical "
    "baseline. The frozen 17-vs-25 comparison is historical selection evidence."
)
LEGACY_FEATURE_CONTRACT_VERSION = GENOME_25.version
PRD_FEATURES = PRD_30.names
PRD_FEATURE_DTYPES = PRD_30.dtypes
PRD_NULLABLE_FEATURES = PRD_30.nullable
PRD_FEATURE_CLASSES = PRD_30.classes
LEGACY_PRD_FEATURES = GENOME_25.names
LEGACY_PRD_FEATURE_DTYPES = GENOME_25.dtypes
LEGACY_PRD_NULLABLE_FEATURES = GENOME_25.nullable
LEGACY_PRD_FEATURE_CLASSES = GENOME_25.classes

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
IMMUTABLE_EXPERIMENT_EVIDENCE_DIR = (
    ROOT / "models/experiments/genome_experiment_v1"
)
PHASE5_HISTORICAL_EVIDENCE_DIR = ROOT / "models/history/phase5"
PRD_OUTPUT_DIR = ROOT / "models/prd"
PRD_ARTIFACT_PATH = PRD_OUTPUT_DIR / f"{PRD_MODEL_NAME}.model.json"
PRD_RESULTS_PATH = PRD_OUTPUT_DIR / f"{PRD_MODEL_NAME}.results.json"
EXPERIMENT_CANDIDATE_ARTIFACT_PATH = (
    IMMUTABLE_EXPERIMENT_EVIDENCE_DIR / "model_b_genome_25.model.json"
)
IMMUTABLE_HISTORICAL_EVIDENCE_DIRS = (
    IMMUTABLE_EXPERIMENT_EVIDENCE_DIR,
    PHASE5_HISTORICAL_EVIDENCE_DIR,
)
PRD_COMPARISON_PATH = IMMUTABLE_EXPERIMENT_EVIDENCE_DIR / "comparison.json"
PRD_CANDIDATE_RESULTS_PATH = (
    IMMUTABLE_EXPERIMENT_EVIDENCE_DIR / "model_b_genome_25.results.json"
)
PRD_BASELINE_RESULTS_PATH = (
    IMMUTABLE_EXPERIMENT_EVIDENCE_DIR / "model_a_17.results.json"
)
PRD_V2_VALIDATION_PATH = IMMUTABLE_EXPERIMENT_EVIDENCE_DIR / "v2_validation.json"
PRD_RUN_MANIFEST_PATH = IMMUTABLE_EXPERIMENT_EVIDENCE_DIR / "run_manifest.json"
PRD_MODEL_MANIFEST_PATH = ROOT / "models/prd/prd_model_manifest.json"
PRD_NOTEBOOK_WORKFLOW = "notebooks/model/genome_prd_v1"
DEFAULT_COMPARISON_REPRODUCTION_DIR = (
    ROOT / "models/experiments/genome_experiment_reproduction_v1"
)
DEFAULT_PRD_REPRODUCTION_DIR = (
    ROOT / "models/experiments/xgboost_genome_prd_v1_reproduction"
)
PHASE5_HISTORICAL_BASELINE = {
    "name": "phase5_baseline_17",
    "model": (
        PHASE5_HISTORICAL_EVIDENCE_DIR / "xgboost_baseline_v1.json"
    ).relative_to(ROOT).as_posix(),
    "metadata": (
        PHASE5_HISTORICAL_EVIDENCE_DIR / "xgboost_baseline_v1.metadata.json"
    ).relative_to(ROOT).as_posix(),
    "evaluation": (
        PHASE5_HISTORICAL_EVIDENCE_DIR / "xgboost_baseline_v1.evaluation.json"
    ).relative_to(ROOT).as_posix(),
    "error_analysis": (
        PHASE5_HISTORICAL_EVIDENCE_DIR / "xgboost_baseline_v1.error_analysis.json"
    ).relative_to(ROOT).as_posix(),
}
PRD_TEST_QUARTERS = ("2014Q1", "2014Q2", "2014Q3", "2014Q4", "2015Q1")
QUARTERLY_IMPROVEMENT_DIRECTIONS = {
    "PR-AUC": 1,
    "ROC-AUC": 1,
    "Log Loss": -1,
    "Brier": -1,
}
_SMALL_INTEGER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)

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


def historical_artifact_provenance() -> dict[str, object]:
    """Return the frozen provenance record for the promoted Genome artifact."""
    return {
        "feature_implementation_base_commit": FEATURE_IMPLEMENTATION_BASE_COMMIT,
        "experiment_code_status_at_artifact_creation": "uncommitted",
        "artifact_provenance": ARTIFACT_PROVENANCE_STATEMENT,
        "promotion_commit": None,
    }


def notebook_artifact_provenance() -> dict[str, object]:
    """Return provenance for an artifact trained by the canonical notebooks."""
    return {
        "artifact_origin": "canonical notebook workflow",
        "notebook_workflow": PRD_NOTEBOOK_WORKFLOW,
        "source_revision": None,
        "source_experiment": (
            IMMUTABLE_EXPERIMENT_EVIDENCE_DIR.relative_to(ROOT).as_posix()
        ),
        "reproducibility_note": (
            "Deterministic for the same data, environment, seed, and code; "
            "XGBoost serialization is not guaranteed byte-identical across "
            "library or platform versions."
        ),
        "promotion_commit": None,
    }


def quarterly_robustness_contract(
    comparison: dict[str, object],
) -> dict[str, object]:
    """Evaluate quarterly deltas against the canonical ordered quarter contract."""
    raw_deltas = comparison["test_quarter_absolute_deltas_model_b_minus_a"]
    if not isinstance(raw_deltas, dict):
        raise ValueError("comparison quarter deltas must be an object")

    expected = tuple(PRD_TEST_QUARTERS)
    actual = set(raw_deltas)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        details = []
        if missing:
            details.append(f"missing configured quarters: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected quarters: {', '.join(extra)}")
        raise ValueError(
            "comparison quarter set differs from config: " + "; ".join(details)
        )

    improved_quarters = []
    for quarter in expected:
        quarter_deltas = raw_deltas[quarter]
        if not isinstance(quarter_deltas, dict):
            raise ValueError(f"comparison deltas for {quarter} must be an object")
        improved = all(
            metric in quarter_deltas
            and direction * float(quarter_deltas[metric]) > 0
            for metric, direction in QUARTERLY_IMPROVEMENT_DIRECTIONS.items()
        )
        if improved:
            improved_quarters.append(quarter)

    return {
        "configured_quarters": list(expected),
        "configured_quarter_count": len(expected),
        "improved_quarters": improved_quarters,
        "improved_quarter_count": len(improved_quarters),
        "all_configured_quarters_improved": len(improved_quarters) == len(expected),
    }


def quarterly_robustness_claim(
    comparison: dict[str, object], historical_name: str
) -> str:
    """Describe quarterly robustness from the shared executable contract."""
    facts = quarterly_robustness_contract(comparison)
    if facts["all_configured_quarters_improved"]:
        count = int(facts["configured_quarter_count"])
        count_text = (
            _SMALL_INTEGER_WORDS[count]
            if count < len(_SMALL_INTEGER_WORDS)
            else str(count)
        )
        quarter_text = "quarter" if count == 1 else "quarters"
        return (
            f"all {count_text} out-of-time test {quarter_text} improved on "
            "PR-AUC, ROC-AUC, "
            f"log loss, and Brier versus {historical_name}"
        )
    return (
        "test-quarter metric deltas versus the historical comparator "
        "are recorded in the comparison artifact"
    )


def canonical_model_identity() -> dict[str, str]:
    """Split the canonical name into family, version, and full identifier."""
    family, separator, version = PRD_MODEL_NAME.rpartition("_v")
    if separator != "_v" or not family or not version.isdigit():
        raise ValueError("PRD_MODEL_NAME must look like '<family>_v<version>'")
    return {
        "model_name": family,
        "model_version": version,
        "canonical_name": PRD_MODEL_NAME,
    }


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
