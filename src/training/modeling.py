"""Scoring helpers and optional historical training reproduction.

Notebooks are the primary training and evaluation workflow. This module
enforces the PRD feature contract for scoring, hashes artifacts, validates
v2 feature inputs, and retains ``run_model`` for optional CLI reproduction
of historical experiment evidence. It is not a training framework.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
import xgboost as xgb
from xgboost import XGBClassifier

from src.features.expanding import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS
from src.features.genome import GENOME_FEATURE_COLUMNS
from src.training import prd_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_ratings_source(path: Path, expected_sha256: str) -> None:
    """Bind positional ratingEventId label reconstruction to an exact source."""
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("ratings source is absent or does not match its SHA-256")


def validate_output_directory(
    output_dir: Path,
    *,
    allow_immutable_evidence: bool = False,
) -> Path:
    """Reject writes into the immutable promotion-evidence directory."""
    resolved = output_dir.resolve()
    immutable = prd_config.IMMUTABLE_EXPERIMENT_EVIDENCE_DIR.resolve()
    targets_evidence = resolved == immutable or immutable in resolved.parents
    if targets_evidence and not allow_immutable_evidence:
        raise ValueError("refusing to write inside immutable PRD experiment evidence")
    return resolved


def prepare_feature_matrix(
    features: pd.DataFrame,
    expected_features: tuple[str, ...],
) -> np.ndarray:
    """Require exact names, order, and dtypes before conversion for XGBoost."""
    if tuple(features.columns) != expected_features:
        raise ValueError("model input feature names or order do not match the contract")
    expected_dtypes = {
        name: prd_config.PRD_FEATURE_DTYPES[name] for name in expected_features
    }
    actual_dtypes = {name: str(features[name].dtype) for name in expected_features}
    if actual_dtypes != expected_dtypes:
        raise ValueError("model input feature dtypes do not match the contract")
    return features.to_numpy(dtype=np.float32, copy=True)


def predict_prd(model: XGBClassifier, features: pd.DataFrame) -> np.ndarray:
    """Score only a frame satisfying the canonical ordered PRD contract."""
    matrix = prepare_feature_matrix(features, prd_config.PRD_FEATURES)
    return model.predict_proba(matrix)[:, 1]


def _filters(split_name: str) -> list[tuple[str, str, pd.Timestamp]] | None:
    start, end = prd_config.PRD_SPLITS[split_name]
    result = []
    if start is not None:
        result.append(("timestamp", ">=", start))
    if end is not None:
        result.append(("timestamp", "<", end))
    return result or None


def load_split(
    feature_path: Path,
    rating_values: np.ndarray,
    split_name: str,
    feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prd_config.validate_split_contract()
    frame = pd.read_parquet(
        feature_path,
        columns=["ratingEventId", "timestamp", *feature_columns],
        filters=_filters(split_name),
    )
    start, end = prd_config.PRD_SPLITS[split_name]
    if start is not None:
        frame = frame.loc[frame["timestamp"] >= start]
    if end is not None:
        frame = frame.loc[frame["timestamp"] < end]
    event_ids = frame["ratingEventId"].to_numpy(dtype=np.uint64, copy=False)
    if event_ids.min() < 1 or event_ids.max() > len(rating_values):
        raise AssertionError("ratingEventId is outside the canonical ratings range")
    labels = (
        rating_values[event_ids - 1] >= prd_config.PRD_TARGET_THRESHOLD
    ).astype(np.int8)
    timestamps = frame["timestamp"].to_numpy(copy=True)
    matrix = prepare_feature_matrix(frame.loc[:, feature_columns], feature_columns)
    return matrix, labels, timestamps


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    return {
        "Rows": len(labels),
        "Prevalence": float(labels.mean()),
        "ROC-AUC": float(roc_auc_score(labels, probability)),
        "PR-AUC": float(average_precision_score(labels, probability)),
        "Log Loss": float(log_loss(labels, probability, labels=[0, 1])),
        "Brier": float(brier_score_loss(labels, probability)),
    }


def _calibration(
    labels: np.ndarray, probability: np.ndarray
) -> tuple[dict[str, dict[str, float | int]], float]:
    bin_ids = np.minimum((probability * 10).astype(np.int8), 9)
    bins: dict[str, dict[str, float | int]] = {}
    weighted_error = 0.0
    for bin_id in range(10):
        mask = bin_ids == bin_id
        rows = int(mask.sum())
        if rows == 0:
            continue
        average = float(probability[mask].mean())
        observed = float(labels[mask].mean())
        bins[f"{bin_id / 10:.1f}-{(bin_id + 1) / 10:.1f}"] = {
            "rows": rows,
            "average_prediction": average,
            "observed_prevalence": observed,
            "absolute_gap": abs(average - observed),
        }
        weighted_error += rows * abs(average - observed)
    return bins, weighted_error / len(labels)


def _period_metrics(
    labels: np.ndarray, probability: np.ndarray, timestamps: np.ndarray
) -> dict[str, dict[str, float | int]]:
    quarters = pd.Series(timestamps).dt.to_period("Q").astype(str).to_numpy()
    return {
        quarter: metrics(labels[quarters == quarter], probability[quarters == quarter])
        for quarter in sorted(np.unique(quarters))
    }


def _gain_importance(
    model: XGBClassifier, feature_columns: tuple[str, ...]
) -> list[dict[str, float | int | str]]:
    raw = model.get_booster().get_score(importance_type="gain")
    rows = [
        {"feature": name, "gain": float(raw.get(f"f{index}", 0.0))}
        for index, name in enumerate(feature_columns)
    ]
    rows.sort(key=lambda row: (-float(row["gain"]), str(row["feature"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def validate_v2(
    v1_path: Path | None,
    v2_path: Path,
    v2_metadata_path: Path,
    ratings_path: Path,
    *,
    expected_ratings_sha256: str = prd_config.RATINGS_SOURCE_SHA256,
) -> dict[str, Any]:
    """Stream full-artifact invariants and enforce source identity."""
    validate_ratings_source(ratings_path, expected_ratings_sha256)
    metadata = json.loads(v2_metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["predictorCount"] != len(prd_config.PRD_FEATURES)
        or metadata["predictorNames"] != list(prd_config.PRD_FEATURES)
    ):
        raise AssertionError("v2 metadata does not contain the exact PRD predictors")
    metadata_rating = metadata["canonicalSources"]["ratings"]
    if (
        metadata_rating["path"] != ratings_path.resolve().relative_to(
            prd_config.ROOT
        ).as_posix()
        or metadata_rating["sha256"] != expected_ratings_sha256
    ):
        raise ValueError("v2 metadata ratings source differs from the PRD contract")
    v2 = pq.ParquetFile(v2_path)
    if v2.metadata.num_rows != metadata["rowCount"]:
        raise AssertionError("feature artifact row counts differ")
    original_17_exact_match: bool | None = None
    if v1_path is not None:
        v1 = pq.ParquetFile(v1_path)
        if v1.metadata.num_rows != v2.metadata.num_rows:
            raise AssertionError("v1 and v2 feature artifact row counts differ")
        comparison_columns = [
            "ratingEventId", "userId", "movieId", "timestamp", *BASE_FEATURE_COLUMNS
        ]
        v1_batches = v1.iter_batches(batch_size=250_000, columns=comparison_columns)
        v2_batches = v2.iter_batches(batch_size=250_000, columns=comparison_columns)
        for left, right in zip(v1_batches, v2_batches, strict=True):
            if not left.equals(right):
                raise AssertionError(
                    "an original Phase 5 feature or row identity changed"
                )
        original_17_exact_match = True

    row_count = v2.metadata.num_rows
    seen = np.zeros(row_count, dtype=np.bool_)
    null_counts = {column: 0 for column in GENOME_FEATURE_COLUMNS}
    previous_timestamp: np.datetime64 | None = None
    scan_columns = ["ratingEventId", "timestamp", *GENOME_FEATURE_COLUMNS]
    for batch in v2.iter_batches(batch_size=250_000, columns=scan_columns):
        frame = batch.to_pandas()
        event_ids = frame["ratingEventId"].to_numpy(dtype=np.uint64, copy=False)
        if event_ids.min() < 1 or event_ids.max() > row_count:
            raise AssertionError("v2 ratingEventId is outside the canonical range")
        positions = event_ids - 1
        if seen[positions].any():
            raise AssertionError("v2 contains duplicate ratingEventId values")
        seen[positions] = True
        timestamps = frame["timestamp"].to_numpy(copy=False)
        if previous_timestamp is not None and timestamps[0] < previous_timestamp:
            raise AssertionError("v2 timestamps are not nondecreasing")
        if (timestamps[1:] < timestamps[:-1]).any():
            raise AssertionError("v2 timestamps are not nondecreasing")
        previous_timestamp = timestamps[-1]
        for column in GENOME_FEATURE_COLUMNS:
            null_counts[column] += int(frame[column].isna().sum())
    if not seen.all():
        raise AssertionError("v2 does not conserve every canonical ratingEventId")

    ratings = pd.read_parquet(ratings_path, columns=["rating"])["rating"]
    if len(ratings) != row_count:
        raise AssertionError("ratings and v2 row counts differ")
    return {
        "row_count": row_count,
        "predictor_count": len(prd_config.PRD_FEATURES),
        "target_prevalence": float(
            ratings.ge(prd_config.PRD_TARGET_THRESHOLD).mean()
        ),
        "artifact_size_bytes": v2_path.stat().st_size,
        "original_17_exact_match": original_17_exact_match,
        "rating_event_ids_complete_unique": True,
        "timestamps_nondecreasing": True,
        "genome_null_rates": {
            column: null_counts[column] / row_count
            for column in GENOME_FEATURE_COLUMNS
        },
    }


def run_model(
    name: str,
    feature_columns: tuple[str, ...],
    feature_path: Path,
    rating_values: np.ndarray,
    output_dir: Path,
    *,
    allow_immutable_evidence: bool = False,
) -> dict[str, Any]:
    output_dir = validate_output_directory(
        output_dir, allow_immutable_evidence=allow_immutable_evidence
    )
    np.random.seed(int(prd_config.PRD_MODEL_PARAMS["random_state"]))
    train_x, train_y, train_timestamps = load_split(
        feature_path, rating_values, "train", feature_columns
    )
    validation_x, validation_y, validation_timestamps = load_split(
        feature_path, rating_values, "validation", feature_columns
    )
    model = XGBClassifier(**prd_config.PRD_MODEL_PARAMS)
    started = time.perf_counter()
    model.fit(
        train_x,
        train_y,
        eval_set=[(train_x, train_y), (validation_x, validation_y)],
        verbose=25,
    )
    runtime = time.perf_counter() - started
    split_metrics = {}
    for split_name, matrix, labels in (
        ("train", train_x, train_y),
        ("validation", validation_x, validation_y),
    ):
        split_metrics[split_name] = metrics(
            labels, model.predict_proba(matrix)[:, 1]
        )
    del train_x, train_y, train_timestamps, validation_x, validation_y
    del validation_timestamps
    gc.collect()

    test_x, test_y, test_timestamps = load_split(
        feature_path, rating_values, "test", feature_columns
    )
    test_probability = model.predict_proba(test_x)[:, 1]
    split_metrics["test"] = metrics(test_y, test_probability)
    calibration_bins, calibration_ece = _calibration(test_y, test_probability)
    result = {
        "name": name,
        "random_seed": prd_config.PRD_MODEL_PARAMS["random_state"],
        "feature_columns": list(feature_columns),
        "model_params": prd_config.serialized_model_params(),
        "missing": "NaN (native XGBoost handling)",
        "decision_threshold": None,
        "decision_threshold_note": "Phase 5 defines no classification threshold",
        "training_runtime_seconds": runtime,
        "best_iteration_zero_based": int(model.best_iteration),
        "effective_tree_count": int(model.best_iteration) + 1,
        "metrics": split_metrics,
        "test_quarter_metrics": _period_metrics(
            test_y, test_probability, test_timestamps
        ),
        "test_calibration_ece_10_uniform_bins": calibration_ece,
        "test_calibration_bins": calibration_bins,
        "gain_importance": _gain_importance(model, feature_columns),
        "package_versions": {
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    model.save_model(output_dir / f"{name}.model.json")
    write_json(output_dir / f"{name}.results.json", result)
    del test_x, test_y, test_timestamps, test_probability, model
    gc.collect()
    return result


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
