"""Historical CLI: reproduce the controlled 17-vs-25 Genome experiment.

Notebooks are the primary training interface. This module writes reproductions
outside the immutable ``models/experiments/genome_experiment_v1/`` directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.training.feature_contracts import BASELINE_17, GENOME_25
from src.training import prd_config
from src.training.modeling import (
    json_value,
    run_model,
    validate_output_directory,
    validate_v2,
    write_json,
)


def ensure_safe_output_directory(output_dir: Path) -> Path:
    """Protect the immutable promotion evidence from accidental regeneration."""
    try:
        return validate_output_directory(output_dir)
    except ValueError as error:
        raise ValueError(
            f"{error}; choose another --output-dir"
        ) from error


def _comparison(model_a: dict[str, Any], model_b: dict[str, Any]) -> dict[str, Any]:
    metric_names = ("ROC-AUC", "PR-AUC", "Log Loss", "Brier")
    split_deltas = {
        split: {
            metric: model_b["metrics"][split][metric]
            - model_a["metrics"][split][metric]
            for metric in metric_names
        }
        for split in ("train", "validation", "test")
    }
    quarter_deltas = {
        quarter: {
            metric: model_b["test_quarter_metrics"][quarter][metric]
            - model_a["test_quarter_metrics"][quarter][metric]
            for metric in metric_names
        }
        for quarter in model_a["test_quarter_metrics"]
    }
    split_deltas["test"]["relative_PR-AUC"] = (
        split_deltas["test"]["PR-AUC"] / model_a["metrics"]["test"]["PR-AUC"]
    )
    return {
        "model_a": model_a,
        "model_b": model_b,
        "absolute_metric_deltas_model_b_minus_a": split_deltas,
        "test_quarter_absolute_deltas_model_b_minus_a": quarter_deltas,
        "training_runtime_delta_seconds": (
            model_b["training_runtime_seconds"]
            - model_a["training_runtime_seconds"]
        ),
        "calibration_ece_delta_model_b_minus_a": (
            model_b["test_calibration_ece_10_uniform_bins"]
            - model_a["test_calibration_ece_10_uniform_bins"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-features",
        type=Path,
        default=prd_config.ROOT / "data/features/rating_features_v1.parquet",
    )
    parser.add_argument(
        "--v2-features",
        type=Path,
        default=prd_config.FEATURE_ARTIFACT_PATH,
    )
    parser.add_argument(
        "--v2-metadata",
        type=Path,
        default=prd_config.FEATURE_METADATA_PATH,
    )
    parser.add_argument(
        "--ratings", type=Path, default=prd_config.RATINGS_SOURCE_PATH
    )
    parser.add_argument(
        "--saved-baseline",
        type=Path,
        default=(
            prd_config.ROOT
            / prd_config.PHASE5_HISTORICAL_BASELINE["evaluation"]
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=prd_config.DEFAULT_COMPARISON_REPRODUCTION_DIR,
    )
    args = parser.parse_args()
    try:
        output_dir = ensure_safe_output_directory(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_v2(
        args.v1_features,
        args.v2_features,
        args.v2_metadata,
        args.ratings,
        expected_ratings_sha256=prd_config.RATINGS_SOURCE_SHA256,
        feature_contract=GENOME_25,
    )
    write_json(output_dir / "v2_validation.json", validation)
    rating_values = pd.read_parquet(args.ratings, columns=["rating"])[
        "rating"
    ].to_numpy(dtype=np.float32, copy=False)

    model_a = run_model(
        "model_a_17",
        BASELINE_17.names,
        args.v2_features,
        rating_values,
        output_dir,
    )
    saved = json.loads(args.saved_baseline.read_text(encoding="utf-8"))["metrics"]
    for split in ("train", "validation", "test"):
        for metric in ("ROC-AUC", "PR-AUC", "Log Loss", "Brier"):
            delta = abs(model_a["metrics"][split][metric] - saved[split][metric])
            if delta > 1e-4:
                raise RuntimeError(
                    f"Model A failed reproduction: {split} {metric} delta={delta}"
                )

    model_b = run_model(
        "model_b_genome_25",
        GENOME_25.names,
        args.v2_features,
        rating_values,
        output_dir,
    )
    comparison = _comparison(model_a, model_b)
    comparison["v2_validation"] = validation
    write_json(output_dir / "comparison.json", comparison)
    print(json.dumps(json_value(comparison), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
