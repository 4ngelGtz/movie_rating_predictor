"""Reproduce the canonical 25-feature PRD model without training Model A."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.training import prd_config
from src.training.modeling import run_model, validate_output_directory, validate_v2
from src.training.prd import validate_prd_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--output-dir",
        type=Path,
        default=prd_config.DEFAULT_PRD_REPRODUCTION_DIR,
    )
    args = parser.parse_args()
    output_dir = validate_output_directory(args.output_dir)
    manifest = validate_prd_manifest(ratings_path=args.ratings)
    validate_v2(
        None,
        args.v2_features,
        args.v2_metadata,
        args.ratings,
        expected_ratings_sha256=manifest["data_sources"]["ratings"]["sha256"],
    )
    rating_values = pd.read_parquet(args.ratings, columns=["rating"])[
        "rating"
    ].to_numpy(dtype=np.float32, copy=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(
        "xgboost_genome_prd_v1_reproduction",
        prd_config.PRD_FEATURES,
        args.v2_features,
        rating_values,
        output_dir,
    )
    print(
        f"reproduced xgboost_genome_prd_v1: "
        f"test PR-AUC={result['metrics']['test']['PR-AUC']:.6f}"
    )


if __name__ == "__main__":
    main()
