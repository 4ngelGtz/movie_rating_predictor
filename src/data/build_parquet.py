"""Validate MovieLens CSVs and create typed, round-trip-checked Parquet files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.api.types import is_datetime64_any_dtype

from .schemas import SCHEMAS, columns_for, expected_dtypes


ROOT = Path(__file__).resolve().parents[2]


def validate_dataframe(df: pd.DataFrame, dataset: str, schema: dict[str, Any]) -> None:
    """Validate columns, nullability, keys, and simple domain invariants."""
    expected, actual = set(columns_for(schema)), set(df.columns)
    if missing := expected - actual:
        raise ValueError(f"{dataset}: missing required columns: {sorted(missing)}")
    if unexpected := actual - expected:
        raise ValueError(f"{dataset}: unexpected columns: {sorted(unexpected)}")

    dtype_violations = {
        col: (expected_dtype, str(df[col].dtype))
        for col, expected_dtype in expected_dtypes(schema).items()
        if not (
            (expected_dtype == "datetime64" and is_datetime64_any_dtype(df[col].dtype))
            or str(df[col].dtype) == expected_dtype
        )
    }
    if dtype_violations:
        raise ValueError(f"{dataset}: dtype violations (expected, actual): {dtype_violations}")

    for column in schema["timestamps"]:
        if df[column].dt.tz is not None:
            raise ValueError(f"{dataset}: {column} must be timezone-naive")

    non_nullable = expected - schema["nullable"]
    if violations := {col: int(df[col].isna().sum()) for col in non_nullable if df[col].isna().any()}:
        raise ValueError(f"{dataset}: nulls in non-nullable columns: {violations}")

    for key in schema["unique"]:
        duplicate_rows = int(df.duplicated(list(key), keep=False).sum())
        if duplicate_rows:
            raise ValueError(f"{dataset}: {duplicate_rows} rows violate unique key {key}")

    for column in ("userId", "movieId", "tagId", "imdbId", "tmdbId"):
        if column in df and (df[column].dropna() <= 0).any():
            raise ValueError(f"{dataset}: {column} must contain only positive values")
    for column, allowed in schema.get("allowed_values", {}).items():
        if invalid := set(df[column].dropna().unique()) - allowed:
            raise ValueError(f"{dataset}: invalid {column} values: {sorted(invalid)}")
    if dataset == "genome_scores" and not df["relevance"].between(0, 1).all():
        raise ValueError("genome_scores: relevance must be between 0 and 1")


def validate_input_values(df: pd.DataFrame, dataset: str, schema: dict[str, Any]) -> None:
    """Validate wide input values before any compact numeric conversion."""
    expected, actual = set(columns_for(schema)), set(df.columns)
    if missing := expected - actual:
        raise ValueError(f"{dataset}: missing required columns: {sorted(missing)}")
    if unexpected := actual - expected:
        raise ValueError(f"{dataset}: unexpected columns: {sorted(unexpected)}")

    non_nullable = expected - schema["nullable"]
    if violations := {col: int(df[col].isna().sum()) for col in non_nullable if df[col].isna().any()}:
        raise ValueError(f"{dataset}: nulls in non-nullable columns: {violations}")

    for column in schema["timestamps"]:
        if not is_datetime64_any_dtype(df[column].dtype):
            raise ValueError(f"{dataset}: {column} contains invalid timestamps")
        if df[column].dt.tz is not None:
            raise ValueError(f"{dataset}: {column} must be timezone-naive")

    for column, target_dtype in schema["dtypes"].items():
        if "int" not in target_dtype.lower():
            continue
        values = df[column].dropna()
        maximum = np.iinfo(np.dtype(target_dtype.lower())).max
        if (values <= 0).any():
            raise ValueError(f"{dataset}: {column} must contain only positive values")
        if (values > maximum).any():
            raise ValueError(f"{dataset}: {column} exceeds maximum for {target_dtype}: {maximum}")

    for column, allowed in schema.get("allowed_values", {}).items():
        if invalid := set(df[column].dropna().unique()) - allowed:
            raise ValueError(f"{dataset}: invalid {column} values: {sorted(invalid)}")
    if dataset == "genome_scores" and not df["relevance"].between(0, 1).all():
        raise ValueError("genome_scores: relevance must be between 0 and 1")


def read_and_validate_csv(path: Path, dataset: str, schema: dict[str, Any]) -> pd.DataFrame:
    """Read a CSV with explicit dtypes and validate the resulting frame."""
    try:
        df = pd.read_csv(
            path,
            dtype=schema["read_dtypes"],
            parse_dates=list(schema["timestamps"]),
            date_format="%Y-%m-%d %H:%M:%S",
            keep_default_na=False,
            na_values=schema.get("na_values"),
        )
    except (OSError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{dataset}: could not read {path}: {exc}") from exc
    validate_input_values(df, dataset, schema)
    df = df.astype(schema["dtypes"])
    validate_dataframe(df, dataset, schema)
    return df


def verify_round_trip(
    source: pd.DataFrame, parquet_path: Path, dataset: str, schema: dict[str, Any]
) -> pd.DataFrame:
    """Read Parquet back and verify rows, columns, dtypes, and null counts."""
    restored = pd.read_parquet(parquet_path, engine="pyarrow")
    if len(restored) != len(source):
        raise ValueError(f"{dataset}: row count changed: {len(source)} -> {len(restored)}")
    if set(restored.columns) != set(columns_for(schema)):
        raise ValueError(f"{dataset}: Parquet columns do not match expected columns")
    if list(restored.columns) != list(source.columns):
        raise ValueError(f"{dataset}: Parquet column order changed")
    try:
        pd.testing.assert_frame_equal(source, restored, check_dtype=True, check_exact=True)
    except AssertionError as exc:
        raise ValueError(f"{dataset}: Parquet values or row order changed: {exc}") from exc

    if pq.read_schema(parquet_path).names != list(source.columns):
        raise ValueError(f"{dataset}: Parquet physical column order changed")
    return restored


def convert_dataset(
    dataset: str, schema: dict[str, Any], raw_dir: Path, processed_dir: Path
) -> dict[str, Any]:
    """Convert and verify one source, returning summary metrics."""
    raw_path = raw_dir / schema["raw_file"]
    parquet_path = processed_dir / schema["parquet_file"]
    if not raw_path.is_file():
        raise FileNotFoundError(f"{dataset}: raw source not found: {raw_path}")

    source = read_and_validate_csv(raw_path, dataset, schema)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=processed_dir, prefix=f".{parquet_path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        source.to_parquet(temporary_path, engine="pyarrow", compression="snappy", index=False)
        restored = verify_round_trip(source, temporary_path, dataset, schema)
        os.replace(temporary_path, parquet_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    raw_mb = raw_path.stat().st_size / 1024**2
    parquet_mb = parquet_path.stat().st_size / 1024**2
    return {
        "dataset": dataset,
        "raw_rows": len(source),
        "parquet_rows": len(restored),
        "raw_size_mb": raw_mb,
        "parquet_size_mb": parquet_mb,
        "compression_ratio": raw_mb / parquet_mb,
        "memory_usage_mb": source.memory_usage(deep=True).sum() / 1024**2,
        "validation_status": "passed",
    }


def build_all(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """Convert all declared sources, stopping clearly on the first failure."""
    rows = [convert_dataset(name, schema, raw_dir, processed_dir) for name, schema in SCHEMAS.items()]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    summary = build_all(args.raw_dir, args.processed_dir)
    print(summary.to_string(index=False, formatters={
        "raw_size_mb": "{:.2f}".format,
        "parquet_size_mb": "{:.2f}".format,
        "compression_ratio": "{:.2f}x".format,
        "memory_usage_mb": "{:.2f}".format,
    }))


if __name__ == "__main__":
    main()
