"""Reproduce the temporal audit of the canonical processed ratings table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .build_parquet import ROOT


def audit_ratings(
    ratings: pd.DataFrame, *, parquet_storage_unit: str | None = None
) -> dict[str, Any]:
    """Return temporal and duplicate metrics for rating events.

    Duplicate rows are counted beyond their first occurrence. Repeated
    interactions are counted as distinct ``(userId, movieId)`` keys, with a
    separate metric for all rows belonging to those keys.
    """
    required = {"userId", "movieId", "rating", "timestamp"}
    if missing := required - set(ratings.columns):
        raise ValueError(f"ratings: missing required columns: {sorted(missing)}")

    timestamp_counts = ratings["timestamp"].value_counts(sort=False)
    repeated_pair_mask = ratings.duplicated(["userId", "movieId"], keep=False)
    repeated_user_time_mask = ratings.duplicated(["userId", "timestamp"], keep=False)
    timestamps = ratings["timestamp"]
    whole_second_aligned = bool(
        timestamps.dt.microsecond.eq(0).all() and timestamps.dt.nanosecond.eq(0).all()
    )

    return {
        "row_count": int(len(ratings)),
        "unique_users": int(ratings["userId"].nunique()),
        "unique_movies": int(ratings["movieId"].nunique()),
        "min_timestamp": timestamps.min().isoformat(sep=" "),
        "max_timestamp": timestamps.max().isoformat(sep=" "),
        "parquet_timestamp_unit": parquet_storage_unit,
        "declared_source_timestamp_resolution": "1 second",
        "timestamps_aligned_to_declared_resolution": whole_second_aligned,
        "duplicate_full_rows_beyond_first": int(ratings.duplicated().sum()),
        "repeated_user_movie_pairs": int(
            ratings.loc[repeated_pair_mask, ["userId", "movieId"]].drop_duplicates().shape[0]
        ),
        "rows_in_repeated_user_movie_pairs": int(repeated_pair_mask.sum()),
        "timestamps_with_multiple_events": int(timestamp_counts.gt(1).sum()),
        "maximum_events_at_one_timestamp": int(timestamp_counts.max()),
        "users_with_multiple_events_at_same_timestamp": int(
            ratings.loc[repeated_user_time_mask, "userId"].nunique()
        ),
    }


def audit_parquet(path: Path) -> dict[str, Any]:
    """Read only needed columns from Parquet and audit them."""
    arrow_type = pq.read_schema(path).field("timestamp").type
    ratings = pd.read_parquet(
        path, columns=["userId", "movieId", "rating", "timestamp"], engine="pyarrow"
    )
    return audit_ratings(ratings, parquet_storage_unit=str(arrow_type))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=ROOT / "data/processed/ratings.parquet")
    args = parser.parse_args()
    print(json.dumps(audit_parquet(args.ratings), indent=2))


if __name__ == "__main__":
    main()
