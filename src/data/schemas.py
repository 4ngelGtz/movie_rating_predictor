"""Explicit schemas for the MovieLens 20M raw CSV files."""

from __future__ import annotations

from typing import Any


RATING_VALUES = frozenset(value / 2 for value in range(1, 11))

# Timestamps are parsed as timezone-naive pandas datetime64 values. The raw
# strings have no UTC offset, so attaching a timezone would assert information
# not present in the source.
SCHEMAS: dict[str, dict[str, Any]] = {
    "ratings": {
        "raw_file": "rating.csv",
        "parquet_file": "ratings.parquet",
        "dtypes": {"userId": "uint32", "movieId": "uint32", "rating": "float32"},
        "read_dtypes": {"userId": "Int64", "movieId": "Int64", "rating": "float64"},
        "timestamps": ("timestamp",),
        "nullable": set(),
        "unique": (),
        "allowed_values": {"rating": RATING_VALUES},
    },
    "movies": {
        "raw_file": "movie.csv",
        "parquet_file": "movies.parquet",
        "dtypes": {"movieId": "uint32", "title": "string", "genres": "string"},
        "read_dtypes": {"movieId": "Int64", "title": "string", "genres": "string"},
        "timestamps": (),
        "nullable": set(),
        "unique": (("movieId",),),
    },
    "tags": {
        "raw_file": "tag.csv",
        "parquet_file": "tags.parquet",
        "dtypes": {"userId": "uint32", "movieId": "uint32", "tag": "string"},
        "read_dtypes": {"userId": "Int64", "movieId": "Int64", "tag": "string"},
        "timestamps": ("timestamp",),
        "nullable": set(),
        "unique": (),
    },
    "genome_scores": {
        "raw_file": "genome_scores.csv",
        "parquet_file": "genome_scores.parquet",
        "dtypes": {"movieId": "uint32", "tagId": "uint16", "relevance": "float32"},
        "read_dtypes": {"movieId": "Int64", "tagId": "Int64", "relevance": "float64"},
        "timestamps": (),
        "nullable": set(),
        "unique": (("movieId", "tagId"),),
    },
    "genome_tags": {
        "raw_file": "genome_tags.csv",
        "parquet_file": "genome_tags.parquet",
        "dtypes": {"tagId": "uint16", "tag": "string"},
        "read_dtypes": {"tagId": "Int64", "tag": "string"},
        "timestamps": (),
        "nullable": set(),
        "unique": (("tagId",),),
    },
    "links": {
        "raw_file": "link.csv",
        "parquet_file": "links.parquet",
        "dtypes": {"movieId": "uint32", "imdbId": "uint32", "tmdbId": "UInt32"},
        "read_dtypes": {"movieId": "Int64", "imdbId": "Int64", "tmdbId": "Int64"},
        "na_values": {"tmdbId": [""]},
        "timestamps": (),
        "nullable": {"tmdbId"},
        "unique": (("movieId",),),
    },
}


def columns_for(schema: dict[str, Any]) -> tuple[str, ...]:
    """Return all expected columns, independent of source column order."""
    return tuple(schema["dtypes"]) + tuple(schema["timestamps"])


def expected_dtypes(schema: dict[str, Any]) -> dict[str, str]:
    """Return normalized pandas dtype names for every expected column."""
    return {
        **{column: str(pd_dtype) for column, pd_dtype in schema["dtypes"].items()},
        **{column: "datetime64" for column in schema["timestamps"]},
    }
