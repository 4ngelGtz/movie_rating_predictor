"""Materialize the Phase 4 baseline plus Genome addendum from canonical Parquet.

Run from the repository root with ``python -m src.features.materialize``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.build_parquet import validate_dataframe
from src.data.schemas import SCHEMAS
from src.entities.builders import (
    build_genres_and_movie_genres,
    build_movies,
    read_rating_events,
)
from src.features.catalog import catalog_snapshot_id
from src.features.expanding import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS,
    build_expanding_rating_features,
    iter_expanding_rating_features,
)
from src.features.genome import (
    GENOME_FEATURE_COLUMNS,
    build_genome_features,
    iter_genome_features,
)
from src.state.checkpoints import canonical_history_source_id
from src.state.moments import timestamp_to_string


ROOT = Path(__file__).resolve().parents[2]
MATERIALIZATION_SCHEMA_VERSION = 2
FEATURE_CONTRACT_VERSION = "FEATURE_DICTIONARY_V1_GENOME_ADDENDUM"
DEFAULT_OUTPUT = ROOT / "data/features/rating_features_v2.parquet"
DEFAULT_METADATA = ROOT / "data/features/rating_features_v2.metadata.json"
DEFAULT_CHUNK_SIZE = 250_000
FEATURE_COLUMNS = (*BASE_FEATURE_COLUMNS, *GENOME_FEATURE_COLUMNS)

FEATURE_DTYPES = {
    "global_rating_count": "uint64",
    "global_mean_rating": "float32",
    "user_rating_count": "uint64",
    "user_mean_rating": "float32",
    "user_rating_std_pop": "float32",
    "user_seconds_since_last_rating": "float64",
    "movie_rating_count": "uint64",
    "movie_mean_rating": "float32",
    "movie_rating_std_pop": "float32",
    "movie_rating_count_30d": "uint64",
    "user_target_genre_rating_count": "uint64",
    "user_target_genre_mean_rating": "float32",
    "user_target_genre_mean_delta": "float32",
    "movie_genre_count": "uint8",
    "movie_release_year": "Int16",
    "movie_release_year_missing": "bool",
    "movie_age_years": "float32",
    **{column: "float32" for column in GENOME_FEATURE_COLUMNS},
}


def _schema(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(frame[column].dtype) for column in frame.columns}


def _validate_feature_schema(features: pd.DataFrame) -> None:
    """Validate exact output columns, predictor names, and contracted dtypes."""
    expected_columns = (*CONTEXT_COLUMNS, *FEATURE_COLUMNS)
    if tuple(features.columns) != expected_columns:
        raise ValueError("feature output columns do not match the Genome-addendum contract")
    if len(FEATURE_COLUMNS) != 25 or len(FEATURE_DTYPES) != 25:
        raise ValueError(
            "Genome-addendum predictor contract must contain exactly 25 features"
        )
    if tuple(FEATURE_DTYPES) != FEATURE_COLUMNS:
        raise ValueError(
            "feature dtype contract names do not match the Genome-addendum contract"
        )
    violations = {
        column: (expected, str(features[column].dtype))
        for column, expected in FEATURE_DTYPES.items()
        if str(features[column].dtype) != expected
    }
    if violations:
        raise ValueError(f"feature output dtype violations: {violations}")


def validate_materialization(
    features: pd.DataFrame,
    rating_events: pd.DataFrame,
) -> None:
    """Enforce the published event-grain and v1 feature contract."""
    _validate_feature_schema(features)
    if len(features) != len(rating_events):
        raise ValueError("feature output row count does not match canonical events")
    if not features["ratingEventId"].is_unique:
        raise ValueError("feature output ratingEventId values must be unique")
    expected_ids = rating_events["ratingEventId"].sort_values(ignore_index=True)
    actual_ids = features["ratingEventId"].sort_values(ignore_index=True)
    if not actual_ids.equals(expected_ids):
        raise ValueError("feature output does not conserve canonical ratingEventId values")


def build_materialization(
    rating_events: pd.DataFrame,
    canonical_movies: pd.DataFrame,
    movie_genres: pd.DataFrame,
    genome_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    """Build, order, and validate the 17-feature baseline plus Genome block."""
    history_source_id = canonical_history_source_id(rating_events)
    snapshot_id = catalog_snapshot_id(canonical_movies, movie_genres)
    base = build_expanding_rating_features(
        rating_events, movie_genres, canonical_movies
    )
    genome = build_genome_features(rating_events, genome_scores)
    if not base.loc[:, CONTEXT_COLUMNS].equals(genome.loc[:, CONTEXT_COLUMNS]):
        raise ValueError("base and Genome feature rows are not aligned")
    features = pd.concat(
        [base, genome.loc[:, GENOME_FEATURE_COLUMNS]], axis=1
    ).sort_values("ratingEventId", kind="stable", ignore_index=True)
    validate_materialization(features, rating_events)
    return features, history_source_id, snapshot_id


def _source_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(
    rating_events: pd.DataFrame,
    *,
    output_schema: dict[str, str],
    history_source_id: str,
    catalog_snapshot_id_value: str,
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    timestamps = rating_events["timestamp"]
    return {
        "materializationSchemaVersion": MATERIALIZATION_SCHEMA_VERSION,
        "featureContractVersion": FEATURE_CONTRACT_VERSION,
        "rowCount": len(rating_events),
        "predictorCount": len(FEATURE_COLUMNS),
        "predictorNames": list(FEATURE_COLUMNS),
        "minTimestamp": (
            timestamp_to_string(pd.Timestamp(timestamps.min()))
            if len(rating_events)
            else None
        ),
        "maxTimestamp": (
            timestamp_to_string(pd.Timestamp(timestamps.max()))
            if len(rating_events)
            else None
        ),
        "historySourceId": history_source_id,
        "catalogSnapshotId": catalog_snapshot_id_value,
        "outputSchema": output_schema,
        "canonicalSources": {
            name: {
                "path": _source_name(path),
                "sha256": _file_sha256(path),
            }
            for name, path in sorted(source_paths.items())
        },
        "canonicalDerivations": {
            "ratingEvents": "ratings + 1-based source-row ratingEventId",
            "canonicalMovies": "movies left-joined with links",
            "movieGenre": "normalized distinct genres derived from movies.genres",
            "genomeVectors": "complete movieId x sorted tagId vectors from genome_scores",
        },
        "temporalSemantics": (
            "strict-prior timestamp batches for user-dependent features: "
            "event.timestamp < prediction timestamp; Genome vectors are static "
            "external metadata"
        ),
    }


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _publish_staged_pair(
    staged_output: Path,
    staged_metadata: Path,
    output_path: Path,
    metadata_path: Path,
) -> None:
    """Replace both artifacts and restore the prior pair on an exception."""
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for destination in (output_path, metadata_path):
            if destination.exists():
                backup = _temporary_path(destination)
                backup.unlink()
                os.replace(destination, backup)
                backups[destination] = backup
        os.replace(staged_output, output_path)
        published.append(output_path)
        os.replace(staged_metadata, metadata_path)
        published.append(metadata_path)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def publish_materialization(
    features: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    output_path: Path,
    metadata_path: Path,
) -> None:
    """Verify staged artifacts fully, then atomically replace each destination."""
    output_path = output_path.resolve()
    metadata_path = metadata_path.resolve()
    if output_path == metadata_path:
        raise ValueError("feature and metadata output paths must differ")
    temporary_output = _temporary_path(output_path)
    temporary_metadata = _temporary_path(metadata_path)
    try:
        features.to_parquet(
            temporary_output,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        restored = pd.read_parquet(temporary_output, engine="pyarrow")
        pd.testing.assert_frame_equal(
            restored, features, check_dtype=True, check_exact=True
        )
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if json.loads(temporary_metadata.read_text(encoding="utf-8")) != metadata:
            raise ValueError("metadata JSON round trip changed values")
        _publish_staged_pair(
            temporary_output,
            temporary_metadata,
            output_path,
            metadata_path,
        )
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def _write_streamed_features(
    destination: Path,
    *,
    rating_events: pd.DataFrame,
    canonical_movies: pd.DataFrame,
    movie_genres: pd.DataFrame,
    genome_scores: pd.DataFrame,
    chunk_size: int,
) -> dict[str, str]:
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        base_chunks = iter_expanding_rating_features(
            rating_events, movie_genres, canonical_movies, chunk_size=chunk_size
        )
        genome_chunks = iter_genome_features(
            rating_events, genome_scores, chunk_size=chunk_size
        )
        for base_chunk, genome_chunk in zip(base_chunks, genome_chunks, strict=True):
            if not base_chunk.loc[:, CONTEXT_COLUMNS].equals(
                genome_chunk.loc[:, CONTEXT_COLUMNS]
            ):
                raise ValueError("base and Genome feature chunks are not aligned")
            chunk = pd.concat(
                [base_chunk, genome_chunk.loc[:, GENOME_FEATURE_COLUMNS]], axis=1
            )
            _validate_feature_schema(chunk)
            if not chunk["ratingEventId"].is_unique:
                raise ValueError("feature chunk ratingEventId values must be unique")
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    destination,
                    table.schema,
                    compression="snappy",
                )
            writer.write_table(table)
            row_count += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError("feature iterator did not produce an output schema")
    if row_count != len(rating_events):
        raise ValueError("feature output row count does not match canonical events")
    parquet = pq.ParquetFile(destination)
    if parquet.metadata.num_rows != len(rating_events):
        raise ValueError("persisted feature row count does not match canonical events")
    restored_schema = parquet.schema_arrow.empty_table().to_pandas()
    _validate_feature_schema(restored_schema)
    persisted_ids = pd.read_parquet(
        destination, columns=["ratingEventId"], engine="pyarrow"
    )["ratingEventId"]
    if len(persisted_ids) != len(rating_events) or not persisted_ids.is_unique:
        raise ValueError(
            "persisted ratingEventId count or uniqueness does not match canonical events"
        )
    return _schema(restored_schema)


def materialize_from_parquet(
    *,
    ratings_path: Path,
    movies_path: Path,
    links_path: Path,
    genome_scores_path: Path,
    output_path: Path,
    metadata_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Read canonical Parquet inputs and publish the Genome-addendum contract."""
    ratings_path = ratings_path.resolve()
    movies_path = movies_path.resolve()
    links_path = links_path.resolve()
    genome_scores_path = genome_scores_path.resolve()
    source_movies = pd.read_parquet(movies_path, engine="pyarrow")
    links = pd.read_parquet(links_path, engine="pyarrow")
    genome_scores = pd.read_parquet(genome_scores_path, engine="pyarrow")
    validate_dataframe(source_movies, "movies", SCHEMAS["movies"])
    validate_dataframe(links, "links", SCHEMAS["links"])
    validate_dataframe(genome_scores, "genome_scores", SCHEMAS["genome_scores"])
    canonical_movies = build_movies(source_movies, links)
    _, movie_genres = build_genres_and_movie_genres(source_movies)
    rating_events = read_rating_events(
        ratings_path, movies=canonical_movies[["movieId"]]
    )
    validate_dataframe(
        rating_events.loc[:, ["userId", "movieId", "rating", "timestamp"]],
        "ratings",
        SCHEMAS["ratings"],
    )
    output_path = output_path.resolve()
    metadata_path = metadata_path.resolve()
    if output_path == metadata_path:
        raise ValueError("feature and metadata output paths must differ")
    temporary_output = _temporary_path(output_path)
    temporary_metadata = _temporary_path(metadata_path)
    try:
        output_schema = _write_streamed_features(
            temporary_output,
            rating_events=rating_events,
            canonical_movies=canonical_movies,
            movie_genres=movie_genres,
            genome_scores=genome_scores,
            chunk_size=chunk_size,
        )
        metadata = _metadata(
            rating_events,
            output_schema=output_schema,
            history_source_id=canonical_history_source_id(rating_events),
            catalog_snapshot_id_value=catalog_snapshot_id(
                canonical_movies, movie_genres
            ),
            source_paths={
                "ratings": ratings_path,
                "movies": movies_path,
                "links": links_path,
                "genome_scores": genome_scores_path,
            },
        )
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if json.loads(temporary_metadata.read_text(encoding="utf-8")) != metadata:
            raise ValueError("metadata JSON round trip changed values")
        _publish_staged_pair(
            temporary_output,
            temporary_metadata,
            output_path,
            metadata_path,
        )
        return metadata
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratings-path", type=Path, default=ROOT / "data/processed/ratings.parquet"
    )
    parser.add_argument(
        "--movies-path", type=Path, default=ROOT / "data/processed/movies.parquet"
    )
    parser.add_argument(
        "--links-path", type=Path, default=ROOT / "data/processed/links.parquet"
    )
    parser.add_argument(
        "--genome-scores-path",
        type=Path,
        default=ROOT / "data/processed/genome_scores.parquet",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    started = time.perf_counter()
    metadata = materialize_from_parquet(
        ratings_path=args.ratings_path,
        movies_path=args.movies_path,
        links_path=args.links_path,
        genome_scores_path=args.genome_scores_path,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        chunk_size=args.chunk_size,
    )
    elapsed = time.perf_counter() - started
    print(
        f"materialized {metadata['rowCount']:,} rows and "
        f"{metadata['predictorCount']} predictors to {args.output_path} "
        f"in {elapsed:.2f}s"
    )
    print(f"metadata: {args.metadata_path}")
    print(f"historySourceId: {metadata['historySourceId']}")
    print(f"catalogSnapshotId: {metadata['catalogSnapshotId']}")


if __name__ == "__main__":
    main()
