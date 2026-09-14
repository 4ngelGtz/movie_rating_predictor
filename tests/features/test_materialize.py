from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

import src.features.materialize as materialize_module
from src.entities.builders import build_genres_and_movie_genres, build_movies
from src.features.catalog import catalog_snapshot_id
from src.features.expanding import CONTEXT_COLUMNS, FEATURE_COLUMNS as BASE_FEATURE_COLUMNS
from src.features.materialize import FEATURE_COLUMNS
from src.features.materialize import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_DTYPES,
    MATERIALIZATION_SCHEMA_VERSION,
    build_materialization,
    materialize_from_parquet,
)
from src.state.checkpoints import CheckpointHistorySession


def write_inputs(directory: Path, *, invalid_rating: bool = False) -> dict[str, Path]:
    ratings = pd.DataFrame(
        {
            "userId": pd.Series([1, 1, 2, 1], dtype="uint32"),
            "movieId": pd.Series([10, 20, 10, 30], dtype="uint32"),
            "rating": pd.Series(
                [4.0, 3.0, 9.0 if invalid_rating else 2.0, 5.0],
                dtype="float32",
            ),
            "timestamp": pd.Series(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-02-01",
                ],
                dtype="datetime64[ns]",
            ),
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": pd.Series([10, 20, 30], dtype="uint32"),
            "title": pd.Series(
                ["Ten (2000)", "Twenty", "Thirty (2021)"], dtype="string"
            ),
            "genres": pd.Series(
                ["Drama", "Comedy|Drama", "(no genres listed)"], dtype="string"
            ),
        }
    )
    links = pd.DataFrame(
        {
            "movieId": pd.Series([10, 20, 30], dtype="uint32"),
            "imdbId": pd.Series([100, 200, 300], dtype="uint32"),
            "tmdbId": pd.Series([1000, 2000, pd.NA], dtype="UInt32"),
        }
    )
    genome_rows = [
        (movie_id, tag_id, float(tag_id == active_tag))
        for movie_id, active_tag in ((10, 1), (20, 2))
        for tag_id in range(1, 11)
    ]
    genome_scores = pd.DataFrame(
        {
            "movieId": pd.Series([row[0] for row in genome_rows], dtype="uint32"),
            "tagId": pd.Series([row[1] for row in genome_rows], dtype="uint16"),
            "relevance": pd.Series([row[2] for row in genome_rows], dtype="float32"),
        }
    )
    paths = {
        "ratings": directory / "ratings.parquet",
        "movies": directory / "movies.parquet",
        "links": directory / "links.parquet",
        "genome_scores": directory / "genome_scores.parquet",
    }
    ratings.to_parquet(paths["ratings"], index=False)
    movies.to_parquet(paths["movies"], index=False)
    links.to_parquet(paths["links"], index=False)
    genome_scores.to_parquet(paths["genome_scores"], index=False)
    return paths


def run_materialization(
    directory: Path,
    paths: dict[str, Path],
    suffix: str = "",
    *,
    chunk_size: int = 250_000,
):
    output = directory / f"features{suffix}.parquet"
    metadata = directory / f"features{suffix}.metadata.json"
    result = materialize_from_parquet(
        ratings_path=paths["ratings"],
        movies_path=paths["movies"],
        links_path=paths["links"],
        genome_scores_path=paths["genome_scores"],
        output_path=output,
        metadata_path=metadata,
        chunk_size=chunk_size,
    )
    return output, metadata, result


def test_small_fixture_materializes_complete_schema_and_conserves_events(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path)
    output, _, _ = run_materialization(tmp_path, paths, chunk_size=1)

    features = pd.read_parquet(output)
    assert tuple(features.columns) == (*CONTEXT_COLUMNS, *FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 25
    assert tuple(FEATURE_COLUMNS[:17]) == BASE_FEATURE_COLUMNS
    assert len(features) == 4
    assert features["ratingEventId"].is_unique
    assert features["ratingEventId"].tolist() == [1, 2, 3, 4]
    assert {name: str(features[name].dtype) for name in FEATURE_COLUMNS} == FEATURE_DTYPES
    assert features.loc[0, "global_rating_count"] == 0
    assert features.loc[1, "global_rating_count"] == 1
    assert features.loc[2, "global_rating_count"] == 1
    assert pd.isna(features.loc[1, "movie_release_year"])


def test_metadata_contains_contract_schema_and_provenance(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    output, metadata_path, returned = run_materialization(tmp_path, paths)
    metadata = json.loads(metadata_path.read_text())
    features = pd.read_parquet(output)

    assert metadata == returned
    assert metadata["materializationSchemaVersion"] == MATERIALIZATION_SCHEMA_VERSION
    assert metadata["featureContractVersion"] == FEATURE_CONTRACT_VERSION
    assert metadata["rowCount"] == 4
    assert metadata["predictorCount"] == 25
    assert metadata["predictorNames"] == list(FEATURE_COLUMNS)
    assert metadata["minTimestamp"] == "2020-01-01T00:00:00.000000000"
    assert metadata["maxTimestamp"] == "2020-02-01T00:00:00.000000000"
    assert metadata["outputSchema"] == {
        column: str(features[column].dtype) for column in features.columns
    }
    assert len(metadata["historySourceId"]) == 64
    assert len(metadata["catalogSnapshotId"]) == 64
    assert set(metadata["canonicalSources"]) == {
        "genome_scores", "links", "movies", "ratings"
    }
    for name, path in paths.items():
        assert metadata["canonicalSources"][name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    assert set(metadata["canonicalDerivations"]) == {
        "ratingEvents",
        "canonicalMovies",
        "movieGenre",
        "genomeVectors",
    }

    rating_events = pd.read_parquet(paths["ratings"])
    rating_events.insert(0, "ratingEventId", pd.Series([1, 2, 3, 4], dtype="UInt64"))
    rating_events["highRating"] = rating_events["rating"].ge(4.0)
    session = CheckpointHistorySession.from_canonical_events(rating_events)
    assert metadata["historySourceId"] == session.history_source_id
    source_movies = pd.read_parquet(paths["movies"])
    canonical_movies = build_movies(source_movies, pd.read_parquet(paths["links"]))
    _, movie_genres = build_genres_and_movie_genres(source_movies)
    assert metadata["catalogSnapshotId"] == catalog_snapshot_id(
        canonical_movies, movie_genres
    )


def test_repeated_materialization_has_deterministic_semantics(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    output_a, metadata_a, returned_a = run_materialization(tmp_path, paths, "-a")
    output_b, metadata_b, returned_b = run_materialization(tmp_path, paths, "-b")

    pdt.assert_frame_equal(
        pd.read_parquet(output_a), pd.read_parquet(output_b), check_exact=True
    )
    assert output_a.read_bytes() == output_b.read_bytes()
    assert metadata_a.read_bytes() == metadata_b.read_bytes()
    assert returned_a == returned_b


def test_chunked_materialization_matches_reference_features(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    source_ratings = pd.read_parquet(paths["ratings"]).iloc[[3, 1, 2, 0]]
    source_ratings.to_parquet(paths["ratings"], index=False)
    output, _, _ = run_materialization(tmp_path, paths, chunk_size=1)

    source_movies = pd.read_parquet(paths["movies"])
    canonical_movies = build_movies(source_movies, pd.read_parquet(paths["links"]))
    _, movie_genres = build_genres_and_movie_genres(source_movies)
    rating_events = pd.read_parquet(paths["ratings"])
    rating_events.insert(
        0,
        "ratingEventId",
        pd.Series(range(1, len(rating_events) + 1), dtype="UInt64"),
    )
    rating_events["highRating"] = rating_events["rating"].ge(4.0)
    expected, _, _ = build_materialization(
        rating_events, canonical_movies, movie_genres,
        pd.read_parquet(paths["genome_scores"]),
    )
    actual = pd.read_parquet(output).sort_values(
        "ratingEventId", ignore_index=True
    )
    pdt.assert_frame_equal(actual, expected, check_exact=True)


def test_invalid_input_does_not_publish_or_replace_artifacts(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path, invalid_rating=True)
    output = tmp_path / "features.parquet"
    metadata = tmp_path / "features.metadata.json"
    output.write_bytes(b"existing features")
    metadata.write_bytes(b"existing metadata")

    with pytest.raises(ValueError, match="invalid rating"):
        materialize_from_parquet(
            ratings_path=paths["ratings"],
            movies_path=paths["movies"],
            links_path=paths["links"],
            genome_scores_path=paths["genome_scores"],
            output_path=output,
            metadata_path=metadata,
        )

    assert output.read_bytes() == b"existing features"
    assert metadata.read_bytes() == b"existing metadata"
    assert not list(tmp_path.glob(".*.tmp"))


def test_generation_failure_after_a_chunk_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_inputs(tmp_path)
    output = tmp_path / "features.parquet"
    metadata = tmp_path / "features.metadata.json"
    output.write_bytes(b"existing features")
    metadata.write_bytes(b"existing metadata")
    original = materialize_module.iter_expanding_rating_features

    def interrupted(*args, **kwargs):
        yield from original(*args, **kwargs)
        raise RuntimeError("interrupted materialization")

    monkeypatch.setattr(
        materialize_module, "iter_expanding_rating_features", interrupted
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        materialize_from_parquet(
            ratings_path=paths["ratings"],
            movies_path=paths["movies"],
            links_path=paths["links"],
            genome_scores_path=paths["genome_scores"],
            output_path=output,
            metadata_path=metadata,
            chunk_size=1,
        )

    assert output.read_bytes() == b"existing features"
    assert metadata.read_bytes() == b"existing metadata"
    assert not list(tmp_path.glob(".*.tmp"))


def test_second_publication_failure_restores_prior_artifact_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_inputs(tmp_path)
    output = tmp_path / "features.parquet"
    metadata = tmp_path / "features.metadata.json"
    output.write_bytes(b"existing features")
    metadata.write_bytes(b"existing metadata")
    real_replace = materialize_module.os.replace
    failed = False

    def fail_metadata_once(source, destination):
        nonlocal failed
        if (
            not failed
            and Path(destination) == metadata
            and Path(source).name.startswith(f".{metadata.name}.")
        ):
            failed = True
            raise OSError("metadata publication failed")
        real_replace(source, destination)

    monkeypatch.setattr(materialize_module.os, "replace", fail_metadata_once)
    with pytest.raises(OSError, match="metadata publication failed"):
        materialize_from_parquet(
            ratings_path=paths["ratings"],
            movies_path=paths["movies"],
            links_path=paths["links"],
            genome_scores_path=paths["genome_scores"],
            output_path=output,
            metadata_path=metadata,
        )

    assert output.read_bytes() == b"existing features"
    assert metadata.read_bytes() == b"existing metadata"
    assert not list(tmp_path.glob(".*.tmp"))
