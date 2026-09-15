from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.training import prd_config
from src.training.compare_genome import ensure_safe_output_directory
from src.training.modeling import predict_prd, validate_ratings_source, validate_v2
from src.training.prd import PRD_MODEL_MANIFEST, validate_prd_manifest


def manifest_copy() -> dict[str, object]:
    return json.loads(PRD_MODEL_MANIFEST.read_text(encoding="utf-8"))


def feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: pd.Series([0], dtype=dtype)
            for name, dtype in prd_config.PRD_FEATURE_DTYPES.items()
        }
    )


def test_current_prd_registry_contains_ordered_unique_rolling_features() -> None:
    rolling = (
        "movie_rating_count_30d",
        "movie_rating_count_60d",
        "movie_rating_avg_30d",
        "movie_rating_avg_60d",
        "movie_rating_avg_30d_over_60d",
        "movie_rating_count_30d_over_60d",
    )
    assert len(prd_config.PRD_FEATURES) == 30
    assert len(prd_config.PRD_FEATURES) == len(set(prd_config.PRD_FEATURES))
    assert all(name in prd_config.PRD_FEATURES for name in rolling)
    assert tuple(
        name for name in prd_config.PRD_FEATURES if name in rolling
    ) == rolling


def test_prd_manifest_resolves_exact_promoted_model_and_artifacts() -> None:
    manifest = validate_prd_manifest()

    assert manifest["canonical_name"] == prd_config.PRD_MODEL_NAME
    assert manifest["status"] == "PRD"
    assert manifest["feature_contract"]["predictor_count"] == 25
    assert tuple(
        manifest["feature_contract"]["predictor_names"]
    ) == prd_config.LEGACY_PRD_FEATURES
    assert (
        manifest["feature_contract"]["expected_dtypes"]
        == prd_config.LEGACY_PRD_FEATURE_DTYPES
    )
    assert manifest["target"] == prd_config.target_contract()
    assert manifest["artifact"]["feature_names_embedded"] is False
    assert manifest["artifact"]["feature_types_embedded"] is False


def test_canonical_notebook_artifact_paths_are_under_prd() -> None:
    assert prd_config.PRD_ARTIFACT_PATH == (
        prd_config.ROOT / "models/prd/xgboost_genome_prd_v1.model.json"
    )
    assert prd_config.PRD_RESULTS_PATH == (
        prd_config.ROOT / "models/prd/xgboost_genome_prd_v1.results.json"
    )
    assert prd_config.PRD_ARTIFACT_PATH.parent == prd_config.PRD_OUTPUT_DIR
    assert prd_config.PRD_RESULTS_PATH.parent == prd_config.PRD_OUTPUT_DIR
    assert prd_config.EXPERIMENT_CANDIDATE_ARTIFACT_PATH.parent == (
        prd_config.IMMUTABLE_EXPERIMENT_EVIDENCE_DIR
    )


def test_comparison_runner_protects_immutable_evidence_by_default() -> None:
    for immutable_directory in prd_config.IMMUTABLE_HISTORICAL_EVIDENCE_DIRS:
        with pytest.raises(ValueError, match="immutable model evidence"):
            ensure_safe_output_directory(immutable_directory)
        with pytest.raises(ValueError, match="immutable model evidence"):
            ensure_safe_output_directory(immutable_directory / "nested")
    assert (
        prd_config.DEFAULT_COMPARISON_REPRODUCTION_DIR.resolve()
        != prd_config.IMMUTABLE_EXPERIMENT_EVIDENCE_DIR.resolve()
    )
    assert ensure_safe_output_directory(
        prd_config.DEFAULT_COMPARISON_REPRODUCTION_DIR
    ) == prd_config.DEFAULT_COMPARISON_REPRODUCTION_DIR.resolve()


def test_prd_scoring_rejects_permuted_feature_order() -> None:
    class ModelMustNotRun:
        def predict_proba(self, matrix: object) -> object:
            raise AssertionError("model was called before contract validation")

    canonical = feature_frame()
    permuted = canonical.loc[:, [*canonical.columns[1:], canonical.columns[0]]]
    with pytest.raises(ValueError, match="names or order"):
        predict_prd(ModelMustNotRun(), permuted)  # type: ignore[arg-type]


def test_executable_split_drift_causes_manifest_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dict(prd_config.PRD_SPLITS)
    changed["test"] = (pd.Timestamp("2015-01-01"), None)
    monkeypatch.setattr(prd_config, "PRD_SPLITS", changed)
    with pytest.raises(ValueError, match="temporal splits"):
        validate_prd_manifest()


def test_overlapping_executable_splits_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dict(prd_config.PRD_SPLITS)
    changed["validation"] = (
        pd.Timestamp("2011-01-01"),
        pd.Timestamp("2014-01-01"),
    )
    monkeypatch.setattr(prd_config, "PRD_SPLITS", changed)
    with pytest.raises(ValueError, match="overlap or are out of order"):
        validate_prd_manifest()


def test_executable_target_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prd_config, "PRD_TARGET_THRESHOLD", 3.5)
    with pytest.raises(ValueError, match="target contract"):
        validate_prd_manifest()


def test_same_length_changed_ratings_source_is_rejected(tmp_path: Path) -> None:
    ratings_path = tmp_path / "ratings.parquet"
    original = pd.DataFrame({"rating": pd.Series([1.0, 5.0], dtype="float32")})
    original.to_parquet(ratings_path, index=False)
    expected = hashlib.sha256(ratings_path.read_bytes()).hexdigest()
    validate_ratings_source(ratings_path, expected)

    original.iloc[::-1].reset_index(drop=True).to_parquet(ratings_path, index=False)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_v2(
            None,
            tmp_path / "unused-features.parquet",
            tmp_path / "unused-metadata.json",
            ratings_path,
            expected_ratings_sha256=expected,
        )


def test_manifest_model_digest_mismatch_is_rejected() -> None:
    manifest = manifest_copy()
    manifest["artifact"]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="model artifact"):
        validate_prd_manifest(manifest)


def test_manifest_provenance_does_not_claim_an_exact_source_revision() -> None:
    manifest = manifest_copy()
    assert "source_commit" not in manifest
    assert manifest["provenance"]["source_revision"] is None  # type: ignore[index]
    assert manifest["provenance"]["artifact_origin"] == (  # type: ignore[index]
        "canonical notebook workflow"
    )
    assert manifest["provenance"]["promotion_commit"] is None  # type: ignore[index]

    overstated = copy.deepcopy(manifest)
    overstated["source_commit"] = prd_config.FEATURE_IMPLEMENTATION_BASE_COMMIT
    with pytest.raises(ValueError, match="provenance"):
        validate_prd_manifest(overstated)


def test_manifest_feature_contract_drift_is_rejected() -> None:
    manifest = manifest_copy()
    manifest["feature_contract"]["predictor_names"] = manifest[  # type: ignore[index]
        "feature_contract"
    ]["predictor_names"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="names or order"):
        validate_prd_manifest(manifest)

    manifest = manifest_copy()
    manifest["feature_contract"]["nullable_predictors"] = []  # type: ignore[index]
    with pytest.raises(ValueError, match="nullability"):
        validate_prd_manifest(manifest)

    manifest = manifest_copy()
    manifest["feature_contract"]["same_timestamp_rule"] = "changed"  # type: ignore[index]
    with pytest.raises(ValueError, match="same_timestamp_rule"):
        validate_prd_manifest(manifest)
