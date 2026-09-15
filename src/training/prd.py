"""Canonical access and validation for the current PRD model manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier

from src.training import prd_config
from src.training.feature_contracts import GENOME_25, PRD_30, contract_for_version
from src.training.modeling import predict_prd, sha256_file, validate_ratings_source


PRD_MODEL_MANIFEST = prd_config.PRD_MODEL_MANIFEST_PATH


def load_prd_manifest(path: Path = PRD_MODEL_MANIFEST) -> dict[str, Any]:
    """Load the machine-readable default-model pointer."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_prd_model(manifest: dict[str, Any] | None = None) -> XGBClassifier:
    """Load the validated PRD artifact for use with ``predict_prd``."""
    value = validate_prd_manifest(manifest)
    model = XGBClassifier()
    model.load_model(prd_config.ROOT / value["artifact"]["path"])
    return model


def _validate_provenance(value: dict[str, Any], *, root: Path) -> None:
    artifact = root / value["artifact"]["path"]
    expected = (
        prd_config.notebook_artifact_provenance()
        if artifact.resolve() == prd_config.PRD_ARTIFACT_PATH.resolve()
        else prd_config.historical_artifact_provenance()
    )
    if value.get("provenance") != expected or "source_commit" in value:
        raise ValueError("PRD artifact provenance is incomplete or overstated")


def _validate_temporal_robustness(
    evaluation: dict[str, Any],
    comparison: dict[str, Any],
    historical_name: str,
) -> None:
    """Validate the manifest claim using the shared quarterly contract."""
    robustness = prd_config.quarterly_robustness_contract(comparison)
    if not robustness["all_configured_quarters_improved"]:
        configured = robustness["configured_quarter_count"]
        improved = robustness["improved_quarter_count"]
        raise ValueError(
            f"PRD candidate improved every required metric in {improved} of "
            f"{configured} configured test quarters"
        )
    expected_claim = prd_config.quarterly_robustness_claim(
        comparison, historical_name
    )
    if evaluation["temporal_robustness"] != expected_claim:
        raise ValueError(
            "PRD temporal robustness claim differs from comparison evidence"
        )


def validate_prd_manifest(
    manifest: dict[str, Any] | None = None,
    *,
    root: Path = prd_config.ROOT,
    require_materialized_features: bool = False,
    ratings_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the PRD pointer against executable config and saved evidence."""
    value = load_prd_manifest() if manifest is None else manifest
    if value["status"] != "PRD":
        raise ValueError("PRD model manifest status must be PRD")
    if value["canonical_name"] != prd_config.PRD_MODEL_NAME:
        raise ValueError("unexpected canonical PRD model name")
    _validate_provenance(value, root=root)

    contract = value["feature_contract"]
    selected = contract_for_version(contract["version"])
    if selected not in (GENOME_25, PRD_30):
        raise ValueError("unsupported PRD feature contract version")
    contract_names = tuple(contract["predictor_names"])
    if contract_names != selected.names:
        raise ValueError("PRD predictor names or order differ from versioned contract")
    if contract["predictor_count"] != len(selected.names):
        raise ValueError("PRD predictor count differs from its ordered names")
    if contract["expected_dtypes"] != selected.dtypes:
        raise ValueError("PRD predictor dtypes differ from versioned contract")
    if tuple(contract["nullable_predictors"]) != selected.nullable:
        raise ValueError("PRD predictor nullability differs from versioned contract")
    expected_classes = {
        key: list(features) for key, features in selected.classes.items()
    }
    if contract["feature_classes"] != expected_classes:
        raise ValueError("PRD feature classes differ from executable config")
    rules = {
        "genome_metadata_classification": (
            prd_config.GENOME_METADATA_CLASSIFICATION
        ),
        "genome_user_feature_temporal_rule": prd_config.GENOME_TEMPORAL_RULE,
        "same_timestamp_rule": prd_config.SAME_TIMESTAMP_RULE,
        "target_leakage_rule": prd_config.TARGET_LEAKAGE_RULE,
        "genome_missingness": prd_config.GENOME_MISSINGNESS_RULE,
    }
    for key, expected in rules.items():
        if contract[key] != expected:
            raise ValueError(f"PRD feature rule changed: {key}")

    if value["target"] != prd_config.target_contract():
        raise ValueError("PRD target contract differs from executable config")
    prd_config.validate_split_contract()
    if value["temporal_splits"] != prd_config.serialized_splits():
        raise ValueError("PRD temporal splits differ from executable config")
    if value["training"]["parameters"] != prd_config.serialized_model_params():
        raise ValueError("PRD model parameters differ from executable config")
    if value["training"]["random_seed"] != prd_config.PRD_MODEL_PARAMS[
        "random_state"
    ]:
        raise ValueError("PRD random seed differs from executable config")

    sources = value["data_sources"]
    expected_ratings = {
        "path": prd_config.RATINGS_SOURCE_PATH.relative_to(root).as_posix(),
        "sha256": prd_config.RATINGS_SOURCE_SHA256,
    }
    if sources["ratings"] != expected_ratings:
        raise ValueError("PRD ratings source differs from executable config")
    if ratings_path is not None:
        validate_ratings_source(ratings_path, sources["ratings"]["sha256"])
    expected_feature_source = {
        "path": prd_config.FEATURE_ARTIFACT_PATH.relative_to(root).as_posix(),
        "metadata_path": prd_config.FEATURE_METADATA_PATH.relative_to(root).as_posix(),
    }
    if sources["feature_artifact"] != expected_feature_source:
        raise ValueError("PRD feature artifact differs from executable config")
    if expected_feature_source["metadata_path"] != contract[
        "materialization_metadata"
    ]:
        raise ValueError("PRD feature metadata references disagree")

    artifact = root / value["artifact"]["path"]
    allowed_artifacts = {
        prd_config.PRD_ARTIFACT_PATH.resolve(),
        prd_config.EXPERIMENT_CANDIDATE_ARTIFACT_PATH.resolve(),
    }
    if artifact.resolve() not in allowed_artifacts:
        raise ValueError("PRD model path differs from executable config")
    if not artifact.is_file() or sha256_file(artifact) != value["artifact"]["sha256"]:
        raise ValueError("PRD model artifact is absent or does not match its digest")
    model = XGBClassifier()
    model.load_model(artifact)
    booster = model.get_booster()
    if booster.num_features() != contract["predictor_count"]:
        raise ValueError("PRD model artifact has an unexpected feature count")
    names_embedded = value["artifact"]["feature_names_embedded"]
    if names_embedded:
        if tuple(booster.feature_names or ()) != contract_names:
            raise ValueError("embedded model feature names differ from PRD order")
    elif booster.feature_names is not None:
        raise ValueError("manifest incorrectly says model feature names are absent")
    types_embedded = value["artifact"]["feature_types_embedded"]
    if types_embedded is not False or booster.feature_types is not None:
        raise ValueError("manifest incorrectly says model feature types are absent")
    if model.best_iteration != value["training"]["best_iteration_zero_based"]:
        raise ValueError("PRD model artifact has an unexpected best iteration")
    if booster.num_boosted_rounds() != value["training"]["effective_tree_count"]:
        raise ValueError("PRD model artifact has an unexpected tree count")

    experiment = value["source_experiment"]
    for path_key, digest_key in (
        ("comparison", "comparison_sha256"),
        ("candidate_results", "candidate_results_sha256"),
        ("baseline_results", "baseline_results_sha256"),
        ("v2_validation", "v2_validation_sha256"),
        ("run_manifest", "run_manifest_sha256"),
    ):
        source = root / experiment[path_key]
        if not source.is_file() or sha256_file(source) != experiment[digest_key]:
            raise ValueError(f"source experiment {path_key} is absent or changed")

    results_artifact = value.get("results_artifact")
    if results_artifact is None:
        results_path = root / experiment["candidate_results"]
    else:
        results_path = root / results_artifact["path"]
        if (
            not results_path.is_file()
            or sha256_file(results_path) != results_artifact["sha256"]
        ):
            raise ValueError("PRD results artifact is absent or changed")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results["feature_columns"] != contract["predictor_names"]:
        raise ValueError("candidate result feature list differs from PRD contract")
    if results["model_params"] != prd_config.serialized_model_params():
        raise ValueError("candidate result parameters differ from executable config")
    if results["random_seed"] != prd_config.PRD_MODEL_PARAMS["random_state"]:
        raise ValueError("candidate result seed differs from executable config")
    test_metrics = results["metrics"]["test"]
    evaluation = value["evaluation"]
    metric_mapping = {
        "test_rows": "Rows",
        "test_prevalence": "Prevalence",
        "test_pr_auc": "PR-AUC",
        "test_roc_auc": "ROC-AUC",
        "test_log_loss": "Log Loss",
        "test_brier": "Brier",
    }
    for manifest_key, result_key in metric_mapping.items():
        if evaluation[manifest_key] != test_metrics[result_key]:
            raise ValueError(f"PRD evaluation metric changed: {manifest_key}")
    if evaluation["test_ece_10_uniform_bins"] != results[
        "test_calibration_ece_10_uniform_bins"
    ]:
        raise ValueError("PRD calibration metric changed")

    comparison = json.loads(
        (root / experiment["comparison"]).read_text(encoding="utf-8")
    )
    if (
        results_artifact is None
        and comparison["model_b"]["metrics"]["test"] != test_metrics
    ):
        raise ValueError("comparison and candidate test metrics differ")
    if selected == PRD_30:
        if value.get("source_experiment_role") != (
            "historical 17-vs-25 selection evidence only"
        ):
            raise ValueError("current model must classify historical comparison evidence")
        if evaluation["temporal_robustness"] != prd_config.CURRENT_EVALUATION_SCOPE:
            raise ValueError("current model cannot inherit historical robustness claims")
        if evaluation.get("test_quarter_metrics") != results["test_quarter_metrics"]:
            raise ValueError("current model quarterly evaluation differs from its results")
    else:
        _validate_temporal_robustness(
            evaluation, comparison, value["historical_baseline"]["name"]
        )

    for key in ("model", "metadata", "evaluation", "error_analysis"):
        relative_path = value["historical_baseline"][key]
        if not (root / relative_path).is_file():
            raise ValueError(f"historical baseline artifact is missing: {relative_path}")
    if require_materialized_features:
        metadata_path = root / contract["materialization_metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        materialized = contract_for_version(metadata["featureContractVersion"])
        if metadata["predictorCount"] != len(materialized.names):
            raise ValueError("local v2 materialization predictor count changed")
        if metadata["predictorNames"] != list(materialized.names):
            raise ValueError("local v2 materialization differs from PRD contract")
        if not set(selected.names).issubset(materialized.names):
            raise ValueError("local features cannot supply the model's versioned contract")
        if metadata["canonicalSources"]["ratings"] != sources["ratings"]:
            raise ValueError("local v2 ratings source differs from PRD contract")
        output_schema = metadata["outputSchema"]
        if any(
            output_schema[name] != expected
            for name, expected in materialized.dtypes.items()
        ):
            raise ValueError("local v2 dtypes differ from PRD contract")
        if any(materialized.dtypes[name] != dtype
               for name, dtype in selected.dtypes.items()):
            raise ValueError("local features are incompatible with the model dtypes")
    return value


if __name__ == "__main__":
    validated = validate_prd_manifest(
        require_materialized_features=True,
        ratings_path=prd_config.RATINGS_SOURCE_PATH,
    )
    print(
        f"validated {validated['canonical_name']} "
        f"({validated['feature_contract']['predictor_count']} predictors)"
    )


__all__ = [
    "PRD_MODEL_MANIFEST",
    "load_prd_manifest",
    "load_prd_model",
    "predict_prd",
    "validate_prd_manifest",
]
