from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.training import prd_config
from src.training.build_prd_manifest import (
    _temporal_robustness,
    build_canonical_prd_manifest,
    build_prd_manifest,
    check_prd_manifest,
    dumps_prd_manifest,
    main,
    write_prd_manifest,
)
from src.training.modeling import sha256_file
from src.training.prd import (
    PRD_MODEL_MANIFEST,
    _validate_temporal_robustness,
    validate_prd_manifest,
)


def _canonical_kwargs() -> dict[str, object]:
    return {
        "model_path": prd_config.PRD_ARTIFACT_PATH,
        "results_path": prd_config.PRD_CANDIDATE_RESULTS_PATH,
        "comparison_path": prd_config.PRD_COMPARISON_PATH,
        "promotion_date": prd_config.PRD_PROMOTION_DATE,
        "validation_scope": prd_config.PRD_VALIDATION_SCOPE,
        "promote": True,
    }


def test_manifest_generation_is_deterministic() -> None:
    first = dumps_prd_manifest(build_prd_manifest(**_canonical_kwargs()))
    second = dumps_prd_manifest(build_prd_manifest(**_canonical_kwargs()))
    assert first == second


def test_canonical_manifest_is_reproduced_semantically() -> None:
    generated = build_canonical_prd_manifest()
    committed = json.loads(PRD_MODEL_MANIFEST.read_text(encoding="utf-8"))
    assert generated == committed
    assert dumps_prd_manifest(generated) == PRD_MODEL_MANIFEST.read_text(
        encoding="utf-8"
    )


def test_model_digest_is_derived_from_the_artifact() -> None:
    generated = build_canonical_prd_manifest()
    assert generated["artifact"]["sha256"] == sha256_file(
        prd_config.PRD_ARTIFACT_PATH
    )
    assert generated["artifact"]["path"] == (
        "models/experiments/genome_experiment_v1/model_b_genome_25.model.json"
    )
    assert generated["artifact"]["feature_names_embedded"] is False
    assert generated["artifact"]["feature_types_embedded"] is False
    assert generated["training"]["best_iteration_zero_based"] == 599
    assert generated["training"]["effective_tree_count"] == 600


def test_feature_contract_is_derived_from_executable_config() -> None:
    contract = build_canonical_prd_manifest()["feature_contract"]
    assert contract["predictor_count"] == 25
    assert tuple(contract["predictor_names"]) == prd_config.PRD_FEATURES
    assert contract["expected_dtypes"] == prd_config.PRD_FEATURE_DTYPES
    assert tuple(contract["nullable_predictors"]) == prd_config.PRD_NULLABLE_FEATURES
    assert contract["feature_classes"] == {
        key: list(features)
        for key, features in prd_config.PRD_FEATURE_CLASSES.items()
    }
    assert contract["version"] == prd_config.FEATURE_CONTRACT_VERSION
    assert contract["same_timestamp_rule"] == prd_config.SAME_TIMESTAMP_RULE


def test_target_and_splits_are_derived_from_executable_config() -> None:
    generated = build_canonical_prd_manifest()
    assert generated["target"] == prd_config.target_contract()
    assert generated["temporal_splits"] == prd_config.serialized_splits()
    assert generated["training"]["parameters"] == prd_config.serialized_model_params()
    assert generated["training"]["random_seed"] == prd_config.PRD_MODEL_PARAMS[
        "random_state"
    ]


def test_evaluation_metrics_are_extracted_from_candidate_results() -> None:
    generated = build_canonical_prd_manifest()
    results = json.loads(
        prd_config.PRD_CANDIDATE_RESULTS_PATH.read_text(encoding="utf-8")
    )
    test_metrics = results["metrics"]["test"]
    evaluation = generated["evaluation"]
    assert evaluation["test_rows"] == test_metrics["Rows"]
    assert evaluation["test_pr_auc"] == test_metrics["PR-AUC"]
    assert evaluation["test_roc_auc"] == test_metrics["ROC-AUC"]
    assert evaluation["test_log_loss"] == test_metrics["Log Loss"]
    assert evaluation["test_brier"] == test_metrics["Brier"]
    assert evaluation["test_ece_10_uniform_bins"] == results[
        "test_calibration_ece_10_uniform_bins"
    ]
    assert evaluation["temporal_robustness"].startswith(
        "all five out-of-time test quarters improved"
    )
    comparison = json.loads(prd_config.PRD_COMPARISON_PATH.read_text(encoding="utf-8"))
    assert (
        generated["source_experiment"]["comparison_sha256"]
        == hashlib.sha256(prd_config.PRD_COMPARISON_PATH.read_bytes()).hexdigest()
    )
    assert comparison["model_b"]["metrics"]["test"] == test_metrics


def test_promotion_is_not_emitted_implicitly() -> None:
    kwargs = _canonical_kwargs()
    kwargs["promote"] = False
    del kwargs["promotion_date"]
    generated = build_prd_manifest(**kwargs)
    assert generated["status"] != "PRD"
    assert generated["status"] == "CANDIDATE"
    assert generated["promotion_date"] is None
    with pytest.raises(ValueError, match="status must be PRD"):
        validate_prd_manifest(generated)


def test_explicit_promotion_requires_a_promotion_date() -> None:
    kwargs = _canonical_kwargs()
    kwargs["promotion_date"] = None
    with pytest.raises(ValueError, match="promotion_date is required"):
        build_prd_manifest(**kwargs)
    kwargs = _canonical_kwargs()
    kwargs["validation_scope"] = None
    with pytest.raises(ValueError, match="validation_scope is required"):
        build_prd_manifest(**kwargs)


def test_generated_manifest_preserves_historical_provenance() -> None:
    generated = build_canonical_prd_manifest()
    assert generated["provenance"] == prd_config.historical_artifact_provenance()
    assert "source_commit" not in generated
    assert generated["provenance"]["promotion_commit"] is None
    assert generated["provenance"][
        "experiment_code_status_at_artifact_creation"
    ] == "uncommitted"
    validate_prd_manifest(generated)

    overstated = json.loads(json.dumps(generated))
    overstated["source_commit"] = prd_config.FEATURE_IMPLEMENTATION_BASE_COMMIT
    with pytest.raises(ValueError, match="provenance"):
        validate_prd_manifest(overstated)


def test_check_succeeds_against_the_committed_manifest() -> None:
    check_prd_manifest()
    assert main(["--check"]) == 0


def test_check_fails_on_an_altered_manifest_without_writing(
    tmp_path: Path,
) -> None:
    original = PRD_MODEL_MANIFEST.read_bytes()
    altered = tmp_path / "prd_model_manifest.json"
    payload = json.loads(original)
    payload["evaluation"]["test_pr_auc"] = 0.0
    altered.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    before = altered.read_bytes()

    with pytest.raises(ValueError, match="differs from the committed pointer"):
        check_prd_manifest(altered)
    assert main(["--check", "--manifest", str(altered)]) == 1
    assert altered.read_bytes() == before
    assert PRD_MODEL_MANIFEST.read_bytes() == original


def test_write_refuses_to_overwrite_canonical_pointer_without_promotion(
    tmp_path: Path,
) -> None:
    original = PRD_MODEL_MANIFEST.read_bytes()
    candidate = build_prd_manifest(
        model_path=prd_config.PRD_ARTIFACT_PATH,
        results_path=prd_config.PRD_CANDIDATE_RESULTS_PATH,
        comparison_path=prd_config.PRD_COMPARISON_PATH,
        promote=False,
    )
    with pytest.raises(ValueError, match="without promote=True"):
        write_prd_manifest(candidate, PRD_MODEL_MANIFEST)
    assert PRD_MODEL_MANIFEST.read_bytes() == original

    destination = tmp_path / "candidate.json"
    assert main(
        [
            "--write",
            str(destination),
            "--model",
            str(prd_config.PRD_ARTIFACT_PATH),
            "--results",
            str(prd_config.PRD_CANDIDATE_RESULTS_PATH),
            "--comparison",
            str(prd_config.PRD_COMPARISON_PATH),
        ]
    ) == 0
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["status"] == "CANDIDATE"
    assert PRD_MODEL_MANIFEST.read_bytes() == original


@pytest.mark.parametrize(
    "destination",
    [
        prd_config.PRD_ARTIFACT_PATH,
        prd_config.PRD_CANDIDATE_RESULTS_PATH,
        prd_config.PRD_COMPARISON_PATH,
        prd_config.PRD_RUN_MANIFEST_PATH,
        prd_config.PRD_BASELINE_RESULTS_PATH,
        prd_config.PRD_V2_VALIDATION_PATH,
        prd_config.RATINGS_SOURCE_PATH,
        prd_config.FEATURE_ARTIFACT_PATH,
        prd_config.FEATURE_METADATA_PATH,
        *(
            prd_config.ROOT / prd_config.PHASE5_HISTORICAL_BASELINE[key]
            for key in ("model", "metadata", "evaluation", "error_analysis")
        ),
    ],
)
def test_write_rejects_selected_input_artifacts(destination: Path) -> None:
    manifest = build_canonical_prd_manifest()
    original = destination.read_bytes()
    with pytest.raises(ValueError, match="immutable evidence|selected manifest input"):
        write_prd_manifest(manifest, destination)
    assert destination.read_bytes() == original


@pytest.mark.parametrize(
    "immutable_directory", prd_config.IMMUTABLE_HISTORICAL_EVIDENCE_DIRS
)
def test_write_rejects_nested_immutable_evidence_path(
    immutable_directory: Path,
) -> None:
    destination = (
        immutable_directory / "nested" / "candidate_manifest.json"
    )
    with pytest.raises(ValueError, match="inside immutable evidence"):
        write_prd_manifest(build_canonical_prd_manifest(), destination)
    assert not destination.exists()


def test_write_allows_safe_explicit_manifest_destination(tmp_path: Path) -> None:
    destination = tmp_path / "candidate_manifest.json"
    write_prd_manifest(build_canonical_prd_manifest(), destination)
    assert json.loads(destination.read_text(encoding="utf-8")) == (
        build_canonical_prd_manifest()
    )


def _comparison() -> dict[str, object]:
    return json.loads(prd_config.PRD_COMPARISON_PATH.read_text(encoding="utf-8"))


def test_current_quarterly_contract_passes_builder_and_validator() -> None:
    comparison = _comparison()
    claim = _temporal_robustness(comparison, "phase5_baseline_17")
    assert claim.startswith("all five out-of-time test quarters improved")
    _validate_temporal_robustness(
        {"temporal_robustness": claim}, comparison, "phase5_baseline_17"
    )


def test_one_quarter_contract_drives_builder_and_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quarter = prd_config.PRD_TEST_QUARTERS[0]
    comparison = _comparison()
    comparison["test_quarter_absolute_deltas_model_b_minus_a"] = {
        quarter: comparison["test_quarter_absolute_deltas_model_b_minus_a"][quarter]
    }
    monkeypatch.setattr(prd_config, "PRD_TEST_QUARTERS", (quarter,))

    claim = _temporal_robustness(comparison, "phase5_baseline_17")
    assert claim.startswith("all one out-of-time test quarter improved")
    assert "five" not in claim
    _validate_temporal_robustness(
        {"temporal_robustness": claim}, comparison, "phase5_baseline_17"
    )


def test_quarterly_contract_rejects_a_missing_configured_quarter() -> None:
    comparison = _comparison()
    del comparison["test_quarter_absolute_deltas_model_b_minus_a"][
        prd_config.PRD_TEST_QUARTERS[0]
    ]
    with pytest.raises(ValueError, match="missing configured quarters"):
        prd_config.quarterly_robustness_contract(comparison)


def test_quarterly_contract_rejects_an_unexpected_extra_quarter() -> None:
    comparison = _comparison()
    comparison["test_quarter_absolute_deltas_model_b_minus_a"]["2099Q1"] = {
        "PR-AUC": 1.0,
        "ROC-AUC": 1.0,
        "Log Loss": -1.0,
        "Brier": -1.0,
    }
    with pytest.raises(ValueError, match="unexpected quarters: 2099Q1"):
        prd_config.quarterly_robustness_contract(comparison)


def test_validator_rejects_a_configured_quarter_without_full_improvement() -> None:
    comparison = _comparison()
    quarter = prd_config.PRD_TEST_QUARTERS[0]
    comparison["test_quarter_absolute_deltas_model_b_minus_a"][quarter][
        "PR-AUC"
    ] = 0.0
    claim = prd_config.quarterly_robustness_claim(
        comparison, "phase5_baseline_17"
    )
    with pytest.raises(ValueError, match="4 of 5 configured test quarters"):
        _validate_temporal_robustness(
            {"temporal_robustness": claim}, comparison, "phase5_baseline_17"
        )
