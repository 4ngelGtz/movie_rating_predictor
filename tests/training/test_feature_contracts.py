from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.features.materialize import FEATURE_COLUMNS, FEATURE_DTYPES
from src.training import compare_genome, prd_config, train_prd
from src.training.feature_contracts import (
    BASELINE_17, GENOME_25, PRD_30,
)
from src.training.build_prd_manifest import build_prd_manifest
from src.training.modeling import prepare_feature_matrix
from src.training.modeling import validate_v2, sha256_file
from tests.features.test_materialize import write_inputs, run_materialization
from src.training.prd import load_prd_manifest, validate_prd_manifest


def test_explicit_contracts_match_frozen_evidence_and_current_materialization() -> None:
    legacy = load_prd_manifest()["feature_contract"]
    assert len(BASELINE_17.names) == 17
    assert len(GENOME_25.names) == 25
    assert len(PRD_30.names) == len(set(PRD_30.names)) == 30
    assert BASELINE_17.names == GENOME_25.names[:17]
    assert tuple(legacy["predictor_names"]) == GENOME_25.names
    assert legacy["expected_dtypes"] == GENOME_25.dtypes
    assert tuple(legacy["nullable_predictors"]) == GENOME_25.nullable
    assert legacy["feature_classes"] == {
        key: list(names) for key, names in GENOME_25.classes.items()
    }
    assert PRD_30.names == FEATURE_COLUMNS == prd_config.PRD_FEATURES
    assert PRD_30.dtypes == FEATURE_DTYPES


def test_legacy_dtype_validation_does_not_inherit_current_config(monkeypatch) -> None:
    monkeypatch.setattr(prd_config, "PRD_FEATURE_DTYPES", {})
    frame = pd.DataFrame({
        name: pd.Series([0], dtype=dtype)
        for name, dtype in GENOME_25.dtypes.items()
    })
    assert prepare_feature_matrix(frame, GENOME_25.names).shape == (1, 25)
    validate_prd_manifest()


@pytest.mark.parametrize("artifact_contract", [GENOME_25, PRD_30])
def test_historical_dataset_validation_accepts_declared_compatible_contract(
    artifact_contract, tmp_path, monkeypatch,
) -> None:
    paths = write_inputs(tmp_path)
    output, metadata_path, metadata = run_materialization(tmp_path, paths)
    context = ["ratingEventId", "userId", "movieId", "timestamp"]
    frame = pd.read_parquet(output).loc[:, [*context, *artifact_contract.names]]
    frame.to_parquet(output, index=False)
    metadata["predictorCount"] = len(artifact_contract.names)
    metadata["predictorNames"] = list(artifact_contract.names)
    metadata["featureContractVersion"] = artifact_contract.version
    metadata["outputSchema"] = {name: str(frame[name].dtype) for name in frame}
    metadata["canonicalSources"]["ratings"]["path"] = "ratings.parquet"
    metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(prd_config, "ROOT", tmp_path)
    kwargs = {"expected_ratings_sha256": sha256_file(paths["ratings"])}
    result = validate_v2(
        None, output, metadata_path, paths["ratings"],
        feature_contract=GENOME_25, **kwargs,
    )
    assert result["predictor_count"] == 25
    if artifact_contract == PRD_30:
        assert validate_v2(None, output, metadata_path, paths["ratings"], **kwargs)[
            "predictor_count"
        ] == 30
    else:
        with pytest.raises(ValueError, match="cannot supply"):
            validate_v2(None, output, metadata_path, paths["ratings"], **kwargs)
    metadata["featureContractVersion"] = "arbitrary"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="version"):
        validate_v2(None, output, metadata_path, paths["ratings"], **kwargs)


@pytest.mark.parametrize("version", ["arbitrary", PRD_30.version, BASELINE_17.version])
def test_wrong_manifest_version_is_rejected(version) -> None:
    manifest = load_prd_manifest()
    manifest["feature_contract"]["version"] = version
    with pytest.raises(ValueError, match="version|names or order"):
        validate_prd_manifest(manifest)


def test_legacy_manifest_accepts_versioned_current_materialization(monkeypatch) -> None:
    metadata_path = prd_config.FEATURE_METADATA_PATH
    metadata = json.loads(metadata_path.read_text())
    metadata.update({
        "featureContractVersion": PRD_30.version,
        "predictorCount": 30,
        "predictorNames": list(PRD_30.names),
    })
    metadata["outputSchema"].update(PRD_30.dtypes)
    original = Path.read_text

    def read(self, *args, **kwargs):
        if self == metadata_path:
            return json.dumps(metadata)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert validate_prd_manifest(require_materialized_features=True)[
        "feature_contract"
    ]["predictor_count"] == 25
    metadata["featureContractVersion"] = "arbitrary"
    with pytest.raises(ValueError, match="version"):
        validate_prd_manifest(require_materialized_features=True)


@pytest.mark.parametrize("module", [compare_genome, train_prd])
def test_historical_cli_selects_frozen_contract(module, monkeypatch, tmp_path) -> None:
    calls = []
    validations = []
    baseline = json.loads(prd_config.PRD_BASELINE_RESULTS_PATH.read_text())
    monkeypatch.setattr(sys, "argv", ["reproduce", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(module, "validate_v2", lambda *a, **kw: validations.append(
        kw["feature_contract"]
    ))
    monkeypatch.setattr(module.pd, "read_parquet", lambda *a, **kw: pd.DataFrame(
        {"rating": [4.0]}
    ))

    def fake_run(name, names, *args):
        calls.append((name, names))
        return copy.deepcopy(baseline)

    monkeypatch.setattr(module, "run_model", fake_run)
    if module is compare_genome:
        monkeypatch.setattr(module, "_comparison", lambda *a: {})
        monkeypatch.setattr(module, "write_json", lambda *a: None)
    module.main()
    assert validations == [GENOME_25]
    assert calls[-1][1] == GENOME_25.names
    if module is compare_genome:
        assert calls[0][1] == BASELINE_17.names
        assert calls[-1][0] == "model_b_genome_25"


def test_current_manifest_has_own_evaluation_and_no_inherited_lift(
    monkeypatch, tmp_path,
) -> None:
    model_path = tmp_path / "current.model.json"
    results_path = tmp_path / "current.results.json"
    monkeypatch.setattr(prd_config, "PRD_ARTIFACT_PATH", model_path)
    monkeypatch.setattr(prd_config, "PRD_RESULTS_PATH", results_path)
    x = np.zeros((8, 30), dtype=np.float32)
    y = np.array([0, 1] * 4)
    params = {**prd_config.PRD_MODEL_PARAMS, "n_estimators": 2,
              "early_stopping_rounds": 1}
    model = XGBClassifier(**params).fit(x, y, eval_set=[(x, y)], verbose=False)
    model.save_model(model_path)
    results = json.loads(prd_config.PRD_CANDIDATE_RESULTS_PATH.read_text())
    results["feature_columns"] = list(PRD_30.names)
    results["best_iteration_zero_based"] = model.best_iteration
    results["effective_tree_count"] = model.get_booster().num_boosted_rounds()
    # Deliberately poor test-quarter results must not gate feature inclusion.
    for metrics in results["test_quarter_metrics"].values():
        metrics["PR-AUC"] = 0.01
    results_path.write_text(json.dumps(results))
    historical = {
        key: (prd_config.ROOT / value).relative_to(Path("/")).as_posix()
        if key != "name" else value
        for key, value in prd_config.PHASE5_HISTORICAL_BASELINE.items()
    }
    manifest = build_prd_manifest(
        model_path=model_path, results_path=results_path,
        comparison_path=prd_config.PRD_COMPARISON_PATH,
        historical_baseline=historical, root=Path("/"), promote=True,
        promotion_date="2026-09-15", validation_scope="test fixture",
    )
    validate_prd_manifest(manifest, root=Path("/"))
    assert manifest["evaluation"]["test_quarter_metrics"] == results["test_quarter_metrics"]
    assert manifest["evaluation"]["temporal_robustness"] == prd_config.CURRENT_EVALUATION_SCOPE
    manifest["evaluation"]["temporal_robustness"] = load_prd_manifest()["evaluation"][
        "temporal_robustness"
    ]
    with pytest.raises(ValueError, match="cannot inherit"):
        validate_prd_manifest(manifest, root=Path("/"))
