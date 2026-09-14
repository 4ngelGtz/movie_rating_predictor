"""Generate the technical PRD model manifest from artifacts and canonical config.

Notebooks remain the primary training and evaluation workflow. After a notebook
saves the booster and result JSON, call ``build_prd_manifest`` and then
``validate_prd_manifest``. Promotion is a human decision: ``status = "PRD"``
is emitted only when ``promote=True``.

The builder never inspects several models or chooses a winner. The caller
selects the candidate artifact paths.

Intended final notebook cell::

    from pathlib import Path
    from src.training.build_prd_manifest import build_prd_manifest
    from src.training.prd import validate_prd_manifest

    manifest = build_prd_manifest(
        model_path=Path(
            "models/prd/xgboost_genome_prd_v1.model.json"
        ),
        results_path=Path(
            "models/prd/xgboost_genome_prd_v1.results.json"
        ),
        comparison_path=Path(
            "models/experiments/genome_experiment_v1/comparison.json"
        ),
        promotion_date="2026-09-14",
        validation_scope=(
            "offline temporal production baseline; not online business validation"
        ),
        promote=True,
    )
    validate_prd_manifest(manifest)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier

from src.training import prd_config
from src.training.modeling import sha256_file


MANIFEST_SCHEMA_VERSION = 3
CANDIDATE_STATUS = "CANDIDATE"
PROMOTED_STATUS = "PRD"
COMPACT_JSON_MAX_WIDTH = 64
EVALUATION_METRIC_MAPPING = {
    "test_rows": "Rows",
    "test_prevalence": "Prevalence",
    "test_pr_auc": "PR-AUC",
    "test_roc_auc": "ROC-AUC",
    "test_log_loss": "Log Loss",
    "test_brier": "Brier",
}


def dumps_prd_manifest(manifest: dict[str, Any]) -> str:
    """Serialize a manifest with stable indentation and a trailing newline."""
    return _encode_json(manifest, level=0) + "\n"


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _encode_json(value: Any, *, level: int, indent: int = 2) -> str:
    if _is_json_scalar(value):
        return json.dumps(value, allow_nan=False)
    if isinstance(value, list):
        if not value:
            return "[]"
        encoded_items = [
            _encode_json(item, level=level + 1, indent=indent) for item in value
        ]
        if all(_is_json_scalar(item) for item in value):
            compact = "[" + ", ".join(encoded_items) + "]"
            if len(compact) <= COMPACT_JSON_MAX_WIDTH:
                return compact
        pad = " " * (indent * (level + 1))
        close = " " * (indent * level)
        body = ",\n".join(pad + item for item in encoded_items)
        return "[\n" + body + "\n" + close + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        encoded_items = [
            (
                json.dumps(str(key), allow_nan=False),
                _encode_json(item, level=level + 1, indent=indent),
            )
            for key, item in value.items()
        ]
        if all(_is_json_scalar(item) for item in value.values()):
            compact = (
                "{"
                + ", ".join(f"{key}: {item}" for key, item in encoded_items)
                + "}"
            )
            if len(compact) <= COMPACT_JSON_MAX_WIDTH:
                return compact
        pad = " " * (indent * (level + 1))
        close = " " * (indent * level)
        body = ",\n".join(f"{pad}{key}: {item}" for key, item in encoded_items)
        return "{\n" + body + "\n" + close + "}"
    raise TypeError(f"unsupported manifest value: {type(value)!r}")


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required artifact is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"required artifact is not a JSON object: {path}")
    return payload


def _inspect_model(model_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        raise ValueError(f"required artifact is absent: {model_path}")
    model = XGBClassifier()
    model.load_model(model_path)
    booster = model.get_booster()
    return {
        "sha256": sha256_file(model_path),
        "num_features": int(booster.num_features()),
        "best_iteration_zero_based": int(model.best_iteration),
        "effective_tree_count": int(booster.num_boosted_rounds()),
        "feature_names_embedded": booster.feature_names is not None,
        "feature_types_embedded": booster.feature_types is not None,
    }


def _feature_contract(root: Path) -> dict[str, Any]:
    return {
        "version": prd_config.FEATURE_CONTRACT_VERSION,
        "documentation": prd_config.FEATURE_CONTRACT_DOCUMENTATION,
        "materialization_metadata": _relative_path(
            prd_config.FEATURE_METADATA_PATH, root
        ),
        "predictor_count": len(prd_config.PRD_FEATURES),
        "predictor_names": list(prd_config.PRD_FEATURES),
        "expected_dtypes": dict(prd_config.PRD_FEATURE_DTYPES),
        "feature_classes": {
            key: list(features)
            for key, features in prd_config.PRD_FEATURE_CLASSES.items()
        },
        "nullable_predictors": list(prd_config.PRD_NULLABLE_FEATURES),
        "genome_metadata_classification": (
            prd_config.GENOME_METADATA_CLASSIFICATION
        ),
        "genome_user_feature_temporal_rule": prd_config.GENOME_TEMPORAL_RULE,
        "same_timestamp_rule": prd_config.SAME_TIMESTAMP_RULE,
        "target_leakage_rule": prd_config.TARGET_LEAKAGE_RULE,
        "genome_missingness": prd_config.GENOME_MISSINGNESS_RULE,
    }


def _evaluation(
    results: dict[str, Any],
    comparison: dict[str, Any],
    historical_name: str,
) -> dict[str, Any]:
    test_metrics = results["metrics"]["test"]
    evaluation = {
        manifest_key: test_metrics[result_key]
        for manifest_key, result_key in EVALUATION_METRIC_MAPPING.items()
    }
    evaluation["test_ece_10_uniform_bins"] = results[
        "test_calibration_ece_10_uniform_bins"
    ]
    evaluation["temporal_robustness"] = _temporal_robustness(
        comparison, historical_name
    )
    return evaluation


def _temporal_robustness(comparison: dict[str, Any], historical_name: str) -> str:
    return prd_config.quarterly_robustness_claim(comparison, historical_name)


def _source_experiment(
    *,
    comparison_path: Path,
    candidate_results_path: Path,
    baseline_results_path: Path,
    v2_validation_path: Path,
    run_manifest_path: Path,
    root: Path,
) -> dict[str, str]:
    files = {
        "comparison": comparison_path,
        "candidate_results": candidate_results_path,
        "baseline_results": baseline_results_path,
        "v2_validation": v2_validation_path,
        "run_manifest": run_manifest_path,
    }
    experiment: dict[str, str] = {}
    for key, path in files.items():
        if not path.is_file():
            raise ValueError(f"source experiment {key} is absent: {path}")
        experiment[key] = _relative_path(path, root)
        experiment[f"{key}_sha256"] = sha256_file(path)
    return experiment


def _validate_candidate_consistency(
    *,
    model_facts: dict[str, Any],
    results: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    if results["feature_columns"] != contract["predictor_names"]:
        raise ValueError("candidate result feature list differs from PRD contract")
    if results["model_params"] != prd_config.serialized_model_params():
        raise ValueError("candidate result parameters differ from executable config")
    if results["random_seed"] != prd_config.PRD_MODEL_PARAMS["random_state"]:
        raise ValueError("candidate result seed differs from executable config")
    if model_facts["num_features"] != contract["predictor_count"]:
        raise ValueError("model artifact has an unexpected feature count")
    if results["best_iteration_zero_based"] != model_facts[
        "best_iteration_zero_based"
    ]:
        raise ValueError(
            "candidate results best iteration does not match the model artifact"
        )
    if results["effective_tree_count"] != model_facts["effective_tree_count"]:
        raise ValueError(
            "candidate results tree count does not match the model artifact"
        )


def _mismatch_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if left == right:
        return []
    if type(left) is not type(right):
        return [f"{prefix or '$'}: {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, dict):
        paths: list[str] = []
        keys = set(left) | set(right)
        for key in sorted(keys):
            child = f"{prefix}.{key}" if prefix else key
            if key not in left:
                paths.append(f"{child}: missing from generated manifest")
            elif key not in right:
                paths.append(f"{child}: missing from committed manifest")
            else:
                paths.extend(_mismatch_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{prefix}: list length {len(left)} != {len(right)}"]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(
                _mismatch_paths(left_item, right_item, f"{prefix}[{index}]")
            )
        return paths
    return [f"{prefix}: {left!r} != {right!r}"]


def manifests_semantically_equal(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    """Return whether two manifests contain the same JSON values."""
    return left == right


def build_prd_manifest(
    *,
    model_path: Path,
    results_path: Path,
    comparison_path: Path,
    promotion_date: str | None = None,
    promote: bool = False,
    validation_scope: str | None = None,
    baseline_results_path: Path | None = None,
    v2_validation_path: Path | None = None,
    run_manifest_path: Path | None = None,
    experiment_candidate_results_path: Path | None = None,
    historical_baseline: dict[str, str] | None = None,
    root: Path = prd_config.ROOT,
) -> dict[str, Any]:
    """Derive a technical PRD manifest from a caller-selected candidate.

    ``promote=True`` is required to emit ``status = "PRD"``. The builder does
    not compare models or decide which artifact should be promoted.
    """
    if promote:
        if not promotion_date:
            raise ValueError("promotion_date is required when promote=True")
        if not validation_scope:
            raise ValueError("validation_scope is required when promote=True")
        status = PROMOTED_STATUS
    else:
        status = CANDIDATE_STATUS

    experiment_dir = comparison_path.parent
    baseline_results_path = (
        baseline_results_path or experiment_dir / "model_a_17.results.json"
    )
    v2_validation_path = (
        v2_validation_path or experiment_dir / "v2_validation.json"
    )
    run_manifest_path = (
        run_manifest_path or experiment_dir / "run_manifest.json"
    )
    experiment_candidate_results_path = (
        experiment_candidate_results_path
        or experiment_dir / "model_b_genome_25.results.json"
    )
    historical_baseline = dict(
        historical_baseline or prd_config.PHASE5_HISTORICAL_BASELINE
    )

    prd_config.validate_split_contract()
    identity = prd_config.canonical_model_identity()
    contract = _feature_contract(root)
    model_facts = _inspect_model(model_path)
    results = _load_json(results_path)
    comparison = _load_json(comparison_path)
    _validate_candidate_consistency(
        model_facts=model_facts,
        results=results,
        contract=contract,
    )

    is_notebook_artifact = (
        model_path.resolve() == prd_config.PRD_ARTIFACT_PATH.resolve()
    )
    provenance = (
        prd_config.notebook_artifact_provenance()
        if is_notebook_artifact
        else prd_config.historical_artifact_provenance()
    )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION if is_notebook_artifact else 2,
        "model_name": identity["model_name"],
        "model_version": identity["model_version"],
        "canonical_name": identity["canonical_name"],
        "status": status,
        "promotion_date": promotion_date if promote else None,
        "validation_scope": validation_scope,
        "provenance": provenance,
        "artifact": {
            "path": _relative_path(model_path, root),
            "sha256": model_facts["sha256"],
            "feature_names_embedded": model_facts["feature_names_embedded"],
            "feature_types_embedded": model_facts["feature_types_embedded"],
            "input_contract": prd_config.PRD_INPUT_CONTRACT,
            "scoring_interface": prd_config.PRD_SCORING_INTERFACE,
        },
        **(
            {
                "results_artifact": {
                    "path": _relative_path(results_path, root),
                    "sha256": sha256_file(results_path),
                }
            }
            if is_notebook_artifact
            else {}
        ),
        "data_sources": {
            "ratings": {
                "path": _relative_path(prd_config.RATINGS_SOURCE_PATH, root),
                "sha256": prd_config.RATINGS_SOURCE_SHA256,
            },
            "feature_artifact": {
                "path": _relative_path(prd_config.FEATURE_ARTIFACT_PATH, root),
                "metadata_path": _relative_path(
                    prd_config.FEATURE_METADATA_PATH, root
                ),
            },
        },
        "source_experiment": _source_experiment(
            comparison_path=comparison_path,
            candidate_results_path=experiment_candidate_results_path,
            baseline_results_path=baseline_results_path,
            v2_validation_path=v2_validation_path,
            run_manifest_path=run_manifest_path,
            root=root,
        ),
        "historical_baseline": historical_baseline,
        "feature_contract": contract,
        "target": prd_config.target_contract(),
        "temporal_splits": prd_config.serialized_splits(),
        "training": {
            "random_seed": prd_config.PRD_MODEL_PARAMS["random_state"],
            "best_iteration_zero_based": model_facts["best_iteration_zero_based"],
            "effective_tree_count": model_facts["effective_tree_count"],
            "missing_value_behavior": prd_config.MISSING_VALUE_BEHAVIOR,
            "parameters": prd_config.serialized_model_params(),
        },
        "evaluation": _evaluation(
            results, comparison, historical_baseline["name"]
        ),
    }


def build_canonical_prd_manifest(
    *, root: Path = prd_config.ROOT
) -> dict[str, Any]:
    """Build the notebook artifact pointer, or the legacy pointer until it exists."""
    notebook_artifacts_exist = (
        prd_config.PRD_ARTIFACT_PATH.is_file()
        and prd_config.PRD_RESULTS_PATH.is_file()
    )
    return build_prd_manifest(
        model_path=(
            prd_config.PRD_ARTIFACT_PATH
            if notebook_artifacts_exist
            else prd_config.EXPERIMENT_CANDIDATE_ARTIFACT_PATH
        ),
        results_path=(
            prd_config.PRD_RESULTS_PATH
            if notebook_artifacts_exist
            else prd_config.PRD_CANDIDATE_RESULTS_PATH
        ),
        comparison_path=prd_config.PRD_COMPARISON_PATH,
        baseline_results_path=prd_config.PRD_BASELINE_RESULTS_PATH,
        v2_validation_path=prd_config.PRD_V2_VALIDATION_PATH,
        run_manifest_path=prd_config.PRD_RUN_MANIFEST_PATH,
        experiment_candidate_results_path=prd_config.PRD_CANDIDATE_RESULTS_PATH,
        historical_baseline=prd_config.PHASE5_HISTORICAL_BASELINE,
        promotion_date=prd_config.PRD_PROMOTION_DATE,
        validation_scope=prd_config.PRD_VALIDATION_SCOPE,
        promote=True,
        root=root,
    )


def check_prd_manifest(
    manifest_path: Path = prd_config.PRD_MODEL_MANIFEST_PATH,
    *,
    root: Path = prd_config.ROOT,
) -> dict[str, Any]:
    """Generate the expected canonical manifest and compare it without writing."""
    generated = build_canonical_prd_manifest(root=root)
    existing = _load_json(manifest_path)
    if not manifests_semantically_equal(generated, existing):
        mismatch = _mismatch_paths(generated, existing)
        details = "; ".join(mismatch[:12])
        raise ValueError(
            "generated PRD manifest differs from the committed pointer"
            + (f": {details}" if details else "")
        )
    return generated


def write_prd_manifest(
    manifest: dict[str, Any],
    path: Path,
    *,
    root: Path = prd_config.ROOT,
) -> None:
    """Write a manifest without allowing evidence or selected inputs as outputs."""
    destination = path.resolve()
    canonical = prd_config.PRD_MODEL_MANIFEST_PATH.resolve()
    if destination == canonical and manifest.get("status") != PROMOTED_STATUS:
        raise ValueError(
            "refusing to overwrite the canonical PRD manifest without promote=True"
        )

    immutable_directories = tuple(
        directory.resolve()
        for directory in prd_config.IMMUTABLE_HISTORICAL_EVIDENCE_DIRS
    )
    if any(
        destination == directory or directory in destination.parents
        for directory in immutable_directories
    ):
        raise ValueError("refusing to write a manifest inside immutable evidence")

    selected_inputs = _manifest_input_paths(manifest, root=root)
    selected_inputs.add(prd_config.PRD_ARTIFACT_PATH.resolve())
    selected_inputs.add(prd_config.PRD_RESULTS_PATH.resolve())
    if destination in selected_inputs:
        raise ValueError("refusing to overwrite a selected manifest input artifact")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_prd_manifest(manifest), encoding="utf-8")


def _manifest_input_paths(
    manifest: dict[str, Any], *, root: Path
) -> set[Path]:
    """Resolve every artifact path recorded as input evidence in a manifest."""
    relative_paths = [
        manifest["artifact"]["path"],
        manifest["data_sources"]["ratings"]["path"],
        manifest["data_sources"]["feature_artifact"]["path"],
        manifest["data_sources"]["feature_artifact"]["metadata_path"],
    ]
    if "results_artifact" in manifest:
        relative_paths.append(manifest["results_artifact"]["path"])
    experiment = manifest["source_experiment"]
    relative_paths.extend(
        experiment[key]
        for key in (
            "comparison",
            "candidate_results",
            "baseline_results",
            "v2_validation",
            "run_manifest",
        )
    )
    historical = manifest["historical_baseline"]
    relative_paths.extend(
        historical[key]
        for key in ("model", "metadata", "evaluation", "error_analysis")
    )
    return {(root / relative_path).resolve() for relative_path in relative_paths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="generate the canonical manifest in memory and compare it",
    )
    parser.add_argument(
        "--write",
        type=Path,
        help="write a generated manifest to this explicit path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=prd_config.PRD_MODEL_MANIFEST_PATH,
        help="committed manifest compared by --check",
    )
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--promotion-date")
    parser.add_argument("--validation-scope")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--v2-validation", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    args = parser.parse_args(argv)

    if args.check and args.write is not None:
        parser.error("--check and --write are mutually exclusive")
    if not args.check and args.write is None:
        parser.error("one of --check or --write is required")

    try:
        if args.check:
            check_prd_manifest(args.manifest)
            print(f"checked {args.manifest}")
            return 0

        missing = [
            name
            for name, value in (
                ("--model", args.model),
                ("--results", args.results),
                ("--comparison", args.comparison),
            )
            if value is None
        ]
        if missing:
            parser.error(f"{', '.join(missing)} required with --write")

        write_path = args.write
        canonical = prd_config.PRD_MODEL_MANIFEST_PATH.resolve()
        if write_path.resolve() == canonical and not args.promote:
            raise ValueError(
                "refusing to overwrite the canonical PRD manifest without --promote"
            )
        manifest = build_prd_manifest(
            model_path=args.model,
            results_path=args.results,
            comparison_path=args.comparison,
            baseline_results_path=args.baseline_results,
            v2_validation_path=args.v2_validation,
            run_manifest_path=args.run_manifest,
            promotion_date=args.promotion_date,
            validation_scope=args.validation_scope,
            promote=args.promote,
        )
        write_prd_manifest(manifest, write_path)
        print(f"wrote {write_path}")
        return 0
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_canonical_prd_manifest",
    "build_prd_manifest",
    "check_prd_manifest",
    "dumps_prd_manifest",
    "main",
    "manifests_semantically_equal",
    "write_prd_manifest",
]
