from __future__ import annotations

import ast
import json
from pathlib import Path

from src.training import prd_config


NOTEBOOK_ROOT = prd_config.ROOT / "notebooks/model"
HISTORICAL_NAMES = (
    "01_training_dataset.ipynb",
    "02_temporal_splits.ipynb",
    "03_xgboost_baseline.ipynb",
    "04_model_evaluation.ipynb",
    "05_error_analysis.ipynb",
)
PRD_NAMES = (
    "01_training_dataset.ipynb",
    "02_temporal_splits.ipynb",
    "03_xgboost_training.ipynb",
    "04_model_evaluation.ipynb",
    "05_model_artifact_and_manifest.ipynb",
)


def _load_notebook(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["nbformat"] == 4
    assert isinstance(value["cells"], list)
    return value


def test_expected_model_notebook_structure_exists() -> None:
    historical = NOTEBOOK_ROOT / "phase5_baseline_17"
    current = NOTEBOOK_ROOT / "genome_prd_v1"
    assert tuple(path.name for path in sorted(historical.glob("*.ipynb"))) == (
        HISTORICAL_NAMES
    )
    assert tuple(path.name for path in sorted(current.glob("*.ipynb"))) == PRD_NAMES
    assert not tuple(NOTEBOOK_ROOT.glob("*.ipynb"))


def test_all_model_notebooks_are_valid_json_and_code_parses() -> None:
    for path in NOTEBOOK_ROOT.glob("*/*.ipynb"):
        notebook = _load_notebook(path)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=str(path))


def test_prd_notebooks_reference_canonical_modules_and_paths() -> None:
    current = NOTEBOOK_ROOT / "genome_prd_v1"
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in current.glob("*.ipynb")
    )
    assert "src.training" in combined
    assert "prd_config.PRD_FEATURES" in combined
    assert "prd_config.PRD_SPLITS" in combined
    assert "prd_config.PRD_ARTIFACT_PATH" in combined
    assert "prd_config.PRD_RESULTS_PATH" in combined
    assert "build_prd_manifest" in combined
    assert prd_config.FEATURE_ARTIFACT_PATH.is_file()
    assert prd_config.FEATURE_METADATA_PATH.is_file()
