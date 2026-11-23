from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_regime_classification.detectors import VolThresholdDetector
from market_regime_classification.metadata import sha256_file
from market_regime_classification.models import load_model
from market_regime_classification.run import main

EXPECTED_FILES = {
    "features.csv",
    "metrics.csv",
    "metrics.json",
    "agreement.csv",
    "vol_threshold_labels.csv",
    "vol_threshold_probabilities.csv",
    "vol_threshold_regime_stats.csv",
    "true_states.csv",
    "run_metadata.json",
}


def test_cli_synthetic_end_to_end(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    rc = main(
        [
            "--source",
            "synthetic",
            "--n-days",
            "320",
            "--train-days",
            "80",
            "--refit-every",
            "40",
            "--detectors",
            "vol_threshold",
            "--cost-bps",
            "5",
            "--save-models",
            "--no-plot",
            "--outdir",
            str(outdir),
        ]
    )

    assert rc == 0
    assert {path.name for path in outdir.iterdir()} == EXPECTED_FILES | {
        "models"
    }
    model_names = {path.name for path in (outdir / "models").iterdir()}
    assert model_names == {"vol_threshold.json"}

    metrics = pd.read_csv(outdir / "metrics.csv")
    assert {"cost_bps", "gross_cagr", "gross_sharpe", "ann_cost"} <= set(
        metrics.columns
    )
    assert (metrics["cost_bps"] == 5.0).all()
    assert (metrics["ann_cost"] >= 0.0).all()

    metadata = json.loads((outdir / "run_metadata.json").read_text())
    assert metadata["source"] == {"type": "synthetic", "n_days": 320, "seed": 42}
    assert metadata["argv"][:2] == ["--source", "synthetic"]
    assert metadata["config"]["train_days"] == 80
    assert metadata["config"]["refit_every"] == 40
    assert metadata["config"]["detectors"] == ["vol_threshold"]
    assert metadata["config"]["max_stale_days"] == 5
    assert metadata["config"]["cost_bps"] == 5.0
    assert metadata["config"]["save_models"] is True

    recorded = {item["path"]: item for item in metadata["outputs"]}
    assert set(recorded) == (
        EXPECTED_FILES - {"run_metadata.json"}
    ) | {"models/vol_threshold.json"}
    for name, item in recorded.items():
        path = outdir / name
        assert item["sha256"] == sha256_file(path)
        assert item["bytes"] == path.stat().st_size

    model = load_model(outdir / "models" / "vol_threshold.json")
    assert isinstance(model, VolThresholdDetector)
    assert model.n_states == 3
