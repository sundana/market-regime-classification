from __future__ import annotations

import json
from pathlib import Path

from market_regime_classification.metadata import (
    METADATA_FILENAME,
    METADATA_SCHEMA_VERSION,
    collect_run_metadata,
    sha256_file,
    write_run_metadata,
)


def test_collect_run_metadata(tmp_path: Path) -> None:
    output = tmp_path / "metrics.csv"
    output.write_text("detector,sharpe\nhmm,0.5\n", encoding="utf-8")

    metadata = collect_run_metadata(
        argv=["--source", "csv", "--csv", "prices.csv"],
        config={"n_states": 3, "train_days": 756},
        source={"type": "csv", "path": "prices.csv"},
        outputs=[output],
    )

    assert metadata["schema_version"] == METADATA_SCHEMA_VERSION
    assert metadata["package_version"]
    assert metadata["argv"] == ["--source", "csv", "--csv", "prices.csv"]
    assert metadata["config"] == {"n_states": 3, "train_days": 756}
    assert metadata["source"] == {"type": "csv", "path": "prices.csv"}
    assert metadata["outputs"] == [
        {
            "path": "metrics.csv",
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        }
    ]


def test_write_run_metadata_roundtrips(tmp_path: Path) -> None:
    metadata = collect_run_metadata(
        argv=["--source", "synthetic"],
        config={"n_states": 3},
        source={"type": "synthetic"},
        outputs=[],
    )
    path = write_run_metadata(tmp_path, metadata)

    assert path == tmp_path / METADATA_FILENAME
    assert json.loads(path.read_text(encoding="utf-8")) == metadata
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_output_paths_are_relative_to_base_dir(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    output = models_dir / "hmm.json"
    output.write_text("{}", encoding="utf-8")

    metadata = collect_run_metadata(
        argv=[],
        config={},
        source={},
        outputs=[output],
        base_dir=tmp_path,
    )

    assert metadata["outputs"][0]["path"] == "models/hmm.json"


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("same bytes", encoding="utf-8")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64
