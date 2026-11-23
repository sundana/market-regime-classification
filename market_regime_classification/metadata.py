"""Run provenance metadata persisted alongside CLI outputs.

A production run should be reproducible and auditable. ``run_metadata.json``
records the package version, interpreter/platform, git revision, the exact
arguments and configuration used, the data source, and a SHA-256 digest for
every other output file.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__

METADATA_FILENAME = "run_metadata.json"
METADATA_SCHEMA_VERSION = 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git(args: list[str]) -> str | None:
    """Return trimmed git output, or ``None`` when git/repo is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_run_metadata(
    *,
    argv: list[str],
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    outputs: Iterable[str | Path],
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable provenance record for one CLI run.

    When ``base_dir`` is given, output paths are recorded relative to it (for
    example ``models/hmm.json``); otherwise only the file name is recorded.
    """
    commit = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain", "--untracked-files=no"])
    root = Path(base_dir).resolve() if base_dir is not None else None

    files = []
    for output in outputs:
        path = Path(output)
        name = path.name
        if root is not None:
            with contextlib.suppress(ValueError):
                name = path.resolve().relative_to(root).as_posix()
        files.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "argv": list(argv),
        "config": dict(config),
        "source": dict(source),
        "outputs": sorted(files, key=lambda item: str(item["path"])),
    }


def write_run_metadata(
    outdir: str | Path, metadata: Mapping[str, Any]
) -> Path:
    """Write ``metadata`` to ``<outdir>/run_metadata.json`` and return its path."""
    path = Path(outdir) / METADATA_FILENAME
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
