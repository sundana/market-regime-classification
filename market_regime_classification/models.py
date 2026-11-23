"""Save and load fitted regime models as schema-versioned JSON.

JSON is deliberate: model artifacts are small, human-inspectable, diffable, and
cannot execute code on load (unlike pickle). Each payload carries a schema
version, the detector name, the feature/volatility column contract, the
training-window metadata, and the fitted parameters.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .detectors import (
    MODEL_SCHEMA_VERSION,
    BaseDetector,
    GaussianHMMDetector,
    GMMDetector,
    JumpModelDetector,
    VolThresholdDetector,
)

MODEL_FILE_SUFFIX = ".json"

_MODEL_CLASSES: dict[str, type[BaseDetector]] = {
    VolThresholdDetector.name: VolThresholdDetector,
    GMMDetector.name: GMMDetector,
    GaussianHMMDetector.name: GaussianHMMDetector,
    JumpModelDetector.name: JumpModelDetector,
}


def detector_from_dict(payload: Mapping[str, Any]) -> BaseDetector:
    """Rebuild a fitted detector from a serialized payload."""
    if not isinstance(payload, Mapping):
        raise TypeError("model payload must be a mapping")
    name = payload.get("detector")
    if not isinstance(name, str):
        raise ValueError("model payload is missing a string 'detector' field")
    model_class = _MODEL_CLASSES.get(name)
    if model_class is None:
        raise ValueError(
            f"unknown detector {name!r}; expected one of "
            f"{sorted(_MODEL_CLASSES)}"
        )
    return model_class.from_dict(payload)


def save_model(model: object, path: str | Path) -> Path:
    """Serialize a fitted detector to ``path`` and return the written path.

    Parent directories are created as needed. The file is written atomically
    via a temporary file, and non-finite parameters are rejected rather than
    written as invalid JSON.
    """
    if not isinstance(model, BaseDetector):
        raise TypeError("model must be a BaseDetector instance")
    payload = model.to_dict()
    payload["artifact"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": f"market-regime-classification {__version__}",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        temporary.replace(destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def load_model(path: str | Path) -> BaseDetector:
    """Load a fitted detector from a JSON model artifact."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: model payload must be a JSON object")
    return detector_from_dict(payload)


__all__ = [
    "MODEL_FILE_SUFFIX",
    "MODEL_SCHEMA_VERSION",
    "detector_from_dict",
    "load_model",
    "save_model",
]
