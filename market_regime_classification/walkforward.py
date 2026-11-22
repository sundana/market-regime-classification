"""Walk-forward harness.

One pass per detector:

* fit on a trailing window of ``train_days`` observations;
* predict the next ``refit_every`` observations out-of-sample;
* carry the last filtered state forward into the next block (HMM only);
* repeat.

Because training windows only contain past data and predictions are filtered,
every output row is causal. `tests/test_walkforward.py` checks that the
predictions are invariant to truncating the dataset after the row in question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd


class _Detector(Protocol):
    n_states: int

    def fit(self, X: pd.DataFrame, feature_cols: list[str], vol_col: str): ...

    def predict(
        self,
        X: pd.DataFrame,
        feature_cols: list[str],
        init_probs: np.ndarray | None = None,
    ) -> np.ndarray: ...


@dataclass
class WalkForwardResult:
    name: str
    probs: pd.DataFrame  # index = OOS dates, columns = state_0..state_k
    labels: pd.Series  # argmax state per OOS date
    refit_dates: list[pd.Timestamp]

    def __post_init__(self) -> None:
        if len(self.probs) != len(self.labels):
            raise ValueError("probs and labels have inconsistent lengths")


def run_walkforward(
    features: pd.DataFrame,
    detector_factory: Callable[[], _Detector],
    *,
    name: str,
    train_days: int,
    refit_every: int,
    feature_cols: list[str],
    vol_col: str,
) -> WalkForwardResult:
    """Run a detector across a rolling out-of-sample schedule."""
    if train_days < 60:
        raise ValueError("train_days must be at least 60")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    if not features.index.is_monotonic_increasing:
        raise ValueError("features index must be sorted ascending")
    if len(features) <= train_days:
        raise ValueError(
            f"need more than train_days={train_days} rows; got {len(features)}"
        )

    n = len(features)
    all_probs: list[np.ndarray] = []
    all_index: list[pd.Index] = []
    refit_dates: list[pd.Timestamp] = []

    previous_probs: np.ndarray | None = None
    n_states: int | None = None

    for block_start in range(train_days, n, refit_every):
        block_end = min(block_start + refit_every, n)
        train = features.iloc[block_start - train_days : block_start]
        test = features.iloc[block_start:block_end]

        detector = detector_factory()
        detector.fit(train, feature_cols=feature_cols, vol_col=vol_col)
        probs = np.asarray(
            detector.predict(
                test, feature_cols=feature_cols, init_probs=previous_probs
            ),
            dtype=float,
        )

        n_states = detector.n_states
        if probs.shape != (len(test), n_states):
            raise RuntimeError(
                f"{name}: expected {(len(test), n_states)} probabilities, "
                f"got {probs.shape}"
            )
        if not np.isfinite(probs).all():
            raise RuntimeError(f"{name}: non-finite probabilities")
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise RuntimeError(f"{name}: probability rows do not sum to 1")

        all_probs.append(probs)
        all_index.append(test.index)
        refit_dates.append(test.index[0])
        previous_probs = probs[-1].copy()

    assert n_states is not None
    P = pd.DataFrame(
        np.vstack(all_probs),
        index=pd.Index(np.concatenate([idx.to_numpy() for idx in all_index])),
        columns=[f"state_{j}" for j in range(n_states)],
    )
    labels = pd.Series(P.to_numpy().argmax(axis=1), index=P.index, name="state")

    return WalkForwardResult(
        name=name, probs=P, labels=labels, refit_dates=refit_dates
    )
