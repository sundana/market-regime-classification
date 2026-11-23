"""Shared defaults for the baseline pipeline."""

from __future__ import annotations

from dataclasses import dataclass

TRADING_DAYS = 252

# Feature order is stable so that `rv_21` has a known column index after
# subsetting. Detectors still look up the volatility column by name.
FEATURE_ORDER: tuple[str, ...] = (
    "rv_5",
    "rv_21",
    "rv_63",
    "vol_ratio",
    "trend_21",
    "trend_63",
    "drawdown_63",
    "corr_21",
)

DEFAULT_DETECTORS: tuple[str, ...] = ("vol_threshold", "gmm", "hmm", "jump")


@dataclass(frozen=True)
class Config:
    """Model / walk-forward configuration.

    Attributes
    ----------
    n_states:
        Number of regimes. States are sorted by mean realized vol, so state 0
        is always the calmest.
    train_days:
        Length of the trailing training window in trading days.
    refit_every:
        Number of out-of-sample days between refits.
    vol_col:
        Feature used to sort states and (for the rule baseline) to threshold.
    random_state:
        Seed for KMeans / GaussianMixture initialization.
    hmm_n_init, hmm_max_iter:
        Restarts and EM iteration cap for the hand-rolled HMM.
    hmm_stickiness:
        Dirichlet self-transition prior weight (kappa >= 0) encouraging regime
        persistence in the HMM M-step.
    hmm_cov_shrinkage:
        Convex shrinkage weight in [0, 1] toward the diagonal covariance matrix
        in the HMM M-step.
    gmm_n_init:
        Restarts for sklearn's GaussianMixture.
    jump_penalty:
        L1 state-switch penalty (lambda >= 0) for the Statistical Jump Model.
    jump_n_init, jump_max_iter:
        Restarts and coordinate-descent iteration cap for the Jump Model.
    """

    n_states: int = 3
    train_days: int = 756  # ~3 years
    refit_every: int = 63  # ~quarterly
    vol_col: str = "rv_21"
    random_state: int = 42
    hmm_n_init: int = 3
    hmm_max_iter: int = 50
    hmm_stickiness: float = 15.0
    hmm_cov_shrinkage: float = 0.25
    gmm_n_init: int = 3
    jump_penalty: float = 10.0
    jump_n_init: int = 3
    jump_max_iter: int = 30

    def feature_columns(self, available: list[str] | tuple[str, ...]) -> list[str]:
        """Return FEATURE_ORDER columns present in `available`, plus vol_col."""
        cols = [c for c in FEATURE_ORDER if c in available]
        if self.vol_col not in cols:
            if self.vol_col not in available:
                raise KeyError(
                    f"vol_col={self.vol_col!r} missing from features: {list(available)}"
                )
            cols.append(self.vol_col)
        return cols
