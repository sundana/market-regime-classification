"""Feature engineering.

Everything here is a function of data at or before time `t`. In particular:

* volatility is a trailing rolling standard deviation;
* momentum is a trailing log-price difference;
* drawdown uses a trailing rolling maximum;
* correlation uses trailing rolling correlations.

The final frame is dropna'd, so the first ~63 observations are unavailable for
all models equally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def daily_returns(prices: pd.DataFrame, column: str | None = None) -> pd.Series:
    """Simple daily returns for one price column (default: first column)."""
    if column is None:
        column = prices.columns[0]
    return prices[column].pct_change().rename(f"{column}_ret")


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the baseline feature set from a price frame.

    The first column is treated as the primary (equity) asset. Additional
    columns contribute to the cross-asset correlation feature.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")
    if len(prices) < 80:
        raise ValueError(
            f"need at least 80 price rows for a 63-day feature warmup, got {len(prices)}"
        )
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive")

    primary = prices.columns[0]
    log_px = np.log(prices[primary])
    rets = np.log(prices).diff()

    f = pd.DataFrame(index=prices.index)

    for window in (5, 21, 63):
        f[f"rv_{window}"] = (
            rets[primary].rolling(window, min_periods=window).std()
            * np.sqrt(TRADING_DAYS)
        )

    f["vol_ratio"] = f["rv_5"] / f["rv_63"].replace(0.0, np.nan)
    f["trend_21"] = log_px.diff(21)
    f["trend_63"] = log_px.diff(63)

    trailing_high = prices[primary].rolling(63, min_periods=63).max()
    f["drawdown_63"] = prices[primary] / trailing_high - 1.0

    if prices.shape[1] >= 2:
        corrs: list[pd.Series] = []
        cols = list(prices.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corrs.append(
                    rets[cols[i]].rolling(21, min_periods=21).corr(rets[cols[j]])
                )
        if corrs:
            f["corr_21"] = pd.concat(corrs, axis=1).mean(axis=1)

    f = f.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if f.empty:
        raise ValueError("no feature rows survived the warmup")
    return f
