"""Data sources: synthetic regime-switching market, yfinance, and CSV.

All loaders return a price frame with one column per asset. The synthetic
generator additionally returns the latent state series, which is useful for a
sanity check (not for tuning on real data, obviously).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Synthetic market
# ---------------------------------------------------------------------------
def synthetic_market(
    n_days: int = 5_000,
    seed: int = 42,
    end: str = "2026-11-20",
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate a two-asset market with three latent regimes.

    Regimes:
      0 — calm: positive equity drift, low vol, negative stock/bond correlation
      1 — choppy: flat drift, medium vol, roughly uncorrelated
      2 — crisis: negative drift, high vol, positive stock/bond correlation
                  plus occasional jump risk

    Returns
    -------
    prices : DataFrame with columns ["EQ", "BD"]
    true_state : Series of latent regime labels (0/1/2)
    """
    if n_days < 300:
        raise ValueError("n_days must be at least 300")

    rng = np.random.default_rng(seed)

    # Row-stochastic transition matrix.
    trans = np.array(
        [
            [0.9920, 0.0075, 0.0005],
            [0.0200, 0.9690, 0.0110],
            [0.0080, 0.0300, 0.9620],
        ]
    )

    mu_eq = np.array([0.00055, 0.00010, -0.00180])
    sd_eq = np.array([0.0070, 0.0130, 0.0320])

    mu_bd = np.array([0.00012, 0.00008, -0.00060])
    sd_bd = np.array([0.0040, 0.0060, 0.0110])

    # Equity/bond return correlation by regime.
    rho = np.array([-0.25, 0.05, 0.55])

    states = np.empty(n_days, dtype=np.int64)
    states[0] = 0
    for t in range(1, n_days):
        states[t] = rng.choice(3, p=trans[states[t - 1]])

    z_eq = rng.standard_normal(n_days)
    z_idio = rng.standard_normal(n_days)

    eq_ret = mu_eq[states] + sd_eq[states] * z_eq
    bd_ret = mu_bd[states] + sd_bd[states] * (
        rho[states] * z_eq + np.sqrt(1.0 - rho[states] ** 2) * z_idio
    )

    # Rare gap-downs concentrated in the crisis state.
    jump_mask = (rng.random(n_days) < 0.0015) & (states == 2)
    n_jumps = int(jump_mask.sum())
    if n_jumps:
        eq_ret[jump_mask] -= rng.gamma(shape=3.0, scale=0.010, size=n_jumps)

    dates = pd.bdate_range(end=end, periods=n_days)
    prices = pd.DataFrame(
        {
            "EQ": 100.0 * np.exp(np.cumsum(eq_ret)),
            "BD": 100.0 * np.exp(np.cumsum(bd_ret)),
        },
        index=dates,
    )
    true_state = pd.Series(states, index=dates, name="true_state")
    return prices, true_state


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------
def load_yfinance(
    tickers: list[str] | str,
    start: str = "2005-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted daily closes via yfinance.

    A single ticker returns a one-column frame; multiple tickers return one
    column per ticker. Rows are forward-filled after dropping empty rows.
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    if not tickers:
        raise ValueError("at least one ticker is required")

    import yfinance as yf  # imported lazily so the rest works offline

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            "yfinance returned no data (check tickers / network / date range)"
        )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError(f"unexpected yfinance columns: {raw.columns}")
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    prices = prices.sort_index()
    prices = prices.dropna(how="all").ffill()
    # Drop leading rows where any series is still missing.
    prices = prices.dropna(how="any")
    if prices.empty:
        raise RuntimeError("no usable price rows after cleaning")
    return prices


def load_csv(path: str | Path, date_col: str | None = None) -> pd.DataFrame:
    """Load a date-indexed price CSV.

    The file is expected to contain a date column (used as index) and one or
    more price columns. With `date_col=None`, the first column is treated as
    the date/index.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if date_col is None:
        date_col = df.columns[0]
    if date_col not in df.columns:
        raise KeyError(f"date column {date_col!r} not found in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("CSV has no numeric price columns")
    numeric = numeric.dropna(how="all").ffill().dropna(how="any")
    if numeric.empty:
        raise ValueError("CSV has no usable rows after cleaning")
    return numeric
