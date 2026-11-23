"""Data sources: synthetic regime-switching market, yfinance, and CSV.

All loaders return a price frame with one column per asset. The synthetic
generator additionally returns the latent state series, which is useful for a
sanity check (not for tuning on real data, obviously).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
DEFAULT_MAX_STALE_DAYS = 5


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
def _validate_price_frame(
    prices: pd.DataFrame,
    *,
    source: str,
    max_stale_days: int = DEFAULT_MAX_STALE_DAYS,
) -> pd.DataFrame:
    """Validate and clean a raw price frame shared by the real-data loaders.

    Rules
    -----
    * the index must be a :class:`pandas.DatetimeIndex` with no NaT and no
      duplicate dates;
    * columns must be unique, numeric, entirely finite and strictly positive;
    * history before the first date on which *every* asset has a price is
      trimmed (assets are never back-filled);
    * interior gaps are forward-filled for at most ``max_stale_days`` rows;
      longer gaps raise instead of silently carrying a stale price;
    * at least two rows must remain.

    Parameters
    ----------
    prices:
        Raw price frame as returned by a data source.
    source:
        Human-readable source label used in error messages.
    max_stale_days:
        Maximum number of consecutive missing observations that may be
        forward-filled. ``0`` disables filling.

    Returns
    -------
    A cleaned, float-typed copy of ``prices`` sorted by date.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError(f"{source}: price frame is empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(f"{source}: price index must be a DatetimeIndex")
    if prices.index.hasnans:
        raise ValueError(f"{source}: price index contains NaT")
    if prices.index.has_duplicates:
        dupes = prices.index[prices.index.duplicated()].unique()
        preview = ", ".join(str(d) for d in dupes[:5])
        raise ValueError(f"{source}: duplicate dates in index (e.g. {preview})")
    if prices.columns.has_duplicates:
        dupes = prices.columns[prices.columns.duplicated()].unique()
        raise ValueError(f"{source}: duplicate price columns: {list(dupes)}")
    if max_stale_days < 0:
        raise ValueError("max_stale_days must be >= 0")

    try:
        prices = prices.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: price columns must be numeric") from exc

    prices = prices.sort_index().dropna(how="all")
    if prices.empty:
        raise ValueError(f"{source}: no usable price rows")

    if np.isinf(prices.to_numpy()).any():
        raise ValueError(f"{source}: prices contain infinite values")

    # Assets may list at different dates. Trim to the first date on which all
    # of them have an observation; do not invent pre-listing history.
    all_missing = prices.isna().all()
    missing_assets = [str(column) for column in all_missing[all_missing].index]
    if missing_assets:
        raise ValueError(f"{source}: no usable prices for columns {missing_assets}")
    complete = prices.dropna(how="any")
    if complete.empty:
        raise ValueError(f"{source}: no date has a price for every asset")
    prices = prices.loc[complete.index[0] :]

    missing_cells = int(prices.isna().to_numpy().sum())
    if missing_cells:
        if max_stale_days:
            prices = prices.ffill(limit=max_stale_days)
        remaining = prices.isna()
        if remaining.to_numpy().any():
            bad = {
                str(col): int(count)
                for col, count in remaining.sum().items()
                if count
            }
            first_bad = prices.index[remaining.any(axis=1)][0]
            raise ValueError(
                f"{source}: gaps longer than max_stale_days={max_stale_days} "
                f"remain for {bad}; first affected date {first_bad.date()}. "
                "Repair the input, load more history, or raise max_stale_days."
            )
        logger.warning(
            "%s: forward-filled %d missing price cell(s) for at most %d day(s)",
            source,
            missing_cells,
            max_stale_days,
        )

    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: prices contain NaN or infinite values")
    non_positive = np.argwhere(values <= 0.0)
    if len(non_positive):
        row, col = non_positive[0]
        raise ValueError(
            f"{source}: non-positive price for {prices.columns[col]!r} on "
            f"{prices.index[row].date()}: {values[row, col]}"
        )
    if len(prices) < 2:
        raise ValueError(f"{source}: need at least 2 price rows, got {len(prices)}")
    return prices


def _extract_close_prices(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract the adjusted-close block from a yfinance response."""
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0)
        field = next((f for f in ("Close", "Adj Close") if f in fields), None)
        if field is None:
            raise RuntimeError(f"unexpected yfinance columns: {raw.columns}")
        prices = raw[field].copy()
    else:
        field = next((f for f in ("Close", "Adj Close") if f in raw.columns), None)
        if field is None:
            raise RuntimeError(f"unexpected yfinance columns: {raw.columns}")
        prices = raw[[field]].copy()
        prices.columns = tickers[:1]

    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise RuntimeError(f"yfinance did not return prices for {missing}")
    return prices.loc[:, tickers]


def load_yfinance(
    tickers: list[str] | str,
    start: str = "2005-01-01",
    end: str | None = None,
    *,
    max_stale_days: int = DEFAULT_MAX_STALE_DAYS,
    retries: int = 3,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Download adjusted daily closes via yfinance.

    A single ticker returns a one-column frame; multiple tickers return one
    column per ticker. Transient download failures are retried with exponential
    backoff, then the result goes through :func:`_validate_price_frame`.
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("at least one ticker is required")
    if retries < 1:
        raise ValueError("retries must be >= 1")

    import yfinance as yf  # imported lazily so the rest works offline

    raw: pd.DataFrame | None = None
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
                timeout=timeout,
                multi_level_index=True,
            )
        except Exception as exc:  # yfinance raises a variety of I/O errors
            last_error = exc
            raw = None
        if raw is not None and not raw.empty:
            break
        if attempt < retries - 1:
            delay = float(2**attempt)
            logger.warning(
                "yfinance download attempt %d/%d failed; retrying in %.0fs",
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)

    if raw is None or raw.empty:
        message = "yfinance returned no data (check tickers / network / date range)"
        if last_error is not None:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)

    prices = _extract_close_prices(raw, tickers)
    return _validate_price_frame(
        prices,
        source=f"yfinance {','.join(tickers)}",
        max_stale_days=max_stale_days,
    )


def load_csv(
    path: str | Path,
    date_col: str | None = None,
    *,
    max_stale_days: int = DEFAULT_MAX_STALE_DAYS,
) -> pd.DataFrame:
    """Load a date-indexed price CSV.

    The file is expected to contain a date column (used as index) and one or
    more price columns. With ``date_col=None``, the first column is treated as
    the date/index. Rows are passed through :func:`_validate_price_frame`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV {path} is empty")
    if df.columns.has_duplicates:
        raise ValueError(f"CSV {path}: duplicate column names")

    if date_col is None:
        index_name = df.columns[0]
    else:
        if date_col not in df.columns:
            raise KeyError(f"date column {date_col!r} not found in {path}")
        index_name = date_col

    parsed = pd.to_datetime(df[index_name], errors="raise")
    if parsed.isna().any():
        raise ValueError(
            f"CSV {path}: date column {index_name!r} contains missing values"
        )

    df = df.copy()
    df[index_name] = parsed
    df = df.set_index(index_name).sort_index()
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("CSV has no numeric price columns")
    return _validate_price_frame(
        numeric,
        source=f"CSV {path}",
        max_stale_days=max_stale_days,
    )
