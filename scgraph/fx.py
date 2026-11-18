"""Daily FX to USD and conversion of local grid returns into USD returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import prices

# yfinance symbol, and whether the quote is USD per unit (True) or units per USD (False)
PAIRS = {"EUR": ("EURUSD=X", True), "JPY": ("JPY=X", False), "KRW": ("KRW=X", False),
         "TWD": ("TWD=X", False), "HKD": ("HKD=X", False)}


def usd_per_unit(currency: str, start: str, end: str) -> pd.Series:
    if currency == "USD":
        return pd.Series(dtype=float)
    sym, direct = PAIRS[currency]
    s = prices.fetch(sym, start, end)
    return s if direct else 1.0 / s


def fx_grid_returns(currency: str, grid: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Change in USD value of one unit of `currency` per US grid day (FX dated by calendar day, compounded over gaps)."""
    if currency == "USD":
        return pd.Series(0.0, index=grid.index)
    r = np.log(usd_per_unit(currency, start, end).dropna()).diff().dropna()
    # assign each calendar date to the first grid day on or after it
    pos = np.searchsorted(grid.index.to_numpy("datetime64[ns]"), r.index.to_numpy("datetime64[ns]"), side="left")
    ok = pos < len(grid)
    agg = pd.Series(r.to_numpy()[ok], index=pos[ok]).groupby(level=0).sum()
    out = pd.Series(0.0, index=grid.index)
    out.iloc[agg.index.to_numpy()] = agg.to_numpy()
    return np.expm1(out)


def to_usd(local_ret: pd.Series, fx_ret: pd.Series) -> pd.Series:
    """USD return = (1 + local)(1 + fx change) - 1; NaN local stays NaN (no zero fill)."""
    return (1.0 + local_ret) * (1.0 + fx_ret) - 1.0
