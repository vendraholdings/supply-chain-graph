"""Daily adjusted closes from yfinance with an on-disk cache (git-ignored; vendor prices are never committed)."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .config import ROOT

CACHE = ROOT / "cache" / "prices"


def _path(ticker: str) -> Path:
    return CACHE / f"{ticker.replace('/', '_')}.parquet"


def fetch(ticker: str, start: str, end: str, force: bool = False, retries: int = 3) -> pd.Series:
    """Adjusted close in local currency indexed by session date (tz-naive). Empty Series if not found."""
    p = _path(ticker)
    if p.exists() and not force:
        return pd.read_parquet(p)["close"]
    import yfinance as yf

    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    s = pd.Series(dtype=float, name="close")
    for k in range(retries):
        try:
            df = yf.Ticker(ticker).history(start=start, end=end_excl, auto_adjust=True, actions=False)
            if len(df):
                idx = pd.DatetimeIndex(df.index)
                idx = idx.tz_localize(None) if idx.tz is not None else idx
                s = pd.Series(df["Close"].to_numpy(float), index=idx.normalize(), name="close")
                s = s[~s.index.duplicated(keep="last")].dropna()
                break
        except Exception:  # noqa: BLE001 - network errors are retried
            pass
        time.sleep(2 * (k + 1))
    CACHE.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(p)
    return s


def load_all(tickers, start: str, end: str) -> pd.DataFrame:
    """Wide frame of local-currency closes, one column per ticker."""
    cols = {t: fetch(t, start, end) for t in tickers}
    return pd.DataFrame(cols).sort_index()
