"""Time-zone-aware alignment of listings onto the US trading-day grid.

Grid day t is a US (NYSE) session. A listing's session that closes at UTC time c is assigned to the first grid day
t with G[t-1] < c <= G[t], where G are the NYSE close timestamps. Asian and European sessions dated t therefore
fall inside grid day t (known before the US open of day t+1's information set is not used), a US after-close
shock reaches Asian listings on grid day t+1, and holidays give NaN (never zero): the next session's return spans
the gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAL = {"EU": "XPAR", "US": "XNYS", "JP": "XTKS", "TW": "XTAI", "KR": "XKRX", "HK": "XHKG"}
EU_CAL = {".AS": "XAMS", ".PA": "XPAR", ".DE": "XFRA"}
# fallback close times (local clock) when a price date is not a calendar session
FALLBACK = {"XNYS": ("America/New_York", 16, 0), "XTKS": ("Asia/Tokyo", 15, 30), "XTAI": ("Asia/Taipei", 13, 30),
            "XKRX": ("Asia/Seoul", 15, 30), "XHKG": ("Asia/Hong_Kong", 16, 0), "XAMS": ("Europe/Amsterdam", 17, 30),
            "XPAR": ("Europe/Paris", 17, 30), "XFRA": ("Europe/Berlin", 17, 30)}
_cache: dict = {}


def calendar_code(ticker: str, country: str) -> str:
    for suf, code in EU_CAL.items():
        if ticker.endswith(suf):
            return code
    return CAL[country]


def _closes(code: str, start: str, end: str) -> pd.Series:
    key = (code, start, end)
    if key not in _cache:
        import exchange_calendars as xc

        cal = xc.get_calendar(code)
        lo = max(pd.Timestamp(start), cal.first_session)
        hi = min(pd.Timestamp(end), cal.last_session)
        sess = cal.sessions_in_range(lo, hi)
        closes = cal.closes.loc[sess]
        cl = pd.DatetimeIndex(closes)
        cl = cl.tz_convert("UTC").tz_localize(None) if cl.tz is not None else cl
        idx = sess.tz_localize(None) if sess.tz is not None else sess
        _cache[key] = pd.Series(cl.to_numpy(), index=idx)
    return _cache[key]


def close_utc(dates: pd.DatetimeIndex, code: str) -> pd.DatetimeIndex:
    """UTC close time (tz-naive) per session date; fallback to a nominal clock time off-calendar."""
    dates = pd.DatetimeIndex(dates).normalize()
    c = _closes(code, str(dates.min().date() - pd.Timedelta(days=7)), str(dates.max().date() + pd.Timedelta(days=7)))
    out = pd.Series(c.reindex(dates).to_numpy(), index=dates)
    miss = out.isna()
    if miss.any():
        tz, h, m = FALLBACK[code]
        loc = (dates[miss] + pd.Timedelta(hours=h, minutes=m)).tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        out[miss.to_numpy()] = loc.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(out.to_numpy())


def us_grid(start: str, end: str) -> pd.DataFrame:
    """US session dates and their UTC closes."""
    c = _closes("XNYS", str((pd.Timestamp(start) - pd.Timedelta(days=10)).date()), end)
    c = c.sort_index()
    df = pd.DataFrame({"close_utc": c.to_numpy(), "prev_close_utc": c.shift(1).to_numpy()}, index=pd.DatetimeIndex(c.index))
    return df.loc[pd.Timestamp(start):]


def to_grid_returns(close: pd.Series, code: str, grid: pd.DataFrame) -> pd.Series:
    """Simple returns of a listing on the US grid (NaN where no session of the listing closes in the grid interval)."""
    close = close.dropna()
    close = close[close > 0]
    if len(close) < 2:
        return pd.Series(np.nan, index=grid.index)
    lr = np.log(close).diff().dropna()
    c = close_utc(lr.index, code)
    G = grid["close_utc"].to_numpy("datetime64[ns]")
    pos = np.searchsorted(G, np.asarray(c, dtype="datetime64[ns]"), side="left")  # first t with G[t] >= c
    ok = pos < len(G)
    prev0 = grid["prev_close_utc"].iloc[0]
    if pd.notna(prev0):  # sessions closing before the first grid interval are dropped, not folded into day 0
        ok &= ~((pos == 0) & (np.asarray(c, dtype="datetime64[ns]") <= np.datetime64(prev0)))
    s = pd.Series(lr.to_numpy()[ok], index=pos[ok])
    agg = s.groupby(level=0).sum()
    out = pd.Series(np.nan, index=grid.index)
    out.iloc[agg.index.to_numpy()] = np.expm1(agg.to_numpy())
    return out
