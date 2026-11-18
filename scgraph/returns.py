"""USD return matrix on the US grid for the whole universe, plus market and sector benchmarks."""
from __future__ import annotations

import pandas as pd

from . import calendars, config, fx, prices

BENCH = {"SPY": ("SPY", "US"), "SOXX": ("SOXX", "US"), "N225": ("^N225", "JP"), "TWII": ("^TWII", "TW"),
         "KS11": ("^KS11", "KR"), "HSI": ("^HSI", "HK"), "SX5E": ("^STOXX50E", "EU")}
BENCH_CCY = {"US": "USD", "JP": "JPY", "TW": "TWD", "KR": "KRW", "HK": "HKD", "EU": "EUR"}
CACHE = config.ROOT / "cache" / "returns_usd.parquet"


def build(cfg: dict | None = None, force: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    cfg = cfg or config.load()
    uni = pd.read_csv(config.ROOT / "data" / "universe.csv")
    grid = calendars.us_grid(cfg["price_start"], cfg["build_end"])
    fxr = {c: fx.fx_grid_returns(c, grid, cfg["price_start"], cfg["build_end"]) for c in ["USD", "EUR", "JPY", "KRW", "TWD", "HKD"]}
    out = {}
    for r in uni.itertuples():
        px = prices.fetch(r.ticker, cfg["price_start"], cfg["build_end"])
        loc = calendars.to_grid_returns(px, calendars.calendar_code(r.ticker, r.country), grid)
        out[r.ticker] = fx.to_usd(loc, fxr[r.currency])
    for name, (sym, ctry) in BENCH.items():
        px = prices.fetch(sym, cfg["price_start"], cfg["build_end"])
        loc = calendars.to_grid_returns(px, calendars.calendar_code(sym, ctry), grid)
        out["BM_" + name] = fx.to_usd(loc, fxr[BENCH_CCY[ctry]])
    df = pd.DataFrame(out)
    df = df.loc[df.index <= pd.Timestamp(cfg["build_end"])]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df
