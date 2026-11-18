"""Verify every seed ticker with yfinance and write data/ticker_report.csv and data/universe.csv."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config, prices  # noqa: E402

ROOT = config.ROOT
CORP = re.compile(r"\b(inc|corp|corporation|co|ltd|limited|plc|ag|se|sa|nv|n\.v|holdings?|group|company|technology|technologies|the|de|class|a|adr)\b\.?", re.I)


def tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", CORP.sub(" ", s.lower())) if len(w) > 1}


def info(ticker: str) -> dict:
    import yfinance as yf

    for k in range(3):
        try:
            i = yf.Ticker(ticker).info or {}
            if i:
                return i
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 * (k + 1))
    return {}


def main() -> None:
    cfg = config.load()
    end = pd.Timestamp(cfg["build_end"])
    seed = pd.read_csv(ROOT / "data" / "universe_seed.csv")
    rows = []
    for r in seed.itertuples():
        s = prices.fetch(r.ticker, cfg["price_start"], cfg["build_end"])
        i = info(r.ticker)
        rec = dict(ticker=r.ticker, seed_name=r.name, tier=r.tier, country=r.country, currency_seed=r.currency,
                   found=len(s) > 0, quote_type=i.get("quoteType"), exchange=i.get("exchange"),
                   currency=i.get("currency"), long_name=i.get("longName") or i.get("shortName"),
                   market_cap=i.get("marketCap"),
                   first_date=s.index.min().date().isoformat() if len(s) else None,
                   last_date=s.index.max().date().isoformat() if len(s) else None, n_obs=len(s))
        nm_ok = bool(tokens(r.name) & tokens(rec["long_name"] or ""))
        rec["name_match"] = nm_ok
        if not len(s):
            status, reason = "not_found", "no price history returned"
        elif rec["quote_type"] not in (None, "EQUITY"):
            status, reason = "not_found", f"quoteType {rec['quote_type']}"
        elif (end - s.index.max()).days > 10:
            status, reason = "stale", f"last bar {rec['last_date']}"
        elif (s.index.max() - s.index.min()).days < 3 * 365:
            status, reason = "short_history", f"first bar {rec['first_date']}"
        else:
            status, reason = "ok", ""
        if status in ("ok", "short_history") and not nm_ok:
            reason = (reason + "; " if reason else "") + "name mismatch, review"
        rec.update(status=status, reason=reason)
        rows.append(rec)
        print(r.ticker, status, rec["long_name"], rec["currency"], flush=True)
    rep = pd.DataFrame(rows)
    rep.to_csv(ROOT / "data" / "ticker_report.csv", index=False)
    print(rep.status.value_counts().to_string())


if __name__ == "__main__":
    main()
