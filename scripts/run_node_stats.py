"""Derived per-company statistics for the site's node panel (no price series): beta and correlation to the global semiconductor
index, annualised volatility, and the mean correlation with sourced neighbours. Daily US-dollar returns on the US trading-day grid."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config, edges as edges_mod, returns  # noqa: E402


def main() -> None:
    R = returns.build(config.load())
    uni = pd.read_csv(config.ROOT / "data" / "universe.csv")
    E = edges_mod.load().drop_duplicates(["supplier", "customer"])
    C = R[list(uni.ticker)].corr(min_periods=250)
    rows = []
    for t in uni.ticker:
        r = R[t].dropna()
        m = R["BM_SOXX"].reindex(r.index)
        ok = m.notna()
        nb = list(E[E.supplier == t].customer) + list(E[E.customer == t].supplier)
        cn = [C.loc[t, x] for x in nb if x in C.columns and not np.isnan(C.loc[t, x])]
        rows.append(dict(ticker=t, n_days=int(len(r)), beta_soxx=round(float(np.cov(r[ok], m[ok])[0, 1] / m[ok].var()), 2) if ok.sum() > 250 else None,
                         corr_soxx=round(float(r[ok].corr(m[ok])), 2) if ok.sum() > 250 else None, ann_vol_pct=round(float(100 * r.std() * np.sqrt(252)), 1),
                         mean_corr_neighbours=round(float(np.mean(cn)), 2) if cn else None, n_neighbours=len(cn)))
    pd.DataFrame(rows).to_csv(config.ROOT / "results" / "node_stats.csv", index=False)
    print(len(rows), "companies")


if __name__ == "__main__":
    main()
