"""Spillover (Diebold-Yilmaz) and lead-lag results on cap-weighted USD tier indices."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config, graph as gr, leadlag as ll, returns, spillover as sp  # noqa: E402

D, RES = config.ROOT / "data", config.ROOT / "results"


def main() -> None:
    cfg = config.load()
    sc = cfg["spillover"]
    R = returns.build(cfg)
    uni = pd.read_csv(D / "universe.csv")
    mcap = gr.market_caps_usd(pd.read_csv(D / "ticker_report.csv"), cfg)
    Rs = R[[c for c in R.columns if not c.startswith("BM_")]]
    ti = gr.tier_index(Rs, uni, mcap)
    ti = ti.dropna(how="all")
    X = sp.residualise(ti, R["BM_SPY"]).dropna()
    out: dict = dict(n_days=int(len(X)), tiers=list(X.columns))
    r = sp.spillover(X.to_numpy(), sc["horizon"], sc["max_lag"])
    stable_note = None
    if not r["stable"]:
        stable_note = "11-variable VAR unstable: reported on 6 merged tiers"
        Xm = sp.merge_tiers(ti)
        Xm = sp.residualise(Xm, R["BM_SPY"]).dropna()
        r = sp.spillover(Xm.to_numpy(), sc["horizon"], sc["max_lag"])
        X = Xm
    names = list(X.columns)
    th = pd.DataFrame(r["theta"] * 100, index=names, columns=names)
    th["from_others"] = r["from_others"]
    th.loc["to_others"] = list(r["to_others"]) + [r["total"]]
    th.loc["net"] = list(r["net"]) + [np.nan]
    th.round(2).to_csv(RES / "spillover_static.csv")
    out["static"] = dict(total_index=r["total"], var_lag=r["lag"], stable=r["stable"], note=stable_note, tiers_used=names)
    rolled = sp.rolling(X, sc["window"], 5, sc["horizon"], sc["max_lag"])
    rolled.to_parquet(RES / "spillover_rolling.parquet")
    out["rolling"] = dict(n_windows=int(len(rolled)), total_min=float(rolled.total.min()), total_max=float(rolled.total.max()),
                          share_stable=float(rolled.stable.mean()))
    # weekly robustness: compound five consecutive grid days
    wk = (1 + X).groupby(np.arange(len(X)) // 5).prod() - 1
    rw = sp.spillover(wk.to_numpy(), 4, sc["max_lag"])
    out["weekly"] = dict(total_index=rw["total"], n_weeks=int(len(wk)), var_lag=rw["lag"], stable=rw["stable"])
    # lead-lag across tiers (raw tier index returns, not residualised)
    Y = ti[[c for c in ti.columns if ti[c].notna().mean() > 0.9]].dropna()
    lo, hi = ll.bootstrap_ci(Y.to_numpy(), reps=300, mean_block=10, seed=cfg["seed"])
    t = ll.table(Y, lo, hi)
    t.to_csv(RES / "leadlag_tiers.csv", index=False)
    nz = t[t.lag != 0]
    out["leadlag"] = dict(n_pairs=int(len(Y.columns) * (len(Y.columns) - 1)), n_days=int(len(Y)),
                          significant_nonzero_lag_share=float(nz.significant.mean()),
                          expected_share_if_null=0.05, strongest=nz.reindex(nz["corr"].abs().sort_values(ascending=False).index).head(5)[
                              ["lag", "leader", "follower", "corr"]].round(3).to_dict("records"))
    # Asia-before-US: same-grid-day correlation between the Asian and US halves of each tier (mechanical by construction)
    asia = uni[uni.country.isin(["JP", "TW", "KR", "HK"])]
    us = uni[uni.country == "US"]
    ia, iu = gr.tier_index(Rs, asia, mcap), gr.tier_index(Rs, us, mcap)
    rows = []
    for a in ia.columns:
        for u in iu.columns:
            d = pd.concat([ia[a], iu[u]], axis=1).dropna()
            if len(d) > 500:
                rows.append(dict(asia_tier=a, us_tier=u, corr_lag0=float(d.corr().iloc[0, 1]), n=len(d)))
    pd.DataFrame(rows).to_csv(RES / "asia_us_lag0.csv", index=False)
    out["asia_us_lag0_median_corr"] = float(np.median([x["corr_lag0"] for x in rows]))
    (RES / "spillover_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
