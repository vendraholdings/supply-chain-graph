"""Compute every result of the first stage: graph, centrality, momentum (primary, costs, grid, placebo) and diagnostics."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overfit import cscv as cs, sharpe as sh  # noqa: E402

from scgraph import config, edges as edges_mod, graph as gr, momentum as mo, returns  # noqa: E402

D = config.ROOT / "data"
OUT = D / "derived"
RES = config.ROOT / "results"


def monthly(R: pd.DataFrame, col: str) -> pd.Series:
    s = R[col].dropna()
    m = (1 + s).groupby([s.index.year, s.index.month]).prod() - 1
    m.index = pd.PeriodIndex([f"{y}-{mo_:02d}" for y, mo_ in m.index], freq="M")
    return m


def main() -> None:
    cfg = config.load()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(exist_ok=True)
    R = returns.build(cfg)
    uni = pd.read_csv(D / "universe.csv")
    rep = pd.read_csv(D / "ticker_report.csv")
    E = edges_mod.load()
    summary: dict = dict(build_end=cfg["build_end"], n_companies=int(len(uni)), n_edge_rows=int(len(E)),
                         methods=E["method"].value_counts().to_dict())
    # ---- graph and centrality ------------------------------------------------------------------------------
    mcap = gr.market_caps_usd(rep, cfg)
    G = gr.build(E, uni)
    cent = gr.centrality(G)
    cent.to_csv(OUT / "centrality.csv", index=False)
    gr.tier_flow(G).to_csv(OUT / "tier_flow.csv")
    gr.export_json(G, E, mcap, cfg["build_end"], OUT / "graph.json")
    connected = [n for n in G if G.degree(n) > 0]
    summary["graph"] = dict(nodes=G.number_of_nodes(), connected_nodes=len(connected), edges=G.number_of_edges(),
                            disclosed_weight_share=float(np.mean([d["disclosed"] for *_, d in G.edges(data=True)])) if G.number_of_edges() else None,
                            pagerank_vs_mcap=gr.rank_corr_with_mcap(cent[cent.ticker.isin(connected)], mcap),
                            strength_vs_mcap=gr.rank_corr_with_mcap(cent[cent.ticker.isin(connected)], mcap, "out_strength"))
    tier = uni.set_index("ticker")["tier"].to_dict()
    country = uni.set_index("ticker")["country"].to_dict()
    # ---- momentum --------------------------------------------------------------------------------------------
    Rs = R[[c for c in R.columns if not c.startswith("BM_")]]
    eng = mo.Engine(Rs, country, cfg)
    EA = mo.edge_arrays(E, eng.tickers, tier)
    prim = mo.Spec()
    info = eng.info_set(EA, prim)
    sup_n = np.array([len(eng.signals(EA, k, prim)) for k in range(len(eng.me))])
    summary["information_set"] = dict(first_month_with_edges=str(eng.dates[eng.me[np.flatnonzero(info)[0]]].date()) if info.any() else None,
                                      months_with_at_least_6_suppliers=int((sup_n >= 6).sum()), max_suppliers=int(sup_n.max()))
    fac = pd.concat([monthly(R, "BM_SPY").rename("mkt"), monthly(R, "BM_SOXX").rename("semi")], axis=1)
    res: dict = {}
    for mult in cfg["costs_bps"]["multipliers"]:
        m = eng.run(EA, prim, cost_mult=float(mult))
        res[f"cost_x{mult}"] = dict(n_months=int(len(m)), sharpe=mo.sharpe(m) if len(m) > 2 else None,
                                    mean_monthly=float(m.mean()) if len(m) else None)
        if mult == 1:
            primary = m
    summary["momentum_primary"] = res
    if len(primary) >= 8:
        summary["momentum_primary"]["alpha"] = mo.newey_west_alpha(primary, fac)
    summary["fama_macbeth"] = mo.fama_macbeth(eng, EA, prim)
    primary.rename("net_return").to_frame().to_parquet(RES / "momentum_primary.parquet")
    # ---- trials matrix ------------------------------------------------------------------------------------------
    specs = mo.grid_specs()
    cols = {s.label(): eng.run(EA, s) for s in specs}
    M = pd.concat(cols, axis=1).sort_index() if any(len(v) for v in cols.values()) else pd.DataFrame(index=pd.PeriodIndex([], freq="M"))
    M.astype("float32").to_parquet(RES / "momentum_trials.parquet")
    active = M[prim.label()].dropna().index
    summary["trials"] = dict(n_trials=len(specs), months_total=int(len(M)), months_primary=int(len(active)))
    # ---- placebo: degree-preserving rewiring within tiers -----------------------------------------------------
    tier_ix = {t: i for i, t in enumerate(sorted(set(tier.values())))}
    grp = np.array([tier_ix[tier[eng.tickers[s]]] * 20 + tier_ix[tier[eng.tickers[c]]] for s, c in zip(EA.sup, EA.cus)])
    if len(primary) >= 8:
        rng = np.random.default_rng(cfg["seed"])
        real = mo.sharpe(primary)
        draws = []
        for _ in range(cfg["rewiring_draws"]):
            m = eng.run(mo.rewire(EA, grp, rng), prim)
            draws.append(mo.sharpe(m) if len(m) > 2 else np.nan)
        d = np.array(draws)
        d = d[~np.isnan(d)]
        pd.Series(d, name="sharpe").to_frame().to_parquet(RES / "placebo_rewiring.parquet")
        summary["placebo"] = dict(n=int(len(d)), real_sharpe=real, mean=float(d.mean()), sd=float(d.std(ddof=1)),
                                  p_ge_real=float(np.mean(d >= real)))
    else:
        summary["placebo"] = dict(note="skipped: primary strategy has fewer than 8 active months")
    # ---- overfitting diagnostics ------------------------------------------------------------------------------
    X = M.loc[active].fillna(0.0)
    T = len(X)
    s = min(16, (T // 12) // 2 * 2)
    if s >= 4:
        r = cs.cscv(X.to_numpy(float), s=s)
        srs = X.to_numpy(float).mean(0) / np.where(X.to_numpy(float).std(0, ddof=1) > 0, X.to_numpy(float).std(0, ddof=1), np.nan)
        srs = np.nan_to_num(srs)
        best = int(np.argmax(srs))
        dsr, sr0 = sh.dsr(X.iloc[:, best].to_numpy(float), srs)
        summary["overfitting"] = dict(cscv=r.summary(), best_trial=X.columns[best], dsr=float(dsr), sr0_per_period=float(sr0), T=int(T), n_trials=int(X.shape[1]))
    else:
        summary["overfitting"] = dict(note=f"skipped: {T} months give fewer than 4 CSCV blocks of 12 months", T=int(T))
    summary["runtime_s"] = round(time.time() - t0, 1)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float)[:3500])


if __name__ == "__main__":
    main()
