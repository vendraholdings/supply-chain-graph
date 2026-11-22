"""Build the static site into site/dist: vendored scripts only, data embedded, no external requests."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config, edges as edges_mod, momentum as mo  # noqa: E402

ROOT = config.ROOT
DIST = ROOT / "site" / "dist"
RES = ROOT / "results"
BG, PANEL, INK, LINK = "#0b0f17", "#121826", "#e5e7eb", "#4cc9f0"


def turbo_hex(n: int) -> list[str]:
    cm = matplotlib.colormaps["turbo"]
    return [matplotlib.colors.to_hex(cm(0.1 + 0.85 * i / (n - 1))) for i in range(n)]  # drop the near-black end for contrast on the dark page


def page(data: dict) -> str:
    tpl = (ROOT / "site" / "template.html").read_text()
    return tpl.replace("%%DATA%%", json.dumps(data, separators=(",", ":"), default=float))


def main() -> None:
    import plotly

    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "vendor").mkdir(parents=True)
    for f in (ROOT / "site" / "vendor").iterdir():
        shutil.copy(f, DIST / "vendor" / f.name)
    shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", DIST / "vendor" / "plotly.min.js")
    g = json.loads((ROOT / "data" / "derived" / "graph.json").read_text())
    st = pd.read_csv(RES / "spillover_static.csv", index_col=0)
    names = [c for c in st.columns if c != "from_others"]
    ro = pd.read_parquet(RES / "spillover_rolling.parquet")
    ll = pd.read_csv(RES / "leadlag_tiers.csv")
    tiers = sorted(set(ll.leader))
    lagmats = {}
    for lag, d in ll.groupby("lag"):
        M = [[None] * len(tiers) for _ in tiers]
        for r in d.itertuples():
            M[tiers.index(r.leader)][tiers.index(r.follower)] = round(r.corr, 3)
        lagmats[int(lag)] = M
    summ = json.loads((RES / "summary.json").read_text())
    sps = json.loads((RES / "spillover_summary.json").read_text())
    evs = json.loads((RES / "event_summary.json").read_text())
    uni = pd.read_csv(ROOT / "data" / "universe.csv")
    edges = edges_mod.load().drop_duplicates(["supplier", "customer"])
    tier_ids = [t["id"] for t in g["tiers"]]
    ctys = sorted(uni["country"].unique())
    cm = [[int(((uni.tier == t) & (uni.country == c)).sum()) for c in ctys] for t in tier_ids]
    cty = dict(zip(uni.ticker, uni.country))
    pc = (edges.supplier.map(cty) + " to " + edges.customer.map(cty)).value_counts().sort_values()
    cent = pd.read_csv(ROOT / "data" / "derived" / "centrality.csv").merge(
        pd.DataFrame(g["nodes"]).rename(columns={"ticker": "ticker"})[["ticker", "tier", "mcap_usd"]], on="ticker")
    cent = cent[(cent.in_degree + cent.out_degree) > 0].copy()
    cent["mcap_rank"] = cent.mcap_usd.rank(ascending=False)
    cent["pr_rank"] = cent.pagerank.rank(ascending=False)
    flow = pd.read_csv(ROOT / "data" / "derived" / "tier_flow.csv", index_col=0)
    caar = pd.read_csv(RES / "event_caar.csv")
    caar_d = {}
    for (rel, tr), d in caar.groupby(["relation", "tier"]):
        if rel != "event_node":
            caar_d.setdefault(rel, {})[tr] = dict(day=d.day.tolist(), caar=(100 * d.caar).round(3).tolist(), n=int(d.n.iloc[0]))
    draws = pd.read_csv(RES / "event_placebo_draws.csv")
    plc = {rel: dict(draws=(100 * d.caar_end).round(3).tolist(), observed=round(100 * evs[rel]["caar_end"], 3), n=evs[rel]["n"],
                     rejection=evs[rel]["placebo"]["rejection_rate_5pct"]) for rel, d in draws.groupby("relation")}
    ret = pd.read_parquet(RES / "momentum_primary.parquet")
    trials = pd.read_parquet(RES / "momentum_trials.parquet")
    full = trials.loc[:, trials.notna().mean() >= 0.9].dropna()
    f5 = dict(dates=[str(d) for d in ret.index], gross=(1 + ret["gross_return"]).cumprod().round(4).tolist(), net=(1 + ret["net_return"]).cumprod().round(4).tolist())
    f6 = dict(draws=pd.read_parquet(RES / "placebo_rewiring.parquet")["sharpe"].round(3).tolist(), observed=round(summ["placebo"]["real_sharpe"], 3),
              trials=sorted(round(mo.sharpe(full[c].to_numpy()), 3) for c in full.columns),
              sr0=round(summ["overfitting"]["sr0_per_period"] * mo.PPY ** 0.5, 3), pbo=round(summ["overfitting"]["cscv"]["pbo"], 2),
              dsr=round(summ["overfitting"]["dsr"], 2), p=round(summ["placebo"]["p_ge_real"], 2))
    sc = RES / "shock_case.json"
    ns = RES / "node_stats.csv"
    nstats = pd.read_csv(ns).set_index("ticker").where(lambda d: d.notna(), None).to_dict("index") if ns.exists() else {}
    data = dict(graph=g, nstats=nstats, f5=f5, f6=f6, shock=json.loads(sc.read_text()) if sc.exists() else None, turbo=[[i / 10, matplotlib.colors.to_hex(matplotlib.colormaps["turbo"](i / 10))] for i in range(11)],
                f2=dict(tiers=tier_ids, countries=ctys, matrix=cm, pair_labels=pc.index.tolist(), pair_counts=pc.tolist()),
                f3=dict(rows=cent[["ticker", "pagerank", "tier", "mcap_rank", "pr_rank"]].round(5).to_dict("records")),
                f4=dict(names=list(flow.columns), z=[[None if v == 0 else round(v, 2) for v in r] for r in flow.to_numpy(float)]),
                f7=caar_d, f8=plc, tier_colors=turbo_hex(len(g["tiers"])),
                spill=dict(names=names, theta=st.loc[names, names].round(2).values.tolist(),
                           net=st.loc["net", names].round(2).tolist(), total=sps["static"]["total_index"]),
                rolling=dict(dates=[d.strftime("%Y-%m-%d") for d in ro.index], total=ro["total"].round(2).tolist(),
                             net={c[4:]: ro[c].round(2).tolist() for c in ro.columns if c.startswith("net_")}),
                leadlag=dict(tiers=tiers, lags=sorted(lagmats), mats=lagmats),
                facts=dict(n_nodes=summ["graph"]["nodes"], n_connected=summ["graph"]["connected_nodes"], n_edges=summ["graph"]["edges"],
                           n_edge_rows=summ["n_edge_rows"], build_end=summ["build_end"], spill_total=sps["static"]["total_index"],
                           spill_weekly=sps["weekly"]["total_index"], n_shocks=evs["n_shocks"],
                           max_suppliers=summ["information_set"]["max_suppliers"], sr_net=round(summ["momentum_primary"]["cost_x1"]["sharpe"], 2),
                           pbo=round(summ["overfitting"]["cscv"]["pbo"], 2)),
                colors=dict(bg=BG, panel=PANEL, ink=INK, link=LINK))
    (DIST / "index.html").write_text(page(data))
    print("site built:", DIST / "index.html", round((DIST / "index.html").stat().st_size / 1e3), "kB")


if __name__ == "__main__":
    main()
