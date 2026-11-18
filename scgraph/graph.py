"""Directed supplier -> customer graph: construction, centrality, tier flow matrix and the graph.json export."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from . import edges as edges_mod

TIERS = {"T01": "Wafers", "T02": "Industrial gases", "T03": "Photoresists and chemicals", "T04": "CMP, materials, masks",
         "T05": "Equipment", "T06": "EDA and IP", "T07": "Foundry", "T08": "Memory", "T09": "OSAT and substrates",
         "T10": "Chip designers and IDMs", "T11": "Hyperscalers, OEMs, servers"}


def latest_pairs(edges: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """One row per (supplier, customer): the latest filing that is usable at `as_of` (all filings if None)."""
    e = edges if as_of is None else edges_mod.edges_as_of(edges, as_of)
    if e.empty:
        return e
    e = e.sort_values("filing_date")
    return e.groupby(["supplier", "customer"], as_index=False).tail(1).reset_index(drop=True)


def tier_median_weight(edges: pd.DataFrame, universe: pd.DataFrame) -> dict[str, float]:
    t = universe.set_index("ticker")["tier"]
    d = edges.assign(tier=edges["supplier"].map(t)).dropna(subset=["pct_revenue"])
    return d.groupby("tier")["pct_revenue"].median().to_dict()


def build(edges: pd.DataFrame, universe: pd.DataFrame, as_of=None, weights: str = "disclosed") -> nx.DiGraph:
    G = nx.DiGraph()
    for r in universe.itertuples():
        G.add_node(r.ticker, name=r.name, tier=r.tier, country=r.country)
    pairs = latest_pairs(edges, as_of)
    med = tier_median_weight(edges, universe)
    tier = universe.set_index("ticker")["tier"].to_dict()
    for r in pairs.itertuples():
        if r.supplier not in G or r.customer not in G:
            continue
        if weights == "unit":
            w = 1.0
        else:
            w = r.pct_revenue if pd.notna(r.pct_revenue) else med.get(tier[r.supplier], 1.0)
        G.add_edge(r.supplier, r.customer, weight=float(w), disclosed=bool(pd.notna(r.pct_revenue)))
    return G


def centrality(G: nx.DiGraph) -> pd.DataFrame:
    n = list(G.nodes)
    in_s = dict(G.in_degree(weight="weight"))
    out_s = dict(G.out_degree(weight="weight"))
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {k: 1 / len(n) for k in n}
    bt = nx.betweenness_centrality(G) if G.number_of_edges() else {k: 0.0 for k in n}
    return pd.DataFrame({"ticker": n, "in_strength": [in_s[k] for k in n], "out_strength": [out_s[k] for k in n],
                         "pagerank": [pr[k] for k in n], "betweenness": [bt[k] for k in n],
                         "in_degree": [G.in_degree(k) for k in n], "out_degree": [G.out_degree(k) for k in n]})


def rank_corr_with_mcap(cent: pd.DataFrame, mcap: pd.Series, col: str = "pagerank") -> dict:
    from scipy.stats import spearmanr

    d = cent.set_index("ticker")[col].to_frame().join(mcap.rename("mcap")).dropna()
    if len(d) < 5:
        return {"rho": None, "n": int(len(d))}
    rho, p = spearmanr(d[col], d["mcap"])
    return {"rho": float(rho), "p": float(p), "n": int(len(d))}


def tier_flow(G: nx.DiGraph) -> pd.DataFrame:
    tiers = sorted(TIERS)
    M = pd.DataFrame(0.0, index=tiers, columns=tiers)
    for s, c, d in G.edges(data=True):
        M.loc[G.nodes[s]["tier"], G.nodes[c]["tier"]] += d["weight"]
    return M


def market_caps_usd(report: pd.DataFrame, cfg: dict) -> pd.Series:
    """Market capitalisation reported by the vendor at the verification date, converted at the latest FX rate."""
    from . import fx

    out = {}
    last = {}
    for r in report.itertuples():
        if pd.isna(r.market_cap):
            continue
        cur = r.currency if isinstance(r.currency, str) else r.currency_seed
        if cur not in last:
            u = fx.usd_per_unit(cur, cfg["price_start"], cfg["build_end"]) if cur != "USD" else None
            last[cur] = 1.0 if u is None else float(u.dropna().iloc[-1])
        out[r.ticker] = float(r.market_cap) * last[cur]
    return pd.Series(out, name="mcap_usd")


def export_json(G: nx.DiGraph, edges: pd.DataFrame, mcap: pd.Series, built_on: str, path: Path) -> dict:
    src = edges.sort_values("filing_date")
    nodes = [dict(ticker=t, name=d["name"], tier=d["tier"], country=d["country"],
                  mcap_usd=None if t not in mcap.index or pd.isna(mcap[t]) else round(float(mcap[t]), -6))
             for t, d in G.nodes(data=True)]
    out_edges = []
    for s, c, d in G.edges(data=True):
        rows = src[(src.supplier == s) & (src.customer == c)]
        out_edges.append(dict(supplier=s, customer=c, weight=round(d["weight"], 2), disclosed=d["disclosed"],
                              valid_from=rows["available_date"].min(), valid_to=rows["valid_to"].max(),
                              sources=[dict(url=r.source_url, type=r.source_type, filing_date=r.filing_date,
                                            fiscal_year=int(r.fiscal_year)) for r in rows.itertuples()]))
    doc = dict(version=1, built_on=built_on, nodes=nodes, edges=sorted(out_edges, key=lambda e: (e["supplier"], e["customer"])),
               tiers=[dict(id=k, name=v) for k, v in TIERS.items()])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=False))
    return doc


def tier_index(R: pd.DataFrame, universe: pd.DataFrame, mcap: pd.Series) -> pd.DataFrame:
    """Cap-weighted USD tier index returns (static current weights; missing returns drop out of the day's average)."""
    out = {}
    for tier, g in universe.groupby("tier"):
        t = [x for x in g["ticker"] if x in R.columns]
        w = mcap.reindex(t).fillna(mcap.reindex(t).median() if mcap.reindex(t).notna().any() else 1.0).to_numpy(float)
        X = R[t].to_numpy(float)
        m = ~np.isnan(X)
        num = np.nansum(X * w, axis=1)
        den = (m * w).sum(axis=1)
        out[tier] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return pd.DataFrame(out, index=R.index)
