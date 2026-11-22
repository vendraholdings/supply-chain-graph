"""Numbers quoted in the README and the paper, read from committed results and formatted once.

Both documents are rendered from templates with ``<<key>>`` placeholders, so a number cannot differ between them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TIER_NAMES = {"T01": "wafers", "T02": "industrial gases", "T03": "photoresists and chemicals", "T04": "CMP, materials and masks",
              "T05": "equipment", "T06": "EDA and IP", "T07": "foundry", "T08": "memory", "T09": "packaging and substrates",
              "T10": "chip designers and IDMs", "T11": "hyperscalers, OEMs and servers"}


def _j(name: str) -> dict:
    return json.loads((RES / name).read_text())


def pct(x: float, d: int = 1, sign: bool = False) -> str:
    return f"{100 * x:+.{d}f}%" if sign else f"{100 * x:.{d}f}%"


def facts() -> dict[str, str]:
    s, sp, ev = _j("summary.json"), _j("spillover_summary.json"), _j("event_summary.json")
    rep = pd.read_csv(ROOT / "data" / "ticker_report.csv")
    uni = pd.read_csv(ROOT / "data" / "universe.csv")
    st = pd.read_csv(RES / "spillover_static.csv", index_col=0)
    net = st.loc["net"].drop(labels=["from_others"], errors="ignore").dropna().astype(float).sort_values()
    f: dict[str, str] = {}
    f["n_seed"] = str(len(rep))
    f["n_companies"] = str(s["n_companies"])
    f["n_dropped"] = str(len(rep) - len(uni))
    f["n_pairs"] = str(s["graph"]["edges"])
    f["n_edge_rows"] = str(s["n_edge_rows"])
    f["n_connected"] = str(s["graph"]["connected_nodes"])
    f["n_disclosed"] = str(round(s["graph"]["disclosed_weight_share"] * s["graph"]["edges"]))
    f["build_end"] = s["build_end"]
    f["first_edge_month"] = s["information_set"]["first_month_with_edges"][:7]
    f["max_suppliers"] = str(s["information_set"]["max_suppliers"])
    f["n_trials"] = str(s["trials"]["n_trials"])
    f["months_total"] = str(s["trials"]["months_total"])
    f["months_primary"] = str(s["trials"]["months_primary"])
    f["pr_rho"] = f"{s['graph']['pagerank_vs_mcap']['rho']:.2f}"
    f["pr_n"] = str(s["graph"]["pagerank_vs_mcap"]["n"])
    f["str_rho"] = f"{s['graph']['strength_vs_mcap']['rho']:.2f}"
    f["n_days"] = f"{sp['n_days']:,}"
    f["spill_total"] = f"{sp['static']['total_index']:.1f}%"
    f["spill_weekly"] = f"{sp['weekly']['total_index']:.1f}%"
    f["spill_min"] = f"{sp['rolling']['total_min']:.1f}%"
    f["spill_max"] = f"{sp['rolling']['total_max']:.1f}%"
    f["n_windows"] = str(sp["rolling"]["n_windows"])
    top = net.index[-1], net.index[-2]
    bot = net.index[0], net.index[1]
    f["tx1"], f["tx2"] = top
    f["rx1"], f["rx2"] = bot
    f["tx1_name"], f["tx2_name"] = TIER_NAMES[top[0]], TIER_NAMES[top[1]]
    f["rx1_name"], f["rx2_name"] = TIER_NAMES[bot[0]], TIER_NAMES[bot[1]]
    f["tx1_v"], f["tx2_v"] = f"{net[top[0]]:+.1f}", f"{net[top[1]]:+.1f}"
    f["rx1_v"], f["rx2_v"] = f"{net[bot[0]]:+.1f}", f"{net[bot[1]]:+.1f}"
    ll = sp["leadlag"]
    f["ll_pairs"] = str(ll["n_pairs"])
    f["ll_share"] = pct(ll["significant_nonzero_lag_share"])
    f["ll_null"] = pct(ll["expected_share_if_null"], 0)
    f["ll_top_leader"], f["ll_top_follower"] = ll["strongest"][1]["leader"], ll["strongest"][1]["follower"]
    f["ll_top_corr"] = f"{ll['strongest'][1]['corr']:.2f}"
    f["asia_us_corr"] = f"{sp['asia_us_lag0_median_corr']:.2f}"
    f["n_shocks"] = f"{ev['n_shocks']:,}"
    f["node_caar"] = pct(ev["event_node"]["caar_end"], 1, True)
    for k in ("direct_supplier", "direct_customer", "two_hop"):
        e = ev[k]
        f[f"{k}_n"] = str(e["n"])
        f[f"{k}_caar"] = pct(e["caar_end"], 1, True)
        f[f"{k}_t"] = f"{e['t_kp']:.1f}"
        f[f"{k}_rej"] = pct(e["placebo"]["rejection_rate_5pct"])
    ea = pd.read_csv(ROOT / "data" / "edges_auto.csv")
    ec = pd.read_csv(ROOT / "data" / "edges_curated.csv")
    xr = pd.read_csv(ROOT / "data" / "extracts" / "report.csv")
    f["n_auto_rows"] = str(len(ea))
    f["n_auto_pairs"] = str(ea.groupby(["supplier", "customer"]).ngroups)
    f["n_overlap"] = str(len(ea) + len(ec) - s["n_edge_rows"])
    f["n_manual"] = str(int(ea["source_note"].str.contains("manually reviewed").sum()))
    dec = json.loads((ROOT / "data" / "review_decisions.json").read_text())
    f["n_queue_accepted"] = str(sum(v == "accept" for v in dec.values()))
    f["n_queue_rejected"] = str(sum(v == "reject" for v in dec.values()))
    f["n_queue"] = str(len(dec))
    f["n_curated_rows"] = str(len(ec))
    f["n_curated_pairs"] = str(ec.groupby(["supplier", "customer"]).ngroups)
    f["n_filers"] = str(int((xr["n_docs"] > 0).sum()))
    f["n_filings"] = f"{int(xr['n_docs'].sum()):,}"
    f["n_extracts"] = f"{int(xr['n_extracts'].sum()):,}"
    mp, pl, ov = s["momentum_primary"], s["placebo"], s["overfitting"]
    f["sr_gross"] = f"{mp['cost_x0']['sharpe']:.2f}"
    f["sr_net"] = f"{mp['cost_x1']['sharpe']:.2f}"
    f["sr_net2"] = f"{mp['cost_x2']['sharpe']:.2f}"
    f["alpha_ann"] = pct(mp["alpha"]["alpha_annual"], 1, True)
    f["alpha_t"] = f"{mp['alpha']['t']:.2f}"
    f["fm_t"] = f"{s['fama_macbeth']['t']:.1f}"
    f["plc_mean"] = f"{pl['mean']:.2f}"
    f["plc_sd"] = f"{pl['sd']:.2f}"
    f["plc_p"] = f"{pl['p_ge_real']:.2f}"
    f["pbo"] = f"{ov['cscv']['pbo']:.2f}"
    f["dsr"] = f"{ov['dsr']:.2f}"
    f["n_trials_used"] = str(ov["n_trials"])
    f["sr0_ann"] = f"{ov['sr0_per_period'] * 12 ** 0.5:.2f}"
    f["oos_loss"] = pct(ov["cscv"]["prob_oos_loss"], 0)
    f["placebo_draws"] = f"{ev['direct_supplier']['placebo']['draws']:,}"
    return f


def fmt(text: str, f: dict | None = None) -> str:
    """Replace <<key>> placeholders; an unknown key raises."""
    f = f or facts()

    def sub(m: re.Match) -> str:
        if m.group(1) not in f:
            raise KeyError(m.group(1))
        return f[m.group(1)]

    return re.sub(r"<<\s*([A-Za-z0-9_]+)\s*>>", sub, text)
