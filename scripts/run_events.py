"""Event study: price shocks (|r| > 3 trailing-60-day SD) and abnormal returns of linked companies, with placebo."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config, edges as edges_mod, events as ev, returns  # noqa: E402

D, RES = config.ROOT / "data", config.ROOT / "results"
LOCAL = {"US": "BM_SPY", "JP": "BM_N225", "TW": "BM_TWII", "KR": "BM_KS11", "HK": "BM_HSI", "EU": "BM_SX5E"}


def main() -> None:
    cfg = config.load()
    ec = cfg["event"]
    R = returns.build(cfg)
    uni = pd.read_csv(D / "universe.csv")
    tick = list(uni.ticker)
    country = dict(zip(uni.ticker, uni.country))
    tier = dict(zip(uni.ticker, uni.tier))
    Y = R[tick].to_numpy(float)
    col = {t: i for i, t in enumerate(tick)}
    F = {col[t]: R[[LOCAL[country[t]], "BM_SOXX"]].fillna(0.0).to_numpy(float) for t in tick}
    E = edges_mod.load()
    dates = R.index
    T = len(dates)
    rows, groups = [], {}
    for t in tick:
        for p in ev.find_shocks(R[t], ec["z_threshold"]):
            if p < 280 or p > T - 25:
                continue
            sgn = 1 if R[t].iloc[p] > 0 else -1
            live = edges_mod.edges_as_of(E, dates[p])
            sup1 = set(live[live.customer == t].supplier)
            cus1 = set(live[live.supplier == t].customer)
            sup2 = set(live[live.customer.isin(sup1)].supplier) - sup1 - {t}
            cus2 = set(live[live.supplier.isin(cus1)].customer) - cus1 - {t}
            for rel, S in (("direct_supplier", sup1), ("direct_customer", cus1), ("two_hop", sup2 | cus2)):
                for x in S:
                    if x in col:
                        d0 = p + ev.day0_offset(country[t], country[x])
                        if d0 < T - 22:
                            groups.setdefault(rel, []).append((col[x], d0, sgn, tier[x]))
            groups.setdefault("event_node", []).append((col[t], p, sgn, tier[t]))
    out: dict = dict(n_shocks=len(groups.get("event_node", [])), edges_used=int(len(E)))
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    draw_rows = []
    for rel, items in groups.items():
        pairs = [(c, d, s) for c, d, s, _ in items]
        r = ev.study(Y, F, pairs, ec["car_horizon"], tuple(ec["estimation_window"]))
        entry = dict(n=r["n"], t_kp=None if np.isnan(r["t_kp"]) else r["t_kp"],
                     caar_end=None if r["caar"] is None else float(r["caar"][-1]))
        if r["n"] >= 10 and rel != "event_node":
            pl = ev.placebo(Y, F, [col[t] for t in tick], [country[t] for t in tick], r["n"], ec["placebo_draws"], rng,
                            ec["car_horizon"], tuple(ec["estimation_window"]))
            draw_rows += [dict(relation=rel, draw=i, caar_end=float(v)) for i, v in enumerate(pl["caar_draws"])]
            entry["placebo"] = dict(rejection_rate_5pct=pl["rejection_rate"], mean_caar=pl["mean_caar"], se=pl["se_caar"], draws=pl["draws"])
        if r["caar"] is not None:
            for h, v in enumerate(r["caar"]):
                rows.append(dict(relation=rel, tier="all", day=h, caar=float(v), n=r["n"]))
        for tr in sorted({x[3] for x in items}):
            sub = [(c, d, s) for c, d, s, tt in items if tt == tr]
            rt = ev.study(Y, F, sub, ec["car_horizon"], tuple(ec["estimation_window"]))
            if rt["caar"] is not None and rt["n"] >= 5:
                for h, v in enumerate(rt["caar"]):
                    rows.append(dict(relation=rel, tier=tr, day=h, caar=float(v), n=rt["n"]))
        out[rel] = entry
    pd.DataFrame(rows).to_csv(RES / "event_caar.csv", index=False)
    pd.DataFrame(draw_rows).to_csv(RES / "event_placebo_draws.csv", index=False)
    (RES / "event_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
