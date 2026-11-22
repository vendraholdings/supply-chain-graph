"""One documented shock case for the site: the largest absolute one-day move among equipment (T05) and memory (T08) listings
with at least five linked companies at that date (rule fixed here, not chosen by hand). Writes results/shock_case.json with the
cumulative abnormal return of every linked company on days 0 to 10 (market model with local index and SOXX, as in the event study)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config, edges as edges_mod, events as ev, returns  # noqa: E402

D, RES = config.ROOT / "data", config.ROOT / "results"
LOCAL = {"US": "BM_SPY", "JP": "BM_N225", "TW": "BM_TWII", "KR": "BM_KS11", "HK": "BM_HSI", "EU": "BM_SX5E"}
TIERS, MIN_LINKED, HORIZON = ("T05", "T08"), 5, 10


def main() -> None:
    cfg = config.load()
    R = returns.build(cfg)
    uni = pd.read_csv(D / "universe.csv")
    tick = list(uni.ticker)
    country, tier, name = dict(zip(uni.ticker, uni.country)), dict(zip(uni.ticker, uni.tier)), dict(zip(uni.ticker, uni.name))
    Y = R[tick].to_numpy(float)
    col = {t: i for i, t in enumerate(tick)}
    F = {col[t]: R[[LOCAL[country[t]], "BM_SOXX"]].fillna(0.0).to_numpy(float) for t in tick}
    E = edges_mod.load()
    dates, T = R.index, len(R.index)
    est = tuple(cfg["event"]["estimation_window"])
    best = None
    for t in (x for x in tick if tier[x] in TIERS):
        for p in ev.find_shocks(R[t], cfg["event"]["z_threshold"]):
            if p < 280 or p > T - 25:
                continue
            live = edges_mod.edges_as_of(E, dates[p])
            sup1, cus1 = set(live[live.customer == t].supplier), set(live[live.supplier == t].customer)
            sup2 = set(live[live.customer.isin(sup1)].supplier) - sup1 - {t}
            cus2 = set(live[live.supplier.isin(cus1)].customer) - cus1 - {t}
            linked = {x: "direct supplier" for x in sup1} | {x: "direct customer" for x in cus1}
            linked |= {x: "two hops" for x in (sup2 | cus2) if x not in linked}
            paths = {}
            for x, rel in linked.items():
                if x not in col:
                    continue
                d0 = p + ev.day0_offset(country[t], country[x])
                Xd = ev.design(F[col[x]])
                fit = ev.fit_market_model(Y[:, col[x]], Xd, d0 + est[0], d0 + est[1])
                cp = None if fit is None else ev.car_path(ev.abnormal(Y[:, col[x]], Xd, fit), d0, HORIZON)
                if cp is not None:
                    paths[x] = dict(relation=rel, tier=tier[x], name=name[x], car=[round(100 * float(v), 3) for v in cp])
            if len(paths) >= MIN_LINKED:
                r = float(R[t].iloc[p])
                if best is None or abs(r) > abs(best["event"]["return_pct"] / 100):
                    best = dict(event=dict(ticker=t, name=name[t], tier=tier[t], date=str(dates[p].date()), return_pct=round(100 * r, 2)), nodes=paths,
                                rule=f"largest absolute one-day move among {', '.join(TIERS)} listings with at least {MIN_LINKED} linked companies with usable estimates")
    if best is None:
        print("no event satisfies the rule")
        return
    (RES / "shock_case.json").write_text(json.dumps(best, indent=1))
    print(json.dumps(best["event"]), len(best["nodes"]), "linked")


if __name__ == "__main__":
    main()
