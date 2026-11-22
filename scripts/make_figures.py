"""Figures that depend only on committed results (figures 2 to 4 and 7 to 11; figures 5 and 6 need an active momentum strategy)."""
from __future__ import annotations

import sys
from pathlib import Path

import json

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config, plots  # noqa: E402

RES = config.ROOT / "results"


def main() -> None:
    out = RES / "figures"
    st = pd.read_csv(RES / "spillover_static.csv", index_col=0)
    ro = pd.read_parquet(RES / "spillover_rolling.parquet")
    ll = pd.read_csv(RES / "leadlag_tiers.csv")
    uni = pd.read_csv(config.ROOT / "data" / "universe.csv")
    edges = pd.read_csv(config.ROOT / "data" / "edges_curated.csv")
    der = config.ROOT / "data" / "derived"
    cent = pd.read_csv(der / "centrality.csv")
    nodes = pd.DataFrame(json.loads((der / "graph.json").read_text())["nodes"])
    flow = pd.read_csv(der / "tier_flow.csv", index_col=0)
    caar = pd.read_csv(RES / "event_caar.csv")
    draws = pd.read_csv(RES / "event_placebo_draws.csv")
    esum = json.loads((RES / "event_summary.json").read_text())
    for mode in ("light", "dark"):
        for p in (plots.fig_country_tier(uni, edges, out, mode), plots.fig_centrality(cent, nodes, out, mode),
                  plots.fig_tier_flow(flow, out, mode), plots.fig_event_fans(caar, out, mode),
                  plots.fig_placebo_panels(draws, esum, out, mode),
                  plots.fig_spillover_table(st, out, mode), plots.fig_spillover_rolling(ro, out, mode), plots.fig_leadlag_grid(ll, out, mode)):
            print(p.name)


if __name__ == "__main__":
    main()
