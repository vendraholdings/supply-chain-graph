"""Figures that depend only on committed results (9 to 11 now; the network, momentum and event figures follow with the data)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config, plots  # noqa: E402

RES = config.ROOT / "results"


def main() -> None:
    out = RES / "figures"
    st = pd.read_csv(RES / "spillover_static.csv", index_col=0)
    ro = pd.read_parquet(RES / "spillover_rolling.parquet")
    ll = pd.read_csv(RES / "leadlag_tiers.csv")
    for mode in ("light", "dark"):
        for p in (plots.fig_spillover_table(st, out, mode), plots.fig_spillover_rolling(ro, out, mode), plots.fig_leadlag_grid(ll, out, mode)):
            print(p.name)


if __name__ == "__main__":
    main()
