"""Build the static site into site/dist: vendored scripts only, data embedded, no external requests."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config  # noqa: E402

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
    data = dict(graph=g, tier_colors=turbo_hex(len(g["tiers"])),
                spill=dict(names=names, theta=st.loc[names, names].round(2).values.tolist(),
                           net=st.loc["net", names].round(2).tolist(), total=sps["static"]["total_index"]),
                rolling=dict(dates=[d.strftime("%Y-%m-%d") for d in ro.index], total=ro["total"].round(2).tolist(),
                             net={c[4:]: ro[c].round(2).tolist() for c in ro.columns if c.startswith("net_")}),
                leadlag=dict(tiers=tiers, lags=sorted(lagmats), mats=lagmats),
                facts=dict(n_nodes=summ["graph"]["nodes"], n_connected=summ["graph"]["connected_nodes"], n_edges=summ["graph"]["edges"],
                           n_edge_rows=summ["n_edge_rows"], build_end=summ["build_end"], spill_total=sps["static"]["total_index"],
                           spill_weekly=sps["weekly"]["total_index"], n_shocks=evs["n_shocks"],
                           max_suppliers=summ["information_set"]["max_suppliers"]),
                colors=dict(bg=BG, panel=PANEL, ink=INK, link=LINK))
    (DIST / "index.html").write_text(page(data))
    print("site built:", DIST / "index.html", round((DIST / "index.html").stat().st_size / 1e3), "kB")


if __name__ == "__main__":
    main()
