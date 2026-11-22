"""Smoke tests: the committed-results figures render in both themes."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_make_figures_runs_and_writes_all_figures():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_figures.py")], check=True, capture_output=True)
    out = ROOT / "results" / "figures"
    for stem in ("fig02_country_tier", "fig03_centrality", "fig04_tier_flow", "fig07_event_fans", "fig08_placebo",
                 "fig09_spillover_table", "fig10_spillover_rolling", "fig11_leadlag"):
        for mode in ("light", "dark"):
            assert (out / f"{stem}_{mode}.png").stat().st_size > 5000


def test_placebo_draws_match_summary_counts():
    import pandas as pd
    d = pd.read_csv(ROOT / "results" / "event_placebo_draws.csv")
    s = json.loads((ROOT / "results" / "event_summary.json").read_text())
    for rel in ("direct_supplier", "direct_customer", "two_hop"):
        assert len(d[d.relation == rel]) == s[rel]["placebo"]["draws"]
