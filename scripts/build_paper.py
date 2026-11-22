"""Render paper/paper.typ from the template and the results, copy the light figures, compile paper/paper.pdf.

    uv run python scripts/build_paper.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typst

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scgraph.facts import fmt  # noqa: E402


def main() -> None:
    fig = ROOT / "paper" / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "results" / "figures").glob("fig*_light.png"):
        shutil.copy(p, fig / p.name)
    (ROOT / "paper" / "paper.typ").write_text(fmt((ROOT / "paper" / "paper.typ.tmpl").read_text()))
    typst.compile(str(ROOT / "paper" / "paper.typ"), output=str(ROOT / "paper" / "paper.pdf"))
    print("paper/paper.pdf written")


if __name__ == "__main__":
    main()
