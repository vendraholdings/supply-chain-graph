"""Render README.md from README.md.tmpl and the committed results."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scgraph.facts import fmt  # noqa: E402

(ROOT / "README.md").write_text(fmt((ROOT / "README.md.tmpl").read_text()))
print("README.md written")
