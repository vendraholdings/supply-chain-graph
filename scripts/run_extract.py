"""make extract: EDGAR annual reports -> data/extracts/*.jsonl (rule extracts) -> data/edges_auto.csv and the review queue.

Needs the private SEC_USER_AGENT variable (else config.yaml is used). The model pass runs only when SCG_LLM_BASE_URL and
SCG_LLM_MODEL are set (SCG_LLM_MODE=live fills the cache, replay reads it); otherwise edges come from the rules alone.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scgraph import auto_edges, config  # noqa: E402
from scgraph.edgar import EdgarClient  # noqa: E402
from scgraph.extract_llm import LLM  # noqa: E402
from scgraph.extract_pipeline import read_universe, run  # noqa: E402
from scgraph.extract_rules import load_aliases, read_extracts  # noqa: E402

DATA = config.ROOT / "data"


def main() -> int:
    aliases = load_aliases()
    out = DATA / "extracts"
    if "--offline" in sys.argv:
        res = {"stopped": None}
    else:
        res = run(EdgarClient(), read_universe(), aliases, out)
        if res["stopped"]:
            print("STOPPED:", res["stopped"])
    extracts = []
    for p in sorted(out.glob("*.jsonl")):
        extracts += read_extracts(p)
    use_llm = bool(os.environ.get("SCG_LLM_BASE_URL") and os.environ.get("SCG_LLM_MODEL")) or os.environ.get("SCG_LLM_MODE") == "replay"
    llm = LLM() if use_llm else None
    print(f"{len(extracts)} extracts; model pass: {'on (' + llm.mode + ')' if llm else 'off, rules only'}")
    dec_path = DATA / "review_decisions.json"
    decisions = json.loads(dec_path.read_text()) if dec_path.exists() else {}
    df, queue = auto_edges.build(extracts, llm, aliases, decisions)
    df.to_csv(DATA / "edges_auto.csv", index=False)
    auto_edges.write_queue(queue, DATA / "review_queue.csv")
    print(f"edges_auto.csv: {len(df)} rows; review queue: {len(queue)}")
    return 1 if res["stopped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
