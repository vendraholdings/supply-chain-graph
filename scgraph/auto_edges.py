"""Automated edges: rule extracts reconciled with the cached LLM pass; disagreements wait in a review queue."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from . import edges as edges_mod
from .extract_llm import LLM, CacheMiss, reconcile


def candidate_rows(extracts: list[dict]) -> list[dict]:
    """Extracts that can become edges: resolved customer other than the supplier, a revenue basis and a stated share."""
    return [x for x in extracts if x["customer"] and x["customer"] != x["supplier"] and x["basis"] == "revenue"
            and x["pct"] is not None]


def build(extracts: list[dict], llm: LLM, aliases, decisions: dict[str, str] | None = None):
    """Returns (edges_auto DataFrame ready for finalize, review queue list). Fully determined by its inputs."""
    decisions = decisions or {}
    rows, queue = [], []
    for x in candidate_rows(extracts):
        key = f"{x['supplier']}>{x['customer']}:{x['accession']}"
        try:
            status = reconcile(x, llm.extract(x["text"]), aliases)
        except CacheMiss:
            status = "no_llm_response"
        accepted = status == "agree" or decisions.get(key) == "accept"
        if not accepted:
            if decisions.get(key) != "reject":
                queue.append(dict(key=key, supplier=x["supplier"], customer=x["customer"], pct=x["pct"], status=status,
                                  source_url=x["source_url"], text=x["text"][:400]))
            continue
        rows.append(dict(supplier=x["supplier"], customer=x["customer"], pct_revenue=x["pct"], fiscal_year=x["fiscal_year"],
                         filing_date=x["filing_date"], source_type=x["form"] if x["form"] in ("10-K", "20-F") else "annual_report",
                         source_url=x["source_url"], source_note=f"Rule match, {'model-confirmed' if status == 'agree' else 'manually reviewed'}"
                         + (f"; table row with {len(x['pct_all'])} columns, first column used" if x.get("table_row") else ""),
                         method="rule", confidence="medium" if x.get("table_row") else "high", reviewed=True))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=edges_mod.COLUMNS), queue
    df = df.drop_duplicates(["supplier", "customer", "fiscal_year", "source_url"]).sort_values(
        ["supplier", "customer", "filing_date", "fiscal_year"]).reset_index(drop=True)
    return edges_mod.finalize(df), sorted(queue, key=lambda q: q["key"])


def write_queue(queue: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "supplier", "customer", "pct", "status", "source_url", "text"])
        w.writeheader()
        w.writerows(queue)
