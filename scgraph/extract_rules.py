"""Rule-based extraction of named-customer concentration disclosures from annual-report text.

Each extract records the filing, character offsets and the matched unit (up to 2,000 characters). Customers are
resolved to universe tickers through data/aliases.csv; anonymised ("Customer A") disclosures stay unresolved.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from . import config

PCT = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d{1,2})?)\s?%")
ANON = re.compile(r"\b(?:Customer|Client)\s+([A-Z])\b")
TRIGGER = re.compile(r"accounted for|account for|represent|comprised|constituted|10\s?%|customer", re.I)
REV = re.compile(r"revenue|net sales|sales", re.I)
REC = re.compile(r"receivable", re.I)
MAX_UNIT = 2000


def load_aliases(path: Path | None = None) -> list[tuple[str, str]]:
    path = path or config.ROOT / "data" / "aliases.csv"
    rows = [(r["alias"], r["ticker"]) for r in csv.DictReader(open(path))]
    return sorted(rows, key=lambda x: -len(x[0]))


def alias_hits(unit: str, aliases: list[tuple[str, str]]) -> list[tuple[int, int, str]]:
    """Non-overlapping case-sensitive whole-word alias matches, longest alias first."""
    taken: list[tuple[int, int, str]] = []
    for alias, tk in aliases:
        for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", unit):
            if all(m.end() <= a or m.start() >= b for a, b, _ in taken):
                taken.append((m.start(), m.end(), tk))
    return sorted(taken)


def units(text: str):
    """Yield (start, unit) for each line, splitting very long lines at sentence ends."""
    pos = 0
    for line in text.split("\n"):
        n = len(line)
        if n > 0:
            if n <= MAX_UNIT:
                yield pos, line
            else:
                off = 0
                for s in re.split(r"(?<=[.;])\s+", line):
                    i = line.find(s, off)
                    yield pos + i, s[:MAX_UNIT]
                    off = i + len(s)
        pos += n + 1


def extract_text(text: str, supplier: str, meta: dict, aliases) -> list[dict]:
    out: list[dict] = []
    recent: list[str] = []
    for start, u in units(text):
        ctx = " ".join(recent[-6:])
        recent.append(u)
        table = " | " in u
        if "%" not in u or not (TRIGGER.search(u) or (table and TRIGGER.search(ctx))):
            continue
        pcts = [(m.start(), float(m.group(1))) for m in PCT.finditer(u)]
        hits = alias_hits(u, aliases)
        anon = [(m.start(), m.end(), "Customer " + m.group(1)) for m in ANON.finditer(u)]
        marks = sorted([(a, b, tk, True) for a, b, tk in hits] + [(a, b, tk, False) for a, b, tk in anon])
        for i, (a, b, tk, resolved) in enumerate(marks):
            nxt = marks[i + 1][0] if i + 1 < len(marks) else len(u)
            after = [p for p in pcts if b <= p[0] < min(nxt, b + 160)]
            if not after:
                continue
            window = u[max(0, a - 120): min(len(u), after[0][0] + 80)]
            basis_text = window + (" " + ctx if table else "")
            basis = "revenue" if REV.search(basis_text) else "receivable" if REC.search(basis_text) else "other"
            out.append(dict(supplier=supplier, **meta, start=start, end=start + len(u), text=u[:MAX_UNIT],
                            customer_raw=u[a:b], customer=tk if resolved else None, pct=after[0][1],
                            pct_all=[p[1] for p in after], basis=basis, anonymised=not resolved,
                            table_row=len(after) > 1))
    return out


def write_extracts(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in sorted(rows, key=lambda r: (r["supplier"], r["filing_date"], r["start"], r["customer_raw"])):
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_extracts(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]
