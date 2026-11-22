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
THRESH = re.compile(r"10\s?%\s+or\s+more|more than\s+10\s?%|over\s+10\s?%|at least\s+10\s?%|exceed(?:ed|ing)?\s+10\s?%|in excess of\s+10\s?%|"
                    r"individually[^.]{0,60}10\s?%|each[^.]{0,80}10\s?%|10\s?%\s+of\s+(?:our\s+)?(?:total\s+|net\s+)*(?:revenue|sales)\s+or\s+more", re.I)
NEGATION = re.compile(r"\b(?:did|do|does|had|has|have)\s+not\b|\bless than\s+\d|\bbelow\s+\d|\bno customer\b|\bnot\s+(?:represent|account|exceed|comprise)", re.I)
AGG = re.compile(r"\b(?:combined|aggregate|together|in total|top\s+(?:\w+\s+)?customers|(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:largest|top|major)"
                 r"|(?:top|largest)\s+(?:two|three|four|five|six|seven|eight|nine|ten|\d+))\b", re.I)
MARKET = re.compile(r"market share|(?:of|in)\s+the\s+(?:total\s+|global\s+|worldwide\s+)*(?:foundry\s+|semiconductor\s+)?market|competitor|worldwide", re.I)
ABBREV = re.compile(r"\b(?:Inc|Corp|Co|Ltd|Cos|Lt|No|S\.A|N\.V|L\.P|U\.S)\.|\d\.\d")
MAX_UNIT = 2000


def sentence_break(s: str) -> bool:
    return bool(re.search(r"\.\s+[A-Z(]", ABBREV.sub("", s)))


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
            shared = False
            # "A and B accounted for 52% and 35% of sales, respectively": names and shares are paired by position
            s_ = i
            while s_ > 0 and not any(marks[s_ - 1][1] <= q[0] < marks[s_][0] for q in pcts):
                s_ -= 1
            e_ = i
            while e_ + 1 < len(marks) and not any(marks[e_][1] <= q[0] < marks[e_ + 1][0] for q in pcts):
                e_ += 1
            nxt_e = marks[e_ + 1][0] if e_ + 1 < len(marks) else len(u)
            pe = [q for q in pcts if marks[e_][1] <= q[0] < min(nxt_e, marks[e_][1] + 160)]
            if e_ > s_ and len(pe) > 1 and "respectively" in u[pe[-1][0]: pe[-1][0] + 90].lower():
                last: dict[str, int] = {}
                for j in range(s_, e_ + 1):
                    last[marks[j][2]] = j
                order = sorted(last.values())
                if len(order) > len(pe):      # more names than shares ("top three customers were A, B and C, with A and B accounting for 56% and 32%"): align from the end
                    order = order[-len(pe):]
                if i not in order:
                    continue
                after = [pe[order.index(i)]]
            elif not after:   # a list of names followed by one share: "Apple, Samsung and Xiaomi each comprised 10% or more"
                tail = [p for p in pcts if b <= p[0] < b + 160 and "." not in re.sub(r"\d\.\d|Inc\.|Corp\.|Co\.|Ltd\.", "", u[b:p[0]])]
                if tail and len(marks) > i + 1:
                    after, shared = tail, True
            if not after:
                continue
            if sentence_break(u[b:after[0][0]]) or AGG.search(u[b:after[0][0]]) or MARKET.search(u[max(0, a - 120): after[0][0] + 80]):
                continue
            window = u[max(0, a - 120): min(len(u), after[0][0] + 80)]
            if NEGATION.search(window):
                continue
            basis_text = window + (" " + ctx if table else "")
            basis = "revenue" if REV.search(basis_text) else "receivable" if REC.search(basis_text) else "other"
            threshold = bool(THRESH.search(window))
            pct = after[0][1]
            out.append(dict(supplier=supplier, **meta, start=start, end=start + len(u), text=u[:MAX_UNIT],
                            customer_raw=u[a:b], customer=tk if resolved else None,
                            pct=None if (threshold or shared) else pct, threshold=pct if threshold else None,
                            pct_all=[p[1] for p in after], basis=basis, anonymised=not resolved,
                            table_row=len(after) > 1, shared=shared))
    return out


def write_extracts(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in sorted(rows, key=lambda r: (r["supplier"], r["filing_date"], r["start"], r["customer_raw"])):
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_extracts(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]
