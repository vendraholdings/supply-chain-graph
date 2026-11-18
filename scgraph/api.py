"""Reusable lookup interface over data/derived/graph.json (used by later projects to map news to affected companies)."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pandas as pd

from . import config

DEFAULT = config.ROOT / "data" / "derived" / "graph.json"


def load(path: Path | str | None = None) -> dict:
    return json.loads(Path(path or DEFAULT).read_text())


def neighbours(ticker: str, depth: int = 1, direction: str = "both", as_of=None, graph: dict | None = None) -> list[dict]:
    """Companies within `depth` hops of `ticker`.

    direction: "suppliers" (upstream), "customers" (downstream) or "both". Returns dicts
    {ticker, relation ("supplier" | "customer"), hops, weight, sources}, where weight is the disclosed revenue share
    (or tier median where undisclosed) of the first-hop edge on the path, and sources are the filings behind that edge.
    `as_of` keeps only edges whose valid_from is before it and whose valid_to is after it.
    """
    g = graph or load()
    t = pd.Timestamp(as_of) if as_of is not None else None
    E = [e for e in g["edges"] if t is None or (pd.Timestamp(e["valid_from"]) < t < pd.Timestamp(e["valid_to"]))]
    up: dict[str, list[dict]] = {}
    down: dict[str, list[dict]] = {}
    for e in E:
        down.setdefault(e["supplier"], []).append(e)    # supplier -> its customers
        up.setdefault(e["customer"], []).append(e)      # customer -> its suppliers
    out: list[dict] = []
    for rel, adj, nxt in (("supplier", up, "supplier"), ("customer", down, "customer")):
        if direction not in ("both", rel + "s"):
            continue
        seen = {ticker}
        q = deque([(ticker, 0)])
        while q:
            cur, h = q.popleft()
            if h == depth:
                continue
            for e in adj.get(cur, []):
                n = e[nxt]
                if n in seen:
                    continue
                seen.add(n)
                out.append(dict(ticker=n, relation=rel, hops=h + 1, weight=e["weight"], sources=e["sources"]))
                q.append((n, h + 1))
    return sorted(out, key=lambda d: (d["hops"], d["relation"], d["ticker"]))
