"""Edge schema, validation, point-in-time availability and the merge of curated and automated edges."""
from __future__ import annotations

from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import calendars, config

COLUMNS = ["edge_id", "supplier", "customer", "pct_revenue", "fiscal_year", "filing_date", "available_date", "valid_to",
           "source_type", "source_url", "source_note", "method", "confidence", "reviewed"]
SOURCE_TYPES = {"10-K", "20-F", "annual_report", "press_release"}
METHODS = {"curated", "rule", "llm"}
REQUIRED = ["supplier", "customer", "fiscal_year", "filing_date", "available_date", "source_type", "source_url"]


def next_us_session(dates) -> pd.Series:
    """First NYSE session strictly after each date (the first day a filing's content can be traded on)."""
    d = pd.to_datetime(pd.Series(dates))
    g = calendars.us_grid(str((d.min() - pd.Timedelta(days=5)).date()), str((d.max() + pd.Timedelta(days=10)).date()))
    idx = g.index.to_numpy("datetime64[ns]")
    pos = np.searchsorted(idx, d.to_numpy("datetime64[ns]"), side="right")
    return pd.Series(g.index[pos], index=d.index)


def valid_url(u) -> bool:
    if not isinstance(u, str):
        return False
    p = urlparse(u)
    return p.scheme in ("http", "https") and bool(p.netloc) and "." in p.netloc


def validate(df: pd.DataFrame) -> list[str]:
    errs = []
    for c in REQUIRED:
        bad = df[c].isna() | (df[c].astype(str).str.strip() == "")
        if bad.any():
            errs.append(f"{int(bad.sum())} rows with empty {c}")
    if (~df["source_url"].map(valid_url)).any():
        errs.append("invalid source_url")
    if (~df["source_type"].isin(SOURCE_TYPES)).any():
        errs.append("unknown source_type")
    if (~df["method"].isin(METHODS)).any():
        errs.append("unknown method")
    if (pd.to_datetime(df["available_date"]) <= pd.to_datetime(df["filing_date"])).any():
        errs.append("available_date must be after filing_date")
    if df["edge_id"].duplicated().any():
        errs.append("duplicate edge_id")
    return errs


def finalize(df: pd.DataFrame, supplier_filings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add available_date, valid_to and edge_id to rows carrying supplier, customer, fiscal_year, filing_date."""
    df = df.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["available_date"] = next_us_session(df["filing_date"]).values
    df = df.sort_values(["supplier", "customer", "filing_date"]).reset_index(drop=True)
    vt = []
    for (_, _), g in df.groupby(["supplier", "customer"], sort=False):
        nxt_av = g["available_date"].shift(-1)
        own = g["available_date"] + pd.Timedelta(days=365)
        v = nxt_av.where(nxt_av.notna() & (nxt_av <= own), own)
        vt.append(v)
    df["valid_to"] = pd.concat(vt).reindex(df.index)
    if supplier_filings is not None and len(supplier_filings):
        # an edge ends at the supplier's next annual filing that does not list it
        sf = supplier_filings.assign(filing_date=pd.to_datetime(supplier_filings["filing_date"]))
        for i, r in df.iterrows():
            later = sf[(sf.supplier == r.supplier) & (sf.filing_date > r.filing_date)].sort_values("filing_date")
            listed = set(zip(df.supplier, df.customer, df.filing_date))
            for f in later.itertuples():
                if (r.supplier, r.customer, f.filing_date) not in listed:
                    cap = next_us_session([f.filing_date]).iloc[0]
                    df.at[i, "valid_to"] = min(df.at[i, "valid_to"], cap)
                    break
    df["edge_id"] = [f"{r.supplier}>{r.customer}:{r.fiscal_year}:{r.source_type}" for r in df.itertuples()]
    for c in ("filing_date", "available_date", "valid_to"):
        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return df[COLUMNS]


def edges_as_of(df: pd.DataFrame, when, strict: bool = True) -> pd.DataFrame:
    """Edges usable at `when`: available strictly before it (information set) and not yet expired."""
    t = pd.Timestamp(when)
    av = pd.to_datetime(df["available_date"])
    vt = pd.to_datetime(df["valid_to"])
    ok = (av < t if strict else av <= t) & (vt > t)
    return df[ok]


def load(include=("curated", "rule", "llm")) -> pd.DataFrame:
    parts = []
    for name in ("edges_curated.csv", "edges_auto.csv"):
        p = config.ROOT / "data" / name
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    return df[df["method"].isin(include)].reset_index(drop=True)
