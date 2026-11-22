"""Annual-report extraction pipeline: EDGAR submissions index, filing text, rule extraction, per-supplier extract files.

Resumable: every EDGAR response is cached by the client, so a rerun costs no requests for finished filings.
A rate-limit or access refusal (HTTP 403 or 429) stops all EDGAR work at once and is recorded in the report.
"""
from __future__ import annotations

import csv
from pathlib import Path

from . import config
from .edgar import SEC_SYMBOL, EdgarHTTPError, html_to_text
from .extract_rules import extract_text, write_extracts

STOP_STATUS = {403, 429}
REPORT_FIELDS = ["ticker", "sec_symbol", "cik", "n_filings", "n_docs", "n_extracts", "status"]


def sec_symbol(ticker: str, country: str) -> str | None:
    """Symbol under which a universe listing files with the SEC, or None when it does not."""
    if ticker in SEC_SYMBOL:
        return SEC_SYMBOL[ticker]
    return ticker.replace(".", "-") if country == "US" else None


def fiscal_year(f: dict) -> int:
    rd = f.get("report_date") or ""
    return int(rd[:4]) if len(rd) >= 4 else int(f["filing_date"][:4]) - 1


def process_filing(client, ticker: str, f: dict, aliases) -> list[dict]:
    text = html_to_text(client.get(f["url"]))
    meta = dict(accession=f["accession"], form=f["form"], filing_date=f["filing_date"], report_date=f.get("report_date", ""),
                fiscal_year=fiscal_year(f), source_url=f["url"])
    return extract_text(text, ticker, meta, aliases)


def run(client, universe: list[dict], aliases, out_dir: Path, since_year: int = 2015, log=print) -> dict:
    """Extract every US-filing universe company. Returns {"report": [...], "stopped": None | "HTTP 403 ..."}"""
    out_dir.mkdir(parents=True, exist_ok=True)
    report, stopped = [], None
    try:
        ciks = client.cik_map()
    except EdgarHTTPError as e:
        return {"report": [], "stopped": f"HTTP {e.status} on the ticker index"}
    for u in sorted(universe, key=lambda r: r["ticker"]):
        tk = u["ticker"]
        sym = sec_symbol(tk, u["country"])
        row = dict(ticker=tk, sec_symbol=sym or "", cik="", n_filings=0, n_docs=0, n_extracts=0, status="")
        report.append(row)
        if sym is None:
            row["status"] = "not_sec_filer"
            continue
        cik = ciks.get(sym.upper())
        if cik is None:
            row["status"] = "no_cik"
            continue
        row["cik"] = cik
        rows: list[dict] = []
        try:
            filings = client.annual_filings(cik, since_year)
            row["n_filings"] = len(filings)
            for f in filings:
                try:
                    rows += process_filing(client, tk, f, aliases)
                    row["n_docs"] += 1
                except EdgarHTTPError as e:
                    if e.status in STOP_STATUS:
                        raise
                    log(f"{tk} {f['accession']}: HTTP {e.status}, skipped")
        except EdgarHTTPError as e:
            if e.status in STOP_STATUS:
                stopped = f"HTTP {e.status} while reading {tk}; EDGAR work stopped, no other User-Agent tried"
                row["status"] = f"stopped_http_{e.status}"
                break
            row["status"] = f"http_{e.status}"
            continue
        row["n_extracts"] = len(rows)
        row["status"] = "ok" if row["n_docs"] == row["n_filings"] else "partial"
        write_extracts(rows, out_dir / f"{tk}.jsonl")
        log(f"{tk}: {row['n_docs']}/{row['n_filings']} filings, {len(rows)} extracts")
    with open(out_dir / "report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        w.writeheader()
        w.writerows(report)
    return {"report": report, "stopped": stopped}


def read_universe(path: Path | None = None) -> list[dict]:
    return list(csv.DictReader(open(path or config.ROOT / "data" / "universe.csv")))
