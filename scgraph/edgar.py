"""SEC EDGAR client: shared rate limiter, declared User-Agent, on-disk cache of every response.

Fair-access policy: https://www.sec.gov/os/accessing-edgar-data. The declared User-Agent comes from config.yaml.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import time
from pathlib import Path

from . import config

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
ANNUAL_FORMS = {"10-K", "10-K405", "20-F", "40-F"}
# Foreign listings that file with the SEC under a different symbol
SEC_SYMBOL = {"ASML.AS": "ASML", "2330.TW": "TSM", "2303.TW": "UMC", "3711.TW": "ASX", "STMPA.PA": "STM",
              "000660.KS": None, "005930.KS": None}


class RateLimiter:
    """Sliding-window limiter: at most `rate` requests per second on average, never more than `hard` in any 1 s window."""

    def __init__(self, rate: float = 5.0, hard: int = 10, clock=time.monotonic, sleep=time.sleep):
        self.rate, self.hard = min(rate, 0.9 * hard), hard
        self.clock, self.sleep = clock, sleep
        self.stamps: collections.deque[float] = collections.deque()
        self.last = -1e9

    def acquire(self) -> None:
        now = self.clock()
        wait = max(0.0, self.last + 1.0 / self.rate - now)
        while self.stamps and self.stamps[0] <= now - 1.0:
            self.stamps.popleft()
        if len(self.stamps) >= self.hard:
            wait = max(wait, self.stamps[0] + 1.0 - now)
        if wait > 0:
            self.sleep(wait)
        now = self.clock()
        while self.stamps and self.stamps[0] <= now - 1.0:
            self.stamps.popleft()
        self.stamps.append(now)
        self.last = now


class EdgarClient:
    def __init__(self, cfg: dict | None = None, cache_dir: Path | None = None, session=None, limiter: RateLimiter | None = None):
        cfg = cfg or config.load()
        self.user_agent = cfg["edgar_user_agent"]
        self.limiter = limiter or RateLimiter(cfg["edgar_rate_per_s"])
        self.cache = Path(cache_dir) if cache_dir else config.ROOT / "cache" / "edgar"
        self.cache.mkdir(parents=True, exist_ok=True)
        self._session = session
        self.n_requests = 0

    @property
    def session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _path(self, url: str) -> Path:
        return self.cache / (hashlib.sha256(url.encode()).hexdigest() + ".txt")

    def get(self, url: str) -> str:
        p = self._path(url)
        if p.exists():
            return p.read_text()
        self.limiter.acquire()
        r = self.session.get(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=60)
        self.n_requests += 1
        if r.status_code != 200:
            raise RuntimeError(f"EDGAR request failed: HTTP {r.status_code} for {url}")
        p.write_text(r.text)
        return r.text

    # ---- lookups -------------------------------------------------------------------------------------------
    def cik_map(self) -> dict[str, int]:
        d = json.loads(self.get(TICKERS))
        return {v["ticker"].upper(): int(v["cik_str"]) for v in d.values()}

    def annual_filings(self, cik: int, since_year: int = 2015) -> list[dict]:
        sub = json.loads(self.get(SUBMISSIONS.format(cik=cik)))
        blocks = [sub["filings"]["recent"]]
        for f in sub["filings"].get("files", []):
            blocks.append(json.loads(self.get("https://data.sec.gov/submissions/" + f["name"])))
        out = []
        for b in blocks:
            for i, form in enumerate(b["form"]):
                if form in ANNUAL_FORMS and int(b["filingDate"][i][:4]) >= since_year:
                    acc = b["accessionNumber"][i]
                    out.append(dict(cik=cik, form=form, accession=acc, filing_date=b["filingDate"][i],
                                    report_date=b.get("reportDate", [""] * len(b["form"]))[i],
                                    primary_document=b["primaryDocument"][i],
                                    url=ARCHIVE.format(cik=cik, acc=acc.replace("-", ""), doc=b["primaryDocument"][i])))
        return sorted(out, key=lambda x: x["filing_date"])


def html_to_text(html: str) -> str:
    """Plain text with one line per block element; scripts, styles and hidden inline-XBRL headers removed."""
    from lxml import html as lh

    root = lh.fromstring(html)
    for bad in root.xpath("//script|//style|//*[local-name()='header']"):
        bad.getparent().remove(bad)
    for el in root.iter():
        if el.tag in {"p", "div", "tr", "br", "li", "h1", "h2", "h3", "h4", "table"}:
            el.tail = "\n" + (el.tail or "")
        elif el.tag in {"td", "th"}:
            el.tail = " | " + (el.tail or "")
    txt = root.text_content()
    txt = re.sub(r"[ \t\xa0]+", " ", txt)
    return re.sub(r"\n\s*\n+", "\n", txt)
