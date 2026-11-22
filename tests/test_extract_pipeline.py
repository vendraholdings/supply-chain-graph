import json

import pytest

from scgraph import auto_edges, edges
from scgraph.edgar import EdgarHTTPError
from scgraph.extract_pipeline import fiscal_year, run, sec_symbol
from scgraph.extract_rules import load_aliases, read_extracts

AL = load_aliases()
HTML = "<html><body><p>Apple accounted for 23% of our net revenue in fiscal 2023.</p><p>Nothing else here.</p></body></html>"


class FakeClient:
    def __init__(self, fail_on=None, status=403):
        self.requests, self.fail_on, self.status = [], fail_on, status

    def cik_map(self):
        return {"SWKS": 4127, "AAPL": 320193, "ZZZ": 999}

    def annual_filings(self, cik, since_year=2015):
        return [dict(cik=cik, form="10-K", accession=f"0000004127-{y}-000001", filing_date=f"{y}-11-15", report_date=f"{y}-09-29",
                     primary_document="d.htm", url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{y}/d.htm") for y in (2023, 2024)]

    def get(self, url):
        self.requests.append(url)
        if self.fail_on and self.fail_on in url:
            raise EdgarHTTPError(self.status, url)
        return HTML


UNI = [dict(ticker="SWKS", country="US"), dict(ticker="6857.T", country="JP"), dict(ticker="ZZZZ", country="US"), dict(ticker="TSM", country="US")]


def test_symbol_mapping():
    assert sec_symbol("2330.TW", "TW") == "TSM"
    assert sec_symbol("6857.T", "JP") is None
    assert sec_symbol("BRK.B", "US") == "BRK-B"
    assert sec_symbol("000660.KS", "KR") is None


def test_fiscal_year_falls_back_to_previous_calendar_year():
    assert fiscal_year(dict(report_date="2023-09-29", filing_date="2023-11-15")) == 2023
    assert fiscal_year(dict(report_date="", filing_date="2024-02-20")) == 2023


def test_run_writes_extracts_report_and_skips_non_filers(tmp_path):
    res = run(FakeClient(), UNI, AL, tmp_path, log=lambda *_: None)
    assert res["stopped"] is None
    st = {r["ticker"]: r["status"] for r in res["report"]}
    assert st == {"6857.T": "not_sec_filer", "SWKS": "ok", "TSM": "no_cik", "ZZZZ": "no_cik"}
    ex = read_extracts(tmp_path / "SWKS.jsonl")
    assert len(ex) == 2 and {e["customer"] for e in ex} == {"AAPL"} and ex[0]["fiscal_year"] == 2023
    assert (tmp_path / "report.csv").exists()


@pytest.mark.parametrize("status", [403, 429])
def test_access_refusal_stops_all_work(tmp_path, status):
    c = FakeClient(fail_on="/2024/", status=status)
    res = run(c, UNI + [dict(ticker="ZZZ", country="US")], AL, tmp_path, log=lambda *_: None)
    assert str(status) in res["stopped"]
    assert not any("/999/" in u for u in c.requests)      # nothing requested after the refusal
    assert not (tmp_path / "SWKS.jsonl").exists()


def test_other_http_errors_skip_one_document(tmp_path):
    res = run(FakeClient(fail_on="/2024/", status=404), UNI[:1], AL, tmp_path, log=lambda *_: None)
    assert res["stopped"] is None and res["report"][0]["status"] == "partial" and res["report"][0]["n_docs"] == 1


def test_rules_only_edges_are_marked_not_model_confirmed(tmp_path):
    run(FakeClient(), UNI[:1], AL, tmp_path, log=lambda *_: None)
    ex = read_extracts(tmp_path / "SWKS.jsonl")
    df, queue = auto_edges.build(ex, None, AL)
    assert not queue and len(df) == 2
    assert (df["confidence"] == "medium").all() and (~df["reviewed"]).all() and df["source_note"].str.contains("not model-confirmed").all()
    assert not edges.validate(df)
    assert json.dumps(df["source_url"].tolist())     # sourced
