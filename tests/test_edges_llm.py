import json

import pandas as pd
import pytest

from scgraph import edges
from scgraph.extract_llm import LLM, CacheMiss, cache_key, parse_json, reconcile
from scgraph.extract_rules import load_aliases

AL = load_aliases()


def row(**k):
    d = dict(edge_id="a", supplier="LRCX", customer="MU", pct_revenue=10.0, fiscal_year=2023, filing_date="2024-02-14",
             available_date="2024-02-15", valid_to="2025-02-14", source_type="10-K", source_url="https://example.org/a",
             source_note="n", method="curated", confidence="high", reviewed=True)
    d.update(k)
    return d


def test_validate_accepts_good_and_flags_bad_rows():
    assert edges.validate(pd.DataFrame([row()])) == []
    bad = pd.DataFrame([row(), row(edge_id="b", source_url="not a url"), row(edge_id="c", source_type="blog")])
    errs = edges.validate(bad)
    assert any("source_url" in e for e in errs) and any("source_type" in e for e in errs)
    assert edges.validate(pd.DataFrame([row(source_url="")]))


def test_available_date_is_next_session_and_after_filing():
    out = edges.next_us_session(["2024-02-16", "2024-03-28"])  # Friday; Thursday before Good Friday
    assert str(out.iloc[0].date()) == "2024-02-20"  # Monday 19 Feb is Presidents Day
    assert str(out.iloc[1].date()) == "2024-04-01"


def test_finalize_chains_years_and_point_in_time():
    df = pd.DataFrame([dict(supplier="LRCX", customer="MU", pct_revenue=10, fiscal_year=2022, filing_date="2023-02-15",
                            source_type="10-K", source_url="https://example.org/1", source_note="", method="curated", confidence="high", reviewed=True),
                       dict(supplier="LRCX", customer="MU", pct_revenue=12, fiscal_year=2023, filing_date="2024-02-14",
                            source_type="10-K", source_url="https://example.org/2", source_note="", method="curated", confidence="high", reviewed=True)])
    f = edges.finalize(df)
    assert edges.validate(f) == []
    assert f.loc[0, "valid_to"] == f.loc[1, "available_date"]
    assert len(edges.edges_as_of(f, "2023-06-01")) == 1 and f.loc[0, "edge_id"] in set(edges.edges_as_of(f, "2023-06-01").edge_id)
    assert len(edges.edges_as_of(f, "2023-01-01")) == 0
    # strict: the first usable signal date is the day after availability
    assert len(edges.edges_as_of(f, f.loc[0, "available_date"])) == 0


def test_llm_replay_fails_on_miss_and_live_fills_cache(tmp_path):
    with pytest.raises(CacheMiss):
        LLM(mode="replay", cache_dir=tmp_path).extract("text")

    def post(url, body, headers):
        assert body["temperature"] == 0
        return json.dumps({"customers": [{"name": "Apple Inc.", "percent": 23, "basis": "revenue"}]})

    import os
    os.environ.update(SCG_LLM_BASE_URL="http://localhost:1", SCG_LLM_MODEL="m", SCG_LLM_LABEL="test-label")
    r = LLM(mode="live", cache_dir=tmp_path, post=post).extract("Apple accounted for 23%")
    assert r["customers"][0]["percent"] == 23.0
    r2 = LLM(mode="replay", cache_dir=tmp_path).extract("Apple accounted for 23%")
    assert r2 == r
    blob = "".join(p.read_text() for p in tmp_path.iterdir())
    assert "localhost" not in blob and "http" not in blob


def test_cache_key_depends_on_label_version_text():
    assert cache_key("a", "t") != cache_key("b", "t") != cache_key("a", "u")
    assert cache_key("a", "t", "v1") != cache_key("a", "t", "v2")


def test_parse_json_is_defensive_and_reconcile():
    assert parse_json("not json") == {"customers": []}
    assert parse_json('x {"customers": [{"name": "A", "percent": "n/a"}]} y')["customers"][0]["percent"] is None
    ex = dict(customer="AAPL", pct=23.0, pct_all=[23.0, 20.0])
    assert reconcile(ex, {"customers": [{"name": "Apple", "percent": 20.2, "basis": "revenue"}]}, AL) == "agree"
    assert reconcile(ex, {"customers": [{"name": "Apple", "percent": 40, "basis": "revenue"}]}, AL) == "disagree"
