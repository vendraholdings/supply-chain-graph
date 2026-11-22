import json


from scgraph import auto_edges, edges
from scgraph.extract_llm import LLM, cache_key
from scgraph.extract_rules import extract_text, load_aliases

AL = load_aliases()
META = dict(accession="0001-24-000001", form="10-K", filing_date="2024-02-20", report_date="2023-12-31", fiscal_year=2023,
            source_url="https://example.org/10k.htm")


def seed_cache(tmp_path, text, customers):
    key = cache_key("lbl", text)
    (tmp_path / f"{key}.json").write_text(json.dumps({"prompt_version": "v1", "label": "lbl", "response": {"customers": customers}}))


def make(tmp_path, llm_customers):
    t = "Apple accounted for 23% of net revenue in fiscal 2023."
    ex = extract_text(t, "SWKS", META, AL)
    seed_cache(tmp_path, ex[0]["text"], llm_customers)
    import os
    os.environ["SCG_LLM_LABEL"] = "lbl"
    return ex, LLM(mode="replay", cache_dir=tmp_path)


def test_agreement_makes_a_reviewed_edge_and_is_deterministic(tmp_path):
    ex, llm = make(tmp_path, [{"name": "Apple Inc.", "percent": 23, "basis": "revenue"}])
    a, q = auto_edges.build(ex, llm, AL)
    b, _ = auto_edges.build(list(reversed(ex)), llm, AL)
    assert len(a) == 1 and q == [] and edges.validate(a) == []
    assert a.to_csv(index=False) == b.to_csv(index=False)
    assert a.loc[0, "method"] == "rule" and bool(a.loc[0, "reviewed"])


def test_disagreement_goes_to_queue_then_decision_decides(tmp_path):
    ex, llm = make(tmp_path, [{"name": "Apple", "percent": 40, "basis": "revenue"}])
    a, q = auto_edges.build(ex, llm, AL)
    assert a.empty and len(q) == 1 and q[0]["status"] == "disagree"
    acc, q2 = auto_edges.build(ex, llm, AL, {q[0]["key"]: "accept"})
    assert len(acc) == 1 and q2 == []
    rej, q3 = auto_edges.build(ex, llm, AL, {q[0]["key"]: "reject"})
    assert rej.empty and q3 == []


def test_replay_miss_is_queued_not_silently_accepted(tmp_path):
    ex = extract_text("Apple accounted for 23% of net revenue.", "SWKS", META, AL)
    a, q = auto_edges.build(ex, LLM(mode="replay", cache_dir=tmp_path), AL)
    assert a.empty and q[0]["status"] == "no_llm_response"


def test_receivable_and_anonymised_excluded():
    ex = extract_text("Apple represented 15% of accounts receivable.\nCustomer A accounted for 31% of net sales.", "SWKS", META, AL)
    assert auto_edges.candidate_rows(ex) == []


def test_threshold_edge_has_empty_share(tmp_path):
    ex = extract_text("Revenues from Apple, Samsung and Xiaomi each comprised 10% or more of our revenues.", "QCOM", META, AL)
    seed_cache(tmp_path, ex[0]["text"], [{"name": "Apple", "percent": None, "basis": "revenue"}, {"name": "Samsung", "percent": None, "basis": "revenue"}])
    import os
    os.environ["SCG_LLM_LABEL"] = "lbl"
    a, q = auto_edges.build(ex, LLM(mode="replay", cache_dir=tmp_path), AL)
    assert len(a) == 2 and a["pct_revenue"].isna().all() and edges.validate(a) == []


def test_load_has_unique_edge_ids_and_validates():
    df = edges.load()
    assert not edges.validate(df)
    assert df["edge_id"].is_unique
