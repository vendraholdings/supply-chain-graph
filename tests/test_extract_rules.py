from scgraph.extract_rules import alias_hits, extract_text, load_aliases

AL = load_aliases()
META = dict(accession="0000000000-24-000001", form="10-K", filing_date="2024-02-20", report_date="2023-12-31",
            fiscal_year=2023, source_url="https://example.org/doc.htm")


def run(text, supplier="LRCX"):
    return extract_text(text, supplier, META, AL)


def test_sentence_with_named_customer_and_revenue_basis():
    r = run("Samsung Electronics accounted for approximately 22% of our total revenue in fiscal 2023.")
    assert len(r) == 1 and r[0]["customer"] == "005930.KS" and r[0]["pct"] == 22.0 and r[0]["basis"] == "revenue"


def test_table_row_marks_multiple_years():
    r = run("Customers accounted for the following share of net revenue\nTSMC | 23% | 19% | 17%")
    assert r and r[0]["customer"] == "2330.TW" and r[0]["table_row"] and r[0]["pct_all"] == [23.0, 19.0, 17.0]


def test_anonymised_customer_stays_unresolved():
    r = run("Customer A accounted for 31% of net sales.")
    assert r and r[0]["customer"] is None and r[0]["anonymised"]


def test_receivable_basis_and_self_not_special():
    r = run("Apple represented 15% of accounts receivable.")
    assert r[0]["basis"] == "receivable"


def test_no_percentage_no_extract():
    assert run("We sell to Intel and Micron.") == []


def test_alias_longest_match_wins_and_word_boundary():
    h = alias_hits("HP Inc. and Hewlett-Packard; SHP is not HP", AL)
    assert [x[2] for x in h].count("HPQ") >= 2
    assert not any(s == 0 and e == 2 and t == "HPQ" for s, e, t in alias_hits("HP Inc.", AL))  # 'HP Inc.' wins over 'HP'


def test_list_of_names_with_threshold_gives_named_customers_without_share():
    r = run("In fiscal 2025, revenues from Apple, Samsung and Xiaomi each comprised 10% or more of our consolidated revenues.", "QCOM")
    got = {x["customer"] for x in r}
    assert {"AAPL", "005930.KS"} <= got and all(x["pct"] is None and x["threshold"] == 10.0 for x in r)


def test_negation_is_not_an_extract():
    assert run("ASML did not represent 10% or more of sales in 2024.", "ICHR") == []
    assert run("Apple accounted for less than 10% of net revenue.") == []


def test_respectively_pairs_names_and_shares_by_position():
    ex = run("In 2020, our top two customers were Lam Research and Applied Materials, with Lam Research and Applied Materials accounting for approximately 52% and 35% of sales, respectively.", "ICHR")
    got = {e["customer"]: e["pct"] for e in ex if e["pct"] is not None}
    assert got == {"LRCX": 52.0, "AMAT": 35.0}


def test_aggregate_share_is_not_attributed_to_one_customer():
    assert not [e for e in run("In 2023, our top three customers were Applied Materials, Lam Research, and ASML, accounting for a combined 82% of sales.", "ICHR") if e["pct"] is not None]
    assert not run("Customers include Qualcomm and Novatek. In 2015, our top ten customers accounted for 57.3% of our net operating revenues.", "2303.TW")


def test_market_share_of_a_competitor_is_not_a_customer_share():
    assert not run("Taiwan Semiconductor Manufacturing Company, Limited (\u201cTSMC\u201d) at $75.8 billion of revenue in 2022 accounted for more than 50% of the total market.", "GFS")


def test_respectively_with_more_names_than_shares_aligns_from_the_end():
    ex = run("In 2018, our top three customers were Lam Research, Applied Materials, and ASML, with Lam Research and Applied Materials accounting for approximately 56% and 32% of sales, respectively.", "ICHR")
    got = {e["customer"]: e["pct"] for e in ex if e["pct"] is not None}
    assert got == {"LRCX": 56.0, "AMAT": 32.0}
