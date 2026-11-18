import numpy as np
import pandas as pd
import pytest

from scgraph import api, edges, graph

UNI = pd.DataFrame(dict(ticker=["A", "B", "C", "D"], name=["a", "b", "c", "d"], tier=["T05", "T07", "T10", "T11"],
                        country=["US"] * 4, currency=["USD"] * 4))


def E(rows):
    d = pd.DataFrame(rows, columns=["supplier", "customer", "pct_revenue", "fiscal_year", "filing_date"])
    d["source_type"], d["source_url"], d["source_note"] = "10-K", "https://example.org/x", ""
    d["method"], d["confidence"], d["reviewed"] = "curated", "high", True
    return edges.finalize(d)


@pytest.fixture
def setup():
    e = E([("A", "B", 20, 2022, "2023-02-10"), ("B", "C", None, 2022, "2023-02-12"), ("C", "D", 10, 2022, "2023-02-15"),
           ("A", "B", 25, 2023, "2024-02-10")])
    G = graph.build(e, UNI)
    return e, G


def test_latest_pair_and_median_weight(setup):
    e, G = setup
    assert G["A"]["B"]["weight"] == 25 and G["A"]["B"]["disclosed"]
    assert G["B"]["C"]["disclosed"] is False and G["B"]["C"]["weight"] == 1.0   # no disclosed share in tier T07


def test_centrality_and_flow(setup):
    e, G = setup
    c = graph.centrality(G).set_index("ticker")
    assert c.loc["C", "betweenness"] > 0 and c.loc["A", "out_strength"] == 25
    f = graph.tier_flow(G)
    assert f.loc["T05", "T07"] == 25 and f.values.sum() == sum(d["weight"] for *_, d in G.edges(data=True))


def test_point_in_time_graph(setup):
    e, _ = setup
    assert graph.build(e, UNI, as_of="2023-06-01")["A"]["B"]["weight"] == 20
    assert graph.build(e, UNI, as_of="2023-01-01").number_of_edges() == 0


def test_neighbours_depth_direction_and_sources(setup, tmp_path):
    e, G = setup
    p = tmp_path / "g.json"
    doc = graph.export_json(G, e, pd.Series({"A": 1e9}), "2026-10-30", p)
    r = api.neighbours("B", depth=2, graph=doc)
    got = {(x["ticker"], x["relation"], x["hops"]) for x in r}
    assert got == {("A", "supplier", 1), ("C", "customer", 1), ("D", "customer", 2)}
    assert all(x["sources"] and x["sources"][0]["url"].startswith("https://") for x in r)
    assert [x["ticker"] for x in api.neighbours("B", direction="suppliers", graph=doc)] == ["A"]
    assert api.neighbours("B", as_of="2020-01-01", graph=doc) == []
    assert api.load(p)["nodes"][0]["ticker"] == "A"


def test_rank_corr_and_tier_index():
    cent = pd.DataFrame(dict(ticker=list("ABCDE"), pagerank=[.1, .2, .3, .4, .5]))
    assert graph.rank_corr_with_mcap(cent, pd.Series(dict(A=1, B=2, C=3, D=4, E=5)))["rho"] == pytest.approx(1.0)
    R = pd.DataFrame(np.random.default_rng(0).normal(0, .01, (5, 3)), columns=["A", "B", "C"])
    ti = graph.tier_index(R, UNI[UNI.ticker.isin(["A", "B", "C"])].assign(tier="T05"), pd.Series(dict(A=1., B=1., C=2.)))
    assert np.isclose(ti["T05"].iloc[0], (R.iloc[0, 0] + R.iloc[0, 1] + 2 * R.iloc[0, 2]) / 4)
