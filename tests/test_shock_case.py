import json

from scgraph import config


def test_shock_case_follows_the_stated_rule():
    d = json.loads((config.ROOT / "results" / "shock_case.json").read_text())
    assert d["event"]["tier"] in ("T05", "T08")
    assert len(d["nodes"]) >= 5
    assert all(len(v["car"]) == 11 for v in d["nodes"].values())
    assert d["event"]["ticker"] not in d["nodes"]
