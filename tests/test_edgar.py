import json

import pytest

from scgraph import config
from scgraph.edgar import EdgarClient, RateLimiter, html_to_text

REQUIRED_UA = "vendraholdings research 263115861+vendraholdings@users.noreply.github.com"


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.log = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def run(rate, hard, n):
    c = FakeClock()
    lim = RateLimiter(rate, hard, clock=c.now, sleep=c.sleep)
    for _ in range(n):
        lim.acquire()
        c.log.append(c.t)
    return c.log


@pytest.mark.parametrize("rate,hard", [(5, 10), (50, 10), (1, 10)])
def test_never_more_than_ten_in_any_second(rate, hard):
    log = run(rate, hard, 200)
    for i, t in enumerate(log):
        assert sum(1 for u in log if t <= u < t + 1.0) <= 10


def test_average_rate_is_capped():
    log = run(5, 10, 101)
    assert log[-1] - log[0] >= 100 / 5 - 1e-6


def test_user_agent_matches_required_string_and_is_sent(tmp_path):
    cfg = config.load()
    assert cfg["edgar_user_agent"] == REQUIRED_UA
    seen = {}

    class R:
        status_code, text = 200, json.dumps({"x": 1})

    class S:
        def get(self, url, headers=None, timeout=None):
            seen.update(headers)
            return R()

    c = EdgarClient(cfg, cache_dir=tmp_path, session=S(), limiter=RateLimiter(1000))
    assert c.get("https://data.sec.gov/x") == '{"x": 1}'
    assert seen["User-Agent"] == REQUIRED_UA
    c.get("https://data.sec.gov/x")  # second call served from the cache
    assert c.n_requests == 1


def test_http_error_is_raised(tmp_path):
    class R:
        status_code, text = 403, ""

    class S:
        def get(self, *a, **k):
            return R()

    with pytest.raises(RuntimeError):
        EdgarClient(config.load(), cache_dir=tmp_path, session=S(), limiter=RateLimiter(1000)).get("https://data.sec.gov/x")


def test_html_to_text_keeps_rows():
    t = html_to_text("<html><body><p>Customer A</p><table><tr><td>Apple</td><td>23%</td></tr></table><script>x()</script></body></html>")
    assert "Customer A" in t and "Apple | 23%" in t.replace("\n", " ") and "x()" not in t
