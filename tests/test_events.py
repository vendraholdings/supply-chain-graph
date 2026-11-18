import numpy as np
import pandas as pd

from scgraph import events as ev


def synthetic(T=1200, K=30, seed=0, effect=0.0, eff_days=None):
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0, 0.01, T)
    semi = 0.7 * mkt + rng.normal(0, 0.008, T)
    beta = rng.uniform(0.6, 1.4, K)
    Y = mkt[:, None] * beta + 0.5 * semi[:, None] + rng.normal(0, 0.012, (T, K))
    F = {k: np.c_[mkt, semi] for k in range(K)}
    return Y, F


def test_day0_rule_by_region():
    assert ev.day0_offset("TW", "US") == 0 and ev.day0_offset("US", "TW") == 1
    assert ev.day0_offset("US", "US") == 1 and ev.day0_offset("TW", "JP") == 1 and ev.day0_offset("JP", "EU") == 0


def test_find_shocks_uses_trailing_window_only():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    r.iloc[300] = 0.2
    r.iloc[302] = 0.2                       # within min_gap of the first
    sh = ev.find_shocks(r, z=6.0)
    assert 300 in sh and 302 not in sh
    # the day itself is excluded from its own trailing window: a lone jump after a flat history is flagged
    flat = pd.Series(np.r_[np.random.default_rng(1).normal(0, 0.001, 100), 0.5, np.zeros(50)])
    assert 100 in ev.find_shocks(flat, window=60)


def test_injected_abnormal_return_is_detected():
    Y, F = synthetic()
    pairs = [(k, 600 + 3 * k, 1) for k in range(30)]
    for k, d0, _ in pairs:
        Y[d0:d0 + 5, k] += 0.01             # +5% over five days
    r = ev.study(Y, F, pairs)
    assert r["n"] == 30 and r["caar"][4] > 0.03 and r["t_kp"] > 3


def test_placebo_size_and_centering():
    Y, F = synthetic(T=1500, K=40, seed=1)
    rng = np.random.default_rng(5)
    p = ev.placebo(Y, F, list(range(40)), ["US"] * 40, n_events=25, draws=400, rng=rng)
    assert 0.02 <= p["rejection_rate"] <= 0.09
    assert abs(p["mean_caar"]) < 2 * p["se_caar"]


def test_cross_correlation_adjustment_shrinks_t():
    s = np.random.default_rng(0).normal(0.3, 1, 40)
    assert abs(ev.bmp(s, 0.3)["t_kp"]) < abs(ev.bmp(s, 0.0)["t_kp"])
    assert ev.bmp(np.array([1.0]))["n"] == 1
