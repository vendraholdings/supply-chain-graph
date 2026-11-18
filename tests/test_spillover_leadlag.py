import numpy as np

from scgraph import leadlag as ll
from scgraph import spillover as sp


def simulate(A, S, T, seed=0):
    rng = np.random.default_rng(seed)
    K = A.shape[0]
    L = np.linalg.cholesky(S)
    X = np.zeros((T, K))
    e = rng.standard_normal((T, K)) @ L.T
    for t in range(1, T):
        X[t] = A @ X[t - 1] + e[t]
    return X


A = np.array([[0.3, 0.1, 0.0], [0.0, 0.2, 0.15], [0.1, 0.0, 0.25]])
S = np.array([[1.0, 0.3, 0.1], [0.3, 1.2, 0.2], [0.1, 0.2, 0.8]])


def test_gfevd_rows_sum_to_one_and_matches_known_matrix():
    true = sp.gfevd(sp.ma_coefficients([A], 10), S)
    assert np.allclose(true.sum(1), 1)
    est = sp.spillover(simulate(A, S, 400_000), H=10, max_lag=3)["theta"]
    assert np.abs(est - true).max() < 0.02


def test_independent_system_has_no_spillover():
    r = sp.spillover(simulate(np.diag([0.2, 0.1, 0.3]), np.eye(3), 100_000), H=10)
    assert r["total"] < 1.0 and np.abs(r["net"]).max() < 1.0


def test_net_sums_to_zero_and_total_in_range():
    th = sp.gfevd(sp.ma_coefficients([A], 10), S)
    s = sp.summary(th)
    assert abs(s["net"].sum()) < 1e-9 and 0 < s["total"] < 100


def test_bic_picks_true_lag_and_stability_flag():
    X = simulate(A, S, 50_000)
    assert sp.fit_var(X, 3)[2] == 1 and sp.is_stable([A]) and not sp.is_stable([np.eye(3) * 1.1])


def test_xcorr_sign_convention_and_bootstrap_interval():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(3000)
    y = np.r_[0, x[:-1]] * 0.8 + 0.6 * rng.standard_normal(3000)   # y(t) = x(t-1): x leads y by one
    X = np.c_[x, y]
    C = ll.xcorr(X)
    k = list(ll.LAGS).index(1)
    assert C[k, 0, 1] > 0.7 and abs(C[list(ll.LAGS).index(-1), 0, 1]) < 0.1
    lo, hi = ll.bootstrap_ci(X, reps=60, seed=2)
    assert lo[k, 0, 1] > 0.6 and hi[list(ll.LAGS).index(3), 0, 1] > 0 > lo[list(ll.LAGS).index(3), 0, 1]
