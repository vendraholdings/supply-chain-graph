"""Lead-lag cross-correlations between tier indices on the US grid, with stationary-bootstrap intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = np.arange(-5, 6)


def xcorr(X: np.ndarray, lags=LAGS) -> np.ndarray:
    """C[l, i, j] = corr(X_i(t), X_j(t + lag_l)): positive lag means i leads j."""
    T, K = X.shape
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    out = np.zeros((len(lags), K, K))
    for a, lg in enumerate(lags):
        if lg >= 0:
            out[a] = Z[: T - lg].T @ Z[lg:] / (T - lg - 1)
        else:
            out[a] = Z[-lg:].T @ Z[: T + lg] / (T + lg - 1)
    return out


def stationary_bootstrap_idx(T: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty(T, dtype=np.int64)
    idx[0] = rng.integers(T)
    new = rng.random(T) < p
    starts = rng.integers(T, size=T)
    for t in range(1, T):
        idx[t] = starts[t] if new[t] else (idx[t - 1] + 1) % T
    return idx


def bootstrap_ci(X: np.ndarray, reps: int = 300, mean_block: float = 10.0, seed: int = 0, lags=LAGS):
    """Pointwise 95% intervals for every (lag, i, j) correlation; rows with missing data are dropped first."""
    X = X[~np.isnan(X).any(axis=1)]
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, len(lags)) + (X.shape[1],) * 2)
    for r in range(reps):
        draws[r] = xcorr(X[stationary_bootstrap_idx(len(X), mean_block, rng)], lags)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def table(X: pd.DataFrame, lo: np.ndarray, hi: np.ndarray, lags=LAGS) -> pd.DataFrame:
    C = xcorr(X.dropna().to_numpy(), lags)
    rows = []
    for a, lg in enumerate(lags):
        for i, ci in enumerate(X.columns):
            for j, cj in enumerate(X.columns):
                if i != j:
                    rows.append(dict(lag=int(lg), leader=ci, follower=cj, corr=C[a, i, j], lo=lo[a, i, j], hi=hi[a, i, j],
                                     significant=bool(lo[a, i, j] > 0 or hi[a, i, j] < 0)))
    return pd.DataFrame(rows)
