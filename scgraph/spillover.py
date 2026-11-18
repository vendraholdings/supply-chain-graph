"""Diebold-Yilmaz (2012) spillovers from a generalised forecast-error variance decomposition (Pesaran and Shin 1998)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ma_coefficients(A_list: list[np.ndarray], H: int) -> np.ndarray:
    """MA(infinity) matrices Phi_0..Phi_{H-1} of a VAR(p) with coefficient matrices A_1..A_p."""
    K = A_list[0].shape[0]
    Phi = [np.eye(K)]
    for h in range(1, H):
        Phi.append(sum(A_list[j] @ Phi[h - 1 - j] for j in range(min(len(A_list), h))))
    return np.array(Phi)


def gfevd(Phi: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Row-normalised generalised FEVD. theta[i, j] = share of variable i's forecast-error variance due to shocks to j."""
    K = Sigma.shape[0]
    num = np.zeros((K, K))
    den = np.zeros(K)
    for P in Phi:
        PS = P @ Sigma
        num += PS**2 / np.diag(Sigma)[None, :]
        den += np.einsum("ij,jk,ik->i", P, Sigma, P)
    theta = num / den[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


def fit_var(X: np.ndarray, max_lag: int = 3) -> tuple[list[np.ndarray], np.ndarray, int]:
    """OLS VAR(p) with intercept, p chosen by BIC over 1..max_lag (common estimation sample). Returns (A_list, Sigma_u, p)."""
    T, K = X.shape
    best = None
    for p in range(1, max_lag + 1):
        Y = X[max_lag:]
        Z = np.hstack([np.ones((T - max_lag, 1))] + [X[max_lag - j:T - j] for j in range(1, p + 1)])
        B = np.linalg.lstsq(Z, Y, rcond=None)[0]
        U = Y - Z @ B
        S = U.T @ U / (len(Y))
        bic = np.log(np.linalg.det(S)) + np.log(len(Y)) * (K * K * p) / len(Y)
        if best is None or bic < best[0]:
            Sdf = U.T @ U / (len(Y) - Z.shape[1])
            best = (bic, [B[1 + K * j:1 + K * (j + 1)].T for j in range(p)], Sdf, p)
    return best[1], best[2], best[3]


def is_stable(A_list: list[np.ndarray]) -> bool:
    K, p = A_list[0].shape[0], len(A_list)
    C = np.zeros((K * p, K * p))
    C[:K] = np.hstack(A_list)
    if p > 1:
        C[K:, :-K] = np.eye(K * (p - 1))
    return bool(np.max(np.abs(np.linalg.eigvals(C))) < 1.0)


def spillover(X: np.ndarray, H: int = 10, max_lag: int = 3) -> dict:
    A, S, p = fit_var(X, max_lag)
    theta = gfevd(ma_coefficients(A, H), S)
    return dict(theta=theta, lag=p, stable=is_stable(A), **summary(theta))


def summary(theta: np.ndarray) -> dict:
    K = theta.shape[0]
    off = theta - np.diag(np.diag(theta))
    frm = off.sum(axis=1) * 100            # received from others
    to = off.sum(axis=0) * 100             # transmitted to others
    return dict(total=float(off.sum() / K * 100), from_others=frm, to_others=to, net=to - frm)


def residualise(R: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """Remove the contemporaneous linear market component from each column (OLS on the common sample)."""
    out = {}
    for c in R.columns:
        d = pd.concat([R[c], market], axis=1).dropna()
        x = d.iloc[:, 1].to_numpy()
        y = d.iloc[:, 0].to_numpy()
        b = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
        out[c] = R[c] - (y.mean() - b * x.mean()) - b * market
    return pd.DataFrame(out)


MERGED = {"M1 materials": ["T01", "T02", "T03", "T04"], "M2 equipment": ["T05"], "M3 design": ["T06", "T10"],
          "M4 manufacturing": ["T07", "T09"], "M5 memory": ["T08"], "M6 end markets": ["T11"]}


def merge_tiers(ti: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: ti[v].mean(axis=1) for k, v in MERGED.items() if all(x in ti.columns for x in v)})


def rolling(X: pd.DataFrame, window: int = 250, step: int = 5, H: int = 10, max_lag: int = 3) -> pd.DataFrame:
    rows = []
    for end in range(window, len(X) + 1, step):
        blk = X.iloc[end - window:end].dropna()
        if len(blk) < window * 0.9:
            continue
        r = spillover(blk.to_numpy(), H, max_lag)
        rows.append(dict(date=X.index[end - 1], total=r["total"], stable=r["stable"], **{f"net_{c}": v for c, v in zip(X.columns, r["net"])}))
    return pd.DataFrame(rows).set_index("date")
