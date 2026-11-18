"""Event study on the US grid: shock events, market-model abnormal returns, time-zone-aware day 0, BMP test with the
Kolari-Pynnonen cross-correlation adjustment, and a random-date placebo.

Day 0 of an affected listing is its first session closing strictly after the event timestamp. On a grid day the closes
occur in the order Asia, Europe, US, so a listing in a region that closes later than the event node reacts on the same grid
day; one in the same or an earlier-closing region reacts on the next grid day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGION_RANK = {"JP": 0, "TW": 0, "KR": 0, "HK": 0, "EU": 1, "US": 2}


def day0_offset(event_country: str, listing_country: str) -> int:
    """Grid-day offset (0 or 1) of the listing's day 0 relative to the event node's own grid day."""
    return 0 if REGION_RANK[listing_country] > REGION_RANK[event_country] else 1


def find_shocks(r: pd.Series, z: float = 3.0, window: int = 60, min_gap: int = 5) -> list[int]:
    """Positions where |r| exceeds z trailing-window standard deviations (trailing window excludes the day itself)."""
    x = r.to_numpy(float)
    sd = r.rolling(window, min_periods=window // 2).std().shift(1).to_numpy()
    hit = np.flatnonzero(np.abs(np.nan_to_num(x)) > z * np.nan_to_num(sd, nan=np.inf))
    out: list[int] = []
    for h in hit:
        if not out or h - out[-1] >= min_gap:
            out.append(int(h))
    return out


@dataclass
class Fit:
    beta: np.ndarray
    sigma: float
    resid: np.ndarray
    n: int


def design(F: np.ndarray) -> np.ndarray:
    """Regressors: constant, factors at t, local factor lag and lead (Dimson 1979 terms) for the first column."""
    lag = np.vstack([np.full((1, F.shape[1]), np.nan), F[:-1]])
    lead = np.vstack([F[1:], np.full((1, F.shape[1]), np.nan)])
    return np.column_stack([np.ones(len(F)), F, lag[:, [0]], lead[:, [0]]])


def fit_market_model(y: np.ndarray, Xd: np.ndarray, lo: int, hi: int, min_obs: int = 100) -> Fit | None:
    s = slice(max(lo, 0), max(hi + 1, 0))
    ys, Xs = y[s], Xd[s]
    ok = ~(np.isnan(ys) | np.isnan(Xs).any(axis=1))
    if ok.sum() < min_obs:
        return None
    b = np.linalg.lstsq(Xs[ok], ys[ok], rcond=None)[0]
    res = ys[ok] - Xs[ok] @ b
    return Fit(b, float(res.std(ddof=Xs.shape[1])), res, int(ok.sum()))


def abnormal(y: np.ndarray, Xd: np.ndarray, fit: Fit) -> np.ndarray:
    return y - Xd @ fit.beta


def car_path(ar: np.ndarray, d0: int, horizon: int) -> np.ndarray | None:
    """Cumulative abnormal returns for days 0..horizon (missing days contribute zero abnormal return but are counted)."""
    seg = ar[d0:d0 + horizon + 1]
    if len(seg) < horizon + 1:
        return None
    return np.cumsum(np.nan_to_num(seg))


def bmp(scar: np.ndarray, mean_corr: float = 0.0) -> dict:
    """Boehmer-Musumeci-Poulsen t-statistic on standardised CARs, with the Kolari-Pynnonen adjustment for mean cross-correlation."""
    n = len(scar)
    if n < 3:
        return dict(n=n, t=np.nan, t_kp=np.nan)
    t = scar.mean() / (scar.std(ddof=1) / np.sqrt(n))
    adj = np.sqrt((1 - mean_corr) / (1 + (n - 1) * mean_corr)) if mean_corr > -1 / (n - 1) else 1.0
    return dict(n=n, t=float(t), t_kp=float(t * adj))


def mean_cross_corr(resids: list[np.ndarray]) -> float:
    m = min(len(r) for r in resids)
    if len(resids) < 3 or m < 30:
        return 0.0
    M = np.column_stack([r[-m:] for r in resids])
    C = np.corrcoef(M, rowvar=False)
    iu = np.triu_indices_from(C, 1)
    return float(np.nanmean(C[iu]))


def study(Y: np.ndarray, F: dict[int, np.ndarray], pairs: list[tuple[int, int, int]], horizon: int = 20,
          est: tuple[int, int] = (-250, -30)) -> dict:
    """pairs: (listing column, day-0 grid position, sign +1/-1 of the triggering shock). Y: grid days x listings.
    F[col] is the factor matrix (market first) for that listing. Returns standardised CARs, raw CAR paths and the test."""
    scars, paths, resids = [], [], []
    for col, d0, sgn in pairs:
        Xd = design(F[col])
        fit = fit_market_model(Y[:, col], Xd, d0 + est[0], d0 + est[1])
        if fit is None:
            continue
        cp = car_path(abnormal(Y[:, col], Xd, fit), d0, horizon)
        if cp is None:
            continue
        cp = cp * sgn
        paths.append(cp)
        scars.append(cp[-1] / (fit.sigma * np.sqrt(horizon + 1)))
        resids.append(fit.resid)
    if not paths:
        return dict(n=0, caar=None, t=np.nan, t_kp=np.nan, paths=np.empty((0, horizon + 1)))
    P = np.array(paths)
    res = bmp(np.array(scars), mean_cross_corr(resids))
    return dict(**res, caar=P.mean(0), paths=P)


def placebo(Y, F, listing_cols: list[int], countries: list[str], n_events: int, draws: int, rng: np.random.Generator,
            horizon: int = 20, est=(-250, -30), min_pos: int = 260) -> dict:
    """Random event dates for random listings (same count as the real event set): rejection rate of the 5% BMP test and mean CAAR."""
    T = Y.shape[0]
    rej, caar_end, tstat = 0, [], []
    ok = 0
    for _ in range(draws):
        cols = rng.choice(listing_cols, n_events)
        dates = rng.integers(min_pos, T - horizon - 2, n_events)
        sg = rng.choice([-1, 1], n_events)
        r = study(Y, F, [(int(c), int(d), int(g)) for c, d, g in zip(cols, dates, sg)], horizon, est)
        if r["n"] < 3:
            continue
        ok += 1
        tstat.append(r["t_kp"])
        caar_end.append(r["caar"][-1])
        rej += abs(r["t_kp"]) > 1.96
    c = np.array(caar_end)
    return dict(draws=ok, rejection_rate=rej / max(ok, 1), mean_caar=float(c.mean()) if ok else None,
                se_caar=float(c.std(ddof=1) / np.sqrt(ok)) if ok > 1 else None, caar_draws=c, t_draws=np.array(tstat))
