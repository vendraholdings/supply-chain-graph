"""Customer-momentum strategy (Cohen and Frazzini 2008 design) on a point-in-time supplier-customer graph.

Timing. Signals are formed at the US close of the last grid day of a month, using customer returns stamped at or
before that day and edges whose `available_date` is strictly before it. Positions are filled at each listing's next
close after the signal, so a listing earns returns stamped in [i+2, i_next+1] where i is the signal day.
Weights are held at their targets (no drift); costs are charged on the fill day on the traded weight.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

PPY = 12.0
MIN_OBS = 10  # a customer needs this many non-missing days in a month to give a monthly return


@dataclass(frozen=True)
class Spec:
    lag: int = 0
    quantiles: int = 3
    weighting: str = "equal"          # equal | revenue
    edge_set: str = "curated+rule"    # curated | curated+rule
    holding: int = 1

    def label(self) -> str:
        return f"lag{self.lag}_q{self.quantiles}_{self.weighting}_{self.edge_set}_h{self.holding}"


def grid_specs() -> list[Spec]:
    """Declared grid of variants (N = 48), including the primary specification (lag 0, terciles, equal, curated+rule, 1 month)."""
    return [Spec(l, q, w, e, h) for l, q, w, e, h in itertools.product((0, 1), (2, 3), ("equal", "revenue"),
                                                                       ("curated", "curated+rule"), (1, 3, 6))]


@dataclass
class EdgeArrays:
    sup: np.ndarray
    cus: np.ndarray
    av: np.ndarray        # available_date, days since epoch
    vt: np.ndarray        # valid_to
    filed: np.ndarray     # filing_date
    w_rev: np.ndarray     # revenue-share weight (disclosed, else tier median, else 1)
    curated: np.ndarray   # bool

    def subset(self, mask) -> "EdgeArrays":
        return EdgeArrays(*(getattr(self, f)[mask] for f in ("sup", "cus", "av", "vt", "filed", "w_rev", "curated")))


def edge_arrays(edges: pd.DataFrame, tickers: list[str], tier: dict[str, str]) -> EdgeArrays:
    pos = {t: i for i, t in enumerate(tickers)}
    e = edges[edges["supplier"].isin(pos) & edges["customer"].isin(pos)].reset_index(drop=True)
    day = lambda s: pd.to_datetime(e[s]).to_numpy("datetime64[D]").astype("int64")  # noqa: E731
    pct = pd.to_numeric(e["pct_revenue"], errors="coerce")
    med = pct.groupby(e["supplier"].map(tier)).transform("median")
    w = pct.fillna(med).fillna(1.0).clip(lower=0.1).to_numpy(float)
    return EdgeArrays(e["supplier"].map(pos).to_numpy(), e["customer"].map(pos).to_numpy(), day("available_date"),
                      day("valid_to"), day("filing_date"), w, (e["method"] == "curated").to_numpy())


class Engine:
    def __init__(self, R: pd.DataFrame, country: dict[str, str], cfg: dict):
        self.R = R
        self.tickers = list(R.columns)
        self.dates = R.index
        self.A = R.to_numpy(float)
        self.day = self.dates.to_numpy("datetime64[D]").astype("int64")
        ym = self.dates.year * 12 + self.dates.month
        last = np.r_[ym[1:] != ym[:-1], True]
        self.me = np.flatnonzero(last)                   # grid position of each month's last day
        self.month_id = np.cumsum(np.r_[0, (ym[1:] != ym[:-1]).astype(int)])
        self.cust_ret = self._month_returns()
        ow = cfg["costs_bps"]["oneway"]
        self.bps = np.array([ow.get(country.get(t, "US"), 10.0) for t in self.tickers], float)
        self.borrow = cfg["costs_bps"]["borrow_annual"] / 1e4 / 252

    def _month_returns(self) -> np.ndarray:
        out = np.full((len(self.me), self.A.shape[1]), np.nan)
        start = 0
        for k, end in enumerate(self.me):
            blk = self.A[start:end + 1]
            n = np.sum(~np.isnan(blk), axis=0)
            comp = np.nanprod(1.0 + blk, axis=0) - 1.0
            comp[n < MIN_OBS] = np.nan
            out[k] = comp
            start = end + 1
        return out

    def _edges_at(self, E: EdgeArrays, k: int, spec: Spec):
        t = self.day[self.me[k]]
        m = (E.av < t) & (E.vt > t)
        if spec.edge_set == "curated":
            m &= E.curated
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            return idx
        key = E.sup[idx].astype(np.int64) * len(self.tickers) + E.cus[idx]
        order = np.lexsort((-E.filed[idx], key))        # latest filing first within a pair
        idx, key = idx[order], key[order]
        first = np.r_[True, key[1:] != key[:-1]]
        return idx[first]

    def info_set(self, E: EdgeArrays, spec: Spec) -> np.ndarray:
        return np.array([len(self._edges_at(E, k, spec)) for k in range(len(self.me))])

    def signals(self, E: EdgeArrays, k: int, spec: Spec) -> dict[int, float]:
        idx = self._edges_at(E, k, spec)
        kk = k - spec.lag
        if len(idx) == 0 or kk < 0:
            return {}
        r = self.cust_ret[kk][E.cus[idx]]
        w = E.w_rev[idx] if spec.weighting == "revenue" else np.ones(len(idx))
        ok = ~np.isnan(r)
        sig: dict[int, float] = {}
        for s in np.unique(E.sup[idx][ok]):
            sel = ok & (E.sup[idx] == s)
            sig[int(s)] = float(np.sum(w[sel] * r[sel]) / np.sum(w[sel]))
        return sig

    def cohort(self, sig: dict[int, float], q: int) -> np.ndarray | None:
        n = len(sig)
        if n < 2 * q:
            return None
        ids = np.array(list(sig))
        order = ids[np.argsort([sig[i] for i in ids], kind="stable")]
        chunks = np.array_split(order, q)
        w = np.zeros(len(self.tickers))
        w[chunks[-1]] = 1.0 / len(chunks[-1])
        w[chunks[0]] = -1.0 / len(chunks[0])
        return w

    def weights(self, E: EdgeArrays, spec: Spec) -> np.ndarray:
        W = np.zeros_like(self.A)
        T = len(self.dates)
        for k in range(len(self.me) - 1):
            w = self.cohort(self.signals(E, k, spec), spec.quantiles)
            if w is None:
                continue
            a = self.me[k] + 2
            b = min(self.me[min(k + spec.holding, len(self.me) - 1)] + 1, T - 1)
            if a <= b:
                W[a:b + 1] += w / spec.holding
        return W

    def run(self, E: EdgeArrays, spec: Spec, cost_mult: float = 1.0, daily: bool = False):
        W = self.weights(E, spec)
        gross = np.nansum(W * self.A, axis=1)
        dW = np.abs(np.diff(W, axis=0, prepend=0.0))
        cost = np.zeros(len(self.dates))
        cost[:-1] = (dW[1:] * self.bps).sum(axis=1) / 1e4 * cost_mult     # trade executes on the day before the new weights apply
        cost += np.where(W < 0, -W, 0).sum(axis=1) * self.borrow * cost_mult
        net = gross - cost
        active = np.flatnonzero(np.abs(W).sum(axis=1) > 0)
        if len(active) == 0:
            return pd.Series(dtype=float)
        lo, hi = active[0], min(active[-1] + 1, len(net) - 1)
        s = pd.Series(net[lo:hi + 1], index=self.dates[lo:hi + 1])
        if daily:
            return s, pd.Series(gross[lo:hi + 1], index=self.dates[lo:hi + 1])
        m = (1 + s).groupby([s.index.year, s.index.month]).prod() - 1
        m.index = pd.PeriodIndex([f"{y}-{mo:02d}" for y, mo in m.index], freq="M")
        return m


def sharpe(m: pd.Series | np.ndarray) -> float:
    a = np.asarray(m, float)
    return float(np.sqrt(PPY) * a.mean() / a.std(ddof=1)) if len(a) > 2 and a.std(ddof=1) > 0 else float("nan")


def rewire(E: EdgeArrays, group: np.ndarray, rng: np.random.Generator, n_swaps: int | None = None) -> EdgeArrays:
    """Degree-preserving rewiring within groups: swap customers between two edges of the same (supplier tier, customer tier)."""
    cus = E.cus.copy()
    n = len(cus)
    n_swaps = n_swaps or 10 * n
    by = {g: np.flatnonzero(group == g) for g in np.unique(group)}
    keys = [g for g, ix in by.items() if len(ix) >= 2]
    if not keys:
        return E
    for _ in range(n_swaps):
        ix = by[keys[rng.integers(len(keys))]]
        i, j = rng.choice(ix, 2, replace=False)
        if E.sup[i] == cus[j] or E.sup[j] == cus[i] or cus[i] == cus[j]:
            continue
        cus[i], cus[j] = cus[j], cus[i]
    return EdgeArrays(E.sup, cus, E.av, E.vt, E.filed, E.w_rev, E.curated)


def newey_west_alpha(m: pd.Series, factors: pd.DataFrame, lags: int = 3) -> dict:
    import statsmodels.api as sm

    df = pd.concat([m.rename("y"), factors], axis=1, join="inner").dropna()
    X = sm.add_constant(df[factors.columns])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ci = res.conf_int().loc["const"]
    return dict(alpha_monthly=float(res.params["const"]), alpha_annual=float(res.params["const"] * 12),
                t=float(res.tvalues["const"]), ci95_annual=[float(ci[0] * 12), float(ci[1] * 12)], n=int(len(df)),
                betas={c: float(res.params[c]) for c in factors.columns})


def fama_macbeth(eng: Engine, E: EdgeArrays, spec: Spec, min_n: int = 8) -> dict:
    """Monthly cross-sectional regressions of next-month supplier return on the customer signal and the supplier's own lagged return."""
    betas, own = [], []
    for k in range(len(eng.me) - 1):
        sig = eng.signals(E, k, spec)
        if len(sig) < min_n or k < 1:
            continue
        ids = np.array(list(sig))
        y = eng.cust_ret[k + 1][ids]
        x1 = np.array([sig[i] for i in ids])
        x2 = eng.cust_ret[k][ids]
        ok = ~(np.isnan(y) | np.isnan(x2))
        if ok.sum() < min_n:
            continue
        X = np.c_[np.ones(ok.sum()), x1[ok], x2[ok]]
        b = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        betas.append(b[1])
        own.append(b[2])
    b = np.array(betas)
    if len(b) < 6:
        return dict(n_months=int(len(b)), slope=None, t=None)
    import statsmodels.api as sm

    res = sm.OLS(b, np.ones(len(b))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return dict(n_months=int(len(b)), slope=float(b.mean()), t=float(res.tvalues[0]), own_lag_slope=float(np.mean(own)))
