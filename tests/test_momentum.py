import numpy as np
import pandas as pd

from scgraph import momentum as mo
from scgraph.calendars import us_grid

CFG = {"costs_bps": {"oneway": {"US": 10}, "borrow_annual": 50}}


def world(n_sup=12, signal=0.5, seed=0, months=96):
    """Synthetic world: supplier i has customer c_i; customer return in month k leaks into supplier return in month k+1."""
    rng = np.random.default_rng(seed)
    g = us_grid("2014-01-01", "2023-12-31")
    d = g.index
    sup = [f"S{i}" for i in range(n_sup)]
    cus = [f"C{i}" for i in range(n_sup)]
    R = pd.DataFrame(rng.normal(0, 0.01, (len(d), 2 * n_sup)), index=d, columns=sup + cus)
    ym = d.year * 12 + d.month
    mid = pd.Series(ym - ym.min(), index=d)
    # shift: add a monthly customer shock to the supplier's returns spread over the following month
    for i in range(n_sup):
        shock = {}
        for m in np.unique(mid.values):
            days = np.flatnonzero(mid.values == m)
            shock[m] = rng.normal(0, 0.05)
            R.iloc[days, n_sup + i] += shock[m] / len(days)
        for m in np.unique(mid.values)[1:]:
            days = np.flatnonzero(mid.values == m)
            R.iloc[days, i] += signal * shock[m - 1] / len(days)
    edges = pd.DataFrame(dict(supplier=sup, customer=cus, pct_revenue=20.0, method="curated",
                              filing_date="2014-02-10", available_date="2014-02-11", valid_to="2024-12-31"))
    return R, edges, {t: "T" for t in R.columns}, {t: "US" for t in R.columns}


def test_signal_world_earns_positive_sharpe_and_noise_does_not():
    R, e, tier, ctry = world(signal=0.6)
    eng = mo.Engine(R, ctry, CFG)
    s = mo.Spec()
    m = eng.run(mo.edge_arrays(e, eng.tickers, tier), s)
    assert mo.sharpe(m) > 1.0 and len(m) > 90
    R0, e0, tier, ctry = world(signal=0.0, seed=3)
    m0 = mo.Engine(R0, ctry, CFG).run(mo.edge_arrays(e0, list(R0.columns), tier), s)
    assert abs(mo.sharpe(m0)) < 1.2


def test_no_look_ahead_deleting_future_filings_leaves_past_unchanged():
    R, e, tier, ctry = world()
    e2 = pd.concat([e.assign(filing_date="2018-02-10", available_date="2018-02-12", pct_revenue=25.0, supplier=[f"S{i}" for i in range(6)] + ["S0"] * 6, customer=[f"C{i}" for i in range(6)] + ["C1"] * 6)], ignore_index=True)
    full = pd.concat([e, e2], ignore_index=True)
    eng = mo.Engine(R, ctry, CFG)
    a = eng.run(mo.edge_arrays(full, eng.tickers, tier), mo.Spec(weighting="revenue"))
    b = eng.run(mo.edge_arrays(e, eng.tickers, tier), mo.Spec(weighting="revenue"))
    cut = pd.Period("2018-01", "M")
    pd.testing.assert_series_equal(a[a.index < cut], b[b.index < cut], check_names=False)


def test_delaying_availability_never_raises_information_set():
    R, e, tier, ctry = world()
    eng = mo.Engine(R, ctry, CFG)
    base = eng.info_set(mo.edge_arrays(e, eng.tickers, tier), mo.Spec())
    late = e.assign(available_date=pd.to_datetime(e["available_date"]) + pd.Timedelta(days=400))
    delayed = eng.info_set(mo.edge_arrays(late, eng.tickers, tier), mo.Spec())
    assert (delayed <= base).all() and delayed.sum() < base.sum()


def test_edge_not_used_on_its_available_day():
    R, e, tier, ctry = world()
    eng = mo.Engine(R, ctry, CFG)
    E = mo.edge_arrays(e.assign(available_date="2014-02-28"), eng.tickers, tier)
    assert eng.info_set(E, mo.Spec())[0] == 0       # month end 2014-01 is before it
    assert eng.info_set(E, mo.Spec())[1] == 0       # signal day equal to available_date is not strictly after


def test_costs_reduce_returns_and_zero_cost_matches_gross():
    R, e, tier, ctry = world()
    eng = mo.Engine(R, ctry, CFG)
    E = mo.edge_arrays(e, eng.tickers, tier)
    net, gross = eng.run(E, mo.Spec(), daily=True)
    assert net.sum() < gross.sum()
    m0 = eng.run(E, mo.Spec(), cost_mult=0.0)
    assert np.isclose((1 + m0).prod(), (1 + gross).prod(), rtol=1e-6)


def test_rewiring_preserves_degrees_and_grid_has_48_specs():
    R, e, tier, ctry = world()
    E = mo.edge_arrays(e, list(R.columns), tier)
    rng = np.random.default_rng(1)
    E2 = mo.rewire(E, np.zeros(len(E.sup), int), rng)
    assert sorted(E2.cus) == sorted(E.cus) and (E2.sup == E.sup).all() and (E2.cus != E.cus).any()
    assert len(mo.grid_specs()) == 48 and mo.Spec() in mo.grid_specs()
