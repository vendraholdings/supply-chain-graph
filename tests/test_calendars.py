import numpy as np
import pandas as pd

from scgraph import calendars as cal
from scgraph.fx import to_usd


def grid():
    return cal.us_grid("2024-03-01", "2024-03-29")


def series(dates, vals):
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


def test_asian_shock_lands_on_us_day_same_date():
    g = grid()
    d = pd.date_range("2024-03-04", "2024-03-08")  # Mon-Fri, all Tokyo sessions
    px = series(d, [100, 100, 110, 110, 110])      # +10% on Wed 6 March Tokyo close
    r = cal.to_grid_returns(px, "XTKS", g)
    assert abs(r.loc["2024-03-06"] - 0.10) < 1e-9
    assert r.loc["2024-03-07"] == 0 or np.isclose(r.loc["2024-03-07"], 0)


def test_us_after_close_shock_reaches_asia_next_day():
    g = grid()
    # US shock after the close on Tue 5 March is reflected in Tokyo's Wed 6 March close
    d = pd.date_range("2024-03-04", "2024-03-08")
    px = series(d, [100, 100, 100, 110, 110])      # Tokyo moves on Thu 7 March here
    r = cal.to_grid_returns(px, "XTKS", g)
    assert abs(r.loc["2024-03-07"] - 0.10) < 1e-9  # Tokyo Thursday lands on US Thursday
    # a US listing's own return stays on its own day
    us = series(pd.bdate_range("2024-03-04", "2024-03-08"), [100, 100, 110, 110, 110])
    ru = cal.to_grid_returns(us, "XNYS", g)
    assert abs(ru.loc["2024-03-06"] - 0.10) < 1e-9
    # Tokyo reacting to the US Wednesday close does so on Thursday Tokyo, which is US grid Thursday: a day later than the US day
    tk = series(d, [100, 100, 100, 100, 100])
    tk.loc["2024-03-07"] = 110
    r2 = cal.to_grid_returns(tk, "XTKS", g)
    assert r2.loc["2024-03-07"] > 0.09 and r2.loc["2024-03-06"] == 0


def test_holidays_are_nan_not_zero_and_gap_is_compounded():
    g = grid()
    # Good Friday 29 March 2024: NYSE closed, so it is not a grid day. Tokyo sessions on 28 and 29 March both open.
    assert pd.Timestamp("2024-03-29") not in g.index
    d = pd.to_datetime(["2024-03-27", "2024-03-28", "2024-03-29", "2024-04-01"])
    g2 = cal.us_grid("2024-03-20", "2024-04-05")
    px = series(d, [100, 101, 102, 104])
    r = cal.to_grid_returns(px, "XTKS", g2)
    assert not r.loc["2024-03-29":"2024-03-29"].size
    # Tokyo 29 March session (a US holiday) is compounded into US Monday 1 April together with Tokyo 1 April
    assert np.isclose((1 + r.loc["2024-04-01"]), 104 / 101)
    # Tokyo-holiday days are NaN, never zero
    d3 = pd.to_datetime(["2024-03-18", "2024-03-19", "2024-03-21", "2024-03-22"])  # Tokyo closed 20 March (equinox)
    r3 = cal.to_grid_returns(series(d3, [100, 101, 102, 103]), "XTKS", g2)
    assert np.isnan(r3.loc["2024-03-20"])


def test_usd_conversion():
    local = pd.Series([0.1, np.nan])
    fx = pd.Series([-0.1, 0.05])
    out = to_usd(local, fx)
    assert np.isclose(out.iloc[0], 1.1 * 0.9 - 1) and np.isnan(out.iloc[1])
