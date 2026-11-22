"""Static figures from committed results (light for the paper, dark for the site and README)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import theme  # noqa: E402
from .graph import TIERS  # noqa: E402

SHORT = {k: f"{k} {v.split(',')[0][:14]}" for k, v in TIERS.items()}


def _save(fig, out: Path, name: str, mode: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}_{mode}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_spillover_table(static: pd.DataFrame, out: Path, mode: str, number: int = 9) -> Path:
    theme.use(mode)
    names = [c for c in static.columns if c != "from_others"]
    M = static.loc[names, names].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(M, cmap=theme.CMAP, vmin=0, vmax=20)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < 12 or M[i, j] > 45 else "black")
    ax.set_xlabel("Shock originates in tier")
    ax.set_ylabel("Variance of tier explained (row)")
    ax.set_title(f"Figure {number}. Generalised forecast-error variance decomposition, 10-day horizon (% of row)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("Share of forecast-error variance (%)")
    return _save(fig, out, f"fig{number:02d}_spillover_table", mode)


def fig_spillover_rolling(rolling: pd.DataFrame, out: Path, mode: str, number: int = 10) -> Path:
    theme.use(mode)
    net = [c for c in rolling.columns if c.startswith("net_")]
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.4]))
    a.plot(rolling.index, rolling["total"], color=theme.LGB)
    a.set_ylabel("Total spillover index (%)")
    a.set_title(f"Figure {number}. Rolling 250-day total spillover index and net directional spillover by tier")
    cols = plt.get_cmap(theme.CMAP)(np.linspace(0.05, 0.95, len(net)))
    for c, col in zip(net, cols):
        b.plot(rolling.index, rolling[c], color=col, lw=1.1, label=c[4:])
    b.axhline(0, color=theme.MUTED, lw=0.8)
    b.set_ylabel("Net directional spillover (points)")
    b.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, out, f"fig{number:02d}_spillover_rolling", mode)


def fig_leadlag_grid(table: pd.DataFrame, out: Path, mode: str, number: int = 11) -> Path:
    theme.use(mode)
    tiers = sorted(set(table.leader))
    lags = sorted(set(table.lag))
    fig, axes = plt.subplots(len(tiers), len(tiers), figsize=(11, 9), sharex=True, sharey=True)
    for i, a in enumerate(tiers):
        for j, b in enumerate(tiers):
            ax = axes[i, j]
            ax.grid(False)
            if a != b:
                d = table[(table.leader == a) & (table.follower == b)].sort_values("lag")
                ax.fill_between(d.lag, d.lo, d.hi, color=theme.SHADE, alpha=0.5, lw=0)
                ax.plot(d.lag, d["corr"], color=theme.LGB, lw=1)
                ax.axhline(0, color=theme.MUTED, lw=0.4)
            else:
                ax.set_facecolor(theme.BG)
            if j == 0:
                ax.set_ylabel(a, rotation=0, ha="right", fontsize=7)
            if i == len(tiers) - 1:
                ax.set_xlabel(b, fontsize=7)
            ax.tick_params(labelsize=5)
    fig.suptitle(f"Figure {number}. Cross-correlation of tier index returns by lag (days {lags[0]} to {lags[-1]}); "
                 "row leads column at positive lags; bands are 95% stationary-bootstrap intervals", fontsize=9, x=0.02, ha="left")
    return _save(fig, out, f"fig{number:02d}_leadlag", mode)


def fig_placebo_hist(draws: np.ndarray, real: float, out: Path, mode: str, number: int = 8, label: str = "CAAR") -> Path:
    theme.use(mode)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(draws, bins=40, color=theme.SHADE)
    ax.axvline(real, color=theme.AMBER, lw=2, label=f"Observed {label}")
    ax.set_xlabel(f"{label} over days 0 to 20 (random dates)")
    ax.set_ylabel("Draws")
    ax.set_title(f"Figure {number}. Placebo distribution on random dates and the observed {label}")
    ax.legend()
    return _save(fig, out, f"fig{number:02d}_placebo", mode)


REL_LABEL = {"direct_supplier": "Direct suppliers of the event node", "direct_customer": "Direct customers of the event node",
             "two_hop": "Two hops from the event node"}


def _tier_colors(tiers):
    keys = list(TIERS)
    cm = plt.get_cmap(theme.CMAP)
    return {t: cm(0.05 + 0.9 * keys.index(t) / (len(keys) - 1)) for t in tiers}


def fig_country_tier(universe: pd.DataFrame, edges: pd.DataFrame, out: Path, mode: str, number: int = 2) -> Path:
    """Listings by tier and country (left) and curated supplier-customer pairs by country pair (right)."""
    theme.use(mode)
    tiers = list(TIERS)
    countries = sorted(universe["country"].unique())
    M = np.zeros((len(tiers), len(countries)))
    for _, r in universe.iterrows():
        M[tiers.index(r["tier"]), countries.index(r["country"])] += 1
    cty = dict(zip(universe["ticker"], universe["country"]))
    pairs = edges.drop_duplicates(["supplier", "customer"])
    cc = pairs.assign(k=pairs["supplier"].map(cty) + " to " + pairs["customer"].map(cty)).groupby("k").size().sort_values()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.55))
    im = a.imshow(M, cmap=theme.CMAP, vmin=0, vmax=max(M.max(), 1))
    a.set_xticks(range(len(countries)), countries)
    a.set_yticks(range(len(tiers)), [f"{t} {TIERS[t]}" for t in tiers])
    a.grid(False)
    for i in range(len(tiers)):
        for j in range(len(countries)):
            if M[i, j]:
                a.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] < 0.25 * M.max() else "black")
    a.set_title("Listings by tier and country")
    fig.colorbar(im, ax=a, fraction=0.046, label="Number of listings")
    b.barh(cc.index, cc.values, color=theme.LGB)
    b.set_xlabel("Supplier-customer pairs")
    b.set_title(f"Sourced pairs by country (n = {int(cc.sum())})")
    b.xaxis.get_major_locator().set_params(integer=True)
    fig.suptitle(f"Figure {number}. Universe and sourced supplier-customer pairs by country and tier", fontsize=11, x=0.02, ha="left")
    return _save(fig, out, f"fig{number:02d}_country_tier", mode)


def fig_centrality(cent: pd.DataFrame, nodes: pd.DataFrame, out: Path, mode: str, number: int = 3) -> Path:
    """PageRank of the connected nodes (bar) and PageRank rank against market-cap rank (scatter)."""
    theme.use(mode)
    d = cent.merge(nodes[["ticker", "name", "tier", "mcap_usd"]], on="ticker")
    d = d[(d["in_degree"] + d["out_degree"]) > 0].copy()
    d["mcap_rank"] = d["mcap_usd"].rank(ascending=False)
    d["pr_rank"] = d["pagerank"].rank(ascending=False)
    top = d.sort_values("pagerank").tail(15)
    col = _tier_colors(d["tier"].unique())
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    a.barh(top["ticker"], top["pagerank"], color=[col[t] for t in top["tier"]])
    a.set_xlabel("PageRank (damping 0.85)")
    a.set_title(f"Connected nodes by PageRank (n = {len(d)})")
    b.scatter(d["mcap_rank"], d["pr_rank"], c=[col[t] for t in d["tier"]], s=40)
    for _, r in d[d["pr_rank"] < 10].iterrows():
        b.annotate(r["ticker"], (r["mcap_rank"], r["pr_rank"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = len(d) + 1
    b.plot([0, lim], [0, lim], color=theme.MUTED, lw=0.8, ls="--")
    b.set_xlim(0, lim)
    b.set_ylim(0, lim)
    b.set_xlabel("Market-capitalisation rank among connected nodes")
    b.set_ylabel("PageRank rank")
    b.set_title("Centrality rank against size rank (labels omitted for the tied bottom group)", fontsize=9)
    fig.suptitle(f"Figure {number}. Centrality in the sourced graph (descriptive; 16 pairs)", fontsize=11, x=0.02, ha="left")
    return _save(fig, out, f"fig{number:02d}_centrality", mode)


def fig_tier_flow(flow: pd.DataFrame, out: Path, mode: str, number: int = 4) -> Path:
    """Tier-to-tier flow of disclosed revenue share (supplier tier in rows, customer tier in columns)."""
    theme.use(mode)
    names = list(flow.columns)
    M = flow.loc[names, names].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    Mm = np.ma.masked_where(M == 0, M)
    im = ax.imshow(Mm, cmap=theme.CMAP, vmin=0, vmax=max(M.max(), 1))
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if M[i, j]:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_xlabel("Customer tier")
    ax.set_ylabel("Supplier tier")
    ax.set_title(f"Figure {number}. Disclosed revenue share by tier pair (sum of pair weights; blank = no sourced pair)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Sum of disclosed shares (% of supplier revenue)")
    return _save(fig, out, f"fig{number:02d}_tier_flow", mode)


def fig_event_fans(caar: pd.DataFrame, out: Path, mode: str, number: int = 7) -> Path:
    """CAAR paths over days 0 to 20 by relationship (panels) and tier (lines)."""
    theme.use(mode)
    rels = [r for r in REL_LABEL if r in set(caar["relation"])]
    fig, axes = plt.subplots(1, len(rels), figsize=(4.2 * len(rels), 4.0), sharey=True, squeeze=False)
    col = _tier_colors(sorted(set(caar["tier"]) - {"all"}))
    for ax, rel in zip(axes[0], rels):
        d = caar[caar["relation"] == rel]
        for t, g in d[d["tier"] != "all"].groupby("tier"):
            ax.plot(g["day"], 100 * g["caar"], color=col[t], lw=1.1, label=f"{t} (n={int(g['n'].iloc[0])})")
        a = d[d["tier"] == "all"]
        ax.plot(a["day"], 100 * a["caar"], color=theme.INK, lw=2.2, label=f"All (n={int(a['n'].iloc[0])})")
        ax.axhline(0, color=theme.MUTED, lw=0.8)
        ax.set_title(REL_LABEL[rel], fontsize=9)
        ax.set_xlabel("Days after event")
        ax.legend(fontsize=7, loc="best")
    axes[0][0].set_ylabel("Cumulative average abnormal return (%)")
    fig.suptitle(f"Figure {number}. Sign-adjusted CAAR of linked companies after price shocks (sourced pairs)",
                 fontsize=10, x=0.02, ha="left")
    return _save(fig, out, f"fig{number:02d}_event_fans", mode)


def fig_placebo_panels(draws: pd.DataFrame, summary: dict, out: Path, mode: str, number: int = 8) -> Path:
    """Placebo CAAR (day 20) distribution on random dates by relationship, with the observed CAAR."""
    theme.use(mode)
    rels = [r for r in REL_LABEL if r in set(draws["relation"])]
    fig, axes = plt.subplots(1, len(rels), figsize=(4.2 * len(rels), 3.8), sharey=True, squeeze=False)
    for ax, rel in zip(axes[0], rels):
        v = 100 * draws.loc[draws["relation"] == rel, "caar_end"].to_numpy()
        ax.hist(v, bins=40, color=theme.SHADE)
        obs = 100 * summary[rel]["caar_end"]
        ax.axvline(obs, color=theme.AMBER, lw=2, label=f"Observed {obs:+.1f}%")
        ax.set_title(REL_LABEL[rel], fontsize=9)
        ax.set_xlabel("CAAR at day 20 on random dates (%)")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Draws")
    fig.suptitle(f"Figure {number}. Placebo distribution of CAAR on random dates, with the observed value", fontsize=10, x=0.02, ha="left")
    return _save(fig, out, f"fig{number:02d}_placebo", mode)


def fig_momentum(ret: pd.DataFrame, out: Path, mode: str, number: int = 5) -> Path:
    """Cumulative growth of the primary customer-momentum strategy, gross and net of costs (monthly, log scale)."""
    theme.use(mode)
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    idx = ret.index.to_timestamp() if hasattr(ret.index, "to_timestamp") else ret.index
    ax.plot(idx, (1 + ret["gross_return"]).cumprod(), color=theme.MUTED, lw=1.5, label="Gross of costs")
    ax.plot(idx, (1 + ret["net_return"]).cumprod(), color=theme.AMBER, lw=2.0, label="Net of costs and borrow")
    ax.axhline(1.0, color=theme.GRID, lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.set_title(f"Figure {number}. Customer-momentum strategy, primary specification, long-short, equal weight", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    return _save(fig, out, f"fig{number:02d}_momentum", mode)


def fig_momentum_placebo(placebo: np.ndarray, real: float, trial_sr: pd.Series, sr0_annual: float, out: Path, mode: str, number: int = 6) -> Path:
    """Left: real net Sharpe against the rewired-edge placebo distribution. Right: net Sharpe of each full-coverage trial."""
    theme.use(mode)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a.hist(placebo, bins=40, color=theme.SHADE)
    a.axvline(real, color=theme.AMBER, lw=2, label=f"Observed {real:+.2f}")
    a.set_xlabel("Annualised net Sharpe ratio, rewired edges")
    a.set_ylabel("Draws")
    a.legend(fontsize=8)
    t = trial_sr.sort_values()
    b.scatter(t.to_numpy(), np.arange(len(t)), s=18, color=theme.LINK)
    b.axvline(sr0_annual, color=theme.AMBER, lw=1.6, label=f"Expected maximum under no skill ({sr0_annual:.2f})")
    b.axvline(0, color=theme.MUTED, lw=0.8)
    b.set_yticks([])
    b.set_xlabel("Annualised net Sharpe ratio, one dot per specification")
    b.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Figure {number}. Observed Sharpe ratio against 1,000 degree-preserving rewirings and the specification grid", fontsize=10, x=0.02, ha="left")
    return _save(fig, out, f"fig{number:02d}_momentum_placebo", mode)
