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
