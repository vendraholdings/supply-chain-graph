"""Shared figure theme: one palette and font for the site, the paper and the README.

Two modes: ``use("light")`` (white page, for the paper) and ``use("dark")`` (site, README, hero).
Both use the perceptually ordered ``turbo`` scale for every continuous colour mapping.
"""
from __future__ import annotations

import matplotlib as mpl

CMAP = "turbo"            # matplotlib
COLORSCALE = "Turbo"      # Plotly
FONT = ["DejaVu Sans"]

PALETTES = {
    "light": dict(INK="#1f2933", MUTED="#6b7280", GRID="#e5e7eb", BG="#ffffff", PANEL="#ffffff",
                  LGB="#1f6fb2", MLX="#c8553d", ACCENT="#2f7d6d", SHADE="#d9e2ec",
                  AMBER="#b7791f", GREEN="#2f7d6d", VIOLET="#6b4fbb", LINK="#1f6fb2"),
    "dark": dict(INK="#e5e7eb", MUTED="#9aa5b1", GRID="#1f2937", BG="#0b0f17", PANEL="#121826",
                 LGB="#4cc9f0", MLX="#f3722c", ACCENT="#90be6d", SHADE="#5b6b82",
                 AMBER="#f9c74f", GREEN="#90be6d", VIOLET="#b388ff", LINK="#4cc9f0"),
}
FAMILY_LABEL = {"lgb": "LightGBM", "mlx": "Neural network (MLP)"}

MODE = "light"
INK = MUTED = GRID = BG = PANEL = LGB = MLX = ACCENT = SHADE = AMBER = GREEN = VIOLET = LINK = ""
FAMILY: dict[str, str] = {}


def use(mode: str = "light") -> None:
    """Select the palette and apply the matplotlib rc settings."""
    global MODE, FAMILY
    if mode not in PALETTES:
        raise ValueError(f"unknown theme mode {mode!r}")
    MODE = mode
    globals().update(PALETTES[mode])
    FAMILY = {"lgb": LGB, "mlx": MLX}
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG, "savefig.edgecolor": BG,
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 10.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK,
        "figure.dpi": 100, "savefig.dpi": 200, "lines.linewidth": 1.6,
        "image.cmap": CMAP,
    })


def apply() -> None:
    """Re-apply the current mode (kept for callers that only need the rc settings)."""
    use(MODE)


def _axis(scene_axis: dict) -> dict:
    base = dict(backgroundcolor=PANEL, showbackground=True, gridcolor=GRID, gridwidth=1, zerolinecolor=GRID,
                linecolor=GRID, color=INK, showspikes=False)
    base.update(scene_axis)
    return base


def plotly_layout(**kw) -> dict:
    """Plotly layout in the current mode; a ``scene`` argument gets themed axes."""
    base = dict(font=dict(family="DejaVu Sans, Helvetica, Arial, sans-serif", size=12, color=INK),
                paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(l=10, r=10, t=40, b=10),
                hoverlabel=dict(bgcolor=PANEL, bordercolor=GRID, font=dict(color=INK)))
    scene = dict(kw.pop("scene", {}))
    scene.setdefault("bgcolor", BG)
    for a in "xyz":
        title = scene.pop(f"{a}axis_title", None)
        ax = _axis(dict(scene.pop(f"{a}axis", {})))
        if title is not None:
            ax["title"] = dict(text=title)
        scene[f"{a}axis"] = ax
    base["scene"] = scene
    base.update(kw)
    return base


use("light")
