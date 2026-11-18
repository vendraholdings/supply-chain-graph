"""Download (or reuse from cache) prices and FX, then build the USD return matrix on the US grid."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import returns  # noqa: E402

if __name__ == "__main__":
    df = returns.build(force=True)
    cov = df.notna().mean().sort_values()
    print(df.shape, "grid days x series; lowest coverage:")
    print(cov.head(8).round(3).to_string())
