"""Turn data/edges_curated_raw.csv (researched sources) into data/edges_curated.csv with point-in-time dates."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config, edges  # noqa: E402

D = config.ROOT / "data"


def main() -> None:
    raw = pd.read_csv(D / "edges_curated_raw.csv")
    uni = set(pd.read_csv(D / "universe.csv")["ticker"])
    drop = raw[~(raw.supplier.isin(uni) & raw.customer.isin(uni))]
    raw = raw[raw.supplier.isin(uni) & raw.customer.isin(uni)].copy()
    raw["method"], raw["reviewed"] = "curated", True
    raw["pct_revenue"] = pd.to_numeric(raw["pct_revenue"], errors="coerce")
    df = edges.finalize(raw)
    errs = edges.validate(df)
    if errs:
        raise SystemExit("validation failed: " + "; ".join(errs))
    df.to_csv(D / "edges_curated.csv", index=False)
    print(len(df), "curated edge rows,", df[["supplier", "customer"]].drop_duplicates().shape[0], "pairs;",
          len(drop), "rows outside the universe dropped")


if __name__ == "__main__":
    main()
