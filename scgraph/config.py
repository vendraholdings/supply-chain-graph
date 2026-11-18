from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())
