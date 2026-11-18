"""LLM extraction over committed rule extracts, with a committed response cache.

The endpoint is any OpenAI-compatible chat API configured by environment variables only:
  SCG_LLM_BASE_URL, SCG_LLM_MODEL, SCG_LLM_BEARER (optional), SCG_LLM_LABEL (model label recorded in the cache),
  SCG_LLM_MODE = replay (default: read the cache, fail on a miss) | live (call the endpoint, fill the cache).
No host name is stored in code, configuration or cache files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from . import config

PROMPT_VERSION = "v1"
CACHE_DIR = config.ROOT / "data" / "llm_cache"
SYSTEM = ("You read one passage from a company's annual report. List every customer named in the passage together with the "
          "share it accounts for. Answer with JSON only, in the form "
          '{"customers": [{"name": string, "percent": number or null, "basis": "revenue" | "receivables" | "other"}]}. '
          "Use only information in the passage. Do not list anonymised customers (for example 'Customer A'). "
          "If no named customer with a stated share appears, answer {\"customers\": []}.")


def cache_key(label: str, text: str, version: str = PROMPT_VERSION) -> str:
    return hashlib.sha256(json.dumps([version, label, text], ensure_ascii=False).encode()).hexdigest()


class CacheMiss(RuntimeError):
    pass


class LLM:
    def __init__(self, mode: str | None = None, cache_dir: Path | None = None, post=None):
        self.mode = mode or os.environ.get("SCG_LLM_MODE", "replay")
        self.label = os.environ.get("SCG_LLM_LABEL", "open-weights-local")
        self.dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self._post = post

    def _call(self, text: str) -> str:
        base = os.environ["SCG_LLM_BASE_URL"].rstrip("/")
        body = {"model": os.environ["SCG_LLM_MODEL"], "temperature": 0, "max_tokens": 600,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
        headers = {"Authorization": "Bearer " + os.environ["SCG_LLM_BEARER"]} if os.environ.get("SCG_LLM_BEARER") else {}
        if self._post is not None:
            return self._post(base + "/chat/completions", body, headers)
        import requests

        r = requests.post(base + "/chat/completions", json=body, headers=headers, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def extract(self, text: str) -> dict:
        key = cache_key(self.label, text)
        p = self.dir / f"{key}.json"
        if p.exists():
            return json.loads(p.read_text())["response"]
        if self.mode != "live":
            raise CacheMiss(f"no cached response for key {key[:12]} (SCG_LLM_MODE=replay)")
        raw = self._call(text)
        resp = parse_json(raw)
        self.dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "label": self.label, "response": resp},
                                sort_keys=True, indent=1))
        return resp


def parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    custs = []
    for c in d.get("customers", []) if isinstance(d, dict) else []:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            pct = c.get("percent")
            custs.append({"name": c["name"][:80], "percent": float(pct) if isinstance(pct, (int, float)) else None,
                          "basis": c.get("basis") if c.get("basis") in ("revenue", "receivables", "other") else "other"})
    return {"customers": custs}


def resolve(name: str, aliases) -> str | None:
    from .extract_rules import alias_hits

    h = alias_hits(name, aliases)
    return h[0][2] if h else None


def reconcile(extract: dict, resp: dict, aliases, tol: float = 0.5) -> str:
    """'agree' if the model names the same ticker and share (within tol) as the rule; otherwise 'disagree'."""
    if extract["customer"] is None:
        return "disagree"
    for c in resp["customers"]:
        if resolve(c["name"], aliases) != extract["customer"]:
            continue
        if extract["pct"] is None:        # named above a threshold, or a share given jointly: the name is what must agree
            return "agree"
        if c["percent"] is not None and any(abs(c["percent"] - p) <= tol for p in extract.get("pct_all", [extract["pct"]])):
            return "agree"
    return "disagree"
