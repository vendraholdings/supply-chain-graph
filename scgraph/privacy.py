"""Generic privacy scan over tracked files: hosts, private addresses, home paths, keys, env files."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ALLOWED_EMAIL = "263115861+vendraholdings@users.noreply.github.com"
PATTERNS = {
    "private ip": re.compile(r"\b(?:100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])|10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
    "tailnet host": re.compile(r"\.ts\.net\b|tailscale", re.I),
    "home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "api key": re.compile(r"(?i)\b(?:api[_-]?key|secret|passw(?:or)?d|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}"),
}
SKIP_SUFFIX = {".png", ".gif", ".pdf", ".parquet", ".woff", ".woff2", ".ico", ".jpg", ".mp4"}


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    return [root / p for p in out.splitlines() if p]


def scan_text(text: str, name: str = "") -> list[str]:
    hits = []
    for label, pat in PATTERNS.items():
        for m in pat.finditer(text):
            s = m.group(0)
            if label == "email" and (s == ALLOWED_EMAIL or s.endswith("@users.noreply.github.com") and "vendraholdings" in s):
                continue
            hits.append(f"{name}: {label}: {s[:60]}")
    return hits


def scan(root: Path) -> list[str]:
    hits: list[str] = []
    for f in tracked_files(root):
        if f.suffix.lower() in SKIP_SUFFIX or not f.is_file() or f.name == "privacy.py":
            continue
        if f.name.startswith(".env") or f.suffix in {".key", ".pem"}:
            hits.append(f"{f.name}: secret-like file is tracked")
            continue
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        hits += scan_text(text, str(f.relative_to(root)))
    return hits


if __name__ == "__main__":
    h = scan(Path(__file__).resolve().parents[1])
    print("\n".join(h) if h else "privacy scan: clean")
    sys.exit(1 if h else 0)
