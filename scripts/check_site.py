"""Open the built site at desktop and phone sizes; fail on console errors, page errors, failed or external requests."""
from __future__ import annotations

import functools
import http.server
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config  # noqa: E402

DIST = config.ROOT / "site" / "dist"
SIZES = {"desktop": (1440, 900), "phone": (390, 844)}


def chromium_path() -> str | None:
    """Newest locally cached headless Chromium (SCG_CHROMIUM overrides); None lets Playwright use its own."""
    import glob
    import os

    if os.environ.get("SCG_CHROMIUM"):
        return os.environ["SCG_CHROMIUM"]
    c = sorted(glob.glob(os.path.expanduser("~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell")))
    c += sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux*/chrome-headless-shell")))
    return c[-1] if c else None


def main(shots: Path | None = None) -> int:
    from playwright.sync_api import sync_playwright

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DIST))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    bad = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chromium_path(), args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        for name, (w, h) in SIZES.items():
            pg = b.new_page(viewport=dict(width=w, height=h))
            pg.on("console", lambda m, n=name: bad.append((n, "console", m.text)) if m.type == "error" else None)
            pg.on("pageerror", lambda e, n=name: bad.append((n, "pageerror", str(e))))
            pg.on("requestfailed", lambda r, n=name: bad.append((n, "failed", r.url)))
            pg.on("request", lambda r, n=name: bad.append((n, "external", r.url)) if not r.url.startswith(f"http://127.0.0.1:{port}") and not r.url.startswith("data:") and not r.url.startswith("blob:") else None)
            pg.goto(f"http://127.0.0.1:{port}/index.html", wait_until="networkidle")
            pg.wait_for_timeout(2500)
            if pg.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1"):
                bad.append((name, "layout", "horizontal page scroll"))
            if shots:
                pg.screenshot(path=str(shots / f"site_{name}.png"), full_page=False)
        b.close()
    srv.shutdown()
    for x in bad:
        print("PROBLEM", *x)
    print("site check:", "clean" if not bad else f"{len(bad)} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else None))
