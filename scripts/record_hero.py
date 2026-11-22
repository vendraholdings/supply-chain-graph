"""Record the rotating 3D network from headless Chromium into docs/hero.gif (1200 x 675)."""
from __future__ import annotations

import functools
import http.server
import io
import math
import sys
import threading
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scgraph import config  # noqa: E402
from scripts.check_site import DIST, chromium_path  # noqa: E402

OUT = config.ROOT / "docs" / "hero.gif"
FRAMES, W, H = 72, 1200, 675


def main() -> None:
    from playwright.sync_api import sync_playwright

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DIST))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    frames = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chromium_path(), args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = b.new_page(viewport=dict(width=W, height=H))
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/index.html?hero=1", wait_until="networkidle")
        pg.wait_for_timeout(6000)  # layout settles and freezes
        R = pg.evaluate("(()=>{const p=G.cameraPosition();return Math.hypot(p.x,p.z)||600})()")
        R = max(R, 300.0)
        for i in range(FRAMES):
            a = 2 * math.pi * i / FRAMES
            pg.evaluate(f"(G.cameraPosition({{x:{R * math.sin(a):.2f},y:{0.35 * R:.2f},z:{R * math.cos(a):.2f}}}),0)")
            pg.wait_for_timeout(120)
            frames.append(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
        b.close()
    srv.shutdown()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=70, loop=0, optimize=True)
    print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
