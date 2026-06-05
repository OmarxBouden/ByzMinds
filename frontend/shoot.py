"""Headless screenshot loop for the frontend so design can be iterated visually
(not built blind). Serves frontend/ on a local port, drives the app with
Playwright, and writes PNGs of each view to frontend/_shots/.

    python frontend/shoot.py            # default run, all views
"""
from __future__ import annotations

import functools
import http.server
import socketserver
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

FRONTEND = Path(__file__).resolve().parent
SHOTS = FRONTEND / "_shots"
PORT = 8799


def _serve():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(FRONTEND))
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    SHOTS.mkdir(exist_ok=True)
    httpd = _serve()
    time.sleep(0.4)
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
        pg.wait_for_timeout(1800)  # data load + first render
        pg.screenshot(path=str(SHOTS / "01_civ_glassbox.png"))
        pg.click("#mode-toggle"); pg.wait_for_timeout(700)
        pg.screenshot(path=str(SHOTS / "02_civ_public.png"))
        pg.click('.tab[data-tab="replay"]'); pg.wait_for_timeout(700)
        pg.screenshot(path=str(SHOTS / "03_replay.png"))
        pg.click('.tab[data-tab="metrics"]'); pg.wait_for_timeout(1000)
        pg.screenshot(path=str(SHOTS / "04_metrics.png"))
        b.close()
    httpd.shutdown()
    print("wrote:", sorted(s.name for s in SHOTS.glob("*.png")))
    if errors:
        print("PAGE ERRORS:")
        for e in errors[:12]:
            print("  ", e)


if __name__ == "__main__":
    main()
