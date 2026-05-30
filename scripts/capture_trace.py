"""Headless capture of the /trace.html execution tracer for the README.

Usage: python scripts/capture_trace.py
Saves: docs/trace_screenshot.png  (full-page, ~1400px wide)
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "trace_screenshot.png"
URL = "http://127.0.0.1:8000/trace.html"
SAMPLE = "Show me the top outages."


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")

        page.wait_for_selector(".sample", timeout=15_000)
        page.locator(".sample", has_text=SAMPLE).first.click()

        page.wait_for_selector("button:has-text('Reveal all'):not([disabled])", timeout=90_000)
        page.click("button:has-text('Reveal all')")
        page.wait_for_timeout(800)  # let the cards finish rendering

        page.screenshot(path=str(OUT), full_page=True)
        print(f"saved: {OUT}")
        browser.close()


if __name__ == "__main__":
    main()
