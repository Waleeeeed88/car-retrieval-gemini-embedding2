from __future__ import annotations

import os
from urllib.error import URLError
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.ui


def _app_url() -> str:
    return os.getenv("CAR_RETRIEVAL_UI_URL", "http://127.0.0.1:8501")


def _require_server(url: str) -> None:
    try:
        with urlopen(url, timeout=2):
            return
    except URLError:
        pytest.skip(f"Streamlit app is not reachable at {url}")


def test_text_search_flow_renders_results_and_images() -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    url = _app_url()
    _require_server(url)

    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        page.get_by_role("textbox").fill("family suv cargo")
        page.get_by_role("button", name="Search").click()

        expect = sync_api.expect
        expect(page.get_by_text("Results")).to_be_visible(timeout=15000)
        expect(page.locator("img").first).to_be_visible(timeout=15000)
        details = page.get_by_text("Details:", exact=False).first
        expect(details).to_be_visible(timeout=15000)
        details.click()
        expect(page.get_by_text("Summary")).to_be_visible(timeout=15000)
        expect(page.get_by_text("Specs")).to_be_visible(timeout=15000)
        expect(page.get_by_text("Finance")).to_be_visible(timeout=15000)
        browser.close()
