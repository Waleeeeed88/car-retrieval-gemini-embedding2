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


def test_streamlit_smoke_page_loads() -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    url = _app_url()
    _require_server(url)

    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        expect = sync_api.expect
        expect(page.get_by_text("Advanced settings")).to_be_visible()
        expect(page.get_by_text("Ask a text question")).to_be_visible()
        expect(page.get_by_role("button", name="Search")).to_be_visible()
        browser.close()
