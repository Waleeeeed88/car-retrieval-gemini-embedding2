from __future__ import annotations

import os
from pathlib import Path
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


def _fixtures_root() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "sample_files"


@pytest.mark.parametrize(
    ("mode_label", "file_name"),
    [
        ("Upload an image", "query_image.jpg"),
        ("Upload a PDF", "query_brochure.pdf"),
        ("Upload a finance markdown file", "query_finance.md"),
    ],
)
def test_upload_modes_accept_fixture_files(mode_label: str, file_name: str) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    url = _app_url()
    _require_server(url)
    file_path = _fixtures_root() / file_name

    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        page.get_by_label(mode_label).check()
        page.locator("input[type='file']").set_input_files(str(file_path))
        expect = sync_api.expect
        expect(page.get_by_role("button", name="Search")).to_be_visible()
        browser.close()
