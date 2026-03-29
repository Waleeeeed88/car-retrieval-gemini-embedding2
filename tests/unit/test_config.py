from __future__ import annotations

from pathlib import Path

import pytest

from config import PROJECT_ROOT, get_settings


@pytest.mark.unit
def test_relative_paths_resolve_from_project_root() -> None:
    settings = get_settings(dataset_root="./tests/fixtures/sample_dataset", data_dir="./tests-output")

    assert settings.dataset_root == (PROJECT_ROOT / "tests" / "fixtures" / "sample_dataset").resolve()
    assert settings.data_dir == (PROJECT_ROOT / "tests-output").resolve()


@pytest.mark.unit
def test_absolute_paths_remain_absolute(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    data_dir = tmp_path / "data"
    settings = get_settings(dataset_root=dataset_root, data_dir=data_dir)

    assert settings.dataset_root == dataset_root.resolve()
    assert settings.data_dir == data_dir.resolve()


@pytest.mark.unit
def test_supported_spec_order_includes_pdf_markdown_and_txt() -> None:
    settings = get_settings()

    assert settings.supported_pdf_files == ("spec.pdf", "specs.md", "pdf.txt")
