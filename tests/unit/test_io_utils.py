from __future__ import annotations

from pathlib import Path

import pytest

from config import get_settings
from utils.io_utils import build_car_record, list_car_directories, load_json, read_text, resolve_spec_path


@pytest.mark.unit
def test_list_car_directories_finds_fixture_cars(sample_dataset_root: Path) -> None:
    car_dirs = list_car_directories(sample_dataset_root)

    assert [path.name for path in car_dirs] == [
        "acme_electric_sedan_2026",
        "acme_family_suv_2026",
        "acme_work_truck_2026",
    ]


@pytest.mark.unit
def test_resolve_spec_path_prefers_specs_md(sample_dataset_root: Path) -> None:
    settings = get_settings(dataset_root=sample_dataset_root)
    car_dir = sample_dataset_root / "acme_family_suv_2026"

    spec_path = resolve_spec_path(car_dir, settings.supported_pdf_files)

    assert spec_path.name == "specs.md"


@pytest.mark.unit
def test_text_and_json_helpers_load_expected_fixture_content(sample_dataset_root: Path) -> None:
    car_dir = sample_dataset_root / "acme_electric_sedan_2026"

    metadata = load_json(car_dir / "metadata.json")
    summary = read_text(car_dir / "summary.md")

    assert metadata["model"] == "Voltline"
    assert "all-electric sedan" in summary.lower()


@pytest.mark.unit
def test_build_car_record_prioritizes_front_image(sample_dataset_root: Path) -> None:
    settings = get_settings(dataset_root=sample_dataset_root)
    car_record = build_car_record(sample_dataset_root / "acme_work_truck_2026", settings)

    assert car_record["pdf_path"].name == "specs.md"
    assert car_record["image_paths"][0].name == "front.jpg"
