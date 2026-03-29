from __future__ import annotations

from pathlib import Path

import pytest

from utils.feature_utils import build_feature_payload
from utils.io_utils import read_text


@pytest.mark.unit
def test_build_feature_payload_extracts_expected_family_signals(sample_dataset_root: Path) -> None:
    car_dir = sample_dataset_root / "acme_family_suv_2026"
    metadata = {
        "make": "Acme",
        "model": "Family Cruiser",
        "year": 2026,
        "category": "SUV",
        "drivetrain": "AWD",
        "fuel_type": "Gasoline",
        "price_range": "$34,000-$39,000",
        "market": "US",
        "finance_snapshot": {"msrp": "$35,900"},
    }

    payload = build_feature_payload(
        metadata,
        summary_text=read_text(car_dir / "summary.md"),
        finance_text=read_text(car_dir / "finance.md"),
        pdf_text=read_text(car_dir / "specs.md"),
        image_path=car_dir / "images" / "front.jpg",
    )

    assert "awd" in [value.lower() for value in payload["features"]]
    assert any("suv" in value.lower() for value in payload["features"])
    assert any("cargo" in value.lower() for value in payload["spec_facts"])


@pytest.mark.unit
def test_build_feature_payload_handles_missing_image_without_crashing() -> None:
    payload = build_feature_payload(
        {"make": "Acme", "model": "Missing", "year": 2026},
        summary_text="",
        finance_text="",
        pdf_text="",
        image_path=None,
    )

    assert payload["colors"] == []
    assert payload["image_profile_lines"] == []
    assert payload["image_quality_flags"] == []


@pytest.mark.unit
def test_build_feature_payload_emits_low_resolution_flag(sample_dataset_root: Path) -> None:
    car_dir = sample_dataset_root / "acme_electric_sedan_2026"
    payload = build_feature_payload(
        {"make": "Acme", "model": "Voltline", "year": 2026},
        summary_text="electric sedan",
        finance_text="premium EV lease",
        pdf_text="340 mile range",
        image_path=car_dir / "images" / "front.jpg",
    )

    assert "low native source resolution" in payload["image_quality_flags"]
