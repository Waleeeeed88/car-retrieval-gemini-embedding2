from __future__ import annotations

from pathlib import Path

import pytest

import app


@pytest.mark.integration
def test_load_car_context_reads_specs_markdown_and_image(sample_dataset_root: Path) -> None:
    if hasattr(app.load_car_context, "clear"):
        app.load_car_context.clear()

    context = app.load_car_context(str(sample_dataset_root), "acme_family_suv_2026")

    assert context["metadata"]["model"] == "Family Cruiser"
    assert "family suv" in context["summary_text"].lower()
    assert "finance payment" in context["finance_text"].lower()
    assert "265 hp" in context["spec_text"]
    assert context["primary_image"] is not None
    assert Path(context["primary_image"]).name == "front.jpg"
