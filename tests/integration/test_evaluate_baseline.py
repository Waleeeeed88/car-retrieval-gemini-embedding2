from __future__ import annotations

import pytest

import evaluate


@pytest.mark.integration
def test_run_baseline_evaluation_returns_expected_summary(sample_vectors, sample_metadata_records) -> None:
    rows, summary = evaluate.run_baseline_evaluation(sample_vectors, sample_metadata_records)

    assert len(rows) == len(sample_metadata_records)
    assert summary["overall"]["query_count"] == len(sample_metadata_records)
    assert 0.0 <= summary["overall"]["top_1_exact_car_match_rate"] <= 1.0
    assert 0.0 <= summary["overall"]["top_3_exact_car_match_rate"] <= 1.0
    assert summary["overall"]["average_rank_of_correct_car"] >= 1.0
    assert "summary" in summary["by_modality"]
    assert "image" in summary["by_modality"]
