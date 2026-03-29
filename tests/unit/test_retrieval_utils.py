from __future__ import annotations

import pytest

from utils.retrieval_utils import first_correct_rank, rank_records


@pytest.mark.unit
def test_rank_records_returns_expected_family_match_first(sample_vectors, sample_metadata_records, fake_embedder) -> None:
    query_vector = fake_embedder.embed_text("family suv cargo monthly payment")
    ranked = rank_records(query_vector, sample_vectors, sample_metadata_records, modality_filter="all")

    assert ranked
    assert ranked[0]["car_id"] == "acme_family_suv_2026"


@pytest.mark.unit
def test_rank_records_respects_modality_filter(sample_vectors, sample_metadata_records, fake_embedder) -> None:
    query_vector = fake_embedder.embed_text("electric sedan range")
    ranked = rank_records(query_vector, sample_vectors, sample_metadata_records, modality_filter="finance")

    assert ranked
    assert all(record["modality"] == "finance" for record in ranked)
    assert ranked[0]["car_id"] == "acme_electric_sedan_2026"


@pytest.mark.unit
def test_rank_records_excludes_item_id(sample_vectors, sample_metadata_records, fake_embedder) -> None:
    query_vector = fake_embedder.embed_text("work truck towing payload")
    exclude_id = "acme_work_truck_2026::finance"
    ranked = rank_records(
        query_vector,
        sample_vectors,
        sample_metadata_records,
        modality_filter="all",
        exclude_item_id=exclude_id,
    )

    assert all(record["item_id"] != exclude_id for record in ranked)


@pytest.mark.unit
def test_first_correct_rank_returns_first_matching_rank() -> None:
    results = [
        {"car_id": "wrong", "rank": 1},
        {"car_id": "target", "rank": 2},
        {"car_id": "target", "rank": 3},
    ]

    assert first_correct_rank(results, "target") == 2
