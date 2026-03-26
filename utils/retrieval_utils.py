from __future__ import annotations

from typing import Any

import numpy as np

from utils.embedding_utils import normalize_vector

ALLOWED_MODALITY_FILTERS = {"all", "image", "pdf", "finance", "summary"}


def build_car_label(record: dict[str, Any]) -> str:
    year = record.get("year", "")
    make = record.get("make", "")
    model = record.get("model", "")
    return " ".join(part for part in [str(year), make, model] if part).strip()


def _candidate_indices(
    metadata: list[dict[str, Any]],
    *,
    modality_filter: str,
    exclude_item_id: str | None,
) -> list[int]:
    if modality_filter not in ALLOWED_MODALITY_FILTERS:
        raise ValueError(f"Unsupported modality filter: {modality_filter}")

    indices: list[int] = []
    for index, record in enumerate(metadata):
        if exclude_item_id and record.get("item_id") == exclude_item_id:
            continue
        if modality_filter != "all" and record.get("modality") != modality_filter:
            continue
        indices.append(index)
    return indices


def rank_records(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    modality_filter: str = "all",
    exclude_item_id: str | None = None,
) -> list[dict[str, Any]]:
    candidate_indices = _candidate_indices(
        metadata,
        modality_filter=modality_filter,
        exclude_item_id=exclude_item_id,
    )
    if not candidate_indices:
        return []

    normalized_query = normalize_vector(query_vector)
    candidate_vectors = vectors[candidate_indices]
    scores = candidate_vectors @ normalized_query
    order = np.argsort(-scores)

    ranked_results: list[dict[str, Any]] = []
    for rank_position, position in enumerate(order, start=1):
        metadata_index = candidate_indices[position]
        record = dict(metadata[metadata_index])
        record["score"] = float(scores[position])
        record["rank"] = rank_position
        record["car_label"] = build_car_label(record)
        ranked_results.append(record)
    return ranked_results


def top_k_results(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    top_k: int,
    modality_filter: str = "all",
    exclude_item_id: str | None = None,
) -> list[dict[str, Any]]:
    return rank_records(
        query_vector,
        vectors,
        metadata,
        modality_filter=modality_filter,
        exclude_item_id=exclude_item_id,
    )[:top_k]


def first_correct_rank(results: list[dict[str, Any]], expected_car_id: str) -> int | None:
    for result in results:
        if result.get("car_id") == expected_car_id:
            return int(result["rank"])
    return None

