from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_settings
from utils.embedding_utils import GeminiEmbedder, average_embeddings, weighted_average_embeddings
from utils.feature_utils import (
    build_embedding_context_text,
    build_feature_payload,
    build_image_embedding_brief,
)
from utils.io_utils import (
    build_car_record,
    configure_logging,
    extract_pdf_text,
    get_pdf_page_count,
    list_car_directories,
    read_text,
    relative_to_dataset,
    save_index_artifacts,
    split_pdf_into_chunks,
)


def build_item_record(
    *,
    car_record: dict[str, Any],
    modality: str,
    file_path: Path,
    dataset_root: Path,
    item_suffix: str,
    view_type: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = car_record["metadata"]
    car_id = car_record["car_id"]
    record = {
        "item_id": f"{car_id}::{item_suffix}",
        "car_id": car_id,
        "modality": modality,
        "file_path": relative_to_dataset(file_path, dataset_root),
        "make": metadata.get("make", ""),
        "model": metadata.get("model", ""),
        "year": metadata.get("year", ""),
        "category": metadata.get("category")
        or metadata.get("body_type")
        or metadata.get("vehicle_type")
        or metadata.get("segment"),
        "drivetrain": metadata.get("drivetrain"),
        "fuel_type": metadata.get("fuel_type"),
        "price_range": metadata.get("price_range"),
        "market": metadata.get("market"),
        "view_type": view_type,
    }
    if extra:
        record.update(extra)
    return record


def embed_car_items(
    car_record: dict[str, Any],
    *,
    embedder: GeminiEmbedder,
    dataset_root: Path,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    vectors: list[np.ndarray] = []
    metadata_records: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []
    failures: list[str] = []
    primary_image_path = car_record["image_paths"][0] if car_record["image_paths"] else None

    summary_text = read_text(car_record["summary_path"]) if car_record["summary_path"].exists() else ""
    finance_text = read_text(car_record["finance_path"]) if car_record["finance_path"].exists() else ""
    pdf_text = extract_pdf_text(car_record["pdf_path"], max_pages=8) if car_record["pdf_path"].exists() else ""
    feature_payload = build_feature_payload(
        car_record["metadata"],
        summary_text=summary_text,
        finance_text=finance_text,
        pdf_text=pdf_text,
        image_path=primary_image_path,
    )

    text_items = [
        ("summary", car_record["summary_path"], "summary", None, summary_text),
        ("finance", car_record["finance_path"], "finance", None, finance_text),
    ]

    for modality, path, suffix, view_type, text in text_items:
        if not path.exists():
            failures.append(f"Missing {modality} file: {path}")
            continue
        try:
            enriched_text = build_embedding_context_text(
                modality=modality,
                metadata=car_record["metadata"],
                feature_payload=feature_payload,
                original_text=text,
            )
            vector = embedder.embed_text(enriched_text)
            record = build_item_record(
                car_record=car_record,
                modality=modality,
                file_path=path,
                dataset_root=dataset_root,
                item_suffix=suffix,
                view_type=view_type,
                extra={
                    "text_length": len(text),
                    "preview": text[:240],
                    "colors": feature_payload["colors"],
                    "features": feature_payload["features"],
                    "spec_facts": feature_payload["spec_facts"],
                    "image_traits": feature_payload["image_traits"],
                    "image_profile": feature_payload["image_profile_lines"],
                    "image_quality_flags": feature_payload["image_quality_flags"],
                },
            )
            vectors.append(vector)
            metadata_records.append(record)
            debug_records.append({**record, "embedding": vector.tolist()})
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Failed to embed {path}: {exc}")

    pdf_path = car_record["pdf_path"]
    if pdf_path.exists():
        try:
            pdf_context_text = build_embedding_context_text(
                modality="pdf",
                metadata=car_record["metadata"],
                feature_payload=feature_payload,
                original_text=pdf_text[:4000],
            )
            if pdf_path.suffix.lower() in {".txt", ".md"}:
                vector = embedder.embed_text(pdf_context_text)
                page_count = None
                chunk_count = None
            else:
                pdf_vector = embedder.embed_pdf(pdf_path)
                pdf_context_vector = embedder.embed_text(pdf_context_text)
                vector = average_embeddings([pdf_vector, pdf_context_vector])
                page_count = get_pdf_page_count(pdf_path)
                chunk_count = len(
                    split_pdf_into_chunks(pdf_path, embedder.settings.pdf_max_pages_per_request)
                )
            record = build_item_record(
                car_record=car_record,
                modality="pdf",
                file_path=pdf_path,
                dataset_root=dataset_root,
                item_suffix="pdf",
                extra={
                    "page_count": page_count,
                    "pdf_chunk_count": chunk_count,
                    "colors": feature_payload["colors"],
                    "features": feature_payload["features"],
                    "spec_facts": feature_payload["spec_facts"],
                    "image_traits": feature_payload["image_traits"],
                    "image_profile": feature_payload["image_profile_lines"],
                    "image_quality_flags": feature_payload["image_quality_flags"],
                },
            )
            vectors.append(vector)
            metadata_records.append(record)
            debug_records.append({**record, "embedding": vector.tolist()})
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Failed to embed {pdf_path}: {exc}")
    else:
        failures.append(f"Missing PDF file: {pdf_path}")

    for image_path in car_record["image_paths"]:
        try:
            image_vector = embedder.embed_image(image_path)
            image_context_text = build_embedding_context_text(
                modality="image",
                metadata=car_record["metadata"],
                feature_payload=feature_payload,
                original_text=build_image_embedding_brief(
                    metadata=car_record["metadata"],
                    feature_payload=feature_payload,
                    view_type=image_path.stem,
                ),
            )
            image_context_vector = embedder.embed_text(image_context_text)
            vector = weighted_average_embeddings(
                [
                    (image_vector, embedder.settings.image_vector_weight),
                    (image_context_vector, embedder.settings.image_context_weight),
                ]
            )
            record = build_item_record(
                car_record=car_record,
                modality="image",
                file_path=image_path,
                dataset_root=dataset_root,
                item_suffix=f"image::{image_path.stem}",
                view_type=image_path.stem,
                extra={
                    "colors": feature_payload["colors"],
                    "features": feature_payload["features"],
                    "spec_facts": feature_payload["spec_facts"],
                    "image_traits": feature_payload["image_traits"],
                    "image_profile": feature_payload["image_profile_lines"],
                    "image_quality_flags": feature_payload["image_quality_flags"],
                },
            )
            vectors.append(vector)
            metadata_records.append(record)
            debug_records.append({**record, "embedding": vector.tolist()})
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Failed to embed {image_path}: {exc}")

    return vectors, metadata_records, debug_records, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the multimodal Gemini embedding index.")
    parser.add_argument("--dataset-root", type=str, help="Override the dataset root path.")
    parser.add_argument("--output-dir", type=str, help="Override the output data directory.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings(dataset_root=args.dataset_root, data_dir=args.output_dir)
    configure_logging(args.verbose)

    car_dirs = list_car_directories(settings.dataset_root)
    if not car_dirs:
        raise RuntimeError(f"No car folders were found under {settings.dataset_root}")

    embedder = GeminiEmbedder(settings)

    all_vectors: list[np.ndarray] = []
    all_metadata: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []
    failures: list[str] = []

    for car_dir in tqdm(car_dirs, desc="Embedding cars"):
        car_record = build_car_record(car_dir, settings)
        vectors, metadata_records, debug_items, car_failures = embed_car_items(
            car_record,
            embedder=embedder,
            dataset_root=settings.dataset_root,
        )
        all_vectors.extend(vectors)
        all_metadata.extend(metadata_records)
        debug_records.extend(debug_items)
        failures.extend(car_failures)

    if not all_vectors:
        raise RuntimeError("No items were embedded successfully.")

    vector_matrix = np.vstack(all_vectors).astype(np.float32)
    save_index_artifacts(
        vectors=vector_matrix,
        metadata=all_metadata,
        debug_records=debug_records,
        settings=settings,
    )

    modality_counts = Counter(record["modality"] for record in all_metadata)
    print(f"Indexed {len(all_metadata)} items from {len(car_dirs)} cars.")
    for modality, count in sorted(modality_counts.items()):
        print(f"  {modality:<7} {count}")
    print(f"Vectors saved to: {settings.embeddings_npy_path}")
    print(f"Metadata saved to: {settings.index_metadata_path}")
    if failures:
        print("\nCompleted with warnings:")
        for message in failures:
            print(f"  - {message}")


if __name__ == "__main__":
    main()
