from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

from config import get_settings
from utils.embedding_utils import GeminiEmbedder
from utils.io_utils import (
    configure_logging,
    ensure_directory,
    load_index_artifacts,
    load_query_variants,
    read_text,
    save_dataframe,
    save_json,
)
from utils.retrieval_utils import first_correct_rank, rank_records
from utils.transform_utils import build_text_variants, generate_image_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality and robustness.")
    parser.add_argument("--data-dir", type=str, help="Override the output data directory.")
    parser.add_argument("--dataset-root", type=str, help="Override the dataset root path.")
    parser.add_argument("--robustness", action="store_true", help="Run robustness experiments too.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def category_for_record(record: dict[str, Any]) -> str | None:
    return record.get("category") or record.get("body_type") or record.get("vehicle_type")


def evaluate_query_case(
    *,
    query_vector,
    query_record: dict[str, Any],
    vectors,
    metadata,
    exclude_item_id: str | None,
) -> dict[str, Any]:
    ranked = rank_records(
        query_vector,
        vectors,
        metadata,
        modality_filter="all",
        exclude_item_id=exclude_item_id,
    )
    correct_rank = first_correct_rank(ranked, query_record["car_id"])
    effective_rank = correct_rank if correct_rank is not None else len(ranked) + 1
    top_result = ranked[0] if ranked else {}

    query_category = category_for_record(query_record)
    top_category = category_for_record(top_result) if top_result else None
    same_category_top1 = (
        int(query_category == top_category)
        if query_category and top_category
        else None
    )

    return {
        "query_id": query_record["item_id"],
        "query_car_id": query_record["car_id"],
        "query_modality": query_record["modality"],
        "query_file_path": query_record["file_path"],
        "top_1_exact_car_match": int(bool(ranked) and ranked[0]["car_id"] == query_record["car_id"]),
        "top_3_exact_car_match": int(
            any(result["car_id"] == query_record["car_id"] for result in ranked[:3])
        ),
        "correct_car_rank": effective_rank,
        "same_category_top1": same_category_top1,
        "top_result_car_id": top_result.get("car_id"),
        "top_result_modality": top_result.get("modality"),
        "top_result_score": top_result.get("score"),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"query_count": 0}

    top1 = [row["top_1_exact_car_match"] for row in rows]
    top3 = [row["top_3_exact_car_match"] for row in rows]
    ranks = [row["correct_car_rank"] for row in rows]
    category_values = [
        row["same_category_top1"] for row in rows if row["same_category_top1"] is not None
    ]

    summary = {
        "query_count": len(rows),
        "top_1_exact_car_match_rate": sum(top1) / len(top1),
        "top_3_exact_car_match_rate": sum(top3) / len(top3),
        "average_rank_of_correct_car": sum(ranks) / len(ranks),
        "same_category_top1_rate": (
            sum(category_values) / len(category_values) if category_values else None
        ),
    }
    return summary


def run_baseline_evaluation(vectors, metadata) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(metadata, desc="Baseline evaluation")):
        row = evaluate_query_case(
            query_vector=vectors[index],
            query_record=record,
            vectors=vectors,
            metadata=metadata,
            exclude_item_id=record["item_id"],
        )
        rows.append(row)

    summary = {
        "overall": summarize_rows(rows),
        "by_modality": {},
    }
    for modality in sorted({row["query_modality"] for row in rows}):
        modality_rows = [row for row in rows if row["query_modality"] == modality]
        summary["by_modality"][modality] = summarize_rows(modality_rows)
    return rows, summary


def run_image_robustness(settings, embedder, vectors, metadata, baseline_lookup):
    rows: list[dict[str, Any]] = []
    image_records = [record for record in metadata if record["modality"] == "image"]

    for record in tqdm(image_records, desc="Image robustness"):
        image_path = settings.dataset_root / record["file_path"]
        variants = generate_image_variants(image_path)
        baseline = baseline_lookup[record["item_id"]]

        for variant_name, image_bytes in variants.items():
            query_vector = embedder.embed_image_bytes(image_bytes, mime_type="image/jpeg")
            row = evaluate_query_case(
                query_vector=query_vector,
                query_record=record,
                vectors=vectors,
                metadata=metadata,
                exclude_item_id=record["item_id"],
            )
            row.update(
                {
                    "variant_name": variant_name,
                    "baseline_top_1": baseline["top_1_exact_car_match"],
                    "baseline_correct_rank": baseline["correct_car_rank"],
                    "rank_delta_vs_baseline": row["correct_car_rank"] - baseline["correct_car_rank"],
                }
            )
            rows.append(row)

    summary = {}
    for variant_name in sorted({row["variant_name"] for row in rows}):
        variant_rows = [row for row in rows if row["variant_name"] == variant_name]
        variant_summary = summarize_rows(variant_rows)
        baseline_rate = sum(row["baseline_top_1"] for row in variant_rows) / len(variant_rows)
        variant_summary["top_1_drop_vs_original"] = baseline_rate - variant_summary["top_1_exact_car_match_rate"]
        summary[variant_name] = variant_summary
    return rows, summary


def run_text_robustness(settings, embedder, vectors, metadata, baseline_lookup):
    rows: list[dict[str, Any]] = []
    paraphrases = load_query_variants(settings.query_variants_path)
    summary_records = [record for record in metadata if record["modality"] == "summary"]

    for record in tqdm(summary_records, desc="Text robustness"):
        summary_path = settings.dataset_root / record["file_path"]
        finance_path = summary_path.with_name("finance.md")
        summary_text = read_text(summary_path)
        finance_text = read_text(finance_path) if finance_path.exists() else ""

        variants = build_text_variants(
            summary_text,
            finance_text,
            paraphrases=paraphrases.get(record["car_id"], []),
        )
        baseline = baseline_lookup[record["item_id"]]

        for variant_name, text in variants.items():
            query_vector = vectors[metadata.index(record)] if variant_name == "original" else embedder.embed_text(text)
            row = evaluate_query_case(
                query_vector=query_vector,
                query_record=record,
                vectors=vectors,
                metadata=metadata,
                exclude_item_id=record["item_id"],
            )
            row.update(
                {
                    "variant_name": variant_name,
                    "baseline_top_1": baseline["top_1_exact_car_match"],
                    "baseline_correct_rank": baseline["correct_car_rank"],
                    "rank_delta_vs_baseline": row["correct_car_rank"] - baseline["correct_car_rank"],
                }
            )
            rows.append(row)

    summary = {}
    for variant_name in sorted({row["variant_name"] for row in rows}):
        variant_rows = [row for row in rows if row["variant_name"] == variant_name]
        variant_summary = summarize_rows(variant_rows)
        baseline_rate = sum(row["baseline_top_1"] for row in variant_rows) / len(variant_rows)
        variant_summary["top_1_drop_vs_original"] = baseline_rate - variant_summary["top_1_exact_car_match_rate"]
        summary[variant_name] = variant_summary
    return rows, summary


def main() -> None:
    args = parse_args()
    settings = get_settings(dataset_root=args.dataset_root, data_dir=args.data_dir)
    configure_logging(args.verbose)
    ensure_directory(settings.evaluation_dir)

    vectors, metadata = load_index_artifacts(settings)
    baseline_rows, baseline_summary = run_baseline_evaluation(vectors, metadata)
    baseline_lookup = {row["query_id"]: row for row in baseline_rows}

    save_dataframe(settings.evaluation_dir / "baseline_results.csv", baseline_rows)
    save_json(settings.evaluation_dir / "baseline_summary.json", baseline_summary)

    print("Baseline evaluation")
    print(json.dumps(baseline_summary, indent=2))

    if not args.robustness:
        return

    embedder = GeminiEmbedder(settings)
    image_rows, image_summary = run_image_robustness(
        settings,
        embedder,
        vectors,
        metadata,
        baseline_lookup,
    )
    text_rows, text_summary = run_text_robustness(
        settings,
        embedder,
        vectors,
        metadata,
        baseline_lookup,
    )

    save_dataframe(settings.evaluation_dir / "image_robustness_results.csv", image_rows)
    save_dataframe(settings.evaluation_dir / "text_robustness_results.csv", text_rows)
    save_json(settings.evaluation_dir / "image_robustness_summary.json", image_summary)
    save_json(settings.evaluation_dir / "text_robustness_summary.json", text_summary)

    print("\nImage robustness")
    print(json.dumps(image_summary, indent=2))
    print("\nText robustness")
    print(json.dumps(text_summary, indent=2))


if __name__ == "__main__":
    main()

