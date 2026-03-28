from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_settings
from utils.embedding_utils import GeminiEmbedder
from utils.io_utils import configure_logging, load_index_artifacts, read_text
from utils.retrieval_utils import top_k_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the multimodal car index.")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--text", type=str, help="Free-text query.")
    query_group.add_argument("--image", type=str, help="Path to an image query.")
    query_group.add_argument("--pdf", type=str, help="Path to a PDF query.")
    query_group.add_argument("--finance-file", type=str, help="Path to a finance markdown query.")

    parser.add_argument(
        "--modality",
        choices=["all", "image", "pdf", "finance", "summary"],
        default="all",
        help="Restrict retrieval to one indexed modality.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Number of results to display.")
    parser.add_argument("--data-dir", type=str, help="Override the output data directory.")
    parser.add_argument("--dataset-root", type=str, help="Override the dataset root path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def pretty_print_results(results: list[dict[str, object]]) -> None:
    if not results:
        print("No retrieval results were found.")
        return

    header = f"{'Rank':<6}{'Score':<10}{'Car':<36}{'Modality':<10}Path"
    print(header)
    print("-" * len(header))
    for result in results:
        label = result.get("car_label", "")[:34]
        print(
            f"{result['rank']:<6}{result['score']:<10.4f}{label:<36}"
            f"{result['modality']:<10}{result['file_path']}"
        )
        colors = result.get("colors") or []
        features = result.get("features") or []
        image_profile = result.get("image_profile") or []
        if colors:
            print(f"      colors: {', '.join(str(color) for color in colors)}")
        if features:
            print(f"      features: {', '.join(str(feature) for feature in features[:8])}")
        if image_profile:
            print(f"      image: {', '.join(str(line) for line in image_profile[:4])}")


def build_query_embedding(args: argparse.Namespace, embedder: GeminiEmbedder) -> tuple[str, object]:
    if args.text:
        return "text", embedder.embed_text(args.text)
    if args.image:
        return "image", embedder.embed_image(args.image)
    if args.pdf:
        return "pdf", embedder.embed_pdf(args.pdf)
    if args.finance_file:
        finance_text = read_text(Path(args.finance_file))
        return "finance", embedder.embed_text(finance_text)
    raise ValueError("No query input was provided.")


def main() -> None:
    args = parse_args()
    settings = get_settings(
        dataset_root=args.dataset_root,
        data_dir=args.data_dir,
        top_k=args.top_k,
    )
    configure_logging(args.verbose)
    vectors, metadata = load_index_artifacts(settings)
    embedder = GeminiEmbedder(settings)

    query_type, query_vector = build_query_embedding(args, embedder)
    results = top_k_results(
        query_vector,
        vectors,
        metadata,
        top_k=args.top_k or settings.default_top_k,
        modality_filter=args.modality,
    )

    print(f"Query type: {query_type}")
    print(f"Search mode: {args.modality}")
    print()
    pretty_print_results(results)


if __name__ == "__main__":
    main()
