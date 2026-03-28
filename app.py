from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_settings
from utils.embedding_utils import GeminiEmbedder
from utils.io_utils import (
    extract_pdf_text,
    load_index_artifacts,
    load_json,
    read_text,
    resolve_spec_path,
)
from utils.retrieval_utils import build_car_label, rank_records

QUERY_TYPE_OPTIONS = {
    "Ask a text question": "text",
    "Upload an image": "image",
    "Upload a PDF": "pdf",
    "Upload a finance markdown file": "finance",
}

FINANCE_KEYWORDS = {
    "payment",
    "payments",
    "monthly",
    "month",
    "lease",
    "apr",
    "rate",
    "rates",
    "finance",
    "financing",
    "offer",
    "offers",
    "incentive",
    "incentives",
    "signing",
    "down",
    "warranty",
    "affordable",
    "cheap",
    "cost",
    "price",
}

SPEC_KEYWORDS = {
    "spec",
    "specs",
    "horsepower",
    "hp",
    "torque",
    "mpg",
    "range",
    "battery",
    "charging",
    "cargo",
    "towing",
    "seating",
    "seat",
    "seats",
    "awd",
    "fwd",
    "rwd",
    "drivetrain",
    "engine",
    "hybrid",
    "plug",
    "dimensions",
    "size",
    "transmission",
    "efficiency",
    "fuel",
}


@st.cache_resource(show_spinner=False)
def load_search_resources(dataset_root: str | None, data_dir: str | None):
    settings = get_settings(dataset_root=dataset_root, data_dir=data_dir)
    vectors, metadata = load_index_artifacts(settings)
    embedder = GeminiEmbedder(settings)
    return settings, vectors, metadata, embedder


def resolve_car_dir(dataset_root: str | Path, car_id: str) -> Path:
    requested_root = Path(dataset_root)
    project_dataset_root = Path(__file__).resolve().parent / "dataset"
    candidates = [
        requested_root / car_id,
        project_dataset_root / car_id,
    ]
    for candidate in candidates:
        if (candidate / "metadata.json").exists():
            return candidate
    return candidates[0]


@st.cache_data(show_spinner=False)
def load_car_context(dataset_root: str, car_id: str) -> dict[str, object]:
    settings = get_settings(dataset_root=dataset_root)
    car_dir = resolve_car_dir(settings.dataset_root, car_id)
    metadata = load_json(car_dir / "metadata.json")
    summary_path = car_dir / "summary.md"
    finance_path = car_dir / "finance.md"
    spec_path = resolve_spec_path(car_dir, settings.supported_pdf_files)
    images_dir = car_dir / "images"

    image_candidates = sorted(path for path in images_dir.iterdir() if path.is_file()) if images_dir.exists() else []
    primary_image = next((path for path in image_candidates if path.stem == "front"), None)
    if primary_image is None and image_candidates:
        primary_image = image_candidates[0]

    return {
        "metadata": metadata,
        "summary_text": read_text(summary_path) if summary_path.exists() else "",
        "finance_text": read_text(finance_path) if finance_path.exists() else "",
        "spec_text": extract_pdf_text(spec_path, max_pages=8) if spec_path.exists() else "",
        "primary_image": primary_image,
    }


def build_query_vector(
    *,
    query_type: str,
    text_query: str,
    uploaded_file,
    embedder: GeminiEmbedder,
):
    if query_type == "text":
        if not text_query.strip():
            raise ValueError("Enter a text question before searching.")
        return embedder.embed_text(text_query)

    if uploaded_file is None:
        raise ValueError("Upload a file before searching.")

    file_bytes = uploaded_file.getvalue()
    if query_type == "image":
        return embedder.embed_image_bytes(file_bytes, mime_type=uploaded_file.type or "image/jpeg")
    if query_type == "pdf":
        return embedder.embed_pdf_bytes(file_bytes)
    if query_type == "finance":
        try:
            finance_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Finance files must be UTF-8 text.") from exc
        return embedder.embed_text(finance_text)

    raise ValueError(f"Unsupported query type: {query_type}")


def render_uploaded_preview(query_type: str, uploaded_file) -> None:
    if uploaded_file is None:
        return

    if query_type == "image":
        st.image(uploaded_file.getvalue(), caption=uploaded_file.name, use_container_width=True)
        return

    st.caption(f"Uploaded file: `{uploaded_file.name}`")


def tokenize_query(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def detect_query_focus(query_type: str, text_query: str) -> set[str]:
    if query_type == "finance":
        return {"finance"}
    if query_type == "pdf":
        return {"spec"}
    if query_type == "image":
        return {"image"}

    tokens = tokenize_query(text_query)
    focus: set[str] = set()
    if tokens & FINANCE_KEYWORDS:
        focus.add("finance")
    if tokens & SPEC_KEYWORDS:
        focus.add("spec")
    if not focus:
        focus.add("general")
    return focus


def build_car_results(
    query_vector,
    vectors,
    metadata: list[dict[str, object]],
    *,
    top_k: int,
    modality_filter: str,
) -> list[dict[str, object]]:
    ranked_items = rank_records(
        query_vector,
        vectors,
        metadata,
        modality_filter=modality_filter,
    )

    grouped: list[dict[str, object]] = []
    seen_car_ids: set[str] = set()
    for item in ranked_items:
        car_id = str(item["car_id"])
        if car_id in seen_car_ids:
            continue
        seen_car_ids.add(car_id)
        supporting = [
            record
            for record in ranked_items
            if record["car_id"] == car_id
        ][:4]
        grouped.append(
            {
                "rank": len(grouped) + 1,
                "car_id": car_id,
                "car_label": build_car_label(item),
                "score": float(item["score"]),
                "best_match": item,
                "supporting_items": supporting,
            }
        )
        if len(grouped) >= top_k:
            break
    return grouped


def extract_relevant_snippets(text: str, query_text: str, *, max_snippets: int = 3) -> list[str]:
    if not text.strip():
        return []

    tokens = tokenize_query(query_text)
    chunks = []
    for raw_piece in re.split(r"[\n\r]+|(?<=[.!?])\s+", text):
        piece = " ".join(raw_piece.split()).strip(" -")
        if len(piece) < 20:
            continue
        piece_tokens = tokenize_query(piece)
        overlap = len(tokens & piece_tokens)
        if overlap or not tokens:
            chunks.append((overlap, len(piece), piece))

    chunks.sort(key=lambda item: (-item[0], item[1]))
    snippets = []
    seen = set()
    for _, _, piece in chunks:
        if piece in seen:
            continue
        seen.add(piece)
        snippets.append(piece)
        if len(snippets) >= max_snippets:
            break
    return snippets


def finance_lines_from_snapshot(snapshot: dict[str, object], query_text: str) -> list[str]:
    if not snapshot:
        return []

    tokens = tokenize_query(query_text)
    preferred_keys = []
    if not tokens:
        preferred_keys.extend(
            [
                "msrp",
                "apr",
                "finance_payment",
                "lease_payment",
                "lease_term",
                "due_at_signing",
                "incentives",
            ]
        )
    if {"payment", "payments", "monthly", "month"} & tokens:
        preferred_keys.extend(["finance_payment", "lease_payment", "due_at_signing", "lease_term"])
    if {"apr", "rate", "rates"} & tokens:
        preferred_keys.append("apr")
    if {"down", "signing"} & tokens:
        preferred_keys.extend(["finance_down_payment", "due_at_signing"])
    if {"offer", "offers", "incentive", "incentives"} & tokens:
        preferred_keys.append("incentives")
    if "warranty" in tokens:
        preferred_keys.append("warranty")

    preferred_keys.extend(["msrp", "total_before_tax"])

    lines = []
    seen = set()
    for key in preferred_keys:
        value = snapshot.get(key)
        if not value or key in seen:
            continue
        seen.add(key)
        label = key.replace("_", " ").title()
        lines.append(f"{label}: {value}")
    return lines[:6]


def spec_lines_from_metadata(metadata: dict[str, object]) -> list[str]:
    lines = []
    category = metadata.get("category") or metadata.get("body_type") or metadata.get("vehicle_type")
    if category:
        lines.append(f"Category: {category}")
    if metadata.get("drivetrain"):
        lines.append(f"Drivetrain: {metadata['drivetrain']}")
    if metadata.get("fuel_type"):
        lines.append(f"Fuel Type: {metadata['fuel_type']}")
    if metadata.get("price_range"):
        lines.append(f"Price Range: {metadata['price_range']}")
    return lines


def feature_lines_from_result(result: dict[str, object]) -> list[str]:
    lines = []
    colors = result.get("colors") or []
    features = result.get("features") or []
    spec_facts = result.get("spec_facts") or []
    image_traits = result.get("image_traits") or []
    image_profile = result.get("image_profile") or []
    image_quality_flags = result.get("image_quality_flags") or []
    if colors:
        lines.append(f"Colors: {', '.join(str(color) for color in colors)}")
    if features:
        lines.append(f"Features: {', '.join(str(feature) for feature in features[:10])}")
    if spec_facts:
        lines.append(f"Spec facts: {', '.join(str(fact) for fact in spec_facts[:6])}")
    if image_traits:
        lines.append(f"Image traits: {', '.join(str(trait) for trait in image_traits[:6])}")
    if image_profile:
        lines.append(f"Image profile: {', '.join(str(line) for line in image_profile[:6])}")
    if image_quality_flags:
        lines.append(f"Image quality flags: {', '.join(str(flag) for flag in image_quality_flags[:4])}")
    return lines


def render_result_card(
    result: dict[str, object],
    settings,
    *,
    query_type: str,
    query_text: str,
    focus: set[str],
) -> None:
    context = load_car_context(str(settings.dataset_root), str(result["car_id"]))
    metadata = context["metadata"]
    finance_text = str(context["finance_text"])
    summary_text = str(context["summary_text"])
    spec_text = str(context["spec_text"])
    primary_image = context["primary_image"]
    best_match = result["best_match"]

    left, right = st.columns([1, 2])
    with left:
        if primary_image and Path(primary_image).exists():
            st.image(str(primary_image), use_container_width=True)
        else:
            st.caption("No image available")

    with right:
        st.markdown(
            f"**#{result['rank']} | {result['car_label']}**  \n"
            f"Score: `{result['score']:.4f}`  \n"
            f"Best match modality: `{best_match['modality']}`"
        )
        st.caption(f"Matched file: {settings.dataset_root / str(best_match['file_path'])}")

        matched_preview = best_match.get("preview")
        if matched_preview:
            st.write(str(matched_preview))

        feature_lines = feature_lines_from_result(best_match)
        if feature_lines:
            st.markdown("**Indexed attributes**")
            for line in feature_lines:
                st.write(f"- {line}")

        if "finance" in focus:
            finance_snapshot = metadata.get("finance_snapshot", {})
            finance_lines = finance_lines_from_snapshot(finance_snapshot, query_text)
            finance_snippets = extract_relevant_snippets(finance_text, query_text, max_snippets=3)
            if finance_lines or finance_snippets:
                st.markdown("**Finance details**")
                for line in finance_lines:
                    st.write(f"- {line}")
                for snippet in finance_snippets:
                    st.write(f"- {snippet}")

        if "spec" in focus:
            spec_lines = spec_lines_from_metadata(metadata)
            spec_snippets = extract_relevant_snippets(spec_text, query_text, max_snippets=3)
            summary_snippets = extract_relevant_snippets(summary_text, query_text, max_snippets=2)
            if spec_lines or spec_snippets or summary_snippets:
                st.markdown("**Spec details**")
                for line in spec_lines:
                    st.write(f"- {line}")
                for snippet in spec_snippets:
                    st.write(f"- {snippet}")
                for snippet in summary_snippets:
                    st.write(f"- {snippet}")

        if focus == {"general"}:
            summary_snippets = extract_relevant_snippets(summary_text, query_text, max_snippets=3)
            if summary_snippets:
                st.markdown("**Relevant details**")
                for snippet in summary_snippets:
                    st.write(f"- {snippet}")

        support_labels = [
            f"{item['modality']} ({item['score']:.4f})"
            for item in result["supporting_items"]
        ]
        st.caption("Supporting matches: " + ", ".join(support_labels))

    st.divider()


def main() -> None:
    st.set_page_config(page_title="Car Retrieval Demo", layout="wide")
    st.title("Car Retrieval Demo")
    st.write(
        "Search your car dataset with text, image, PDF, or finance-document queries. "
        "Results are grouped by car and include a photo plus finance or spec details when relevant."
    )

    with st.sidebar:
        st.header("Search Settings")
        data_dir = st.text_input("Index directory", value="data")
        dataset_root = st.text_input("Dataset root", value="../dataset")
        modality = st.selectbox(
            "Search mode",
            options=["all", "image", "pdf", "finance", "summary"],
            index=0,
        )
        top_k = st.slider("Top K cars", min_value=1, max_value=15, value=5)
        st.caption("Run `python embed_index.py` first if the index files do not exist yet.")

    query_label = st.radio("Query type", options=list(QUERY_TYPE_OPTIONS.keys()), horizontal=True)
    query_type = QUERY_TYPE_OPTIONS[query_label]

    text_query = ""
    uploaded_file = None

    if query_type == "text":
        text_query = st.text_area(
            "Question",
            height=140,
            placeholder="Example: AWD family SUV with low monthly payment and good cargo space",
        )
    elif query_type == "image":
        uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
    elif query_type == "pdf":
        uploaded_file = st.file_uploader("Upload a PDF brochure", type=["pdf"])
    else:
        uploaded_file = st.file_uploader("Upload a finance markdown file", type=["md", "txt"])

    render_uploaded_preview(query_type, uploaded_file)
    focus = detect_query_focus(query_type, text_query)

    if st.button("Search", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading index and generating query embedding..."):
                settings, vectors, metadata, embedder = load_search_resources(dataset_root, data_dir)
                query_vector = build_query_vector(
                    query_type=query_type,
                    text_query=text_query,
                    uploaded_file=uploaded_file,
                    embedder=embedder,
                )
                car_results = build_car_results(
                    query_vector,
                    vectors,
                    metadata,
                    top_k=top_k,
                    modality_filter=modality,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return

        if not car_results:
            st.warning("No results were returned.")
            return

        st.subheader("Results")
        for result in car_results:
            render_result_card(
                result,
                settings,
                query_type=query_type,
                query_text=text_query,
                focus=focus,
            )

    with st.expander("How to run"):
        st.code(
            "cd car_retrieval\n"
            ".\\.venv\\Scripts\\Activate.ps1\n"
            "streamlit run app.py",
            language="powershell",
        )


if __name__ == "__main__":
    main()
