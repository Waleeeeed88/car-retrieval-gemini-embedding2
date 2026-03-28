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


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

        .stApp {
            background: #0e0e0d;
            color: #e7e5e1;
        }
        .block-container {
            max-width: 1536px;
            padding-top: 6.5rem;
            padding-bottom: 4rem;
        }
        .top-nav {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(14, 14, 13, 0.96);
            border-bottom: 1px solid rgba(72, 72, 69, 0.2);
            margin: -6.5rem -1rem 2.5rem;
            padding: 1.75rem 2.5rem;
            backdrop-filter: blur(10px);
        }
        .top-nav-inner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1536px;
            margin: 0 auto;
        }
        .brand-mark {
            font-family: "Noto Serif", serif;
            color: #e7e5e1;
            font-size: 1.7rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
        }
        .nav-links {
            display: flex;
            gap: 2.5rem;
            align-items: center;
        }
        .nav-link {
            color: #acaba7;
            font-family: "Inter", sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.65rem;
        }
        .nav-link.active {
            color: #e7e5e1;
            border-bottom: 1px solid #fff2de;
            padding-bottom: 0.25rem;
        }
        .hero-frame {
            margin-bottom: 2.5rem;
        }
        .editorial-grid {
            display: grid;
            grid-template-columns: repeat(12, minmax(0, 1fr));
            gap: 1.4rem;
        }
        .hero-media {
            grid-column: span 12;
            overflow: hidden;
            border: 1px solid rgba(72, 72, 69, 0.2);
        }
        .hero-meta {
            grid-column: span 12;
        }
        .hero-kicker {
            font-family: "Inter", sans-serif;
            font-size: 0.62rem;
            letter-spacing: 0.3em;
            text-transform: uppercase;
            color: #acaba7;
            margin-bottom: 1rem;
        }
        .hero-match {
            font-family: "Inter", sans-serif;
            font-size: 0.62rem;
            letter-spacing: 0.12em;
            color: #fff2de;
            background: #252623;
            padding: 0.3rem 0.45rem;
            white-space: nowrap;
        }
        .hero-title {
            font-family: "Noto Serif", serif;
            font-size: clamp(3.2rem, 6vw, 6rem);
            line-height: 0.95;
            font-weight: 400;
            margin: 0;
            color: #e7e5e1;
            letter-spacing: -0.04em;
        }
        .hero-copy {
            margin-top: 1.5rem;
            color: rgba(231, 229, 225, 0.9);
            max-width: 28rem;
            font-size: 1.25rem;
            font-family: "Noto Serif", serif;
            font-style: italic;
            line-height: 1.65;
        }
        .divider-line {
            border-top: 1px solid rgba(72, 72, 69, 0.2);
            margin: 2rem 0;
        }
        .section-label {
            font-family: "Inter", sans-serif;
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.3em;
            color: #acaba7;
            margin-bottom: 0.7rem;
        }
        .section-title {
            font-family: "Noto Serif", serif;
            font-size: clamp(2rem, 3vw, 3rem);
            line-height: 1.05;
            color: #e7e5e1;
            margin: 0 0 0.55rem 0;
        }
        .section-copy {
            color: #acaba7;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }
        .query-shell, .result-shell {
            margin-bottom: 2.5rem;
        }
        .result-title {
            font-family: "Noto Serif", serif;
            font-size: clamp(2.7rem, 5vw, 4.6rem);
            font-weight: 400;
            color: #e7e5e1;
            margin-bottom: 0.6rem;
            letter-spacing: -0.04em;
        }
        .result-subtle {
            font-family: "Inter", sans-serif;
            font-size: 0.62rem;
            letter-spacing: 0.18em;
            color: #acaba7;
            margin-bottom: 1rem;
            text-transform: uppercase;
        }
        .chip-row {
            margin: 0.45rem 0 0.75rem;
            text-align: left;
        }
        .chip {
            display: inline-block;
            border: 1px solid rgba(72, 72, 69, 0.35);
            padding: 0.25rem 0.5rem;
            margin: 0 0.35rem 0.35rem 0;
            font-size: 0.62rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #fff2de;
            background: #252623;
        }
        .mini-heading {
            margin-top: 1rem;
            margin-bottom: 0.55rem;
            font-family: "Inter", sans-serif;
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.3em;
            color: #acaba7;
        }
        .support-line {
            margin-top: 1rem;
            font-size: 0.72rem;
            color: #acaba7;
        }
        .option-card {
            background: transparent;
            border: 1px solid rgba(72, 72, 69, 0.2);
            padding: 1.5rem 1.2rem;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .option-icon {
            width: 56px;
            height: 56px;
            border-radius: 999px;
            background: #e7e5e1;
            color: #0e0e0d;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        .option-title {
            font-family: "Noto Serif", serif;
            color: #e7e5e1;
            font-size: 1.8rem;
            margin-bottom: 0.7rem;
        }
        .option-copy {
            color: #acaba7;
            font-size: 0.92rem;
            line-height: 1.7;
            max-width: 21rem;
        }
        .option-pill {
            margin-top: 1.2rem;
            display: inline-block;
            background: transparent;
            border: 1px solid rgba(72, 72, 69, 0.3);
            padding: 0.35rem 0.75rem;
            color: #acaba7;
            font-size: 0.62rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }
        .about-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.9rem 1.2rem;
            background: #191a18;
            border: 1px solid rgba(72, 72, 69, 0.2);
            color: #e7e5e1;
            font-family: "Inter", sans-serif;
        }
        .center-wrap {
            text-align: center;
        }
        .query-strip {
            background: transparent;
            border-top: 1px solid rgba(72, 72, 69, 0.2);
            padding-top: 1.25rem;
        }
        .meta-stack {
            display: grid;
            gap: 0.85rem;
        }
        .meta-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.95rem 0;
            border-bottom: 1px solid rgba(72, 72, 69, 0.2);
        }
        .meta-key {
            font-family: "Inter", sans-serif;
            font-size: 0.62rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: #acaba7;
        }
        .meta-value {
            color: #e7e5e1;
            font-size: 0.88rem;
        }
        .rationale-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 2rem;
            margin-top: 1.4rem;
        }
        .rationale-item {
            border-left: 1px solid rgba(72, 72, 69, 0.4);
            padding-left: 1rem;
        }
        .rationale-title {
            font-family: "Inter", sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #e7e5e1;
            margin-bottom: 0.8rem;
        }
        .rationale-copy {
            color: #acaba7;
            font-size: 0.88rem;
            line-height: 1.7;
        }
        .asset-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        .asset-row {
            display: flex;
            gap: 1.4rem;
            align-items: flex-start;
        }
        .asset-icon {
            width: 68px;
            height: 92px;
            border: 1px solid rgba(72, 72, 69, 0.15);
            background: #1f201e;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #acaba7;
            font-size: 0.7rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            flex-shrink: 0;
        }
        .asset-copy {
            flex: 1;
            border-bottom: 1px solid rgba(72, 72, 69, 0.2);
            padding-bottom: 1.2rem;
        }
        .asset-title {
            font-family: "Inter", sans-serif;
            font-size: 0.68rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #e7e5e1;
            margin-bottom: 0.4rem;
        }
        .asset-body {
            color: #acaba7;
            font-size: 0.82rem;
            line-height: 1.6;
            margin-bottom: 0.75rem;
        }
        .asset-link {
            color: #fff2de;
            font-family: "Inter", sans-serif;
            font-size: 0.62rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(255, 242, 222, 0.4);
            display: inline-block;
            padding-bottom: 0.15rem;
        }
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div, .stFileUploader section, .stSlider {
            background: #191a18 !important;
            border-color: rgba(72, 72, 69, 0.2) !important;
        }
        .stTextInput input, .stTextArea textarea {
            color: #e7e5e1 !important;
            border-radius: 0 !important;
        }
        .stButton > button, .stFormSubmitButton > button {
            border-radius: 0 !important;
            background: #e7e5e1 !important;
            color: #0e0e0d !important;
            border: none !important;
            font-weight: 600 !important;
            min-height: 3rem;
            text-transform: uppercase !important;
            letter-spacing: 0.16em !important;
            font-size: 0.62rem !important;
        }
        .stRadio label, .stCaption, .stMarkdown, .stTextInput label, .stSelectbox label, .stSlider label {
            color: #d7d3cb;
        }
        div[data-testid="stExpander"] {
            background: transparent;
            border: 1px solid rgba(72, 72, 69, 0.2);
            border-radius: 0;
        }
        div[data-testid="stExpander"] details summary p {
            color: #e7e5e1 !important;
        }
        [data-testid="stImage"] img {
            border-radius: 0;
        }
        @media (min-width: 1024px) {
            .hero-media {
                grid-column: span 8;
            }
            .hero-meta {
                grid-column: span 4;
                padding-left: 2rem;
                padding-bottom: 0.5rem;
            }
        }
        @media (max-width: 900px) {
            .rationale-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_shell_open() -> None:
    return


def render_shell_close() -> None:
    return


def render_chip_row(values: list[str]) -> None:
    if not values:
        return
    chips = "".join(f'<span class="chip">{value}</span>' for value in values)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def render_option_card(*, icon: str, title: str, copy: str, pill: str) -> None:
    return


def render_top_nav() -> None:
    st.markdown(
        """
        <div class="top-nav">
          <div class="top-nav-inner">
            <div class="brand-mark">AERIS</div>
            <div class="nav-links">
              <span class="nav-link">Archive</span>
              <span class="nav-link active">Search</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meta_rows(rows: list[tuple[str, str]]) -> None:
    if not rows:
        return
    parts = ['<div class="meta-stack">']
    for key, value in rows:
        parts.append(
            f'<div class="meta-row"><span class="meta-key">{key}</span><span class="meta-value">{value}</span></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_rationale_items(items: list[tuple[str, str]]) -> None:
    if not items:
        return
    parts = ['<div class="rationale-grid">']
    for title, copy in items[:3]:
        parts.append(
            '<div class="rationale-item">'
            f'<div class="rationale-title">{title}</div>'
            f'<div class="rationale-copy">{copy}</div>'
            '</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_asset_rows(rows: list[tuple[str, str, str, str]]) -> None:
    if not rows:
        return
    parts = ['<div class="asset-grid">']
    for icon, title, body, link_label in rows:
        parts.append(
            '<div class="asset-row">'
            f'<div class="asset-icon">{icon}</div>'
            '<div class="asset-copy">'
            f'<div class="asset-title">{title}</div>'
            f'<div class="asset-body">{body}</div>'
            f'<div class="asset-link">{link_label}</div>'
            '</div></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


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
    finance_snapshot = metadata.get("finance_snapshot", {})
    matched_preview = str(best_match.get("preview") or "").strip()
    if not matched_preview:
        matched_preview = f"A strong retrieval match surfaced through {str(best_match['modality']).upper()} context."

    match_percent = max(0, min(99, int(round(float(result["score"]) * 100))))

    left, right = st.columns([7, 5], gap="large")
    with left:
        if primary_image and Path(primary_image).exists():
            st.image(str(primary_image), use_container_width=True)
        else:
            st.caption("No image available")

    with right:
        st.markdown(
            (
                '<div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:1rem;">'
                '<span class="hero-kicker">Match</span>'
                f'<span class="hero-match">{match_percent}% MATCH</span>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="result-title">{result["car_label"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
        render_chip_row(
            [
                str(best_match["modality"]).upper(),
                str(metadata.get("category") or "vehicle"),
                str(metadata.get("fuel_type") or "fuel n/a"),
            ]
        )

    meta_rows: list[tuple[str, str]] = []
    if metadata.get("drivetrain"):
        meta_rows.append(("Drivetrain", str(metadata["drivetrain"])))
    if metadata.get("fuel_type"):
        meta_rows.append(("Fuel", str(metadata["fuel_type"])))
    if metadata.get("price_range"):
        meta_rows.append(("Price", str(metadata["price_range"])))
    if finance_snapshot.get("msrp"):
        meta_rows.append(("MSRP", str(finance_snapshot["msrp"])))
    if meta_rows:
        render_meta_rows(meta_rows[:4])

    summary_snippets = extract_relevant_snippets(summary_text, query_text, max_snippets=1)
    spec_excerpt = (
        extract_relevant_snippets(spec_text, query_text, max_snippets=1)
        or ["No spec details available."]
    )[0]
    finance_excerpt = (
        finance_lines_from_snapshot(finance_snapshot, query_text)
        or ["No finance details available."]
    )[0]

    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    with st.expander(f"Details: {result['car_label']}", expanded=False):
        if summary_snippets:
            st.markdown("**Summary**")
            st.write(summary_snippets[0])
        st.markdown("**Specs**")
        st.write(spec_excerpt)
        st.markdown("**Finance**")
        st.write(finance_excerpt)


def main() -> None:
    st.set_page_config(
        page_title="Car Retrieval",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_app_styles()
    st.markdown('<div class="hero-kicker">Car Retrieval</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Search</h1>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Text, image, PDF, or finance file.</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    _unused_banner = (
            '<div class="center-wrap"><span class="about-chip">Search modes · text · image · PDF · finance</span></div>',
    )
    default_settings = get_settings()
    with st.expander("Advanced settings", expanded=False):
        left, right, far = st.columns([1.4, 1.4, 1], gap="medium")
        with left:
            dataset_root = st.text_input("Dataset root", value=str(default_settings.dataset_root))
        with right:
            data_dir = st.text_input("Index directory", value=str(default_settings.data_dir))
        with far:
            top_k = st.slider("Top K", min_value=1, max_value=15, value=5)
            modality = st.selectbox(
                "Mode",
                options=["all", "image", "pdf", "finance", "summary"],
                index=0,
            )
        st.caption("Rebuild index with `python embed_index.py` if needed.")

    if "dataset_root" not in locals():
        dataset_root = str(default_settings.dataset_root)
    if "data_dir" not in locals():
        data_dir = str(default_settings.data_dir)
    if "top_k" not in locals():
        top_k = 5
    if "modality" not in locals():
        modality = "all"

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    st.markdown('', unsafe_allow_html=True)
    st.markdown('', unsafe_allow_html=True)
    opt1, opt2 = st.columns(2, gap="large")
    with opt1:
        render_option_card(
            icon="⚡",
            title="Text Search",
            copy="Describe the vehicle you want and let the index surface strong matches from summaries, specs, and finance context.",
            pill="Fastest flow",
        )
    with opt2:
        render_option_card(
            icon="▣",
            title="Reference Search",
            copy="Use an image, brochure, or finance file when you already have a reference vehicle or document to match against.",
            pill="Image · PDF · finance",
        )
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="query-strip">', unsafe_allow_html=True)
    with st.form("search-form", clear_on_submit=False):
        query_label = st.radio("Query type", options=list(QUERY_TYPE_OPTIONS.keys()), horizontal=True, label_visibility="collapsed")
        query_type = QUERY_TYPE_OPTIONS[query_label]

        text_query = ""
        uploaded_file = None

        if query_type == "text":
            text_query = st.text_area(
                "Question",
                height=120,
                label_visibility="collapsed",
                placeholder="AWD family SUV with low monthly payment and good cargo space",
            )
        elif query_type == "image":
            uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        elif query_type == "pdf":
            uploaded_file = st.file_uploader("Upload a PDF brochure", type=["pdf"], label_visibility="collapsed")
        else:
            uploaded_file = st.file_uploader("Upload a finance markdown file", type=["md", "txt"], label_visibility="collapsed")

        submit = st.form_submit_button("Search", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    render_uploaded_preview(query_type, uploaded_file)
    focus = detect_query_focus(query_type, text_query)

    if submit:
        try:
            with st.spinner("Searching index..."):
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

        top_line = f"{len(car_results)} results" if len(car_results) != 1 else "1 result"
        st.markdown('<div style="height:3rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Retrieved Set</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
        st.markdown('', unsafe_allow_html=True)
        render_chip_row(
            [
                top_line,
                f"mode {modality}",
                f"focus {', '.join(sorted(focus))}",
            ]
        )
        for result in car_results:
            render_result_card(
                result,
                settings,
                query_type=query_type,
                query_text=text_query,
                focus=focus,
            )

    with st.expander("Run locally", expanded=False):
        st.code(
            "cd car_retrieval\n"
            ".\\.venv\\Scripts\\Activate.ps1\n"
            "streamlit run app.py",
            language="powershell",
        )


if __name__ == "__main__":
    main()
