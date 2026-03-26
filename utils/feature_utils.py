from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

COLOR_PALETTE = {
    "black": (30, 30, 30),
    "white": (235, 235, 235),
    "gray": (145, 145, 145),
    "silver": (190, 190, 200),
    "red": (190, 45, 45),
    "blue": (55, 90, 190),
    "green": (55, 135, 70),
    "yellow": (220, 190, 40),
    "orange": (220, 120, 35),
    "brown": (120, 80, 50),
    "beige": (205, 185, 145),
}

FEATURE_PATTERNS = {
    "awd": r"\bawd\b|all[- ]wheel drive",
    "fwd": r"\bfwd\b|front[- ]wheel drive",
    "rwd": r"\brwd\b|rear[- ]wheel drive",
    "4wd": r"\b4wd\b|four[- ]wheel drive",
    "hybrid": r"\bhybrid\b",
    "plug_in_hybrid": r"plug[- ]in hybrid|phev",
    "electric": r"\ball[- ]electric\b|\bfully electric\b|\belectric vehicle\b|\bbattery electric\b",
    "turbo": r"\bturbo\b",
    "sedan": r"\bsedan\b",
    "suv": r"\bsuv\b",
    "hatchback": r"\bhatchback\b",
    "wagon": r"\bwagon\b",
    "pickup": r"\bpickup\b|truck",
    "minivan": r"\bminivan\b",
    "coupe": r"\bcoupe\b",
    "convertible": r"\bconvertible\b",
    "three_row": r"3-row|three-row",
    "apple_carplay": r"apple carplay",
    "android_auto": r"android auto",
    "blind_spot_monitoring": r"blind[- ]spot",
    "adaptive_cruise_control": r"adaptive cruise",
    "lane_keep_assist": r"lane[- ]keep|lane keeping",
    "remote_start": r"remote start|remote engine start",
    "heated_seats": r"heated seats?",
    "ventilated_seats": r"ventilated seats?",
    "panoramic_roof": r"panoramic roof|panoramic moonroof|panoramic sunroof",
    "wireless_charging": r"wireless charging",
    "third_row": r"third row",
    "captains_chairs": r"captain'?s chairs?",
}

FINANCE_SNAPSHOT_KEYS = (
    "msrp",
    "total_before_tax",
    "apr",
    "finance_payment",
    "finance_down_payment",
    "lease_payment",
    "lease_term",
    "due_at_signing",
    "incentives",
    "warranty",
)


def _normalize_phrase(value: str) -> str:
    return value.replace("_", " ").strip()


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(value.strip())
    return ordered


def extract_color_mentions(*texts: str) -> list[str]:
    combined = " ".join(texts).lower()
    found: list[str] = []
    for color in COLOR_PALETTE:
        if re.search(rf"\b{re.escape(color)}\b", combined):
            found.append(color)
    return _deduplicate(found)


def extract_feature_keywords(*texts: str) -> list[str]:
    combined = " ".join(texts).lower()
    found: list[str] = []
    for label, pattern in FEATURE_PATTERNS.items():
        if re.search(pattern, combined):
            found.append(_normalize_phrase(label))
    return _deduplicate(found)


def _nearest_palette_name(rgb: tuple[int, int, int]) -> str:
    def distance(color_name: str) -> int:
        target = COLOR_PALETTE[color_name]
        return sum((rgb[index] - target[index]) ** 2 for index in range(3))

    return min(COLOR_PALETTE, key=distance)


def detect_image_colors(image_path: Path, *, max_colors: int = 3) -> list[str]:
    if not image_path.exists():
        return []

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        left = int(width * 0.2)
        top = int(height * 0.2)
        right = max(left + 1, int(width * 0.8))
        bottom = max(top + 1, int(height * 0.8))
        cropped = rgb.crop((left, top, right, bottom))
        reduced = cropped.resize((80, 80))
        quantized = reduced.quantize(colors=8, method=Image.Quantize.MEDIANCUT).convert("RGB")
        color_counts = quantized.getcolors(maxcolors=80 * 80) or []

    ranked = sorted(color_counts, key=lambda item: item[0], reverse=True)
    names = [_nearest_palette_name(tuple(color)) for _, color in ranked]
    return _deduplicate(names)[:max_colors]


def extract_spec_facts(*texts: str) -> list[str]:
    combined = " ".join(texts)
    fact_patterns = [
        ("horsepower", r"(\d{2,4})\s*(?:hp|horsepower)\b"),
        ("torque", r"(\d{2,4})\s*(?:lb[.-]?\s*ft|lb-ft|lb ft)\b"),
        ("mpg", r"(\d{2,3})\s*mpg\b"),
        ("range", r"(\d{2,4})\s*(?:mile|miles)\s+range\b"),
        ("towing", r"(\d[\d,]{2,6})\s*(?:lb|lbs|pounds)\b"),
        ("seating", r"\b(\d)\s*(?:passenger|passengers|seat|seats)\b"),
        ("cargo", r"(\d{1,3}(?:\.\d+)?)\s*(?:cu\.?\s*ft|cubic feet)\b"),
        ("battery", r"(\d{1,3}(?:\.\d+)?)\s*kwh\b"),
    ]

    facts: list[str] = []
    for label, pattern in fact_patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if not match:
            continue
        raw_value = match.group(1).replace(",", "")
        numeric_value = float(raw_value)
        if label == "horsepower" and not (70 <= numeric_value <= 1500):
            continue
        if label == "torque" and not (70 <= numeric_value <= 2000):
            continue
        if label == "mpg" and not (5 <= numeric_value <= 200):
            continue
        if label == "range" and not (20 <= numeric_value <= 1000):
            continue
        if label == "towing" and not (500 <= numeric_value <= 25000):
            continue
        if label == "seating" and not (2 <= numeric_value <= 9):
            continue
        if label == "cargo" and not (1 <= numeric_value <= 300):
            continue
        if label == "battery" and not (1 <= numeric_value <= 300):
            continue

        value = str(int(numeric_value)) if numeric_value.is_integer() else raw_value
        if label == "horsepower":
            facts.append(f"{value} hp")
        elif label == "torque":
            facts.append(f"{value} lb-ft torque")
        elif label == "mpg":
            facts.append(f"{value} mpg")
        elif label == "range":
            facts.append(f"{value} mile range")
        elif label == "towing":
            facts.append(f"{value} lb towing")
        elif label == "seating":
            facts.append(f"{value} seats")
        elif label == "cargo":
            facts.append(f"{value} cu ft cargo")
        elif label == "battery":
            facts.append(f"{value} kWh battery")
    return _deduplicate(facts)


def describe_image_visual_traits(image_path: Path) -> list[str]:
    if not image_path.exists():
        return []

    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((120, 120))
        width, height = rgb.size
        pixels = list(rgb.getdata())

    brightness = sum((r + g + b) / 3 for r, g, b in pixels) / len(pixels)
    saturation = sum((max(r, g, b) - min(r, g, b)) for r, g, b in pixels) / len(pixels)
    aspect_ratio = width / max(height, 1)

    traits: list[str] = []
    if aspect_ratio >= 1.2:
        traits.append("landscape image")
    elif aspect_ratio <= 0.8:
        traits.append("portrait image")
    else:
        traits.append("square-ish image")

    if brightness >= 185:
        traits.append("bright image")
    elif brightness <= 90:
        traits.append("dark image")
    else:
        traits.append("balanced brightness")

    if saturation >= 75:
        traits.append("high saturation")
    elif saturation <= 35:
        traits.append("muted saturation")
    else:
        traits.append("moderate saturation")

    return traits


def build_feature_payload(
    metadata: dict[str, Any],
    *,
    summary_text: str,
    finance_text: str,
    pdf_text: str,
    image_path: Path | None = None,
) -> dict[str, Any]:
    finance_snapshot = metadata.get("finance_snapshot") or {}
    snapshot_text = " ".join(
        str(finance_snapshot.get(key, "")).strip()
        for key in FINANCE_SNAPSHOT_KEYS
    )

    image_colors = detect_image_colors(image_path) if image_path else []
    image_traits = describe_image_visual_traits(image_path) if image_path else []
    text_colors = extract_color_mentions(summary_text, finance_text, pdf_text, snapshot_text)
    colors = _deduplicate(image_colors + text_colors)

    raw_features = extract_feature_keywords(
        summary_text,
        finance_text,
        pdf_text,
        snapshot_text,
        str(metadata.get("model", "")),
        str(metadata.get("make", "")),
        str(metadata.get("market", "")),
        str(metadata.get("category", "")),
        str(metadata.get("drivetrain", "")),
        str(metadata.get("fuel_type", "")),
    )

    category = (
        metadata.get("category")
        or metadata.get("body_type")
        or metadata.get("vehicle_type")
        or metadata.get("segment")
    )
    if category:
        raw_features.append(str(category))
    if metadata.get("drivetrain"):
        raw_features.append(str(metadata["drivetrain"]))
    if metadata.get("fuel_type"):
        raw_features.append(str(metadata["fuel_type"]))

    features = _deduplicate(raw_features)
    spec_facts = extract_spec_facts(summary_text, finance_text, pdf_text, snapshot_text)

    return {
        "colors": colors,
        "features": features,
        "spec_facts": spec_facts,
        "image_traits": image_traits,
        "searchable_attributes": {
            "make": metadata.get("make"),
            "model": metadata.get("model"),
            "year": metadata.get("year"),
            "market": metadata.get("market"),
            "category": category,
            "drivetrain": metadata.get("drivetrain"),
            "fuel_type": metadata.get("fuel_type"),
            "price_range": metadata.get("price_range"),
        },
    }


def build_embedding_context_text(
    *,
    modality: str,
    metadata: dict[str, Any],
    feature_payload: dict[str, Any],
    original_text: str = "",
) -> str:
    finance_snapshot = metadata.get("finance_snapshot") or {}
    attributes = feature_payload.get("searchable_attributes", {})
    colors = ", ".join(feature_payload.get("colors", [])) or "not specified"
    features = ", ".join(feature_payload.get("features", [])) or "not specified"
    spec_facts = ", ".join(feature_payload.get("spec_facts", [])) or "not specified"
    image_traits = ", ".join(feature_payload.get("image_traits", [])) or "not specified"

    lines = [
        f"Document type: {modality}",
        f"Car: {metadata.get('year', '')} {metadata.get('make', '')} {metadata.get('model', '')}".strip(),
        f"Market: {attributes.get('market') or 'not specified'}",
        f"Category: {attributes.get('category') or 'not specified'}",
        f"Drivetrain: {attributes.get('drivetrain') or 'not specified'}",
        f"Fuel type: {attributes.get('fuel_type') or 'not specified'}",
        f"Price range: {attributes.get('price_range') or 'not specified'}",
        f"Colors: {colors}",
        f"Features: {features}",
        f"Spec facts: {spec_facts}",
        f"Image traits: {image_traits}",
    ]

    if modality in {"finance", "summary", "pdf"} and finance_snapshot:
        for key in FINANCE_SNAPSHOT_KEYS:
            value = finance_snapshot.get(key)
            if value:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")

    if original_text.strip():
        lines.append("Original content:")
        lines.append(original_text.strip())

    return "\n".join(lines)
