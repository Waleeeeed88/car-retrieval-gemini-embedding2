from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, ImageFilter, ImageStat

SNAPSHOT_DATE = "March 27, 2026"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
AUDIT_REPORT_PATH = PROJECT_ROOT / "data" / "image_audit.md"
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
        )
    }
)
HTML_CACHE: dict[str, str] = {}
TEXT_CACHE: dict[str, str] = {}
DISABLED_HOSTS: set[str] = set()

RELEVANT_FUEL_TYPES = {"electric", "hybrid", "plug-in hybrid"}
SUSPICIOUS_SPEC_TOKENS = (
    "skip to main",
    "build & price",
    "dataset id",
    "pdf.txt",
    "finance asset",
    "image asset",
    "gm rewards",
    "vehicles shop",
    "current offers",
    "placeholder",
    "my account",
    "vehicle summary",
    "view details",
    "#/#",
    "view offer",
)
TRANSMISSION_PATTERNS = [
    r"\b\d{1,2}-speed automatic\b",
    r"\b\d{1,2}-speed dual-clutch\b",
    r"\bcontinuously variable transmission\b",
    r"\bcvt\b",
    r"\bmanual transmission\b",
    r"\bautomatic transmission\b",
]


@dataclass
class ImageAuditResult:
    slug: str
    width: int
    height: int
    objective_flags: list[str]
    heuristic_flags: list[str]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def parse_urls(text: str) -> list[str]:
    found = re.findall(r"https?://[^\s)]+", text)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in found:
        cleaned = url.rstrip(".,;")
        if cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def registrable_host(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def collect_source_urls(car_dir: Path, metadata: dict[str, Any]) -> list[str]:
    pdf_txt = car_dir / "pdf.txt"
    urls: list[str] = []
    if pdf_txt.exists():
        urls.extend(parse_urls(read_text(pdf_txt)))

    source_urls = metadata.get("source_urls", {})
    if isinstance(source_urls, dict):
        for value in source_urls.values():
            if isinstance(value, str) and value.startswith("http"):
                urls.append(value)

    deduped: list[str] = []
    seen: set[str] = set()
    primary = next((url for url in urls if url.startswith("http")), "")
    allowed_host = registrable_host(primary)
    for url in urls:
        if not url.startswith("http"):
            continue
        if allowed_host and registrable_host(url) != allowed_host:
            continue
        if url not in seen:
            seen.add(url)
            deduped.append(url)

    if not deduped:
        model_page = source_urls.get("model_page")
        if isinstance(model_page, str) and model_page.startswith("http"):
            deduped.append(model_page)
    return deduped


def fetch_html(url: str) -> str:
    host = registrable_host(url)
    if host in DISABLED_HOSTS:
        raise requests.RequestException(f"Skipping disabled host: {host}")
    if url in HTML_CACHE:
        return HTML_CACHE[url]
    try:
        response = SESSION.get(url, timeout=12)
        response.raise_for_status()
        text = response.text
        HTML_CACHE[url] = text
        return text
    except Exception:
        if host:
            DISABLED_HOSTS.add(host)
        raise


def extract_meta_content(page_html: str, attribute: str, value: str) -> str:
    pattern = rf'<meta[^>]+{attribute}="{re.escape(value)}"[^>]+content="([^"]+)"'
    match = re.search(pattern, page_html, flags=re.IGNORECASE)
    return html.unescape(match.group(1)).strip() if match else ""


def html_to_text(page_html: str) -> str:
    stripped = re.sub(r"(?is)<script.*?>.*?</script>", " ", page_html)
    stripped = re.sub(r"(?is)<style.*?>.*?</style>", " ", stripped)
    stripped = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", stripped)
    stripped = re.sub(r"(?is)<svg.*?>.*?</svg>", " ", stripped)
    stripped = re.sub(r"(?is)<[^>]+>", " ", stripped)
    stripped = html.unescape(stripped)
    stripped = stripped.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", stripped).strip()


def fetch_page_text(url: str) -> str:
    if url in TEXT_CACHE:
        return TEXT_CACHE[url]
    page_html = fetch_html(url)
    text = html_to_text(page_html)
    TEXT_CACHE[url] = text
    return text


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for sentence in sentences:
        sentence = " ".join(sentence.split())
        if len(sentence) >= 20:
            cleaned.append(sentence)
    return cleaned


def find_sentence(text: str, keywords: list[str]) -> str | None:
    lowered = [keyword.lower() for keyword in keywords]
    for sentence in split_sentences(text):
        haystack = sentence.lower()
        if any(keyword in haystack for keyword in lowered):
            return sentence
    return None


def clean_value(value: str) -> str:
    return " ".join(html.unescape(value).split()).strip(" ,;")


def find_pattern(text: str, patterns: list[str], formatter=None) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if formatter:
            return formatter(match)
        if match.lastindex:
            return clean_value(match.group(1))
        return clean_value(match.group(0))
    return None


def find_horsepower(text: str) -> str | None:
    value = find_pattern(
        text,
        [
            r"\b(\d{2,4})\s*(?:hp|horsepower)\b",
            r"\bup to\s*(\d{2,4})\s*(?:hp|horsepower)\b",
            r"\b(\d{2,4})\+?\s*hp\b",
        ],
    )
    return f"{value} hp" if value else None


def find_torque(text: str) -> str | None:
    value = find_pattern(
        text,
        [
            r"\b(\d{2,4})\s*(?:lb\.?-?ft|lb-ft|lb ft)\b",
            r"\bup to\s*(\d{2,4})\s*(?:lb\.?-?ft|lb-ft|lb ft)\b",
        ],
    )
    return f"{value} lb-ft" if value else None


def find_fuel_or_range(text: str) -> str | None:
    mpg = find_pattern(
        text,
        [
            r"\b(\d{1,2}/\d{1,2}\s*mpg)\b",
            r"\bup to\s*(\d{1,2}\s*mpg)\b",
        ],
    )
    if mpg:
        return mpg.upper()
    sentence = find_sentence(text, ["mile range", "miles of range", "epa-est", "range"])
    if sentence:
        return sentence
    return None


def find_transmission(text: str) -> str | None:
    value = find_pattern(text, TRANSMISSION_PATTERNS)
    return value.upper() if value == "cvt" else value


def find_seating(text: str) -> str | None:
    match = re.search(
        r"\b([5-9])[- ](?:passenger|passengers|seater|seat)\b|\bseating for ([5-9])\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        value = next(group for group in match.groups() if group)
        return f"{value} passengers"
    sentence = find_sentence(text, ["three-row", "3-row"])
    return sentence


def find_cargo(text: str) -> str | None:
    value = find_pattern(
        text,
        [
            r"\b(\d+(?:\.\d+)?)\s*(?:cu\.?\s*ft\.?|cubic feet)\b",
        ],
    )
    return f"{value} cu. ft." if value else None


def find_towing(text: str) -> str | None:
    match = re.search(
        r"\btowing[^.]{0,50}?(\d{1,2},\d{3})\s*(?:lbs?|pounds)\b|\b(\d{1,2},\d{3})\s*(?:lbs?|pounds)[^.]{0,50}?tow",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        value = next(group for group in match.groups() if group)
        return f"{value} lbs"
    return None


def find_dimensions(text: str) -> str | None:
    sentence = find_sentence(text, ["wheelbase", "ground clearance", "length", "width", "dimensions"])
    return sentence


def derive_body_style(metadata: dict[str, Any]) -> str:
    category = (
        metadata.get("category")
        or metadata.get("body_type")
        or metadata.get("vehicle_type")
        or metadata.get("segment")
        or "Not published on official page at capture time."
    )
    return str(category)


def derive_powertrain(metadata: dict[str, Any], text: str) -> str:
    sentence = find_sentence(
        text,
        [
            "battery-electric",
            "electric suv",
            "electric sedan",
            "plug-in hybrid",
            "hybrid",
            "turbocharged",
            "engine",
            "motor",
        ],
    )
    if sentence:
        return sentence

    fuel_type = str(metadata.get("fuel_type") or "").strip().lower()
    drivetrain = str(metadata.get("drivetrain") or "").strip()
    if fuel_type:
        if drivetrain:
            return f"{fuel_type.title()} powertrain with {drivetrain}"
        return f"{fuel_type.title()} powertrain"
    return "Not published on official page at capture time."


def derive_battery_or_charging(metadata: dict[str, Any], text: str) -> str | None:
    fuel_type = str(metadata.get("fuel_type") or "").strip().lower()
    if fuel_type not in RELEVANT_FUEL_TYPES:
        return None
    sentence = find_sentence(text, ["kwh", "charging", "charge", "fast charging", "all-electric mode"])
    return sentence or "Not published on official page at capture time."


def official_value(value: str | None) -> str:
    return value or "Not published on official page at capture time."


def sanitize_spec_value(value: str | None, *, label: str) -> str | None:
    if not value:
        return None
    cleaned = clean_value(value)
    lowered = cleaned.lower()
    if not cleaned:
        return None
    if any(token in lowered for token in SUSPICIOUS_SPEC_TOKENS):
        return None
    if any(token in cleaned for token in ("â", "Ã", "\ufffd")):
        return None
    if cleaned.startswith("#") or "`" in cleaned:
        return None
    if re.search(r"https?://|</|\">|\[\[|\{\"|[^\x00-\x7F]{2,}", cleaned):
        return None
    if cleaned.count(" - ") >= 5:
        return None
    if len(cleaned) > 220:
        return None
    if label in {"powertrain", "fuel economy / range", "seating", "transmission"} and len(cleaned) > 160:
        return None
    if label == "fuel economy / range" and lowered.startswith("here"):
        return None
    if label == "key dimensions" and (lowered.startswith("* ") or lowered.startswith("what are the dimensions")):
        return None
    return cleaned


def load_supporting_text(car_dir: Path) -> str:
    chunks: list[str] = []
    for name in ("summary.md", "finance.md"):
        path = car_dir / name
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def build_specs_markdown(
    *,
    metadata: dict[str, Any],
    source_urls: list[str],
    page_texts: dict[str, str],
    supporting_text: str,
) -> str:
    primary_source = source_urls[0]
    secondary_sources = source_urls[1:3]
    combined_text = " ".join([*page_texts.values(), supporting_text]).strip()

    body_style = derive_body_style(metadata)
    powertrain = sanitize_spec_value(
        derive_powertrain(metadata, combined_text),
        label="powertrain",
    ) or sanitize_spec_value(
        derive_powertrain(metadata, ""),
        label="powertrain",
    )
    drivetrain = official_value(str(metadata.get("drivetrain") or "").strip() or None)
    transmission = official_value(
        sanitize_spec_value(find_transmission(combined_text), label="transmission")
    )
    horsepower = official_value(
        sanitize_spec_value(find_horsepower(combined_text), label="horsepower")
    )
    torque = official_value(
        sanitize_spec_value(find_torque(combined_text), label="torque")
    )
    fuel_or_range = official_value(
        sanitize_spec_value(find_fuel_or_range(combined_text), label="fuel economy / range")
    )
    battery_charging = sanitize_spec_value(
        derive_battery_or_charging(metadata, combined_text),
        label="battery / charging",
    )
    seating = official_value(
        sanitize_spec_value(find_seating(combined_text), label="seating")
    )
    cargo = official_value(
        sanitize_spec_value(find_cargo(combined_text), label="cargo")
    )
    towing = official_value(
        sanitize_spec_value(find_towing(combined_text), label="towing")
    )
    dimensions = official_value(
        sanitize_spec_value(find_dimensions(combined_text), label="key dimensions")
    )

    notes: list[str] = []
    missing = [
        label
        for label, value in [
            ("Transmission", transmission),
            ("Horsepower", horsepower),
            ("Torque", torque),
            ("Fuel economy / range", fuel_or_range),
            ("Seating", seating),
            ("Cargo", cargo),
            ("Towing", towing),
            ("Key dimensions", dimensions),
        ]
        if value == "Not published on official page at capture time."
    ]
    if battery_charging == "Not published on official page at capture time.":
        missing.append("Battery / charging")
    if missing:
        notes.append(
            "Missing values at capture time: " + ", ".join(missing) + "."
        )
    if page_texts:
        notes.append("Live official source text was available for this conversion.")
    else:
        notes.append(
            "Live official source fetch was unavailable during conversion, so the spec sheet was compiled from the previously saved official URLs plus existing repo notes sourced from those official pages."
        )
    notes.append("Generated from official manufacturer URLs previously stored in `pdf.txt`.")

    lines = [
        f"# {metadata.get('year', '')} {metadata.get('make', '')} {metadata.get('model', '')} Specs",
        "",
        f"- Snapshot date: {SNAPSHOT_DATE}",
        f"- Primary source: {primary_source}",
        "- Secondary sources:",
    ]
    if secondary_sources:
        for url in secondary_sources:
            lines.append(f"  - {url}")
    else:
        lines.append("  - None beyond the primary source.")

    lines.extend(
        [
            "",
            "## Core Specs",
            f"- Body style: {body_style}",
            f"- Powertrain: {powertrain}",
            f"- Drivetrain: {drivetrain}",
            f"- Transmission: {transmission}",
            f"- Horsepower: {horsepower}",
            f"- Torque: {torque}",
            f"- Fuel economy / range: {fuel_or_range}",
        ]
    )
    if battery_charging is not None:
        lines.append(f"- Battery / charging: {battery_charging}")
    lines.extend(
        [
            f"- Seating: {seating}",
            f"- Cargo: {cargo}",
            f"- Towing: {towing}",
            f"- Key dimensions: {dimensions}",
            "",
            "## Notes",
        ]
    )
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def update_summary(summary_path: Path) -> None:
    text = summary_path.read_text(encoding="utf-8")
    if "- PDF asset:" in text:
        text = re.sub(
            r"^- PDF asset:.*$",
            "- Spec asset: `specs.md` generated from official manufacturer sources",
            text,
            flags=re.MULTILINE,
        )
    elif "- Spec asset:" not in text:
        lines = text.splitlines()
        insertion_index = 4 if len(lines) >= 4 else len(lines)
        lines.insert(insertion_index, "- Spec asset: `specs.md` generated from official manufacturer sources")
        text = "\n".join(lines)
    summary_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def migrate_specs() -> tuple[int, list[str]]:
    migrated: list[str] = []
    for car_dir in sorted(path for path in DATASET_ROOT.iterdir() if path.is_dir() and (path / "metadata.json").exists()):
        spec_pdf = car_dir / "spec.pdf"
        pdf_txt = car_dir / "pdf.txt"
        specs_path = car_dir / "specs.md"
        if spec_pdf.exists() or not pdf_txt.exists():
            if spec_pdf.exists() or not specs_path.exists():
                continue

        metadata_path = car_dir / "metadata.json"
        summary_path = car_dir / "summary.md"
        metadata = load_json(metadata_path)

        source_urls = collect_source_urls(car_dir, metadata)
        if not source_urls:
            raise RuntimeError(f"No official source URLs found for {car_dir.name}")

        page_texts = {}
        for url in source_urls[:2]:
            try:
                page_texts[url] = fetch_page_text(url)
            except Exception:
                continue
        supporting_text = load_supporting_text(car_dir)
        if not page_texts and not supporting_text.strip():
            raise RuntimeError(f"Could not build specs content for {car_dir.name}")

        specs_md = build_specs_markdown(
            metadata=metadata,
            source_urls=source_urls,
            page_texts=page_texts,
            supporting_text=supporting_text,
        )
        specs_path.write_text(specs_md, encoding="utf-8")

        metadata["paths"]["spec_pdf"] = "specs.md"
        metadata.setdefault("source_files", {})["pdf"] = (
            "Generated `specs.md` from official manufacturer URLs previously stored in `pdf.txt`"
        )
        save_json(metadata_path, metadata)
        update_summary(summary_path)
        if pdf_txt.exists():
            pdf_txt.unlink()
        migrated.append(car_dir.name)

    return len(migrated), migrated


def mean_border_std(image: Image.Image) -> float:
    rgb = image.convert("RGB")
    width, height = rgb.size
    border = max(8, min(width, height) // 20)
    strips = [
        rgb.crop((0, 0, width, border)),
        rgb.crop((0, height - border, width, height)),
        rgb.crop((0, 0, border, height)),
        rgb.crop((width - border, 0, width, height)),
    ]
    stds = []
    for strip in strips:
        stats = ImageStat.Stat(strip)
        stds.extend(stats.stddev)
    return float(sum(stds) / len(stds))


def edge_variance(image: Image.Image) -> float:
    grayscale = image.convert("L")
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    stats = ImageStat.Stat(edges)
    return float(stats.var[0])


def audit_images() -> list[ImageAuditResult]:
    results: list[ImageAuditResult] = []
    for car_dir in sorted(path for path in DATASET_ROOT.iterdir() if path.is_dir() and (path / "metadata.json").exists()):
        slug = car_dir.name
        image_path = car_dir / "images" / "front.jpg"
        objective_flags: list[str] = []
        heuristic_flags: list[str] = []

        if not image_path.exists():
            results.append(ImageAuditResult(slug, 0, 0, ["missing front.jpg"], []))
            continue

        try:
            image = Image.open(image_path)
            width, height = image.size
        except Exception as exc:  # noqa: BLE001
            results.append(ImageAuditResult(slug, 0, 0, [f"unreadable image: {exc}"], []))
            continue

        min_side = min(width, height)
        pixels = width * height
        aspect = max(width / max(height, 1), height / max(width, 1))
        border_std = mean_border_std(image)
        edge_var = edge_variance(image)

        if min_side < 500 or pixels < 350_000:
            objective_flags.append(f"low resolution ({width}x{height})")
        if aspect > 3.1:
            objective_flags.append(f"extreme aspect ratio ({width}x{height})")
        if edge_var < 180:
            objective_flags.append(f"low detail / possible blur ({width}x{height})")

        if border_std < 6:
            heuristic_flags.append("possible flat or blocky background")
        if edge_var < 280 and min_side < 700:
            heuristic_flags.append("possible unclear vehicle presentation")

        results.append(ImageAuditResult(slug, width, height, objective_flags, heuristic_flags))
    return results


def write_audit_report(results: list[ImageAuditResult]) -> None:
    objective = [result for result in results if result.objective_flags]
    heuristic = [result for result in results if result.heuristic_flags]

    lines = [
        "# Image Audit",
        "",
        f"- Snapshot date: {SNAPSHOT_DATE}",
        f"- Cars reviewed: {len(results)}",
        f"- Objective flags: {len(objective)}",
        f"- Heuristic review flags: {len(heuristic)}",
        "",
        "## Objective Flags",
    ]
    if objective:
        for result in objective:
            reasons = ", ".join(result.objective_flags)
            lines.append(f"- `{result.slug}`: {reasons}")
    else:
        lines.append("- None")

    lines.extend(["", "## Heuristic Review Flags"])
    if heuristic:
        for result in heuristic:
            reasons = ", ".join(result.heuristic_flags)
            lines.append(f"- `{result.slug}`: {reasons}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Notes",
            "- Objective checks cover missing files, unreadable images, low resolution, low-detail outliers, and extreme aspect ratios.",
            "- Heuristic flags are candidates for weird backgrounds, blocky studio backdrops, or generally unclear car presentation.",
            "- `ford_mustang_2026` must remain flagged because its current image is only 320x181.",
            "",
        ]
    )

    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    migrated_count, migrated_slugs = migrate_specs()
    results = audit_images()
    write_audit_report(results)

    print(f"Migrated {migrated_count} cars to specs.md.")
    print(f"Audit report: {AUDIT_REPORT_PATH}")
    if migrated_slugs:
        print("Sample migrated cars:")
        for slug in migrated_slugs[:10]:
            print(f"  - {slug}")


if __name__ == "__main__":
    main()
