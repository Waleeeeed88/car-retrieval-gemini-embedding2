from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter

from config import Settings

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def list_car_directories(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    return sorted(
        [
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and (path / "metadata.json").exists()
        ]
    )


def list_image_files(images_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def relative_to_dataset(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def resolve_spec_path(car_dir: Path, supported_files: tuple[str, ...]) -> Path:
    for filename in supported_files:
        candidate = car_dir / filename
        if candidate.exists():
            return candidate
    return car_dir / supported_files[0]


def build_car_record(car_dir: Path, settings: Settings) -> dict[str, Any]:
    metadata = load_json(car_dir / "metadata.json")
    image_paths = list_image_files(
        car_dir / "images",
        settings.supported_image_extensions,
    )
    image_paths = sorted(
        image_paths,
        key=lambda path: (0 if path.stem.lower() == "front" else 1, path.name.lower()),
    )
    return {
        "car_dir": car_dir,
        "car_id": str(metadata.get("slug") or metadata.get("id") or car_dir.name),
        "metadata": metadata,
        "summary_path": car_dir / "summary.md",
        "finance_path": car_dir / "finance.md",
        "pdf_path": resolve_spec_path(car_dir, settings.supported_pdf_files),
        "image_paths": image_paths,
    }


def split_pdf_into_chunks(pdf_path: Path, max_pages_per_chunk: int) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    return split_pdf_reader_into_chunks(reader, max_pages_per_chunk=max_pages_per_chunk)


def split_pdf_bytes_into_chunks(pdf_bytes: bytes, max_pages_per_chunk: int) -> list[dict[str, Any]]:
    if not pdf_bytes:
        return []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return split_pdf_reader_into_chunks(reader, max_pages_per_chunk=max_pages_per_chunk)


def split_pdf_reader_into_chunks(
    reader: PdfReader,
    max_pages_per_chunk: int,
) -> list[dict[str, Any]]:
    if not reader.pages:
        return []

    chunks: list[dict[str, Any]] = []
    for start_index in range(0, len(reader.pages), max_pages_per_chunk):
        end_index = min(start_index + max_pages_per_chunk, len(reader.pages))
        writer = PdfWriter()
        for page_index in range(start_index, end_index):
            writer.add_page(reader.pages[page_index])

        buffer = io.BytesIO()
        writer.write(buffer)
        chunks.append(
            {
                "page_start": start_index + 1,
                "page_end": end_index,
                "bytes": buffer.getvalue(),
            }
        )
    return chunks


def get_pdf_page_count(pdf_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def extract_pdf_text(pdf_path: Path, max_pages: int | None = None) -> str:
    if pdf_path.suffix.lower() in {".txt", ".md"}:
        return read_text(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    extracted = []
    for page in pages:
        extracted.append(page.extract_text() or "")
    return "\n".join(extracted).strip()


def load_query_variants(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        logger.info("No paraphrase query file found at %s; skipping paraphrase variants.", path)
        return {}

    payload = load_json(path)
    cleaned: dict[str, list[str]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            cleaned[key] = [str(item).strip() for item in value if str(item).strip()]
    return cleaned


def save_index_artifacts(
    *,
    vectors: np.ndarray,
    metadata: list[dict[str, Any]],
    debug_records: list[dict[str, Any]],
    settings: Settings,
) -> None:
    ensure_directory(settings.data_dir)
    np.save(settings.embeddings_npy_path, vectors.astype(np.float32))
    save_json(settings.index_metadata_path, metadata)
    save_json(settings.embeddings_json_path, debug_records)
    pd.json_normalize(metadata).to_csv(settings.index_metadata_csv_path, index=False)


def load_index_artifacts(settings: Settings) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not settings.embeddings_npy_path.exists():
        raise FileNotFoundError(
            f"Missing embeddings file: {settings.embeddings_npy_path}. Run embed_index.py first."
        )
    if not settings.index_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {settings.index_metadata_path}. Run embed_index.py first."
        )

    vectors = np.load(settings.embeddings_npy_path)
    metadata = load_json(settings.index_metadata_path)
    return vectors, metadata


def save_dataframe(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)
