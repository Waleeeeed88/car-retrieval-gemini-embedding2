from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from config import get_settings
from utils.embedding_utils import normalize_vector
from utils.io_utils import save_json


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_dataset_root(project_root: Path) -> Path:
    return project_root / "tests" / "fixtures" / "sample_dataset"


@pytest.fixture
def sample_files_root(project_root: Path) -> Path:
    return project_root / "tests" / "fixtures" / "sample_files"


def _base_vector(kind: str) -> np.ndarray:
    lookup = {
        "family": np.array([1.0, 0.0, 0.0, 0.15, 0.05, 0.0, 0.0, 0.0], dtype=np.float32),
        "electric": np.array([0.0, 1.0, 0.0, 0.05, 0.15, 0.0, 0.0, 0.0], dtype=np.float32),
        "truck": np.array([0.0, 0.0, 1.0, 0.05, 0.0, 0.15, 0.0, 0.0], dtype=np.float32),
    }
    return lookup[kind].copy()


def _infer_kind_from_text(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["truck", "towing", "payload", "work"]):
        return "truck"
    if any(token in lowered for token in ["electric", "ev", "charging", "range", "sedan"]):
        return "electric"
    return "family"


def _vector_from_seed(seed: str, *, kind: str) -> np.ndarray:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    jitter = np.array([(byte / 255.0) * 0.04 for byte in digest[:8]], dtype=np.float32)
    return normalize_vector(_base_vector(kind) + jitter)


class FakeEmbedder:
    def __init__(self, settings) -> None:
        self.settings = settings

    def embed_text(self, text: str) -> np.ndarray:
        kind = _infer_kind_from_text(text)
        return _vector_from_seed(f"text::{text}", kind=kind)

    def embed_image(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        kind = _infer_kind_from_text(path.stem)
        return _vector_from_seed(f"image::{path.name}", kind=kind)

    def embed_pdf(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        kind = _infer_kind_from_text(path.stem)
        return _vector_from_seed(f"pdf::{path.name}", kind=kind)

    def embed_image_bytes(self, image_bytes: bytes, *, mime_type: str) -> np.ndarray:
        token = image_bytes[:16].hex() + mime_type
        kind = _infer_kind_from_text(token)
        return _vector_from_seed(f"image-bytes::{token}", kind=kind)

    def embed_pdf_bytes(self, pdf_bytes: bytes) -> np.ndarray:
        token = pdf_bytes[:16].hex()
        return _vector_from_seed(f"pdf-bytes::{token}", kind="family")


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(get_settings())


@pytest.fixture
def fake_settings(sample_dataset_root: Path, tmp_path: Path):
    return get_settings(dataset_root=sample_dataset_root, data_dir=tmp_path / "data")


def _car_metadata_map() -> dict[str, dict[str, str]]:
    return {
        "acme_family_suv_2026": {
            "make": "Acme",
            "model": "Family Cruiser",
            "year": "2026",
            "category": "SUV",
            "drivetrain": "AWD",
            "fuel_type": "Gasoline",
            "price_range": "$34,000-$39,000",
            "market": "US",
            "kind": "family",
        },
        "acme_electric_sedan_2026": {
            "make": "Acme",
            "model": "Voltline",
            "year": "2026",
            "category": "Sedan",
            "drivetrain": "RWD",
            "fuel_type": "Electric",
            "price_range": "$46,000-$52,000",
            "market": "US",
            "kind": "electric",
        },
        "acme_work_truck_2026": {
            "make": "Acme",
            "model": "Payload Pro",
            "year": "2026",
            "category": "Truck",
            "drivetrain": "4WD",
            "fuel_type": "Gasoline",
            "price_range": "$41,000-$49,000",
            "market": "US",
            "kind": "truck",
        },
    }


@pytest.fixture
def sample_metadata_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for car_id, meta in _car_metadata_map().items():
        common = {
            "car_id": car_id,
            "make": meta["make"],
            "model": meta["model"],
            "year": meta["year"],
            "category": meta["category"],
            "drivetrain": meta["drivetrain"],
            "fuel_type": meta["fuel_type"],
            "price_range": meta["price_range"],
            "market": meta["market"],
        }
        records.extend(
            [
                {**common, "item_id": f"{car_id}::summary", "modality": "summary", "file_path": f"{car_id}/summary.md"},
                {**common, "item_id": f"{car_id}::finance", "modality": "finance", "file_path": f"{car_id}/finance.md"},
                {**common, "item_id": f"{car_id}::pdf", "modality": "pdf", "file_path": f"{car_id}/specs.md"},
                {
                    **common,
                    "item_id": f"{car_id}::image::front",
                    "modality": "image",
                    "file_path": f"{car_id}/images/front.jpg",
                    "view_type": "front",
                },
            ]
        )
    return records


@pytest.fixture
def sample_vectors(sample_metadata_records: list[dict[str, object]]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    kind_map = _car_metadata_map()
    for record in sample_metadata_records:
        car_id = str(record["car_id"])
        kind = kind_map[car_id]["kind"]
        seed = f"{record['item_id']}::{record['modality']}"
        vectors.append(_vector_from_seed(seed, kind=kind))
    return np.vstack(vectors)


def write_index_artifacts(data_dir: Path, *, vectors: np.ndarray, metadata: list[dict[str, object]]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "embeddings.npy", vectors.astype(np.float32))
    save_json(data_dir / "index_metadata.json", metadata)
    save_json(
        data_dir / "embeddings.json",
        [{**record, "embedding": vector.tolist()} for record, vector in zip(metadata, vectors, strict=True)],
    )
    return data_dir


@pytest.fixture
def sample_data_dir(
    tmp_path: Path,
    sample_vectors: np.ndarray,
    sample_metadata_records: list[dict[str, object]],
) -> Path:
    return write_index_artifacts(tmp_path / "data", vectors=sample_vectors, metadata=sample_metadata_records)


@pytest.fixture
def monkeypatch_embedder(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def _apply(*modules) -> None:
        for module in modules:
            monkeypatch.setattr(module, "GeminiEmbedder", FakeEmbedder)

    return _apply
