from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    repo_root: Path = REPO_ROOT
    dataset_root: Path = Path(os.getenv("CAR_DATASET_ROOT", REPO_ROOT / "dataset"))
    data_dir: Path = Path(os.getenv("CAR_RETRIEVAL_OUTPUT_DIR", PROJECT_ROOT / "data"))
    model_name: str = "gemini-embedding-2-preview"
    embedding_dimension: int = 1536
    default_top_k: int = int(os.getenv("CAR_RETRIEVAL_TOP_K", "5"))
    supported_image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    supported_text_files: tuple[str, ...] = ("summary.md", "finance.md")
    supported_pdf_files: tuple[str, ...] = ("spec.pdf",)
    pdf_max_pages_per_request: int = 6

    @property
    def embeddings_json_path(self) -> Path:
        return self.data_dir / "embeddings.json"

    @property
    def embeddings_npy_path(self) -> Path:
        return self.data_dir / "embeddings.npy"

    @property
    def index_metadata_path(self) -> Path:
        return self.data_dir / "index_metadata.json"

    @property
    def index_metadata_csv_path(self) -> Path:
        return self.data_dir / "index_metadata.csv"

    @property
    def evaluation_dir(self) -> Path:
        return self.data_dir / "evaluation"

    @property
    def query_variants_path(self) -> Path:
        return self.data_dir / "query_variants.json"

    @property
    def query_variants_example_path(self) -> Path:
        return self.data_dir / "query_variants.example.json"


def _resolve_path(value: str | Path | None, fallback: Path) -> Path:
    if value is None:
        return fallback.resolve()
    return Path(value).expanduser().resolve()


def get_settings(
    *,
    dataset_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    top_k: int | None = None,
) -> Settings:
    base = Settings()
    updated = replace(
        base,
        dataset_root=_resolve_path(dataset_root, base.dataset_root),
        data_dir=_resolve_path(data_dir, base.data_dir),
        default_top_k=top_k or base.default_top_k,
    )
    return updated


settings = get_settings()

