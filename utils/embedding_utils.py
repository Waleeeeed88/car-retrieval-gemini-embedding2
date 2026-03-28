from __future__ import annotations

import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from google import genai
from google.genai import types

from config import Settings
from utils.io_utils import split_pdf_bytes_into_chunks, split_pdf_into_chunks

logger = logging.getLogger(__name__)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def average_embeddings(vectors: Iterable[np.ndarray]) -> np.ndarray:
    vectors = [normalize_vector(vector) for vector in vectors]
    if not vectors:
        raise ValueError("No vectors were provided for aggregation.")
    stacked = np.vstack(vectors)
    return normalize_vector(stacked.mean(axis=0))


def weighted_average_embeddings(weighted_vectors: Iterable[tuple[np.ndarray, float]]) -> np.ndarray:
    normalized_vectors: list[np.ndarray] = []
    weights: list[float] = []
    for vector, weight in weighted_vectors:
        if weight <= 0:
            continue
        normalized_vectors.append(normalize_vector(vector))
        weights.append(float(weight))

    if not normalized_vectors:
        raise ValueError("No weighted vectors were provided for aggregation.")

    weight_array = np.asarray(weights, dtype=np.float32)
    weight_array = weight_array / weight_array.sum()
    stacked = np.vstack(normalized_vectors)
    return normalize_vector((stacked * weight_array[:, None]).sum(axis=0))


def detect_mime_type(path: Path, fallback: str) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or fallback


class GeminiEmbedder:
    """Small wrapper around the official google-genai embedding client."""

    def __init__(self, settings: Settings) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your environment or .env file before embedding."
            )

        self.settings = settings
        # Official Gemini API client from `google-genai`.
        self.client = genai.Client(api_key=api_key)
        self.embed_config = types.EmbedContentConfig(
            output_dimensionality=settings.embedding_dimension
        )

    def _embed_parts(self, parts: list[types.Part]) -> np.ndarray:
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                response = self.client.models.embed_content(
                    model=self.settings.model_name,
                    contents=[types.Content(parts=parts)],
                    config=self.embed_config,
                )
                break
            except Exception as exc:
                last_error = exc
                message = str(exc)
                transient = any(code in message for code in ("429", "500", "502", "503", "504"))
                if attempt == 5 or not transient:
                    raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc
                delay_seconds = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "Gemini embedding request failed on attempt %s/5 with transient error. Retrying in %ss.",
                    attempt,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
        else:
            raise RuntimeError(f"Gemini embedding request failed: {last_error}") from last_error

        if not response.embeddings:
            raise RuntimeError("Gemini returned no embeddings.")
        values = np.asarray(response.embeddings[0].values, dtype=np.float32)
        return normalize_vector(values)

    def embed_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text input is empty.")
        part = types.Part.from_text(text=text.strip())
        return self._embed_parts([part])

    def embed_image(self, path: str | Path) -> np.ndarray:
        image_path = Path(path)
        mime_type = detect_mime_type(image_path, "image/jpeg")
        return self.embed_image_bytes(image_path.read_bytes(), mime_type=mime_type)

    def embed_image_bytes(self, image_bytes: bytes, *, mime_type: str) -> np.ndarray:
        if not image_bytes:
            raise ValueError("Image bytes are empty.")
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        return self._embed_parts([part])

    def embed_pdf(self, path: str | Path) -> np.ndarray:
        pdf_path = Path(path)
        chunks = split_pdf_into_chunks(pdf_path, max_pages_per_chunk=self.settings.pdf_max_pages_per_request)
        return self._embed_pdf_chunks(chunks, source_label=str(pdf_path))

    def embed_pdf_bytes(self, pdf_bytes: bytes) -> np.ndarray:
        chunks = split_pdf_bytes_into_chunks(
            pdf_bytes,
            max_pages_per_chunk=self.settings.pdf_max_pages_per_request,
        )
        return self._embed_pdf_chunks(chunks, source_label="uploaded PDF")

    def _embed_pdf_chunks(self, chunks: list[dict[str, object]], *, source_label: str) -> np.ndarray:
        if not chunks:
            raise ValueError(f"PDF has no readable pages: {source_label}")

        chunk_vectors = []
        for chunk in chunks:
            part = types.Part.from_bytes(
                data=chunk["bytes"],
                mime_type="application/pdf",
            )
            chunk_vectors.append(self._embed_parts([part]))
        return average_embeddings(chunk_vectors)
