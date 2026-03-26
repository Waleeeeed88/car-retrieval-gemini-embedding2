from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


def _image_to_jpeg_bytes(image: Image.Image, *, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def slight_crop(image: Image.Image, crop_ratio: float = 0.05) -> Image.Image:
    width, height = image.size
    left = int(width * crop_ratio)
    top = int(height * crop_ratio)
    right = width - left
    bottom = height - top
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((width, height))


def brightness_change(image: Image.Image, factor: float = 1.15) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def mild_blur(image: Image.Image, radius: float = 1.0) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def resize_compression(image: Image.Image, scale: float = 0.8, quality: int = 60) -> bytes:
    width, height = image.size
    resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    return _image_to_jpeg_bytes(resized, quality=quality)


def generate_image_variants(image_path: str | Path) -> dict[str, bytes]:
    with Image.open(image_path) as image:
        base = image.convert("RGB")
        variants = {
            "original": _image_to_jpeg_bytes(base),
            "slight_crop": _image_to_jpeg_bytes(slight_crop(base)),
            "brightness_change": _image_to_jpeg_bytes(brightness_change(base)),
            "mild_blur": _image_to_jpeg_bytes(mild_blur(base)),
            "resize_compression": resize_compression(base),
        }
    return variants


def _first_meaningful_sentence(text: str) -> str:
    for part in text.replace("\r", "\n").split("."):
        cleaned = part.strip().replace("\n", " ")
        if cleaned:
            return cleaned + "."
    return text.strip()


def build_text_variants(
    summary_text: str,
    finance_text: str = "",
    paraphrases: list[str] | None = None,
) -> dict[str, str]:
    variants = {
        "original": summary_text.strip(),
        "short": _first_meaningful_sentence(summary_text)[:220],
        "long": "\n\n".join(part for part in [summary_text.strip(), finance_text.strip()] if part).strip(),
    }

    for index, paraphrase in enumerate(paraphrases or [], start=1):
        cleaned = paraphrase.strip()
        if cleaned:
            variants[f"paraphrase_{index}"] = cleaned

    return variants

