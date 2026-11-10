"""Image validation and base64 encoding service."""

import base64
import io
from typing import Any, Dict, Sequence

from PIL import Image

from qna_bot.core.exceptions import ImageValidationError
from qna_bot.core.logging_config import get_logger

logger = get_logger(__name__)


def validate_and_encode_image(
    file_bytes: bytes,
    filename: str,
    max_size_mb: int = 10,
    allowed_types: Sequence[str] = ("png", "jpg", "jpeg", "webp"),
) -> Dict[str, Any]:
    """Validate image bytes and return a base64-encoded metadata payload."""
    if not file_bytes:
        raise ImageValidationError("Uploaded image file is empty.")

    if len(file_bytes) / (1024 * 1024) > max_size_mb:
        raise ImageValidationError(f"Image exceeds {max_size_mb}MB limit.")

    try:
        with Image.open(io.BytesIO(file_bytes)) as probe:
            probe.verify()

        with Image.open(io.BytesIO(file_bytes)) as img:
            raw_format = (img.format or "PNG").lower()
            normalized_format = "jpg" if raw_format == "jpeg" else raw_format
            allowed_normalized = {ext.lower() for ext in allowed_types}
            if (
                raw_format not in allowed_normalized
                and normalized_format not in allowed_normalized
            ):
                raise ImageValidationError(
                    f"Unsupported image format '{raw_format}'. "
                    f"Allowed types: {', '.join(allowed_types)}."
                )

            mime = f"image/{'jpeg' if normalized_format == 'jpg' else normalized_format}"
            width, height = img.size
    except ImageValidationError:
        raise
    except Exception as exc:
        logger.warning("Failed to validate uploaded image '%s': %s", filename, exc)
        raise ImageValidationError(f"Invalid or corrupted image: {exc}") from exc

    encoded = base64.b64encode(file_bytes).decode("utf-8")
    logger.info(
        "Processed image '%s' (%dx%d, %s)",
        filename,
        width,
        height,
        mime,
    )
    return {
        "name": filename,
        "b64": encoded,
        "mime": mime,
        "format": normalized_format,
        "width": width,
        "height": height,
    }
