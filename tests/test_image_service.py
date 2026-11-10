"""Tests for image validation and base64 encoding service."""

import base64
import io
import unittest

from PIL import Image

from qna_bot.core.exceptions import ImageValidationError
from qna_bot.services.image_service import validate_and_encode_image


def _make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (24, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(12, 34, 56)).save(buf, format=fmt)
    return buf.getvalue()


class TestImageService(unittest.TestCase):
    def test_valid_png_image(self) -> None:
        raw = _make_image_bytes("PNG", (32, 20))
        result = validate_and_encode_image(raw, "sample.png")
        self.assertEqual(result["name"], "sample.png")
        self.assertEqual(result["mime"], "image/png")
        self.assertEqual(result["format"], "png")
        self.assertEqual(result["width"], 32)
        self.assertEqual(result["height"], 20)
        self.assertEqual(base64.b64decode(result["b64"]), raw)

    def test_valid_jpeg_image_normalizes_format(self) -> None:
        raw = _make_image_bytes("JPEG", (10, 10))
        result = validate_and_encode_image(raw, "photo.jpeg")
        self.assertEqual(result["mime"], "image/jpeg")
        self.assertEqual(result["format"], "jpg")

    def test_empty_bytes_raises(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_and_encode_image(b"", "empty.png")

    def test_oversized_image_raises(self) -> None:
        raw = b"x" * (2 * 1024 * 1024)
        with self.assertRaises(ImageValidationError):
            validate_and_encode_image(raw, "big.png", max_size_mb=1)

    def test_corrupted_image_raises(self) -> None:
        with self.assertRaises(ImageValidationError):
            validate_and_encode_image(b"not-an-image", "broken.png")

    def test_disallowed_format_raises(self) -> None:
        raw = _make_image_bytes("BMP", (8, 8))
        with self.assertRaises(ImageValidationError):
            validate_and_encode_image(raw, "bitmap.bmp")


if __name__ == "__main__":
    unittest.main()
