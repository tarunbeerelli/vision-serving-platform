"""Unit tests for image preprocessing pipeline."""

import io

import numpy as np
from PIL import Image
from serving.preprocessing import INPUT_SIZE, decode_and_preprocess


def _make_image_bytes(width: int, height: int, mode: str = "RGB") -> bytes:
    """Create a test image as PNG bytes."""
    img = Image.new(mode, (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    fmt = "PNG" if mode == "RGBA" else "JPEG"
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_output_shape() -> None:
    image_bytes = _make_image_bytes(640, 480)
    result = decode_and_preprocess(image_bytes, "jpeg")
    assert result.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)


def test_output_dtype_is_float32() -> None:
    image_bytes = _make_image_bytes(224, 224)
    result = decode_and_preprocess(image_bytes, "jpeg")
    assert result.dtype == np.float32


def test_normalisation_range() -> None:
    # Normalised values should be in approximately [-1, 1]
    image_bytes = _make_image_bytes(224, 224)
    result = decode_and_preprocess(image_bytes, "jpeg")
    assert result.min() >= -1.5
    assert result.max() <= 1.5


def test_rgba_image_converts_to_rgb() -> None:
    # PNG images often have alpha channel — must be stripped
    buf = io.BytesIO()
    Image.new("RGBA", (224, 224)).save(buf, format="PNG")
    result = decode_and_preprocess(buf.getvalue(), "png")
    assert result.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)


def test_different_input_sizes_are_resized() -> None:
    for size in [32, 128, 512, 1024]:
        image_bytes = _make_image_bytes(size, size)
        result = decode_and_preprocess(image_bytes, "jpeg")
        assert result.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
