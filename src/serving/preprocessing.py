"""
Image preprocessing pipeline.

Converts raw JPEG/PNG bytes into a normalised float32 tensor
ready for ViT inference.

ViT-tiny was trained with these exact normalisation values:
  mean = [0.5, 0.5, 0.5]
  std  = [0.5, 0.5, 0.5]
This maps pixel values from [0, 255] → [-1, 1].
Using different values silently degrades accuracy.
"""

import io

import numpy as np
from PIL import Image

# ViT training normalisation constants
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
INPUT_SIZE = 224


def decode_and_preprocess(image_bytes: bytes, image_format: str) -> np.ndarray:
    """
    Decode image bytes and preprocess for ViT inference.

    Args:
        image_bytes: raw JPEG or PNG bytes
        image_format: "jpeg" or "png"

    Returns:
        float32 numpy array of shape [1, 3, 224, 224]
        ready to send to Triton as pixel_values input
    """
    # Decode bytes → PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # ViT requires RGB — drop alpha channel if PNG has one
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to 224x224 — ViT patch size is fixed at 16x16
    # so input must be exactly 224x224 (14x14 patches)
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)

    # PIL → numpy [H, W, C] uint8
    arr = np.array(image, dtype=np.float32)

    # Normalise: [0, 255] → [0, 1] → [-1, 1]
    arr = arr / 255.0
    arr = (arr - MEAN) / STD

    # [H, W, C] → [C, H, W] → [1, C, H, W] (add batch dim)
    arr = arr.transpose(2, 0, 1)
    arr = arr[np.newaxis, ...]

    return arr  # shape: [1, 3, 224, 224]
