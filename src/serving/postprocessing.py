"""
Postprocessing pipeline.

Converts raw ViT logits into human-readable predictions.
Loads ImageNet class labels once at import time.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

# ImageNet labels — downloaded once, stored locally
LABELS_PATH = Path(__file__).parent / "imagenet_labels.json"


def _load_labels() -> dict[int, str]:
    """Load ImageNet class index → label mapping."""
    if not LABELS_PATH.exists():
        # Download on first use
        import urllib.request

        url = (
            "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
            "/master/imagenet-simple-labels.json"
        )
        urllib.request.urlretrieve(url, LABELS_PATH)

    with open(LABELS_PATH) as f:
        labels_list = json.load(f)

    return {i: label for i, label in enumerate(labels_list)}


# Module-level singleton — loaded once when the server starts
IMAGENET_LABELS: dict[int, str] = _load_labels()


def logits_to_predictions(
    logits: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Convert raw logits to top-k predictions.

    Args:
        logits: float32 array of shape [1, 1000] from Triton
        top_k: number of top predictions to return

    Returns:
        list of dicts with keys: label, confidence, class_id
        sorted by confidence descending
    """
    # Softmax: logits → probabilities
    # Subtract max for numerical stability (prevents exp overflow)
    logits = logits[0]  # remove batch dim: [1000]
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)

    # Get top-k indices sorted by probability descending
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]

    return [
        {
            "label": IMAGENET_LABELS.get(int(idx), f"class_{idx}"),
            "confidence": float(probabilities[idx]),
            "class_id": int(idx),
        }
        for idx in top_k_indices
    ]
