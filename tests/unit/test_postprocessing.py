"""Unit tests for postprocessing pipeline."""

import numpy as np
from serving.postprocessing import IMAGENET_LABELS, logits_to_predictions


def test_returns_top_k_predictions() -> None:
    logits = np.random.randn(1, 1000).astype(np.float32)
    preds = logits_to_predictions(logits, top_k=5)
    assert len(preds) == 5


def test_predictions_sorted_by_confidence() -> None:
    logits = np.random.randn(1, 1000).astype(np.float32)
    preds = logits_to_predictions(logits, top_k=10)
    confidences = [p["confidence"] for p in preds]
    assert confidences == sorted(confidences, reverse=True)


def test_confidences_sum_to_one() -> None:
    logits = np.random.randn(1, 1000).astype(np.float32)
    # All 1000 classes should sum to 1.0
    preds = logits_to_predictions(logits, top_k=1000)
    total = sum(p["confidence"] for p in preds)
    assert abs(total - 1.0) < 1e-5


def test_prediction_keys() -> None:
    logits = np.random.randn(1, 1000).astype(np.float32)
    preds = logits_to_predictions(logits, top_k=1)
    assert set(preds[0].keys()) == {"label", "confidence", "class_id"}


def test_high_logit_gets_high_confidence() -> None:
    logits = np.zeros((1, 1000), dtype=np.float32)
    logits[0, 281] = 100.0  # class 281 = tabby cat, very high logit
    preds = logits_to_predictions(logits, top_k=1)
    assert preds[0]["class_id"] == 281
    assert preds[0]["confidence"] > 0.99


def test_imagenet_labels_loaded() -> None:
    assert len(IMAGENET_LABELS) == 1000
    assert isinstance(IMAGENET_LABELS[0], str)
