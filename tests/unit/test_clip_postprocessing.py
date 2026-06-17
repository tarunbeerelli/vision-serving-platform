"""Unit tests for CLIP cosine similarity ranking."""

import numpy as np
import pytest
from serving.clip_postprocessing import rank_by_similarity


def _random_embedding(dim: int = 512) -> np.ndarray:
    e = np.random.randn(1, dim).astype(np.float32)
    return e / np.linalg.norm(e)


def _random_embeddings(n: int, dim: int = 512) -> np.ndarray:
    e = np.random.randn(n, dim).astype(np.float32)
    return e / np.linalg.norm(e, axis=-1, keepdims=True)


def test_returns_all_labels() -> None:
    img = _random_embedding()
    txt = _random_embeddings(5)
    labels = ["a", "b", "c", "d", "e"]
    result = rank_by_similarity(img, txt, labels)
    assert len(result) == 5


def test_sorted_by_score_descending() -> None:
    img = _random_embedding()
    txt = _random_embeddings(10)
    labels = [f"label_{i}" for i in range(10)]
    result = rank_by_similarity(img, txt, labels)
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_scores_sum_to_one() -> None:
    img = _random_embedding()
    txt = _random_embeddings(5)
    labels = ["a", "b", "c", "d", "e"]
    result = rank_by_similarity(img, txt, labels)
    total = sum(r["score"] for r in result)
    assert abs(total - 1.0) < 1e-5


def test_identical_image_text_gets_high_score() -> None:
    # If image and text embeddings are identical, that label should win
    embedding = _random_embedding()
    txt = _random_embeddings(4)
    txt_with_match = np.vstack([embedding, txt])
    labels = ["exact_match", "other_1", "other_2", "other_3", "other_4"]
    result = rank_by_similarity(embedding, txt_with_match, labels)
    assert result[0]["label"] == "exact_match"
    assert result[0]["score"] > 0.5


def test_label_count_mismatch_raises() -> None:
    img = _random_embedding()
    txt = _random_embeddings(3)
    labels = ["a", "b"]  # only 2 labels but 3 embeddings
    with pytest.raises(AssertionError):
        rank_by_similarity(img, txt, labels)


def test_result_keys() -> None:
    img = _random_embedding()
    txt = _random_embeddings(2)
    labels = ["cat", "dog"]
    result = rank_by_similarity(img, txt, labels)
    assert set(result[0].keys()) == {"label", "score"}
