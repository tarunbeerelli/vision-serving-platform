"""
CLIP postprocessing — cosine similarity ranking.

Takes image and text embeddings from Triton and returns
labels ranked by cosine similarity score.
"""

from typing import Any

import numpy as np


def rank_by_similarity(
    image_embedding: np.ndarray,
    text_embeddings: np.ndarray,
    labels: list[str],
) -> list[dict[str, Any]]:
    """
    Rank labels by cosine similarity to the image embedding.

    Args:
        image_embedding: float32 array [1, 512] from clip_visual
        text_embeddings: float32 array [N, 512] from clip_text
                         where N = number of candidate labels
        labels: list of N text labels in same order as text_embeddings

    Returns:
        list of dicts sorted by score descending:
        [{"label": "a dog", "score": 0.94}, ...]
    """
    assert (
        len(labels) == text_embeddings.shape[0]
    ), f"Label count {len(labels)} != embedding count {text_embeddings.shape[0]}"

    # L2 normalise both — cosine similarity becomes dot product
    img_norm = image_embedding / np.linalg.norm(image_embedding, axis=-1, keepdims=True)
    txt_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

    # Dot product: [1, 512] × [512, N] → [1, N]
    scores = (img_norm @ txt_norm.T)[0]

    # Softmax over scores — converts similarities to probabilities
    # Temperature scaling (100×) sharpens the distribution
    scores = scores * 100.0
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)

    # Sort by probability descending
    ranked_indices = np.argsort(probs)[::-1]

    return [{"label": labels[i], "score": float(probs[i])} for i in ranked_indices]
