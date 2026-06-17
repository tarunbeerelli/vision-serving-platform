"""
Validate exported CLIP ONNX graphs locally on M1.
Verifies embedding shapes and cosine similarity computation.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

VISUAL_PATH = Path("triton_repo/clip_visual/1/model.onnx")
TEXT_PATH = Path("triton_repo/clip_text/1/model.onnx")
EMBEDDING_DIM = 512


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return float(np.sum(a * b, axis=-1)[0])


def validate_visual(path: Path) -> None:
    assert path.exists(), f"Missing: {path}"
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    print("Visual encoder inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape}")

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = sess.run(["image_embeddings"], {"pixel_values": dummy})
    emb = outputs[0]

    assert emb.shape == (1, EMBEDDING_DIM), f"Expected (1, {EMBEDDING_DIM}), got {emb.shape}"
    print(f"✓ Visual encoder output shape: {emb.shape}")


def validate_text(path: Path) -> None:
    assert path.exists(), f"Missing: {path}"
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    print("\nText encoder inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape}")

    dummy_ids = np.zeros((1, 77), dtype=np.int64)
    dummy_mask = np.ones((1, 77), dtype=np.int64)
    outputs = sess.run(["text_embeddings"], {"input_ids": dummy_ids, "attention_mask": dummy_mask})
    emb = outputs[0]

    assert emb.shape == (1, EMBEDDING_DIM), f"Expected (1, {EMBEDDING_DIM}), got {emb.shape}"
    print(f"✓ Text encoder output shape: {emb.shape}")


def validate_similarity() -> None:
    visual_sess = ort.InferenceSession(str(VISUAL_PATH), providers=["CPUExecutionProvider"])
    text_sess = ort.InferenceSession(str(TEXT_PATH), providers=["CPUExecutionProvider"])

    img = np.random.randn(1, 3, 224, 224).astype(np.float32)
    img_emb = visual_sess.run(["image_embeddings"], {"pixel_values": img})[0]

    ids = np.zeros((1, 77), dtype=np.int64)
    mask = np.ones((1, 77), dtype=np.int64)
    txt_emb = text_sess.run(["text_embeddings"], {"input_ids": ids, "attention_mask": mask})[0]

    sim = cosine_similarity(img_emb, txt_emb)
    print(f"\n✓ Cosine similarity (random input): {sim:.4f}")
    print("✓ CLIP graphs are valid")


if __name__ == "__main__":
    validate_visual(VISUAL_PATH)
    validate_text(TEXT_PATH)
    validate_similarity()
