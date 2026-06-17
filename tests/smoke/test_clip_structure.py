"""Smoke tests for CLIP model repository structure."""

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
TRITON_REPO = ROOT / "triton_repo"


def _read(model: str) -> str:
    path = TRITON_REPO / model / "config.pbtxt"
    assert path.exists(), f"Missing: {path}"
    return path.read_text()


def test_clip_visual_config_exists() -> None:
    assert (TRITON_REPO / "clip_visual" / "config.pbtxt").exists()


def test_clip_text_config_exists() -> None:
    assert (TRITON_REPO / "clip_text" / "config.pbtxt").exists()


def test_clip_visual_backend() -> None:
    assert 'backend: "onnxruntime"' in _read("clip_visual")


def test_clip_text_backend() -> None:
    assert 'backend: "onnxruntime"' in _read("clip_text")


def test_clip_visual_embedding_output() -> None:
    assert "image_embeddings" in _read("clip_visual")


def test_clip_text_embedding_output() -> None:
    assert "text_embeddings" in _read("clip_text")


def test_clip_visual_model_dir_exists() -> None:
    assert (TRITON_REPO / "clip_visual" / "1").exists()


def test_clip_text_model_dir_exists() -> None:
    assert (TRITON_REPO / "clip_text" / "1").exists()
