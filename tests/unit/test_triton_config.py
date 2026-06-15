"""
Validates Triton config files without starting Triton.
Catches typos and missing fields before you spin up a GPU instance.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
TRITON_REPO = ROOT / "triton_repo"


def _read(model: str) -> str:
    path = TRITON_REPO / model / "config.pbtxt"
    assert path.exists(), f"Missing: {path}"
    return path.read_text()


def test_vit_fp32_config_exists() -> None:
    assert (TRITON_REPO / "vit_fp32" / "config.pbtxt").exists()


def test_vit_int8_config_exists() -> None:
    assert (TRITON_REPO / "vit_int8" / "config.pbtxt").exists()


def test_vit_fp32_backend_is_onnxruntime() -> None:
    assert 'backend: "onnxruntime"' in _read("vit_fp32")


def test_vit_int8_backend_is_tensorrt() -> None:
    assert 'backend: "tensorrt"' in _read("vit_int8")


def test_vit_fp32_input_name() -> None:
    assert 'name: "pixel_values"' in _read("vit_fp32")


def test_vit_fp32_output_name() -> None:
    assert 'name: "logits"' in _read("vit_fp32")


def test_vit_fp32_dynamic_batching() -> None:
    assert "dynamic_batching" in _read("vit_fp32")


def test_vit_int8_dynamic_batching() -> None:
    assert "dynamic_batching" in _read("vit_int8")


def test_vit_fp32_max_batch_size() -> None:
    assert "max_batch_size: 32" in _read("vit_fp32")


def test_model_artifact_directory_exists() -> None:
    # Triton requires the versioned subdirectory to exist
    # even if the model file isn't there yet
    assert (TRITON_REPO / "vit_fp32" / "1").exists()
    assert (TRITON_REPO / "vit_int8" / "1").exists()
