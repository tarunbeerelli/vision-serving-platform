"""
Unit tests for quantization pipeline.
Run locally on M1 — no GPU, no model download needed.
Tests validate logic, not actual model outputs.
"""

from pathlib import Path

import numpy as np
import pytest


def test_onnx_output_path_is_correct() -> None:
    from quantization.export_onnx import OUTPUT_DIR

    assert str(OUTPUT_DIR) == "triton_repo/vit_fp32/1"


def test_calibration_output_dir_is_correct() -> None:
    from quantization.calibration_dataset import OUTPUT_DIR

    assert str(OUTPUT_DIR) == "quantization/calibration_data"


def test_int8_output_path_is_correct() -> None:
    from quantization.convert_tensorrt import FP32_OUTPUT, INT8_OUTPUT

    assert str(INT8_OUTPUT) == "triton_repo/vit_int8/1/model.plan"
    assert str(FP32_OUTPUT) == "triton_repo/vit_fp32/1/model.plan"


def test_validate_onnx_raises_on_missing_file() -> None:
    from quantization.validate_onnx import validate

    with pytest.raises(AssertionError, match="ONNX file not found"):
        validate(Path("nonexistent/model.onnx"))


def test_calibration_batch_padding() -> None:
    # Verify numpy batch padding logic works correctly
    # without needing real calibration files
    batch_size = 8
    images = [np.zeros((3, 224, 224), dtype=np.float32) for _ in range(3)]
    batch = np.stack(images)
    pad = np.zeros((batch_size - batch.shape[0], *batch.shape[1:]), dtype=np.float32)
    padded = np.concatenate([batch, pad], axis=0)
    assert padded.shape == (8, 3, 224, 224)
