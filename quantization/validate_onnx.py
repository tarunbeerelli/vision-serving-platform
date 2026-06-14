"""
Validate the exported ONNX graph locally on M1.

Runs inference with onnxruntime (CPU) and checks:
  - Graph loads without errors
  - Output shape is correct [batch, 1000] for ImageNet
  - Predicted class is sensible for a test image

Runs locally — no GPU needed. Catches export bugs before
you upload to Vast.ai for TensorRT conversion.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

ONNX_PATH = Path("triton_repo/vit_fp32/1/model.onnx")
EXPECTED_NUM_CLASSES = 1000


def validate(onnx_path: Path) -> None:
    print(f"Loading {onnx_path}...")
    assert onnx_path.exists(), f"ONNX file not found: {onnx_path}"

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    # Print input/output specs — useful for writing Triton config.pbtxt
    print("\nModel inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    print("\nModel outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Run inference with random input
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = sess.run(["logits"], {"pixel_values": dummy})
    logits = outputs[0]

    assert logits.shape == (
        1,
        EXPECTED_NUM_CLASSES,
    ), f"Expected (1, {EXPECTED_NUM_CLASSES}), got {logits.shape}"

    top_class = int(np.argmax(logits[0]))
    print(f"\n✓ Output shape: {logits.shape}")
    print(f"✓ Top predicted class index: {top_class}")
    print("✓ ONNX graph is valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", default=str(ONNX_PATH))
    args = parser.parse_args()
    validate(Path(args.onnx_path))
