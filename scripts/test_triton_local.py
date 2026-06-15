"""
Smoke test against a locally running Triton instance.

Usage:
  # Terminal 1 — start Triton
  docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/triton_repo:/models \
    nvcr.io/nvidia/tritonserver:24.05-py3 \
    tritonserver --model-repository=/models --model-control-mode=explicit \
    --load-model=vit_fp32

  # Terminal 2 — run this script
  python scripts/test_triton_local.py
"""

import sys

import numpy as np
import tritonclient.grpc as grpcclient

TRITON_URL = "localhost:8001"
MODEL_NAME = "vit_fp32"
NUM_CLASSES = 1000


def test_triton_health() -> None:
    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    assert client.is_server_live(), "Triton is not live"
    assert client.is_server_ready(), "Triton is not ready"
    assert client.is_model_ready(MODEL_NAME), f"{MODEL_NAME} is not ready"
    print("✓ Triton health check passed")


def test_vit_inference() -> None:
    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    # Build input tensor — random image, same shape as training
    pixel_values = np.random.randn(1, 3, 224, 224).astype(np.float32)
    inputs = [grpcclient.InferInput("pixel_values", pixel_values.shape, "FP32")]
    inputs[0].set_data_from_numpy(pixel_values)

    outputs = [grpcclient.InferRequestedOutput("logits")]
    response = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

    logits = response.as_numpy("logits")
    assert logits.shape == (1, NUM_CLASSES), f"Unexpected shape: {logits.shape}"

    top_class = int(np.argmax(logits[0]))
    confidence = float(np.max(np.exp(logits[0]) / np.sum(np.exp(logits[0]))))
    print(f"✓ Inference passed — top class: {top_class}, confidence: {confidence:.3f}")


if __name__ == "__main__":
    try:
        test_triton_health()
        test_vit_inference()
        print("\n✓ All Triton smoke tests passed")
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        sys.exit(1)
