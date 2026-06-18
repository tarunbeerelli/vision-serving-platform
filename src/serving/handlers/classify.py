"""
Classify RPC handler.

Orchestrates the full inference pipeline:
  1. Decode + preprocess image
  2. Call Triton via tritonclient
  3. Postprocess logits → top-k predictions
  4. Record Prometheus metrics + structured log
"""

import time

import grpc
import tritonclient.grpc as grpcclient
from serving.logging_config import get_logger
from serving.metrics import (
    INFERENCE_LATENCY,
    INFERENCE_REQUESTS,
    PREDICTION_CLASS,
)
from serving.postprocessing import logits_to_predictions
from serving.preprocessing import decode_and_preprocess
from serving.settings import settings

logger = get_logger(__name__)

# Triton model name → backend mapping
MODEL_NAMES = {
    "fp32": "vit_fp32",
    "int8": "vit_int8",
    "": "vit_fp32",
}


class ClassifyHandler:
    def __init__(self) -> None:
        self._client = grpcclient.InferenceServerClient(url=settings.triton_address)

    def Classify(self, request: object, context: grpc.ServicerContext) -> object:
        from src.generated.inference_pb2 import ClassifyResponse, Prediction  # type: ignore

        start_time = time.perf_counter()

        # Resolve model version
        version_key = request.model_version or ""  # type: ignore
        model_name = MODEL_NAMES.get(version_key, f"vit_{settings.model.default_vit_version}")

        try:
            # 1. Preprocess image bytes → float32 tensor [1, 3, 224, 224]
            pixel_values = decode_and_preprocess(
                request.image.data,  # type: ignore
                request.image.format,  # type: ignore
            )

            # 2. Build Triton input
            triton_input = grpcclient.InferInput("pixel_values", pixel_values.shape, "FP32")
            triton_input.set_data_from_numpy(pixel_values)
            triton_output = grpcclient.InferRequestedOutput("logits")

            # 3. Call Triton
            response = self._client.infer(
                model_name=model_name,
                inputs=[triton_input],
                outputs=[triton_output],
            )

            # 4. Extract Triton compute time from response metadata
            logits = response.as_numpy("logits")
            triton_compute_ms = 0.0  # Triton metadata available in HTTP, not gRPC

            # 5. Postprocess logits → top-k predictions
            top_k = request.top_k if request.top_k > 0 else settings.model.top_k_default  # type: ignore
            raw_preds = logits_to_predictions(logits, top_k=top_k)

            # 6. Record metrics
            inference_ms = (time.perf_counter() - start_time) * 1000
            INFERENCE_REQUESTS.labels(model_version=model_name, status="success").inc()
            INFERENCE_LATENCY.labels(model_version=model_name).observe(inference_ms)

            # Drift detection: record predicted class distribution
            if raw_preds:
                PREDICTION_CLASS.labels(
                    class_id=str(raw_preds[0]["class_id"]),
                    model_version=model_name,
                ).inc()

            # 7. Structured log — queryable in Grafana via Loki
            logger.info(
                "inference_complete",
                model_version=model_name,
                top_class=raw_preds[0]["label"] if raw_preds else "unknown",
                top_confidence=raw_preds[0]["confidence"] if raw_preds else 0.0,
                inference_ms=round(inference_ms, 2),
            )

            predictions = [
                Prediction(
                    label=p["label"],
                    confidence=p["confidence"],
                    class_id=p["class_id"],
                )
                for p in raw_preds
            ]

            return ClassifyResponse(
                predictions=predictions,
                model_version=model_name,
                inference_ms=inference_ms,
                triton_compute_ms=triton_compute_ms,
            )

        except Exception as e:
            INFERENCE_REQUESTS.labels(model_version=model_name, status="error").inc()
            logger.error("inference_failed", error=str(e), model_version=model_name)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from src.generated.inference_pb2 import ClassifyResponse  # type: ignore

            return ClassifyResponse()
