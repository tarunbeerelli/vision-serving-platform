"""
ZeroShot RPC handler — CLIP zero-shot image classification.

Pipeline:
  1. Decode + preprocess image → float32 tensor [1, 3, 224, 224]
  2. Tokenize each label → input_ids + attention_mask [N, 77]
  3. Run clip_visual on image → image_embedding [1, 512]
  4. Run clip_text on all labels in one batch → text_embeddings [N, 512]
  5. Rank labels by cosine similarity → sorted predictions
"""

import time

import grpc
import numpy as np
import tritonclient.grpc as grpcclient
from serving.clip_postprocessing import rank_by_similarity
from serving.logging_config import get_logger
from serving.metrics import INFERENCE_LATENCY, INFERENCE_REQUESTS
from serving.preprocessing import decode_and_preprocess
from serving.settings import settings
from transformers import CLIPTokenizer

logger = get_logger(__name__)

CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"
MAX_LABELS = 100
TEXT_SEQ_LEN = 77


class ZeroShotHandler:
    def __init__(self) -> None:
        self._client = grpcclient.InferenceServerClient(url=settings.triton_address)
        # Tokenizer loaded once at startup
        self._tokenizer = CLIPTokenizer.from_pretrained(CLIP_CHECKPOINT)
        logger.info("zeroshot_handler_ready", checkpoint=CLIP_CHECKPOINT)

    def ZeroShot(self, request: object, context: grpc.ServicerContext) -> object:
        from src.generated.inference_pb2 import ZeroShotPrediction, ZeroShotResponse  # type: ignore

        start_time = time.perf_counter()

        labels = list(request.labels)  # type: ignore
        if not labels:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("At least one label required")
            return ZeroShotResponse()

        if len(labels) > MAX_LABELS:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Maximum {MAX_LABELS} labels allowed")
            return ZeroShotResponse()

        try:
            # 1. Preprocess image
            pixel_values = decode_and_preprocess(
                request.image.data,  # type: ignore
                request.image.format,  # type: ignore
            )

            # 2. Tokenize all labels at once
            tokens = self._tokenizer(
                labels,
                padding="max_length",
                max_length=TEXT_SEQ_LEN,
                truncation=True,
                return_tensors="np",
            )
            input_ids = tokens["input_ids"].astype(np.int64)  # [N, 77]
            attention_mask = tokens["attention_mask"].astype(np.int64)  # [N, 77]

            # 3. Run visual encoder
            visual_input = grpcclient.InferInput("pixel_values", pixel_values.shape, "FP32")
            visual_input.set_data_from_numpy(pixel_values)
            visual_output = grpcclient.InferRequestedOutput("image_embeddings")

            visual_response = self._client.infer(
                model_name="clip_visual",
                inputs=[visual_input],
                outputs=[visual_output],
            )
            image_embedding = visual_response.as_numpy("image_embeddings")

            # 4. Run text encoder — all labels in one batch
            text_ids_input = grpcclient.InferInput("input_ids", input_ids.shape, "INT64")
            text_ids_input.set_data_from_numpy(input_ids)

            text_mask_input = grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64")
            text_mask_input.set_data_from_numpy(attention_mask)

            text_output = grpcclient.InferRequestedOutput("text_embeddings")

            text_response = self._client.infer(
                model_name="clip_text",
                inputs=[text_ids_input, text_mask_input],
                outputs=[text_output],
            )
            text_embeddings = text_response.as_numpy("text_embeddings")

            # 5. Rank by cosine similarity
            ranked = rank_by_similarity(image_embedding, text_embeddings, labels)

            inference_ms = (time.perf_counter() - start_time) * 1000
            INFERENCE_REQUESTS.labels(model_version="clip", status="success").inc()
            INFERENCE_LATENCY.labels(model_version="clip").observe(inference_ms)

            logger.info(
                "zeroshot_complete",
                top_label=ranked[0]["label"] if ranked else "none",
                top_score=ranked[0]["score"] if ranked else 0.0,
                num_labels=len(labels),
                inference_ms=round(inference_ms, 2),
            )

            predictions = [ZeroShotPrediction(label=p["label"], score=p["score"]) for p in ranked]

            return ZeroShotResponse(
                predictions=predictions,
                model_version="clip",
                inference_ms=inference_ms,
            )

        except Exception as e:
            INFERENCE_REQUESTS.labels(model_version="clip", status="error").inc()
            logger.error("zeroshot_failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ZeroShotResponse()
