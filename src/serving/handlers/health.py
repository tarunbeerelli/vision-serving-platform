"""
Health and Models RPC handlers.
Used by k8s liveness/readiness probes and grpcurl debugging.
"""

import grpc
from serving.logging_config import get_logger
from serving.settings import settings

logger = get_logger(__name__)


class HealthHandler:
    """
    Handles Health and Models RPCs.
    Checks Triton is reachable and reports model states.
    """

    def __init__(self) -> None:
        import tritonclient.grpc as grpcclient

        self._client = grpcclient.InferenceServerClient(url=settings.triton_address)

    def Health(self, request: object, context: grpc.ServicerContext) -> object:
        from src.generated.inference_pb2 import HealthResponse  # type: ignore

        try:
            triton_live = self._client.is_server_live()
            triton_ready = self._client.is_server_ready()
            status = "READY" if (triton_live and triton_ready) else "NOT_READY"
        except Exception as e:
            logger.warning("triton_unreachable", error=str(e))
            status = "UNREACHABLE"

        return HealthResponse(
            healthy=(status == "READY"),
            triton_status=status,
            gateway_version="0.1.0",
        )

    def Models(self, request: object, context: grpc.ServicerContext) -> object:
        from src.generated.inference_pb2 import ModelInfo, ModelsResponse  # type: ignore

        try:
            repo_index = self._client.get_model_repository_index()
            models = [
                ModelInfo(
                    name=m.name,
                    version=m.version,
                    state=m.state,
                    backend="tensorrt_plan" if "int8" in m.name else "onnxruntime",
                )
                for m in repo_index
            ]
        except Exception as e:
            logger.error("failed_to_list_models", error=str(e))
            models = []

        return ModelsResponse(models=models)
