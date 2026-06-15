"""
gRPC gateway server entrypoint.

Starts two servers:
  1. gRPC server on gateway.port (50051) — serves inference RPCs
  2. HTTP server on gateway.metrics_port (9090) — Prometheus scrape target

Usage:
  python -m serving.gateway
"""

import signal
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection
from serving.handlers.classify import ClassifyHandler
from serving.handlers.health import HealthHandler
from serving.logging_config import configure_logging, get_logger
from serving.metrics import start_metrics_server
from serving.settings import settings

logger = get_logger(__name__)


def create_server() -> grpc.Server:
    from src.generated import inference_pb2, inference_pb2_grpc  # type: ignore

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            # Allow large image payloads (default 4MB is too small for some images)
            ("grpc.max_receive_message_length", 20 * 1024 * 1024),  # 20MB
            ("grpc.max_send_message_length", 20 * 1024 * 1024),
        ],
    )

    # Register handlers
    classify_handler = ClassifyHandler()
    health_handler = HealthHandler()

    # grpc_tools generates a combined servicer — we use a dispatch class
    class GatewayServicer(inference_pb2_grpc.GatewayServiceServicer):  # type: ignore
        def Classify(self, request, context):  # type: ignore
            return classify_handler.Classify(request, context)

        def ZeroShot(self, request, context):  # type: ignore
            # Phase 9 — not implemented yet
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("ZeroShot not implemented yet")
            from src.generated.inference_pb2 import ZeroShotResponse  # type: ignore

            return ZeroShotResponse()

        def Health(self, request, context):  # type: ignore
            return health_handler.Health(request, context)

        def Models(self, request, context):  # type: ignore
            return health_handler.Models(request, context)

    inference_pb2_grpc.add_GatewayServiceServicer_to_server(GatewayServicer(), server)

    # Reflection lets grpcurl introspect the server without a local .proto file
    SERVICE_NAMES = (
        inference_pb2.DESCRIPTOR.services_by_name["GatewayService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    return server


def serve() -> None:
    configure_logging(settings.gateway.log_level)
    logger.info(
        "gateway_starting",
        port=settings.gateway.port,
        metrics_port=settings.gateway.metrics_port,
        triton_address=settings.triton_address,
    )

    # Start Prometheus metrics HTTP server
    start_metrics_server()

    server = create_server()
    server.add_insecure_port(f"{settings.gateway.host}:{settings.gateway.port}")
    server.start()

    logger.info("gateway_started", port=settings.gateway.port)

    # Graceful shutdown on SIGTERM (k8s sends this before killing the pod)
    def handle_sigterm(*args):  # type: ignore
        logger.info("shutdown_requested")
        server.stop(grace=5)  # 5s grace period for in-flight requests

    signal.signal(signal.SIGTERM, handle_sigterm)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
