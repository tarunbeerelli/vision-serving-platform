"""
Prometheus metrics for the gateway service.

Exposed on a separate HTTP port (metrics_port in settings).
Prometheus scrapes this endpoint — Grafana queries Prometheus.

Why separate port from gRPC:
  Prometheus expects HTTP. gRPC uses HTTP/2 with binary framing.
  They can't share a port cleanly.
"""

from prometheus_client import Counter, Histogram, start_http_server
from serving.settings import settings

# Total inference requests by model version and status
INFERENCE_REQUESTS = Counter(
    "gateway_inference_requests_total",
    "Total inference requests",
    ["model_version", "status"],  # labels: success / error
)

# End-to-end latency (gateway receives request → gateway sends response)
INFERENCE_LATENCY = Histogram(
    "gateway_inference_latency_ms",
    "End-to-end inference latency in milliseconds",
    ["model_version"],
    # Buckets tuned for ViT-tiny INT8 on T4:
    # expect P50~10ms, P99~50ms under load
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000],
)

# Triton-reported GPU compute time only (excludes network + pre/postprocessing)
TRITON_COMPUTE_LATENCY = Histogram(
    "gateway_triton_compute_latency_ms",
    "Triton GPU compute latency in milliseconds",
    ["model_version"],
    buckets=[1, 5, 10, 25, 50, 100, 250],
)

# Prediction class distribution — used for drift detection
# Tracks which ImageNet classes are being predicted
PREDICTION_CLASS = Counter(
    "gateway_prediction_class_total",
    "Predicted class distribution for drift detection",
    ["class_id", "model_version"],
)


def start_metrics_server() -> None:
    """Start Prometheus HTTP server on metrics_port."""
    port = settings.gateway.metrics_port
    start_http_server(port)
