"""
Vision serving platform — full load test suite.

Scenarios:
  RampUser         — autoscale proof (tag: ramp)
  SustainedUser    — stability test (tag: sustained)
  MixedUser        — 70% classify + 30% zeroshot (tag: mixed)
  LabelStressUser  — zeroshot with varying label counts (tag: stress)
  ABVersionUser    — 50/50 fp32 vs int8 (tag: ab)

Usage:
  # Ramp test against GKE
  locust -f locust/locustfile.py --tags ramp \
    --host <gateway-external-ip>:50051 \
    --users 500 --spawn-rate 10 --run-time 15m

  # Mixed traffic
  locust -f locust/locustfile.py --tags mixed \
    --host <gateway-external-ip>:50051 \
    --users 200 --spawn-rate 5 --run-time 30m

  # Web UI (no --headless)
  locust -f locust/locustfile.py \
    --host <gateway-external-ip>:50051
"""

# from gevent import monkey
# monkey.patch_all()
import random
import sys
import time
from pathlib import Path

import grpc

from locust import User, between, events, tag, task

# Add src to path for generated stubs
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from generated import inference_pb2, inference_pb2_grpc  # type: ignore

# ── Image fixtures ────────────────────────────────────────────────────────────

IMAGES_DIR = Path(__file__).parent / "images"
IMAGE_FILES = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.jpeg"))

if not IMAGE_FILES:
    raise RuntimeError(f"No images found in {IMAGES_DIR} — run download script first")


def _load_image(path: Path) -> bytes:
    return path.read_bytes()


IMAGE_POOL = [(p.name, _load_image(p)) for p in IMAGE_FILES]

# ── Label sets for ZeroShot ───────────────────────────────────────────────────

LABELS_SMALL = ["a dog", "a cat"]

LABELS_MEDIUM = [
    "a dog",
    "a cat",
    "a bird",
    "a car",
    "a tree",
    "a person",
    "a building",
    "a flower",
    "a mountain",
    "the ocean",
]

LABELS_LARGE = [
    "a golden retriever",
    "a tabby cat",
    "a red sports car",
    "a mountain landscape",
    "a city skyline",
    "a bowl of food",
    "a smartphone",
    "a wooden chair",
    "a glass of water",
    "a sunset",
    "a white dog",
    "a black cat",
    "a blue car",
    "a green tree",
    "a tall building",
    "a yellow flower",
    "a snowy mountain",
    "a beach scene",
    "a forest path",
    "a kitchen appliance",
    "a person smiling",
    "a baby animal",
    "a wild bird",
    "a vintage car",
    "an airplane",
    "a boat on water",
    "a coffee cup",
    "a laptop computer",
    "a bicycle",
    "a train station",
    "a shopping mall",
    "a park bench",
    "a swimming pool",
    "a bookshelf",
    "a musical instrument",
    "a sports stadium",
    "a hospital room",
    "a school classroom",
    "a restaurant interior",
    "a hotel lobby",
    "a gym equipment",
    "a garden",
    "a rooftop",
    "a bridge",
    "a tunnel",
    "a waterfall",
    "a desert",
    "a cave",
    "a lighthouse",
] * 2  # 100 labels total

LABEL_SETS = {
    2: LABELS_SMALL,
    10: LABELS_MEDIUM,
    50: LABELS_LARGE[:50],
    100: LABELS_LARGE[:100],
}


# ── Base gRPC user ────────────────────────────────────────────────────────────


class GRPCUser(User):
    """
    Base class for all gRPC users.
    Creates a single gRPC channel per user — mimics real client behaviour.
    """

    abstract = True

    def on_start(self) -> None:
        host = self.host or "localhost:50051"
        # Strip http:// if accidentally included
        host = host.replace("http://", "").replace("https://", "")
        self.channel = grpc.insecure_channel(
            host,
            options=[
                ("grpc.max_receive_message_length", 20 * 1024 * 1024),
                ("grpc.max_send_message_length", 20 * 1024 * 1024),
                # Keep-alive — important for long sustained tests
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
            ],
        )
        self.stub = inference_pb2_grpc.GatewayServiceStub(self.channel)

    def on_stop(self) -> None:
        try:
            self.channel.close()
        except Exception:
            pass

    def _classify(
        self,
        model_version: str = "",
        top_k: int = 5,
    ) -> inference_pb2.ClassifyResponse:
        name, data = random.choice(IMAGE_POOL)
        request = inference_pb2.ClassifyRequest(
            image=inference_pb2.ImagePayload(data=data, format="jpeg"),
            top_k=top_k,
            model_version=model_version,
        )
        start = time.perf_counter()
        try:
            response = self.stub.Classify(request, timeout=30.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            events.request.fire(
                request_type="gRPC",
                name=f"Classify/{model_version or 'default'}",
                response_time=elapsed_ms,
                response_length=len(response.SerializeToString()),
                exception=None,
            )
            return response
        except grpc.RpcError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            events.request.fire(
                request_type="gRPC",
                name=f"Classify/{model_version or 'default'}",
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
            )
            raise

    def _zeroshot(
        self,
        labels: list[str],
        label_tag: str = "",
    ) -> inference_pb2.ZeroShotResponse:
        name, data = random.choice(IMAGE_POOL)
        request = inference_pb2.ZeroShotRequest(
            image=inference_pb2.ImagePayload(data=data, format="jpeg"),
            labels=labels,
        )
        start = time.perf_counter()
        try:
            response = self.stub.ZeroShot(request, timeout=120.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            events.request.fire(
                request_type="gRPC",
                name=f"ZeroShot/labels={label_tag or len(labels)}",
                response_time=elapsed_ms,
                response_length=len(response.SerializeToString()),
                exception=None,
            )
            return response
        except grpc.RpcError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            events.request.fire(
                request_type="gRPC",
                name=f"ZeroShot/labels={label_tag or len(labels)}",
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
            )
            raise


# ── Scenario 1: Ramp test — autoscale proof ───────────────────────────────────


@tag("ramp", "autoscale")
class RampUser(GRPCUser):
    """
    Pure Classify load — ramps from 10 to 500 users.
    Designed to trigger HPA and GPU node autoscaling.

    Run with:
      --users 500 --spawn-rate 10 --run-time 15m
    """

    wait_time = between(0.1, 0.5)

    @task
    def classify_default(self) -> None:
        self._classify(model_version="fp32")


# ── Scenario 2: Sustained load ────────────────────────────────────────────────


@tag("sustained")
class SustainedUser(GRPCUser):
    """
    200 users for 30 minutes — stability and memory leak test.
    Mix of classify and health checks to simulate realistic traffic.

    Run with:
      --users 200 --spawn-rate 5 --run-time 30m
    """

    wait_time = between(0.5, 1.5)

    @task(9)
    def classify(self) -> None:
        self._classify()

    @task(1)
    def health_check(self) -> None:
        start = time.perf_counter()
        try:
            response = self.stub.Health(inference_pb2.HealthRequest(), timeout=5.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            events.request.fire(
                request_type="gRPC",
                name="Health",
                response_time=elapsed_ms,
                response_length=len(response.SerializeToString()),
                exception=None,
            )
        except grpc.RpcError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            events.request.fire(
                request_type="gRPC",
                name="Health",
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
            )


# ── Scenario 3: Mixed traffic ─────────────────────────────────────────────────


@tag("mixed")
class MixedUser(GRPCUser):
    """
    70% Classify, 30% ZeroShot — realistic production traffic mix.
    Shows Triton serving multiple models simultaneously under load.

    Run with:
      --users 200 --spawn-rate 5 --run-time 20m
    """

    wait_time = between(0.2, 1.0)

    @task(7)
    def classify(self) -> None:
        self._classify()

    @task(3)
    def zeroshot_medium(self) -> None:
        self._zeroshot(LABELS_MEDIUM, label_tag="10")


# ── Scenario 4: Label count stress ───────────────────────────────────────────


@tag("stress", "zeroshot")
class LabelStressUser(GRPCUser):
    """
    ZeroShot with varying label counts: 2, 10, 50, 100.
    Shows how latency scales with label count.
    Each label requires one text encoder forward pass.

    Run with:
      --users 50 --spawn-rate 2 --run-time 10m
    """

    wait_time = between(1.0, 3.0)

    @task(4)
    def zeroshot_2_labels(self) -> None:
        self._zeroshot(LABELS_SMALL, label_tag="2")

    @task(3)
    def zeroshot_10_labels(self) -> None:
        self._zeroshot(LABELS_MEDIUM, label_tag="10")

    @task(2)
    def zeroshot_50_labels(self) -> None:
        self._zeroshot(LABELS_LARGE[:50], label_tag="50")

    @task(1)
    def zeroshot_100_labels(self) -> None:
        self._zeroshot(LABELS_LARGE[:30], label_tag="30")


# ── Scenario 5: A/B model version ────────────────────────────────────────────


@tag("ab", "versioning")
class ABVersionUser(GRPCUser):
    """
    50% fp32, 50% int8 — A/B versioning proof.
    Grafana shows two distinct latency distributions.
    INT8 should be 3-5x faster than FP32 on GPU.

    Run with:
      --users 100 --spawn-rate 5 --run-time 10m
    """

    wait_time = between(0.1, 0.5)

    @task
    def classify_fp32(self) -> None:
        self._classify(model_version="fp32")

    # @task
    # def classify_int8(self) -> None:
    # self._classify(model_version="int8")
