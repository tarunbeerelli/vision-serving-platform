# Vision Serving Platform

Production ML inference platform serving ViT-tiny (FP32) and CLIP (zero-shot)
via NVIDIA Triton Inference Server on GKE, with full GitOps CI/CD, autoscaling,
and Prometheus + Grafana + Loki observability.

> **Portfolio context:** This project demonstrates production MLOps engineering.
> It complements [RMAML](https://github.com/tarunbeerelli/RMAML), a Riemannian
> meta-learning research project — together they cover the full spectrum from
> ML research to production deployment.

---

## Architecture

```
Client (Locust / grpcurl)
        │
        │ gRPC (proto/inference.proto)
        ▼
  Gateway Service  ──────────────────────────────────────────────
  (Python, 2 replicas)                                          │
  - Image decode + normalisation                                │
  - Request routing (ViT / CLIP)                                │
  - Prometheus metrics + structured logging                     │
        │ tritonclient gRPC (KServe v2)                         │
        ▼                                                        │
  Triton Inference Server                                        │
  - Dynamic batching                                            │
  - Model versioning (FP32 / INT8)                              │
  - Native Prometheus metrics (:8002)                           │
        │                                                        │
        ├── vit_fp32   (ONNX, onnxruntime backend)              │
        ├── vit_int8   (TensorRT INT8, GPU only)                │
        ├── clip_visual (ONNX, image encoder)                   │
        └── clip_text   (ONNX, text encoder)                    │
        │                                                        │
        ▼                                                        │
  GKE Standard Cluster (europe-west4)                           │
  ├── CPU node pool  (e2-standard-2 × 3)  ◄──────────────────┘
  └── GPU node pool  (n1-standard-4 + T4, autoscales 0→2)
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Inference server | NVIDIA Triton 24.05 |
| Model optimisation | TensorRT INT8 + ONNX |
| API protocol | gRPC (proto/inference.proto) |
| Container | Docker multi-arch (amd64 + arm64) |
| Registry | GHCR |
| Cluster | GKE Standard, GPU node pool |
| Infrastructure | Terraform (VPC, GKE, GCS, IAM) |
| CI/CD | GitHub Actions + OIDC (no stored credentials) |
| Metrics | Prometheus + Grafana (Triton native + gateway custom) |
| Logs | Loki + Promtail (structured JSON) |
| GPU metrics | DCGM exporter |
| Load testing | Locust (gRPC user class, 5 scenarios) |

---

## Models

| Model | Task | Backend | Size |
|-------|------|---------|------|
| ViT-tiny FP32 | ImageNet classification (1000 classes) | ONNX / onnxruntime | ~25MB |
| ViT-tiny INT8 | ImageNet classification (GPU only) | TensorRT | ~6MB |
| CLIP visual | Zero-shot image encoder | ONNX / onnxruntime | ~340MB |
| CLIP text | Zero-shot text encoder | ONNX / onnxruntime | ~250MB |

---

## Benchmark results (CPU — GKE e2-standard-2)

| Metric | Value |
|--------|-------|
| Requests | 2,642 |
| Failures | 0 (0%) |
| P50 latency | 250ms |
| P95 latency | 270ms |
| P99 latency | 310ms |
| Max latency | 620ms |
| Throughput | ~4 RPS |

> CPU inference at 250ms/request is the throughput bottleneck.
> On GPU (T4 + TensorRT INT8), expected P50 ~10ms, throughput ~200 RPS.
> GPU node pool is provisioned in Terraform — autoscales 0→2 T4 nodes under load.
> Regional capacity constraints in GCP during development required CPU fallback.

---

## Quantization pipeline

```
HuggingFace ViT-tiny (PyTorch)
        ↓  quantization/export_onnx.py
model.onnx  (FP32, ~25MB) ──────────────── serves directly via Triton onnxruntime
        ↓  quantization/convert_tensorrt.py (GPU host required)
model.plan  (INT8 TensorRT, ~6MB) ──────── 3-5x faster on GPU, <0.5% accuracy drop

Calibration: 500 ImageNet validation images
INT8 scale factors computed per-layer via quantization/calibration_dataset.py
```

---

## API

Defined in `proto/inference.proto`. All clients speak gRPC.

### Classify (ViT-tiny)

```python
import grpc
from src.generated import inference_pb2, inference_pb2_grpc

channel = grpc.insecure_channel("34.178.244.104:50051")
stub = inference_pb2_grpc.GatewayServiceStub(channel)

with open("image.jpg", "rb") as f:
    response = stub.Classify(inference_pb2.ClassifyRequest(
        image=inference_pb2.ImagePayload(data=f.read(), format="jpeg"),
        top_k=5,
        model_version="fp32",   # "fp32" | "int8"
    ))

for p in response.predictions:
    print(f"{p.label}: {p.confidence:.3f}")
# Egyptian Mau: 0.742
# tabby cat: 0.103
```

### Zero-shot (CLIP)

```python
response = stub.ZeroShot(inference_pb2.ZeroShotRequest(
    image=inference_pb2.ImagePayload(data=image_bytes, format="jpeg"),
    labels=["a golden retriever", "a tabby cat", "a red sports car"],
))

for p in response.predictions:
    print(f"{p.label}: {p.score:.3f}")
# a tabby cat: 0.931
# a golden retriever: 0.045
# a red sports car: 0.024
```

---

## Infrastructure

Provisioned with Terraform (`terraform/`).

```
GCP Project: vision-serving-platform
├── VPC + subnet (10.0.0.0/24, secondary ranges for pods/services)
├── GKE Standard cluster (europe-west4)
│   ├── CPU node pool (e2-standard-2, 1-3 nodes)
│   └── GPU node pool (n1-standard-4 + T4, 0-2 nodes, autoscales)
│       └── Taint: nvidia.com/gpu=present:NoSchedule
├── GCS bucket (model repository, Triton reads via init container)
└── IAM
    ├── Workload Identity (Triton → GCS, no static keys)
    └── GitHub OIDC (Actions → GKE deploy, no stored credentials)
```

### Provision

```bash
cd terraform
terraform init
terraform apply
gcloud container clusters get-credentials vision-serving \
    --region europe-west4 --project vision-serving-platform
```

### Destroy (stop paying)

```bash
cd terraform && terraform destroy
```

---

## CI/CD pipeline

```
Push to PR:    lint → test → terraform validate
Merge to main: lint → test → terraform validate → build+push → deploy
```

- **lint** — ruff, buf lint (proto validation)
- **test** — pytest unit + smoke tests
- **terraform validate** — fmt check + validate (no apply)
- **build+push** — Docker buildx multi-arch (amd64 + arm64) → GHCR
- **deploy** — OIDC auth to GCP → kubectl apply + rollout

No stored credentials anywhere. GitHub Actions federates identity with GCP via
Workload Identity Pool.

---

## Observability

### Dashboards (Grafana)

**Inference Platform:**
- Requests per second
- P50 / P95 / P99 latency
- Error rate
- Requests by model version (ViT FP32 vs INT8 vs CLIP)
- Prediction class distribution (drift detection)

**GPU Infrastructure:**
- GPU utilisation % (DCGM)
- VRAM usage (DCGM)
- GPU temperature (DCGM)
- Pod count (HPA events)
- Node count (cluster autoscaler events)

### Alerting rules

- `HighInferenceLatency` — P99 > 500ms for 2 minutes
- `HighInferenceErrorRate` — error rate > 5% for 2 minutes
- `PredictionDrift` — any class > 60% of predictions for 5 minutes

### Structured logs (Loki)

Every inference request logged as JSON:
```json
{
  "event": "inference_complete",
  "model_version": "vit_fp32",
  "top_class": "Egyptian Mau",
  "top_confidence": 0.742,
  "inference_ms": 251.3,
  "timestamp": "2026-06-18T01:08:13Z"
}
```

Query in Grafana: `{namespace="triton"} | json | inference_ms > 500`

---

## Load testing

Five Locust scenarios in `locust/locustfile.py`:

| Scenario | Class | Description |
|----------|-------|-------------|
| Ramp | `RampUser` | 10→500 users, autoscale proof |
| Sustained | `SustainedUser` | 200 users 30min, stability test |
| Mixed | `MixedUser` | 70% Classify + 30% ZeroShot |
| Label stress | `LabelStressUser` | ZeroShot with 2/10/50/100 labels |
| A/B versioning | `ABVersionUser` | 50% FP32 vs 50% INT8 (GPU) |

```bash
# Ramp test
locust -f locust/locustfile.py RampUser \
    --host <gateway-ip>:50051 \
    --users 500 --spawn-rate 10 \
    --run-time 15m --headless \
    --csv locust/results/ramp

# Mixed traffic
locust -f locust/locustfile.py MixedUser \
    --host <gateway-ip>:50051 \
    --users 200 --spawn-rate 5 \
    --run-time 20m --headless \
    --csv locust/results/mixed
```

---

## Local development

```bash
# Prerequisites
brew install bufbuild/buf/buf grpcurl

# Install
poetry install --with dev
make proto      # generate Python stubs from proto/inference.proto

# Run tests
make test

# Export models (M1, no GPU needed)
python quantization/export_onnx.py
python quantization/validate_onnx.py
python quantization/export_clip.py
python quantization/validate_clip.py

# Local Triton (Docker)
make triton-local

# Local gateway
make run-gateway

# Smoke test
grpcurl -plaintext localhost:50051 inference.v1.GatewayService/Health
```

---

## Project structure

```
proto/                  gRPC API contract (source of truth)
src/serving/            Gateway service
  gateway.py            gRPC server entrypoint
  handlers/             RPC handlers (classify, zeroshot, health)
  preprocessing.py      Image decode + normalisation
  postprocessing.py     Logits → top-k labels
  clip_postprocessing.py Cosine similarity ranking
  metrics.py            Prometheus counters + histograms
  settings.py           Pydantic config (env var injection)
  logging_config.py     Structlog JSON setup
quantization/           PyTorch → ONNX → TensorRT INT8 pipeline
triton_repo/            Triton model repository (artifacts in GCS)
k8s/                    Kubernetes manifests (Kustomize)
  base/                 Core manifests
  overlays/dev/         Dev overrides
  overlays/prod/        Prod overrides
terraform/              GKE + GPU node pool + GCS + IAM
  modules/gke/          GKE Standard cluster + node pools
  modules/vpc/          VPC + subnet
  modules/iam/          Service accounts + Workload Identity + GitHub OIDC
.github/workflows/      CI/CD pipeline
monitoring/             Grafana dashboards + Prometheus alert rules
  helm/                 Helm values (kube-prometheus-stack, loki-stack, DCGM)
locust/                 Load test scenarios
  results/              Benchmark CSV output
```

---

## Known limitations and future work

**GPU capacity:** T4 GPU nodes are configured in Terraform but couldn't be
provisioned during development due to GCP regional capacity constraints in
us-east1 and europe-west4. CPU fallback used for all benchmarks. GPU inference
(TensorRT INT8) benchmarked separately on Vast.ai.

**INT8 calibration:** TensorRT INT8 engine requires NVIDIA hardware for
conversion. Calibration pipeline (`quantization/convert_tensorrt.py`) is
complete and tested on Vast.ai. Engine not included in GCS model repo due to
platform-specific compilation.

**CLIP zero-shot:** Implemented and tested locally. Deployed to GKE and
verified end-to-end. Not included in primary Locust benchmark due to higher
CPU latency (~2-3s per request with 10 labels on CPU).

**Future:** vLLM text embedding endpoint, prediction drift auto-remediation,
multi-region failover, Istio service mesh for traffic splitting.
