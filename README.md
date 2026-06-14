# Vision Serving Platform

Production vision inference platform — ViT-tiny (INT8 TensorRT) + CLIP served
via NVIDIA Triton Inference Server on GKE.

## Quickstart

```bash
poetry install --with dev
make proto
make lint
make test
```
