"""
Creates Grafana dashboards via the Grafana API.
Run after Grafana is deployed and has an external IP.

Usage:
  GRAFANA_URL=http://<external-ip> python scripts/create_dashboards.py
"""

import json
import os
import urllib.error
import urllib.request

GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER = "admin"
GRAFANA_PASS = "vision-serving-admin"

INFERENCE_DASHBOARD = {
    "dashboard": {
        "title": "Inference Platform",
        "tags": ["vision-serving"],
        "timezone": "browser",
        "panels": [
            {
                "title": "Requests per second",
                "type": "stat",
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
                "targets": [
                    {
                        "expr": "sum(rate(gateway_inference_requests_total[1m]))",
                        "legendFormat": "RPS",
                    }
                ],
            },
            {
                "title": "P99 latency (ms)",
                "type": "stat",
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.99, rate(gateway_inference_latency_ms_bucket[5m]))",
                        "legendFormat": "P99",
                    }
                ],
            },
            {
                "title": "P50 latency (ms)",
                "type": "stat",
                "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.50, rate(gateway_inference_latency_ms_bucket[5m]))",
                        "legendFormat": "P50",
                    }
                ],
            },
            {
                "title": "Error rate",
                "type": "stat",
                "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0},
                "targets": [
                    {
                        "expr": "rate(gateway_inference_requests_total{status='error'}[5m]) / rate(gateway_inference_requests_total[5m])",
                        "legendFormat": "Error rate",
                    }
                ],
            },
            {
                "title": "Latency distribution",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.50, rate(gateway_inference_latency_ms_bucket[1m]))",
                        "legendFormat": "P50",
                    },
                    {
                        "expr": "histogram_quantile(0.95, rate(gateway_inference_latency_ms_bucket[1m]))",
                        "legendFormat": "P95",
                    },
                    {
                        "expr": "histogram_quantile(0.99, rate(gateway_inference_latency_ms_bucket[1m]))",
                        "legendFormat": "P99",
                    },
                ],
            },
            {
                "title": "Requests by model version",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
                "targets": [
                    {
                        "expr": "sum by (model_version) (rate(gateway_inference_requests_total[1m]))",
                        "legendFormat": "{{ model_version }}",
                    }
                ],
            },
            {
                "title": "Prediction class distribution (drift detection)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 12},
                "targets": [
                    {
                        "expr": "topk(5, sum by (class_id) (rate(gateway_prediction_class_total[5m])))",
                        "legendFormat": "class {{ class_id }}",
                    }
                ],
            },
        ],
        "refresh": "10s",
        "schemaVersion": 38,
    },
    "overwrite": True,
    "folderId": 0,
}

GPU_DASHBOARD = {
    "dashboard": {
        "title": "GPU Infrastructure",
        "tags": ["vision-serving", "gpu"],
        "timezone": "browser",
        "panels": [
            {
                "title": "GPU utilisation (%)",
                "type": "gauge",
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0},
                "targets": [{"expr": "DCGM_FI_DEV_GPU_UTIL", "legendFormat": "GPU util"}],
            },
            {
                "title": "GPU memory used (MB)",
                "type": "gauge",
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0},
                "targets": [{"expr": "DCGM_FI_DEV_FB_USED", "legendFormat": "VRAM used"}],
            },
            {
                "title": "GPU temperature (°C)",
                "type": "gauge",
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0},
                "targets": [{"expr": "DCGM_FI_DEV_GPU_TEMP", "legendFormat": "GPU temp"}],
            },
            {
                "title": "Pod count (HPA)",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                "targets": [
                    {
                        "expr": "count(kube_pod_info{namespace='triton'})",
                        "legendFormat": "pod count",
                    }
                ],
            },
            {
                "title": "Node count",
                "type": "timeseries",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                "targets": [{"expr": "count(kube_node_info)", "legendFormat": "nodes"}],
            },
        ],
        "refresh": "10s",
        "schemaVersion": 38,
    },
    "overwrite": True,
    "folderId": 0,
}


def post_dashboard(dashboard: dict) -> None:
    url = f"{GRAFANA_URL}/api/dashboards/db"
    data = json.dumps(dashboard).encode()
    import base64

    credentials = base64.b64encode(f"{GRAFANA_USER}:{GRAFANA_PASS}".encode()).decode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {credentials}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"✓ Dashboard '{dashboard['dashboard']['title']}' created: {result.get('url')}")


if __name__ == "__main__":
    post_dashboard(INFERENCE_DASHBOARD)
    post_dashboard(GPU_DASHBOARD)
    print("\n✓ All dashboards created")
