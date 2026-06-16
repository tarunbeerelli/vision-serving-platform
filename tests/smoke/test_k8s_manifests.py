"""
Validates k8s manifest structure without a running cluster.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
K8S_BASE = ROOT / "k8s" / "base"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_triton_deployment_has_gpu_toleration() -> None:
    d = _load(K8S_BASE / "triton" / "deployment.yaml")
    tolerations = d["spec"]["template"]["spec"]["tolerations"]
    keys = [t["key"] for t in tolerations]
    assert "nvidia.com/gpu" in keys


def test_triton_deployment_has_init_container() -> None:
    d = _load(K8S_BASE / "triton" / "deployment.yaml")
    init_containers = d["spec"]["template"]["spec"]["initContainers"]
    assert len(init_containers) > 0
    assert init_containers[0]["name"] == "model-downloader"


def test_gateway_deployment_has_configmap_ref() -> None:
    d = _load(K8S_BASE / "gateway" / "deployment.yaml")
    containers = d["spec"]["template"]["spec"]["containers"]
    env_from = containers[0]["envFrom"]
    names = [e["configMapRef"]["name"] for e in env_from]
    assert "gateway-config" in names


def test_triton_service_is_clusterip() -> None:
    s = _load(K8S_BASE / "triton" / "service.yaml")
    assert s["spec"]["type"] == "ClusterIP"


def test_gateway_service_is_loadbalancer() -> None:
    s = _load(K8S_BASE / "gateway" / "service.yaml")
    assert s["spec"]["type"] == "LoadBalancer"


def test_hpa_min_replicas() -> None:
    h = _load(K8S_BASE / "hpa.yaml")
    assert h["spec"]["minReplicas"] == 2
    assert h["spec"]["maxReplicas"] == 8
