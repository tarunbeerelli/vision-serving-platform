"""
Smoke tests — validate structure without running any model code.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent


def test_proto_exists() -> None:
    assert (ROOT / "proto" / "inference.proto").exists()


def test_proto_service_definition() -> None:
    proto = (ROOT / "proto" / "inference.proto").read_text()
    for symbol in [
        "service GatewayService",
        "rpc Classify",
        "rpc ZeroShot",
        "rpc Health",
        "rpc Models",
    ]:
        assert symbol in proto, f"missing: {symbol}"


def test_server_config_exists() -> None:
    assert (ROOT / "configs" / "server.yaml").exists()


def test_server_config_keys() -> None:
    data = yaml.safe_load((ROOT / "configs" / "server.yaml").read_text())
    for key in ["gateway", "triton", "models"]:
        assert key in data, f"missing key: {key}"


def test_package_imports() -> None:
    from serving.logging_config import get_logger
    from serving.settings import settings

    assert settings.gateway.port == 50051
    assert callable(get_logger)
