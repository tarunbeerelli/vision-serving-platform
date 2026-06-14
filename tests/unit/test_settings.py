import pytest
from serving.settings import Settings


def test_default_gateway_port() -> None:
    assert Settings().gateway.port == 50051


def test_default_triton_host() -> None:
    assert Settings().triton.host == "localhost"


def test_default_model_version() -> None:
    assert Settings().model.default_vit_version == "int8"


def test_triton_address_format() -> None:
    assert Settings().triton_address == "localhost:8001"


def test_env_override_triton_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRITON_HOST", "triton-service")
    assert Settings().triton.host == "triton-service"


def test_env_override_gateway_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_PORT", "50052")
    assert Settings().gateway.port == 50052
