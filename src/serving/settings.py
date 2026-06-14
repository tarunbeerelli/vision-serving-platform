"""
Runtime config — pydantic-settings reads env vars automatically.
k8s injects ConfigMap values as env vars, overriding these defaults.

  from serving.settings import settings
  settings.triton_address  →  "triton-service:8001"
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")
    host: str = "0.0.0.0"
    port: int = 50051
    metrics_port: int = 9090
    log_level: str = "INFO"


class TritonSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRITON_", env_file=".env", extra="ignore")
    host: str = "localhost"
    grpc_port: int = 8001
    http_port: int = 8000
    timeout_s: float = 10.0


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODEL_", env_file=".env", extra="ignore")
    default_vit_version: str = "int8"
    default_clip_version: str = "fp32"
    top_k_default: int = 5


class Settings:
    def __init__(self) -> None:
        self.gateway = GatewaySettings()
        self.triton = TritonSettings()
        self.model = ModelSettings()

    @property
    def triton_address(self) -> str:
        return f"{self.triton.host}:{self.triton.grpc_port}"


settings = Settings()
