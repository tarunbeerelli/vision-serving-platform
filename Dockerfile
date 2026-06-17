FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install poetry==1.8.3 --no-cache-dir

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.in-project true && \
    poetry install --only main --no-interaction --no-ansi

FROM python:3.12-slim AS runtime

WORKDIR /app

# grpc_health_probe — used by k8s liveness/readiness probes
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    wget -qO/usr/local/bin/grpc_health_probe \
    https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.19/grpc_health_probe-linux-amd64 && \
    chmod +x /usr/local/bin/grpc_health_probe && \
    apt-get purge -y wget && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/
COPY configs/ ./configs/
COPY proto/ ./proto/

# Regenerate proto stubs inside image
RUN mkdir -p src/generated && \
    .venv/bin/python -m grpc_tools.protoc \
    -I proto \
    --python_out=src/generated \
    --grpc_python_out=src/generated \
    --pyi_out=src/generated \
    proto/inference.proto && \
    touch src/generated/__init__.py && \
    sed -i 's/import inference_pb2/import src.generated.inference_pb2/' \
    src/generated/inference_pb2_grpc.py

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 50051 9090

CMD ["python", "-m", "serving.gateway"]
