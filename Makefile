.PHONY: install proto lint format test clean

install:
	poetry install --with dev,loadtest
	pre-commit install

proto:
	@mkdir -p src/generated
	poetry run python -m grpc_tools.protoc \
		-I proto \
		--python_out=src/generated \
		--grpc_python_out=src/generated \
		--pyi_out=src/generated \
		proto/inference.proto
	@sed -i '' 's/import inference_pb2/import src.generated.inference_pb2/' \
		src/generated/inference_pb2_grpc.py
	@touch src/generated/__init__.py
	@echo "✓ Stubs in src/generated/"

lint:
	poetry run ruff check src tests
	poetry run ruff format --check src tests
	buf lint

format:
	poetry run ruff format src tests
	poetry run ruff check --fix src tests

test:
	poetry run pytest tests/unit tests/smoke -v --tb=short

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf src/generated dist .coverage .ruff_cache .mypy_cache .pytest_cache

export-onnx:
	python quantization/export_onnx.py

validate-onnx:
	python quantization/validate_onnx.py

# Run on Vast.ai only
convert-trt:
	python quantization/convert_tensorrt.py

run-gateway:
	TRITON_HOST=localhost python -m serving.gateway

triton-local:
	docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
		-v $(PWD)/triton_repo:/models \
		nvcr.io/nvidia/tritonserver:24.05-py3 \
		tritonserver --model-repository=/models \
		--model-control-mode=explicit \
		--load-model=vit_fp32

test-triton:
	python scripts/test_triton_local.py

k8s-apply-dev:
	kubectl apply -k k8s/overlays/dev

k8s-apply-prod:
	kubectl apply -k k8s/overlays/prod

k8s-dry-run:
	kubectl apply -k k8s/overlays/dev --dry-run=client

k8s-diff:
	kubectl diff -k k8s/overlays/dev

docker-build:
	docker build -t vision-serving-platform:local .

docker-run:
	docker run --rm -p 50051:50051 -p 9090:9090 \
		-e TRITON_HOST=host.docker.internal \
		vision-serving-platform:local
