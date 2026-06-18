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

install-observability:
	helm repo add prometheus-community \
		https://prometheus-community.github.io/helm-charts
	helm repo add grafana https://grafana.github.io/helm-charts
	helm repo update
	helm upgrade --install kube-prometheus-stack \
		prometheus-community/kube-prometheus-stack \
		--namespace monitoring \
		--create-namespace \
		--values monitoring/helm/prometheus-values.yaml
	helm upgrade --install loki-stack \
		grafana/loki-stack \
		--namespace monitoring \
		--values monitoring/helm/loki-values.yaml

install-dcgm:
	helm repo add gpu-helm-charts \
		https://nvidia.github.io/dcgm-exporter/helm-charts
	helm repo update
	helm upgrade --install dcgm-exporter \
		gpu-helm-charts/dcgm-exporter \
		--namespace monitoring \
		--values monitoring/helm/dcgm-values.yaml

grafana-ip:
	kubectl get svc kube-prometheus-stack-grafana \
		-n monitoring \
		-o jsonpath='{.status.loadBalancer.ingress[0].ip}'

grafana-password:
	kubectl get secret kube-prometheus-stack-grafana \
		-n monitoring \
		-o jsonpath='{.data.admin-password}' | base64 -d

port-forward-grafana:
	kubectl port-forward svc/kube-prometheus-stack-grafana \
		3000:80 -n monitoring

port-forward-prometheus:
	kubectl port-forward svc/kube-prometheus-stack-prometheus \
		9090:9090 -n monitoring

export-clip:
	python quantization/export_clip.py

validate-clip:
	python quantization/validate_clip.py

upload-models:
	bash scripts/upload_models.sh

deploy-gke:
	bash scripts/deploy_to_gke.sh

# Locust scenarios — replace HOST with gateway external IP
locust-ramp:
	locust -f locust/locustfile.py --tags ramp \
		--host $(HOST) \
		--users 500 --spawn-rate 10 \
		--run-time 15m --headless \
		--csv locust/results/ramp

locust-mixed:
	locust -f locust/locustfile.py --tags mixed \
		--host $(HOST) \
		--users 200 --spawn-rate 5 \
		--run-time 20m --headless \
		--csv locust/results/mixed

locust-stress:
	locust -f locust/locustfile.py --tags stress \
		--host $(HOST) \
		--users 50 --spawn-rate 2 \
		--run-time 10m --headless \
		--csv locust/results/stress

locust-ab:
	locust -f locust/locustfile.py --tags ab \
		--host $(HOST) \
		--users 100 --spawn-rate 5 \
		--run-time 10m --headless \
		--csv locust/results/ab

locust-ui:
	locust -f locust/locustfile.py \
		--host $(HOST)
