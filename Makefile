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
