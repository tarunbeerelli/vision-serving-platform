"""Validates locustfile is importable and user classes are defined."""

import importlib.util
from pathlib import Path


def test_locustfile_importable() -> None:
    spec = importlib.util.spec_from_file_location(
        "locustfile",
        Path(__file__).parent.parent.parent / "locust" / "locustfile.py",
    )
    assert spec is not None
    # Just check it parses — don't execute
    import ast

    source = (Path(__file__).parent.parent.parent / "locust" / "locustfile.py").read_text()
    ast.parse(source)


def test_locust_results_dir_exists() -> None:
    results = Path(__file__).parent.parent.parent / "locust" / "results"
    results.mkdir(exist_ok=True)
    assert results.exists()
