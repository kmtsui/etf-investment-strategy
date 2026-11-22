"""Smoke tests verifying package structure and manifest schema."""

import json
from pathlib import Path

import etf_strategy

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_exports_strategy_protocol() -> None:
    assert hasattr(etf_strategy, "StrategyFn")


def test_data_manifest_is_valid_json() -> None:
    manifest_path = REPO_ROOT / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert isinstance(manifest["datasets"], dict)
