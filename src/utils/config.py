"""Tiny YAML config loader. Paths in the config are relative to the repository root."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "config.yaml") -> dict:
    with open(PROJECT_ROOT / "configs" / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Resolve a repo-relative path from the config to an absolute path."""
    return PROJECT_ROOT / path
