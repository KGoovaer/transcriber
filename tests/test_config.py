import os
import pytest
from unittest.mock import patch
from transcriber.config import load_config, DEFAULT_CONFIG


def test_returns_defaults_when_no_config_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config = load_config(str(config_path))
    assert config["default_backend"] == "ollama"
    assert config["default_model"] == "qwen2-audio"


def test_overrides_defaults_with_file_values(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[transcriber]\ndefault_model = "gemma3"\n')
    config = load_config(str(config_path))
    assert config["default_model"] == "gemma3"
    assert config["default_backend"] == "ollama"  # still default


def test_creates_config_file_if_missing(tmp_path):
    config_path = tmp_path / "config.toml"
    load_config(str(config_path))
    assert config_path.exists()
