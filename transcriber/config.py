from __future__ import annotations
import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DEFAULT_CONFIG = {
    "default_backend": "ollama",
    "default_model": "qwen2-audio",
}

DEFAULT_CONFIG_TOML = """\
[transcriber]
default_backend = "ollama"
default_model = "qwen2-audio"
"""


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = os.path.expanduser("~/.transcriber/config.toml")

    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            f.write(DEFAULT_CONFIG_TOML)
        return dict(DEFAULT_CONFIG)

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    section = data.get("transcriber", {})
    return {**DEFAULT_CONFIG, **section}
