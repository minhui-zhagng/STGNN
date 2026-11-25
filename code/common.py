"""Shared argument parsing, configuration loading, and output paths."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# 1. Command-line arguments
def arguments(description):
    """Read the configuration file and edge-feature mode from the command line."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--mode", default="exponential_markov")
    return parser.parse_args()


# 2. Path handling
def resolve_output_path(config, config_path):
    """Resolve relative output paths against the configuration file directory."""
    output_path = Path(config["output_root"])

    if not output_path.is_absolute():
        output_path = config_path.parent / output_path

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# 3. Configuration loading
def load_config(path):
    """Return the configuration dictionary and its resolved output directory."""
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_path = resolve_output_path(config, config_path)
    return config, output_path
