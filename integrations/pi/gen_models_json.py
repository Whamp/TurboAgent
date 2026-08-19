#!/usr/bin/env python3
"""Generate the Pi `models.json` provider block from a TurboAgent config.

Pi (the coding agent) registers custom providers in `~/.pi/agent/models.json`.
This script reads a `turbo-agent.yaml` backend config and prints the JSON for
the `providers.turbo` entry, so the models Pi offers always match the models
the proxy actually runs.

Usage:

    # Print the provider block
    python integrations/pi/gen_models_json.py turbo-agent.yaml

    # Merge it into your pi models.json (creates the file if missing)
    python integrations/pi/gen_models_json.py turbo-agent.yaml --merge \
        ~/.pi/agent/models.json

The proxy ignores the model id a client requests and always runs the backend
models from turbo-agent.yaml, so Pi's model entries are labels that must exist
in the config above.

Note: pi expects camelCase model fields (contextWindow, maxTokens).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# Same defaults as Backend._model_metadata in turbo_agent/proxy/backend.py.
CONTEXT_DEFAULTS = {
    "gemini": 1_000_000,
    "openai": 200_000,
    "anthropic": 200_000,
    "openrouter": 200_000,
    "kimi": 128_000,
    "zai": 128_000,
}

PROVIDER_NAME = "turbo"


def model_entry(model: dict) -> dict:
    name = model["name"]
    prefix = name.split("/", 1)[0] if "/" in name else ""
    return {
        "id": name,
        "name": f"Turbo Agent ({name})",
        "reasoning": True,  # Pi thinking UI; proxy honors client effort
        "thinkingLevelMap": {
            "off": "none",
            "low": "low",
            "medium": "medium",
            "high": "high",
        },
        "input": (["text", "image"] if prefix == "gemini" else ["text"]),
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": model.get("context_window")
        or CONTEXT_DEFAULTS.get(prefix, 128_000),
        "maxTokens": model.get("max_tokens") or 8192,
    }


def provider_block(models: list) -> dict:
    return {
        "providers": {
            PROVIDER_NAME: {
                "baseUrl": "http://localhost:8888/v1",
                "api": "openai-completions",
                "apiKey": "turbo-agent-local",  # the proxy ignores auth
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": True,
                },
                "models": [model_entry(m) for m in models],
            }
        }
    }


def discover_turbo_agent_yaml() -> str:
    """Project ./turbo-agent.yaml, then ~/.config/turbo-agent/turbo-agent.yaml."""
    try:
        from turbo_agent.utils.config import Config
        return str(Config._discover())
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from turbo_agent.utils.config import Config
        return str(Config._discover())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", nargs="?", default=None,
        help="path to turbo-agent.yaml (default: project file, then XDG global)",
    )
    parser.add_argument("--merge", metavar="MODELS_JSON",
                        help="merge into this pi models.json instead of printing")
    args = parser.parse_args()

    try:
        config_path = args.config or discover_turbo_agent_yaml()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    models = cfg.get("backend", {}).get("models", [])
    if not models:
        print(f"no backend.models found in {config_path}", file=sys.stderr)
        return 1

    block = provider_block(models)

    if args.merge:
        path = args.merge
        try:
            with open(path) as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {}
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        existing.setdefault("providers", {})[PROVIDER_NAME] = block["providers"][PROVIDER_NAME]
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
            f.write("\n")
        print(f"merged turbo provider into {path}")
    else:
        print(json.dumps(block, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
