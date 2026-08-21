"""Composition of model execution targets from Turbo Agent configuration."""

import os
from dataclasses import dataclass

from ..utils import Config
from .litellm_adapter import LiteLLMExecutor
from .types import ModelExecutor, ModelTarget, TargetSpec, ThinkingIntent


def _thinking_from_config(value) -> ThinkingIntent:
    """YAML `thinking`: none | low/medium/high | token budget."""
    if value is None:
        return ThinkingIntent()
    if isinstance(value, str):
        if value.lower() == "none":
            return ThinkingIntent(mode="disabled")
        return ThinkingIntent(mode="adaptive", effort=value)
    if isinstance(value, (int, float)):
        return ThinkingIntent(mode="budget", budget_tokens=int(value))
    raise ValueError(f"Invalid thinking value: {value!r}")


@dataclass(frozen=True)
class ExecutionTargets:
    executor: ModelExecutor
    candidate_targets: tuple[ModelTarget, ...]
    default_target: ModelTarget


def build_candidate_execution(
    config: Config,
    executor: ModelExecutor | None = None,
) -> ExecutionTargets:
    """Build the Candidate execution plan from backend.models config.

    Each configured model becomes one opaque ModelTarget; num_candidates
    expands it into repeated Candidate slots. Callers see only targets —
    credentials and endpoints stay inside the executor.
    """
    specs: dict[ModelTarget, TargetSpec] = {}
    candidate_targets: list[ModelTarget] = []
    for index, model in enumerate(config.models):
        target = ModelTarget(id=f"backend-{index}", name=model["name"])
        specs[target] = TargetSpec(
            name=model["name"],
            api_key=model.get("api_key") or None,
            base_url=model.get("base_url") or None,
            temperature=model.get("temperature"),
            max_output_tokens=model.get("max_tokens"),
            thinking=_thinking_from_config(model.get("thinking")),
        )
        candidate_targets.extend([target] * model.get("num_candidates", 1))

    if not candidate_targets:
        # Same error Config.default_model raises for an empty backend list.
        raise ValueError("No models configured under backend.models")

    if executor is None:
        executor = LiteLLMExecutor(specs)
        _apply_legacy_gemini_env(config)
    return ExecutionTargets(
        executor=executor,
        candidate_targets=tuple(candidate_targets),
        default_target=candidate_targets[0],
    )


def _apply_legacy_gemini_env(config: Config) -> None:
    """litellm's gemini/ route resolves GEMINI_API_KEY from the environment
    even when api_key is passed explicitly. Scoped here until that behavior
    is verified removable against a live Gemini route."""
    for model in config.models:
        api_key = model.get("api_key", "")
        if api_key and model.get("name", "").startswith("gemini/"):
            os.environ["GEMINI_API_KEY"] = api_key
