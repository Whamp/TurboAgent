"""Composition of model execution targets from Turbo Agent configuration."""

import os
from dataclasses import dataclass

from ..utils import Config
from .litellm_adapter import LiteLLMExecutor
from .pi_adapter import PiCompanionExecutor
from .types import (
    ModelExecutionRequest,
    ModelExecutionResult,
    ModelExecutor,
    ModelTarget,
    TargetSpec,
    ThinkingIntent,
)


class RoutingExecutor(ModelExecutor):
    """Dispatches each target to the adapter that owns it."""

    def __init__(self, routes: dict[ModelTarget, ModelExecutor]) -> None:
        self._routes = dict(routes)

    def resolve(self, target: ModelTarget):
        return self._routes[target].resolve(target)

    async def complete(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> ModelExecutionResult:
        return await self._routes[target].complete(target, request)

    def stream(self, target: ModelTarget, request: ModelExecutionRequest):
        return self._routes[target].stream(target, request)

    async def aclose(self) -> None:
        for executor in set(self._routes.values()):
            await executor.aclose()


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
    pi_executor: ModelExecutor | None = None,
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
        executor = _compose_executors(config, specs, pi_executor)
        _apply_legacy_gemini_env(config)
    return ExecutionTargets(
        executor=executor,
        candidate_targets=tuple(candidate_targets),
        default_target=candidate_targets[0],
    )


def _compose_executors(
    config: Config,
    specs: dict[ModelTarget, TargetSpec],
    pi_executor: ModelExecutor | None = None,
) -> ModelExecutor:
    """Construct only the adapters required by configured targets.

    A model entry with `executor: pi` runs through the Pi companion; all
    other entries run through LiteLLM. One adapter per kind, routed by
    target.
    """
    routes: dict[ModelTarget, ModelExecutor] = {}
    litellm_specs: dict[ModelTarget, TargetSpec] = {}
    pi_specs: dict[ModelTarget, TargetSpec] = {}
    for target, spec in specs.items():
        entry = _entry_for_target(config, target)
        if entry.get("executor") == "pi":
            pi_specs[target] = spec
        else:
            litellm_specs[target] = spec
    if litellm_specs:
        litellm_executor = LiteLLMExecutor(litellm_specs)
        routes.update({t: litellm_executor for t in litellm_specs})
    if pi_specs:
        pi_adapter = pi_executor or PiCompanionExecutor(pi_specs)
        routes.update({t: pi_adapter for t in pi_specs})
    executors = set(routes.values())
    if len(executors) == 1:
        return next(iter(executors))
    return RoutingExecutor(routes)


def _entry_for_target(config: Config, target: ModelTarget) -> dict:
    index = int(target.id.rsplit("-", 1)[-1])
    return config.models[index]


def _apply_legacy_gemini_env(config: Config) -> None:
    """litellm's gemini/ route resolves GEMINI_API_KEY from the environment
    even when api_key is passed explicitly. Scoped here until that behavior
    is verified removable against a live Gemini route."""
    for model in config.models:
        api_key = model.get("api_key", "")
        if api_key and model.get("name", "").startswith("gemini/"):
            os.environ["GEMINI_API_KEY"] = api_key
