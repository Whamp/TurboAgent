"""Model execution seam types.

See docs/design/model-execution.md. The interface is three entry points:
complete one candidate, stream one candidate, close owned resources.
Callers never see credentials, endpoints, or provider-specific payloads.
"""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, NewType, Protocol

ModelTargetId = NewType("ModelTargetId", str)
JsonObject = Mapping[str, Any]


@dataclass(frozen=True)
class ModelTarget:
    """Opaque handle for one configured backend model target."""

    id: str
    name: str = ""


@dataclass(frozen=True)
class ThinkingIntent:
    mode: Literal["default", "disabled", "adaptive", "budget"] = "default"
    effort: str | None = None
    budget_tokens: int | None = None


@dataclass(frozen=True)
class GenerationIntent:
    """Explicit client-request generation settings. Unset fields fall back to
    the target's configured defaults; the configured output cap always wins."""

    max_output_tokens: int | None = None
    # Which wire field the client used for its output cap. Some backends map
    # only one of the two keys, so the client's choice must survive.
    max_output_tokens_field: Literal["max_tokens", "max_completion_tokens"] = (
        "max_tokens"
    )
    temperature: float | None = None
    top_p: float | None = None
    stop: str | tuple[str, ...] | None = None
    tool_choice: str | JsonObject | None = None
    response_format: JsonObject | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: Mapping[str, int] = field(default_factory=dict)
    thinking: ThinkingIntent = field(default_factory=ThinkingIntent)
    sampling_params: JsonObject = field(default_factory=dict)
    stream_options: JsonObject | None = None


@dataclass(frozen=True)
class ModelExecutionRequest:
    # OpenAI-shaped during the first extraction (see design doc, "Migration
    # constraint"). The result side is canonical from day one.
    messages: tuple[JsonObject, ...]
    tools: tuple[JsonObject, ...] = ()
    generation: GenerationIntent = field(default_factory=GenerationIntent)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: JsonObject


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None


FinishReason = Literal["stop", "length", "tool_use", "content_filter", "other"]


@dataclass(frozen=True)
class AssistantOutput:
    text: str
    tool_calls: tuple[ToolCall, ...]
    thinking: str | None
    finish_reason: FinishReason


@dataclass(frozen=True)
class ModelExecutionResult:
    target: ModelTarget
    output: AssistantOutput
    usage: TokenUsage | None
    response_id: str | None = None


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    id: str
    name: str


@dataclass(frozen=True)
class ToolCallArgumentsDelta:
    id: str
    json_fragment: str


@dataclass(frozen=True)
class ExecutionCompleted:
    result: ModelExecutionResult


ModelExecutionEvent = (
    TextDelta
    | ThinkingDelta
    | ToolCallStarted
    | ToolCallArgumentsDelta
    | ExecutionCompleted
)


class ModelExecutor(Protocol):
    async def complete(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> ModelExecutionResult: ...

    def stream(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> AsyncIterator[ModelExecutionEvent]: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class TargetSpec:
    """Private per-target configuration. Never exposed to callers."""

    name: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    thinking: ThinkingIntent = field(default_factory=ThinkingIntent)
