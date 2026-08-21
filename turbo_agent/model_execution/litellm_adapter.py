"""LiteLLM-backed model execution adapter.

Owns model prefix routing, explicit api_key/base_url forwarding,
GenerationIntent translation, cap clamping, thinking compatibility, response
and stream normalization, and LiteLLM exception classification. Never mutates
process-wide environment or litellm globals.
"""

import asyncio
import copy
import json
from collections.abc import AsyncIterator
from typing import Any

import litellm

from .errors import FailureKind, ModelExecutionError
from .types import (
    AssistantOutput,
    ExecutionCompleted,
    GenerationIntent,
    ModelExecutionRequest,
    ModelExecutionResult,
    ModelExecutor,
    ModelTarget,
    TargetSpec,
    TextDelta,
    ThinkingDelta,
    ThinkingIntent,
    TokenUsage,
    ToolCall,
    ToolCallArgumentsDelta,
    ToolCallStarted,
)

_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "max_tokens": "length",
    "tool_calls": "tool_use",
    "tool_use": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
}

# Ordered: most specific first; APIError is the litellm catch-all.
_ERROR_KINDS: tuple[tuple[type[Exception], FailureKind], ...] = (
    (litellm.exceptions.AuthenticationError, "authentication"),
    (litellm.exceptions.RateLimitError, "rate_limited"),
    (litellm.exceptions.Timeout, "timeout"),
    (litellm.exceptions.APIConnectionError, "unavailable"),
    (litellm.exceptions.BadRequestError, "invalid_request"),
    (litellm.exceptions.APIError, "internal"),
)

_RETRYABLE_KINDS = {"rate_limited", "timeout", "unavailable"}


def _classify_error(exc: Exception) -> FailureKind:
    for exc_type, kind in _ERROR_KINDS:
        if isinstance(exc, exc_type):
            return kind
    return "internal"


def _wrap_error(target: ModelTarget, exc: Exception) -> ModelExecutionError:
    kind = _classify_error(exc)
    return ModelExecutionError(
        target=target,
        kind=kind,
        retryable=kind in _RETRYABLE_KINDS,
        message=f"{type(exc).__name__}: {exc}",
    )


def _effective_thinking(
    spec: TargetSpec,
    intent: GenerationIntent,
) -> ThinkingIntent | None:
    thinking = intent.thinking
    if thinking.mode == "default":
        thinking = spec.thinking
    if thinking.mode in ("default", "disabled"):
        return None
    return thinking


def _effective_max_tokens(
    spec: TargetSpec,
    intent: GenerationIntent,
) -> int | None:
    cap = spec.max_output_tokens
    if intent.max_output_tokens is None:
        return cap
    if cap is None:
        return intent.max_output_tokens
    return min(intent.max_output_tokens, cap)


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        arguments = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        arguments = {"_raw": raw}
    return arguments if isinstance(arguments, dict) else {"_raw": raw}


def _token_usage(usage: dict[str, Any] | None) -> TokenUsage | None:
    if not usage:
        return None
    details = usage.get("completion_tokens_details") or {}
    return TokenUsage(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        reasoning_tokens=details.get("reasoning_tokens"),
    )


class _StreamAccumulator:
    """Folds streaming chunks into canonical events and one final result."""

    def __init__(self, target: ModelTarget) -> None:
        self.target = target
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tools: dict[str, dict[str, str]] = {}
        self._tool_order: list[str] = []
        self._current_tool_id: str | None = None
        self._finish_raw: Any = None
        self._usage: dict[str, Any] | None = None
        self._response_id: str | None = None

    def ingest(self, chunk_dict: dict[str, Any]) -> list[Any]:
        """Consume one chunk; returns the events it produced."""
        events: list[Any] = []
        if chunk_dict.get("usage"):
            self._usage = chunk_dict["usage"]
            self._response_id = chunk_dict.get("id")
        choices = chunk_dict.get("choices") or []
        if not choices:
            return events
        choice = choices[0]
        delta = choice.get("delta") or {}

        content = delta.get("content")
        if content:
            self._text_parts.append(content)
            events.append(TextDelta(text=content))

        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            self._thinking_parts.append(reasoning)
            events.append(ThinkingDelta(text=reasoning))

        for tc_delta in delta.get("tool_calls") or []:
            events.extend(self._ingest_tool_call(tc_delta))

        if choice.get("finish_reason"):
            self._finish_raw = choice["finish_reason"]
        return events

    def _ingest_tool_call(self, tc_delta: dict[str, Any]) -> list[Any]:
        function = tc_delta.get("function") or {}
        tc_id = tc_delta.get("id")
        started: ToolCallStarted | None = None
        if tc_id:
            self._current_tool_id = tc_id
            if tc_id not in self._tools:
                self._tools[tc_id] = {
                    "name": function.get("name", ""),
                    "arguments": "",
                }
                self._tool_order.append(tc_id)
                started = ToolCallStarted(id=tc_id, name=self._tools[tc_id]["name"])
            elif not self._tools[tc_id]["name"]:
                self._tools[tc_id]["name"] = function.get("name", "")
        elif self._current_tool_id is None:
            return []

        fragment = function.get("arguments")
        if fragment:
            self._tools[self._current_tool_id]["arguments"] += fragment
            delta_event = ToolCallArgumentsDelta(
                id=self._current_tool_id, json_fragment=fragment
            )
            return [delta_event, started] if started else [delta_event]
        return [started] if started else []

    def build_result(self) -> ModelExecutionResult:
        tool_calls = tuple(
            ToolCall(
                id=tc_id,
                name=self._tools[tc_id]["name"],
                arguments=_parse_arguments(self._tools[tc_id]["arguments"]),
            )
            for tc_id in self._tool_order
        )
        return ModelExecutionResult(
            target=self.target,
            output=AssistantOutput(
                text="".join(self._text_parts),
                tool_calls=tool_calls,
                thinking="".join(self._thinking_parts) or None,
                finish_reason=_FINISH_REASON_MAP.get(self._finish_raw, "other"),
            ),
            usage=_token_usage(self._usage),
            response_id=self._response_id,
        )


class LiteLLMExecutor(ModelExecutor):
    """Executes candidates through litellm for a fixed set of targets."""

    def __init__(self, targets: Any) -> None:
        self._targets: dict[ModelTarget, TargetSpec] = dict(targets)
        # Injection point for tests; production uses litellm.acompletion.
        self._complete_fn = litellm.acompletion

    def resolve(self, target: ModelTarget) -> TargetSpec:
        try:
            return self._targets[target]
        except KeyError:
            raise KeyError(f"Unknown model target: {target.id}") from None

    async def aclose(self) -> None:
        return None

    # Request building (shared by complete and stream).

    def _build_params(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> dict[str, Any]:
        spec = self.resolve(target)
        generation = request.generation

        params: dict[str, Any] = {
            "model": spec.name,
            # Isolate nested payloads: providers may mutate messages/tools,
            # and concurrent candidates share the caller's request object.
            "messages": copy.deepcopy(list(request.messages)),
        }
        if spec.api_key:
            params["api_key"] = spec.api_key
        if spec.base_url:
            params["base_url"] = spec.base_url

        max_tokens = _effective_max_tokens(spec, generation)
        if max_tokens is not None:
            params[generation.max_output_tokens_field] = max_tokens

        temperature = (
            generation.temperature
            if generation.temperature is not None
            else spec.temperature
        )
        if temperature is not None:
            params["temperature"] = temperature

        for key in ("top_p", "tool_choice", "response_format", "seed"):
            value = getattr(generation, key)
            if value is not None:
                params[key] = value
        if generation.presence_penalty is not None:
            params["presence_penalty"] = generation.presence_penalty
        if generation.frequency_penalty is not None:
            params["frequency_penalty"] = generation.frequency_penalty
        if generation.logit_bias:
            params["logit_bias"] = dict(generation.logit_bias)
        if generation.stop is not None:
            params["stop"] = (
                list(generation.stop)
                if isinstance(generation.stop, tuple)
                else generation.stop
            )
        if request.tools:
            params["tools"] = copy.deepcopy(list(request.tools))
        if generation.stream_options is not None:
            params["stream_options"] = generation.stream_options

        self._apply_thinking(params, spec, generation, max_tokens)
        return params

    @staticmethod
    def _apply_thinking(
        params: dict[str, Any],
        spec: TargetSpec,
        generation: GenerationIntent,
        max_tokens: int | None,
    ) -> None:
        thinking = _effective_thinking(spec, generation)
        if thinking is None:
            return
        if thinking.mode == "adaptive":
            params["reasoning_effort"] = thinking.effort or "high"
            return
        # Budget-based thinking is an Anthropic Messages concept and must fit
        # strictly below the output cap; otherwise drop it rather than
        # shrinking the requested cap.
        budget = thinking.budget_tokens
        is_anthropic = spec.name.startswith("anthropic/")
        fits = max_tokens is None or (budget is not None and budget < max_tokens)
        if thinking.mode == "budget" and budget is not None and fits and is_anthropic:
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}

    # Response normalization.

    @staticmethod
    def _normalize_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for tc in raw_calls or []:
            function = tc.get("function") or {}
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                arguments = _parse_arguments(raw_arguments or "")
            calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=function.get("name", ""),
                    arguments=arguments,
                )
            )
        return tuple(calls)

    @classmethod
    def _normalize_result(
        cls,
        target: ModelTarget,
        response: dict[str, Any],
    ) -> ModelExecutionResult:
        choices = response.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        return ModelExecutionResult(
            target=target,
            output=AssistantOutput(
                text=message.get("content") or "",
                tool_calls=cls._normalize_tool_calls(message.get("tool_calls")),
                thinking=None,
                finish_reason=_FINISH_REASON_MAP.get(
                    choice.get("finish_reason"), "other"
                ),
            ),
            usage=_token_usage(response.get("usage")),
            response_id=response.get("id"),
        )

    async def complete(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> ModelExecutionResult:
        params = self._build_params(target, request)
        try:
            response = await self._complete_fn(**params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _wrap_error(target, exc) from exc
        payload = response.model_dump() if hasattr(response, "model_dump") else response
        return self._normalize_result(target, payload)

    async def stream(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> AsyncIterator[Any]:
        params = self._build_params(target, request)
        params["stream"] = True
        try:
            response = await self._complete_fn(**params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _wrap_error(target, exc) from exc

        accumulator = _StreamAccumulator(target)
        async for chunk in response:
            chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            for event in accumulator.ingest(chunk_dict):
                yield event

        yield ExecutionCompleted(result=accumulator.build_result())
