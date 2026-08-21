"""Model execution seam tests.

These exercise the ModelExecutor contract from docs/design/model-execution.md
through the same seam Backend uses:

- backend model base_url reaches both complete and stream calls
- token/thinking precedence: client intent over target defaults, cap last
- concurrent candidates receive isolated prompts and tools
- provider failures normalize to ModelExecutionError with a safe message
- cancellation propagates to the underlying call as CancelledError
- complete and collected-stream output normalize identically
"""

import asyncio
import copy
import json

import litellm
import pytest

from turbo_agent.model_execution import (
    ExecutionCompleted,
    GenerationIntent,
    LiteLLMExecutor,
    ModelExecutionError,
    ModelExecutionRequest,
    ModelTarget,
    TargetSpec,
    TextDelta,
    ThinkingIntent,
)


def make_target(spec_overrides=None):
    spec = TargetSpec(
        name="openai/dummy",
        api_key="dummy-key",
        base_url="http://127.0.0.1:9999/v1",
        max_output_tokens=2048,
        **(spec_overrides or {}),
    )
    executor = LiteLLMExecutor({ModelTarget("t1"): spec})
    return executor, ModelTarget("t1")


def canned_response(content="hello", finish="stop", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-x",
        "model": "dummy",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish},
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


class FakeCompletion:
    """Stands in for litellm.acompletion and records call kwargs."""

    def __init__(self, response=None):
        self.calls = []
        self.response = response or canned_response()

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.response)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return copy.deepcopy(self._payload)


# ---------------------------------------------------------------------------
# base_url forwarding
# ---------------------------------------------------------------------------


async def test_base_url_reaches_complete_call():
    executor, target = make_target()
    fake = FakeCompletion()
    executor._complete_fn = fake

    await executor.complete(
        target,
        ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
    )

    assert fake.calls[0]["base_url"] == "http://127.0.0.1:9999/v1"
    assert fake.calls[0]["api_key"] == "dummy-key"


async def test_base_url_reaches_stream_call():
    executor, target = make_target()
    fake = FakeCompletion()

    async def fake_stream(**kwargs):
        fake.calls.append(kwargs)
        return _aiter([_stream_chunk("hello")])

    executor._complete_fn = fake_stream

    events = [
        event
        async for event in executor.stream(
            target,
            ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
        )
    ]

    assert fake.calls[0]["base_url"] == "http://127.0.0.1:9999/v1"
    assert any(isinstance(event, TextDelta) for event in events)


# ---------------------------------------------------------------------------
# Token / thinking precedence
# ---------------------------------------------------------------------------


async def test_client_cap_clamped_to_target_cap():
    executor, target = make_target()
    fake = FakeCompletion()
    executor._complete_fn = fake

    await executor.complete(
        target,
        ModelExecutionRequest(
            messages=({"role": "user", "content": "hi"},),
            generation=GenerationIntent(max_output_tokens=8192),
        ),
    )

    assert fake.calls[0]["max_tokens"] == 2048


async def test_target_default_cap_applies_without_client_intent():
    executor, target = make_target()
    fake = FakeCompletion()
    executor._complete_fn = fake

    await executor.complete(
        target,
        ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
    )

    assert fake.calls[0]["max_tokens"] == 2048


async def test_client_effort_replaces_target_thinking_budget():
    executor, target = make_target(
        {"thinking": ThinkingIntent(mode="budget", budget_tokens=1024)},
    )
    fake = FakeCompletion()
    executor._complete_fn = fake

    await executor.complete(
        target,
        ModelExecutionRequest(
            messages=({"role": "user", "content": "hi"},),
            generation=GenerationIntent(
                thinking=ThinkingIntent(mode="adaptive", effort="low"),
            ),
        ),
    )

    assert "thinking" not in fake.calls[0]
    assert fake.calls[0]["reasoning_effort"] == "low"


async def test_disabled_thinking_removes_target_thinking():
    executor, target = make_target(
        {"thinking": ThinkingIntent(mode="adaptive", effort="high")},
    )
    fake = FakeCompletion()
    executor._complete_fn = fake

    await executor.complete(
        target,
        ModelExecutionRequest(
            messages=({"role": "user", "content": "hi"},),
            generation=GenerationIntent(thinking=ThinkingIntent(mode="disabled")),
        ),
    )

    assert "reasoning_effort" not in fake.calls[0]
    assert "thinking" not in fake.calls[0]


async def test_thinking_budget_dropped_when_it_cannot_fit_under_cap():
    executor, target = make_target()
    fake = FakeCompletion()
    executor._complete_fn = fake

    await executor.complete(
        target,
        ModelExecutionRequest(
            messages=({"role": "user", "content": "hi"},),
            generation=GenerationIntent(
                max_output_tokens=1024,
                thinking=ThinkingIntent(mode="budget", budget_tokens=1024),
            ),
        ),
    )

    assert "thinking" not in fake.calls[0]


# ---------------------------------------------------------------------------
# Prompt / tool isolation
# ---------------------------------------------------------------------------


async def test_completion_cannot_mutate_caller_messages():
    executor, target = make_target()

    async def mutating(**kwargs):
        kwargs["messages"][0]["content"] = "corrupted"
        return FakeResponse(canned_response())

    executor._complete_fn = mutating
    messages = [{"role": "user", "content": "original"}]

    await executor.complete(
        target,
        ModelExecutionRequest(messages=tuple(messages)),
    )
    await executor.complete(
        target,
        ModelExecutionRequest(messages=tuple(messages)),
    )

    assert messages[0]["content"] == "original"


async def test_concurrent_candidates_receive_isolated_prompts():
    executor, target = make_target()
    seen = []

    async def recording(**kwargs):
        seen.append(kwargs["messages"][0]["content"])
        await asyncio.sleep(0.01)
        kwargs["messages"][0]["content"] = f"mutated-{len(seen)}"
        return FakeResponse(canned_response())

    executor._complete_fn = recording
    shared = ({"role": "user", "content": "shared-prompt"},)
    request = ModelExecutionRequest(messages=shared)

    await asyncio.gather(
        executor.complete(target, request),
        executor.complete(target, request),
        executor.complete(target, request),
    )

    assert seen == ["shared-prompt", "shared-prompt", "shared-prompt"]


# ---------------------------------------------------------------------------
# Normalized failures
# ---------------------------------------------------------------------------


async def test_auth_failure_normalizes_with_safe_message():
    executor, target = make_target()

    async def failing(**kwargs):
        raise litellm.exceptions.AuthenticationError(
            message="bad key",
            model="dummy",
            llm_provider="openai",
        )

    executor._complete_fn = failing

    with pytest.raises(ModelExecutionError) as exc_info:
        await executor.complete(
            target,
            ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
        )

    assert exc_info.value.kind == "authentication"
    assert exc_info.value.retryable is False
    assert "dummy-key" not in str(exc_info.value)


async def test_rate_limit_normalizes_retryable():
    executor, target = make_target()

    async def failing(**kwargs):
        raise litellm.exceptions.RateLimitError(
            message="slow down",
            model="dummy",
            llm_provider="openai",
        )

    executor._complete_fn = failing

    with pytest.raises(ModelExecutionError) as exc_info:
        await executor.complete(
            target,
            ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
        )

    assert exc_info.value.kind == "rate_limited"
    assert exc_info.value.retryable is True


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancel_propagates_and_cancels_underlying_call():
    executor, target = make_target()
    inner_cancelled = False

    async def slow(**kwargs):
        nonlocal inner_cancelled
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            inner_cancelled = True
            raise

    executor._complete_fn = slow
    task = asyncio.ensure_future(
        executor.complete(
            target,
            ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert inner_cancelled is True


# ---------------------------------------------------------------------------
# Complete / stream equivalence
# ---------------------------------------------------------------------------


def _stream_chunk(content_delta=None, tool_calls=None, finish=None):
    delta = {}
    if content_delta is not None:
        delta["content"] = content_delta
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-x",
        "model": "dummy",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _aiter(items):
    async def generator():
        for item in items:
            yield item

    return generator()


async def test_stream_events_match_complete_result():
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": json.dumps({"command": "ls"})},
        },
    ]
    response = canned_response(
        content="working",
        finish="tool_calls",
        tool_calls=tool_calls,
    )

    executor, target = make_target()
    executor._complete_fn = FakeCompletion(response)

    result = await executor.complete(
        target,
        ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
    )

    async def fake_stream(**kwargs):
        return _aiter(
            [
                _stream_chunk(content_delta="work"),
                _stream_chunk(content_delta="ing"),
                _stream_chunk(
                    tool_calls=[
                        {
                            "id": "call_1",
                            "function": {"name": "Bash", "arguments": '{"command":'},
                        }
                    ]
                ),
                _stream_chunk(
                    tool_calls=[
                        {"function": {"arguments": ' "ls"}'}},
                    ]
                ),
                _stream_chunk(finish="tool_calls"),
                {
                    "id": "chatcmpl-x",
                    "model": "dummy",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            ]
        )

    executor._complete_fn = fake_stream
    events = [
        event
        async for event in executor.stream(
            target,
            ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
        )
    ]

    text = "".join(event.text for event in events if isinstance(event, TextDelta))
    completed = [e for e in events if isinstance(e, ExecutionCompleted)]
    assert text == result.output.text == "working"
    assert len(completed) == 1
    assert completed[-1] is events[-1]
    assert [
        (call.name, call.arguments) for call in completed[0].result.output.tool_calls
    ] == [
        ("Bash", {"command": "ls"}),
    ]
    assert completed[0].result.output.finish_reason == "tool_use"
    assert completed[0].result.usage.output_tokens == 5
