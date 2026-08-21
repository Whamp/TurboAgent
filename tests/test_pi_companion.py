"""Contract tests for the Pi ModelRuntime companion.

These spawn the real Node companion with a scripted fake runtime, exercising
the wire protocol: versioning, complete/stream mapping, multiplexing,
cancellation, crash handling, and credential redaction.
"""

import asyncio
import json
import os
import signal
from pathlib import Path

import pytest

from turbo_agent.model_execution import (
    ExecutionCompleted,
    LiteLLMExecutor,
    ModelExecutionError,
    ModelExecutionRequest,
    ModelTarget,
    TargetSpec,
    TextDelta,
    ThinkingIntent,
)
from turbo_agent.model_execution.pi_adapter import PiCompanionExecutor

FIXTURES = Path(__file__).parent / "fixtures"
COMPANION = (
    Path(__file__).parent.parent / "turbo_agent" / "model_execution" / "pi_companion.mjs"
)


def make_executor(script=None, **kwargs) -> PiCompanionExecutor:
    return PiCompanionExecutor(
        {
            ModelTarget("pi-a"): TargetSpec(
                name="openai-codex/gpt-5.6-sol", api_key=None
            ),
        },
        companion_path=COMPANION,
        setup_module=FIXTURES / "fake_pi_runtime.mjs",
        extra_env={"TURBO_PI_FAKE_SCRIPT": json.dumps(script or {})},
        **kwargs,
    )


async def read_until(proc, match_id, timeout=10.0):
    """Read stdout lines until one carries the requested id; returns it."""
    while True:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
        if not line:
            raise AssertionError("companion closed stdout before matching id")
        msg = json.loads(line)
        if msg.get("id") == match_id:
            return msg


def send(proc, payload):
    proc.stdin.write((json.dumps(payload) + "\n").encode())


@pytest.mark.asyncio
async def test_complete_round_trip():
    ex = make_executor()
    try:
        result = await ex.complete(
            ModelTarget("pi-a"),
            ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
        )
        assert result.output.text == "ok"
        assert result.output.finish_reason == "stop"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.response_id is None or isinstance(result.response_id, str)
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_stream_events_and_final_result():
    ex = make_executor()
    try:
        events = []
        async for ev in ex.stream(
            ModelTarget("pi-a"),
            ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
        ):
            events.append(ev)
        deltas = [e.text for e in events if isinstance(e, TextDelta)]
        assert "".join(deltas) == "hello"
        completed = [e for e in events if isinstance(e, ExecutionCompleted)]
        assert len(completed) == 1
        assert events[-1] is completed[0]
        assert completed[0].result.output.text == "hello"
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_multiplexes_concurrent_calls():
    ex = make_executor({"delayMs": 100})
    try:
        results = await asyncio.gather(
            *[
                ex.complete(
                    ModelTarget("pi-a"),
                    ModelExecutionRequest(
                        messages=({"role": "user", "content": f"m{i}"},),
                    ),
                )
                for i in range(3)
            ]
        )
        assert all(r.output.text == "ok" for r in results)
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_cancel_propagates_to_companion():
    ex = make_executor({"delayMs": 5000})
    try:
        task = asyncio.create_task(
            ex.complete(
                ModelTarget("pi-a"),
                ModelExecutionRequest(messages=({"role": "user", "content": "x"},)),
            )
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_unknown_target_fails_fast():
    ex = make_executor()
    try:
        with pytest.raises(KeyError):
            await ex.complete(
                ModelTarget("nope"),
                ModelExecutionRequest(messages=({"role": "user", "content": "x"},)),
            )
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_crash_fails_inflight_as_unavailable():
    ex = make_executor({"delayMs": 3000})
    try:
        task = asyncio.create_task(
            ex.complete(
                ModelTarget("pi-a"),
                ModelExecutionRequest(messages=({"role": "user", "content": "x"},)),
            )
        )
        await asyncio.sleep(0.4)
        # Kill the companion process abruptly.
        ex._proc.send_signal(signal.SIGKILL)
        with pytest.raises(ModelExecutionError) as exc_info:
            await task
        assert exc_info.value.kind == "unavailable"
        assert exc_info.value.retryable is True
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_never_returns_credentials():
    """The fake runtime plants a secret; it must never cross into results."""
    ex = make_executor()
    try:
        result = await ex.complete(
            ModelTarget("pi-a"),
            ModelExecutionRequest(messages=({"role": "user", "content": "hi"},)),
        )
        blob = json.dumps(result.__dict__, default=str)
        assert "sk-fake-oauth-token" not in blob
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_protocol_version_mismatch_rejected():
    """A v:99 request gets a version error, not silent handling."""
    proc = await asyncio.create_subprocess_exec(
        "node", str(COMPANION),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env={
            **os.environ,
            "TURBO_PI_RUNTIME_SETUP": str(FIXTURES / "fake_pi_runtime.mjs"),
        },
    )
    try:
        send(proc, {"v": 99, "id": "bad", "op": "complete"})
        msg = await read_until(proc, "bad")
        assert msg["event"] == "error"
        assert msg["error"]["kind"] == "invalid_request"
    finally:
        proc.kill()
        await proc.wait()


@pytest.mark.asyncio
async def test_context_mapping_reaches_companion():
    """OpenAI-shaped messages/tools arrive as a Pi Context on the wire."""
    ex = make_executor({"echoContext": True})
    try:
        result = await ex.complete(
            ModelTarget("pi-a"),
            ModelExecutionRequest(
                messages=(
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                ),
                tools=(
                    {
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "description": "run shell",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ),
            ),
        )
        seen = json.loads(result.output.text)
        assert seen["systemPrompt"] == "be brief"
        assert seen["firstRole"] == "user"
        assert seen["toolNames"] == ["Bash"]
    finally:
        await ex.aclose()


@pytest.mark.asyncio
async def test_thinking_intent_maps_to_reasoning_option():
    ex = make_executor()
    try:
        req = ModelExecutionRequest(
            messages=({"role": "user", "content": "hi"},),
            generation=__import__(
                "turbo_agent.model_execution.types", fromlist=["GenerationIntent"]
            ).GenerationIntent(thinking=ThinkingIntent(mode="adaptive", effort="high")),
        )
        await ex.complete(ModelTarget("pi-a"), req)
        sent = ex.last_sent_request()
        assert sent["options"]["reasoning"] == "high"
    finally:
        await ex.aclose()


def _make_config(tmp_path, yaml_text):
    from turbo_agent.utils import Config
    p = tmp_path / "turbo-agent.yaml"
    p.write_text(yaml_text)
    return Config(str(p))


def test_factory_routes_pi_and_litellm_targets(tmp_path):
    """executor: pi entries get a PiCompanionExecutor; others LiteLLM;
    mixed configs get one routing executor."""
    from turbo_agent.model_execution import build_candidate_execution
    from turbo_agent.model_execution.factory import RoutingExecutor

    config = _make_config(tmp_path, """
backend:
  models:
    - name: openai-codex/gpt-5.6-sol
      executor: pi
      num_candidates: 2
    - name: openai/dummy
      api_key: x
      base_url: http://127.0.0.1:1/v1
""")
    targets = build_candidate_execution(config)
    assert isinstance(targets.executor, RoutingExecutor)
    pi_target = targets.candidate_targets[0]
    litellm_target = targets.candidate_targets[2]
    assert isinstance(targets.executor._routes[pi_target], PiCompanionExecutor)
    assert isinstance(targets.executor._routes[litellm_target], LiteLLMExecutor)

    solo = _make_config(tmp_path, """
backend:
  models:
    - name: openai-codex/gpt-5.6-sol
      executor: pi
""")
    solo_targets = build_candidate_execution(solo)
    assert isinstance(solo_targets.executor, PiCompanionExecutor)
    assert solo_targets.default_target.name == "openai-codex/gpt-5.6-sol"


@pytest.mark.asyncio
async def test_mixed_execution_isolates_failures(tmp_path):
    """One LiteLLM candidate failing (dead endpoint) does not discard a
    successful Pi candidate — Backend's concurrent gather semantics over
    the routing executor."""
    from turbo_agent.model_execution import build_candidate_execution
    from turbo_agent.model_execution.types import (
        AssistantOutput,
        ModelExecutionResult,
    )

    class StubPiExecutor:
        """Stands in for the companion; success is covered by contract tests."""

        def resolve(self, target):
            return TargetSpec(name="openai-codex/gpt-5.6-sol")

        async def complete(self, target, request):
            return ModelExecutionResult(
                target=target,
                output=AssistantOutput(
                    text="ok", tool_calls=(), thinking=None, finish_reason="stop"
                ),
                usage=None,
            )

        def stream(self, target, request):
            raise NotImplementedError

        async def aclose(self):
            return None

    config = _make_config(tmp_path, """
backend:
  models:
    - name: openai-codex/gpt-5.6-sol
      executor: pi
    - name: openai/dummy
      api_key: x
      base_url: http://127.0.0.1:9/v1
""")
    targets = build_candidate_execution(config, pi_executor=StubPiExecutor())
    try:
        request = ModelExecutionRequest(
            messages=({"role": "user", "content": "hi"},)
        )
        results = await asyncio.gather(
            targets.executor.complete(targets.candidate_targets[0], request),
            targets.executor.complete(targets.candidate_targets[1], request),
            return_exceptions=True,
        )
        assert not isinstance(results[0], BaseException), results[0]
        assert isinstance(results[1], ModelExecutionError)
        assert results[1].kind in ("unavailable", "timeout", "internal")
        assert results[0].output.text == "ok"
    finally:
        await targets.executor.aclose()
