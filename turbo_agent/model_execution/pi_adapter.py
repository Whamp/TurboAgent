"""Pi ModelRuntime adapter.

Executes candidates through a Node companion process that owns Pi's
ModelRuntime and its subscription credentials. Python speaks versioned
newline-delimited JSON over stdio; credentials never cross into this
process. See docs/design/model-execution.md ("Pi ModelRuntime adapter").
"""

import asyncio
import contextlib
import json
import os
from pathlib import Path

from .errors import FailureKind, ModelExecutionError
from .litellm_adapter import _effective_max_tokens, _effective_thinking
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
    TokenUsage,
    ToolCall,
    ToolCallArgumentsDelta,
    ToolCallStarted,
)

PROTOCOL_VERSION = 1

_DEFAULT_COMPANION = Path(__file__).parent / "pi_companion.mjs"

_ERROR_KIND_MAP: dict[str, tuple[FailureKind, bool]] = {
    "authentication": ("authentication", False),
    "rate_limited": ("rate_limited", True),
    "timeout": ("timeout", True),
    "unavailable": ("unavailable", True),
    "invalid_request": ("invalid_request", False),
    "internal": ("internal", False),
}


def _split_target_name(name: str) -> tuple[str, str]:
    provider, _, model_id = name.partition("/")
    if not model_id:
        raise ValueError(
            f"Pi target name {name!r} must be '<pi-provider>/<pi-model-id>'"
        )
    return provider, model_id


def _companion_dead() -> ModelExecutionError:
    return ModelExecutionError(
        target=ModelTarget("companion"),
        kind="unavailable",
        retryable=True,
        message="Pi companion process exited with calls in flight",
    )


_TEXT_DELTAS = {"text": TextDelta, "thinking": ThinkingDelta}


class PiCompanionExecutor(ModelExecutor):
    """Runs one Node companion and multiplexes candidate calls over stdio.

    The companion starts lazily on the first call and restarts after a
    crash; in-flight calls at crash time fail as retryable `unavailable`
    and are never silently replayed.
    """

    def __init__(
        self,
        targets: object,
        companion_path: Path | None = None,
        setup_module: Path | None = None,
        extra_env: dict[str, str] | None = None,
        node_bin: str = "node",
    ) -> None:
        self._targets: dict[ModelTarget, TargetSpec] = dict(targets)
        self._companion_path = Path(companion_path or _DEFAULT_COMPANION)
        self._node_bin = node_bin
        self._extra_env = {k: v for k, v in (extra_env or {}).items() if v is not None}
        if setup_module is not None:
            self._extra_env["TURBO_PI_RUNTIME_SETUP"] = str(setup_module)
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        # req_id -> {"future": Future for complete | None, "queue": stream Queue | None}
        self._pending: dict[str, dict] = {}
        self._next_id = 0
        self._last_sent: dict | None = None
        self._spawn_lock = asyncio.Lock()

    def resolve(self, target: ModelTarget) -> TargetSpec:
        try:
            return self._targets[target]
        except KeyError:
            raise KeyError(f"Unknown model target: {target.id}") from None

    def last_sent_request(self) -> dict | None:
        """Diagnostics hook: the most recent wire request body."""
        return self._last_sent

    async def aclose(self) -> None:
        proc, reader = self._proc, self._reader_task
        self._proc, self._reader_task = None, None
        if reader:
            reader.cancel()
        if proc and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            await proc.wait()
        self._fail_pending(_companion_dead())

    def _fail_pending(self, exc: ModelExecutionError) -> None:
        pending, self._pending = self._pending, {}
        for entry in pending.values():
            if entry["future"] and not entry["future"].done():
                entry["future"].set_exception(exc)
            if entry["queue"] is not None:
                entry["queue"].put_nowait(exc)

    async def _ensure_proc(self) -> asyncio.subprocess.Process:
        # Concurrent first calls must not spawn multiple companions.
        async with self._spawn_lock:
            if self._proc is not None and self._proc.returncode is None:
                return self._proc
            env = dict(os.environ)
            env.update(self._extra_env)
            self._proc = await asyncio.create_subprocess_exec(
                self._node_bin,
                str(self._companion_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=None,  # companion diagnostics flow to our stderr
                env=env,
            )
            self._reader_task = asyncio.create_task(self._read_loop())
            return self._proc

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                self._dispatch(msg)
        finally:
            # Companion died: fail everything in flight as unavailable.
            self._proc = None
            self._fail_pending(_companion_dead())

    def _dispatch(self, msg: dict) -> None:
        req_id = msg.get("id")
        entry = self._pending.get(req_id)
        if entry is None:
            return  # late events for cancelled/unknown requests are dropped
        event = msg.get("event")
        if event == "error":
            error = msg.get("error") or {}
            kind_name = error.get("kind", "internal")
            if kind_name == "cancelled":
                exc: Exception = asyncio.CancelledError()
            else:
                kind, retryable = _ERROR_KIND_MAP.get(kind_name, ("internal", False))
                exc = ModelExecutionError(
                    target=self._error_target,
                    kind=kind,
                    retryable=retryable,
                    message=str(error.get("message", "companion error")),
                )
            if entry["future"] and not entry["future"].done():
                entry["future"].set_exception(exc)
            if entry["queue"] is not None:
                entry["queue"].put_nowait(exc)
            self._pending.pop(req_id, None)
        elif event == "done":
            if entry["future"] and not entry["future"].done():
                entry["future"].set_result(msg.get("result"))
            if entry["queue"] is not None:
                entry["queue"].put_nowait({"done": msg.get("result")})
            self._pending.pop(req_id, None)
        elif event == "delta" and entry["queue"] is not None:
            entry["queue"].put_nowait({"delta": msg.get("delta")})

    async def _request(
        self,
        target: ModelTarget,
        op: str,
        request: ModelExecutionRequest,
        options: dict,
    ) -> tuple[asyncio.Future, asyncio.Queue | None]:
        spec = self.resolve(target)
        provider, model_id = _split_target_name(spec.name)
        proc = await self._ensure_proc()
        self._next_id += 1
        req_id = f"req-{self._next_id}"
        payload = {
            "v": PROTOCOL_VERSION,
            "id": req_id,
            "op": op,
            "provider": provider,
            "model": model_id,
            "request": {
                "messages": [dict(m) for m in request.messages],
                **(
                    {"tools": [dict(t) for t in request.tools]} if request.tools else {}
                ),
            },
            "options": options,
        }
        self._last_sent = payload
        queue: asyncio.Queue | None = None
        future: asyncio.Future | None = None
        if op == "stream":
            queue = asyncio.Queue()
        else:
            future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = {"future": future, "queue": queue}
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(payload) + "\n").encode())
        await proc.stdin.drain()
        return future, queue

    @staticmethod
    def _build_options(spec: TargetSpec, request: ModelExecutionRequest) -> dict:
        generation: GenerationIntent = request.generation
        options: dict = {}
        max_tokens = _effective_max_tokens(spec, generation)
        if max_tokens is not None:
            options["maxTokens"] = max_tokens
        temperature = (
            generation.temperature
            if generation.temperature is not None
            else spec.temperature
        )
        if temperature is not None:
            options["temperature"] = temperature
        sampling: dict = {}
        if generation.top_p is not None:
            sampling["top_p"] = generation.top_p
        if generation.stop is not None:
            stop = (
                list(generation.stop)
                if isinstance(generation.stop, tuple)
                else generation.stop
            )
            sampling["stop"] = stop
        if sampling:
            options["samplingParams"] = sampling

        thinking = _effective_thinking(spec, generation)
        if thinking is not None and thinking.mode == "adaptive":
            options["reasoning"] = thinking.effort or "high"
        return options

    async def complete(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> ModelExecutionResult:
        spec = self.resolve(target)
        self._error_target = target
        future, _ = await self._request(
            target,
            "complete",
            request,
            self._build_options(spec, request),
        )
        req_id = next(
            (rid for rid, e in self._pending.items() if e["future"] is future), None
        )
        try:
            result = await future
        except asyncio.CancelledError:
            await self._cancel(req_id)
            raise
        return self._to_result(target, result or {})

    async def _cancel(self, req_id: str | None) -> None:
        if req_id is None:
            return
        self._pending.pop(req_id, None)
        proc = self._proc
        if proc is None or proc.returncode is not None or proc.stdin is None:
            return
        proc.stdin.write(
            (
                json.dumps(
                    {
                        "v": PROTOCOL_VERSION,
                        "id": req_id,
                        "op": "cancel",
                    }
                )
                + "\n"
            ).encode()
        )
        await proc.stdin.drain()

    async def stream(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ):
        spec = self.resolve(target)
        self._error_target = target
        _, queue = await self._request(
            target,
            "stream",
            request,
            self._build_options(spec, request),
        )
        assert queue is not None
        req_id = next(
            (rid for rid, e in self._pending.items() if e["queue"] is queue), None
        )
        state = {"current_tool_id": ""}
        try:
            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                if "delta" in item:
                    for event in self._delta_events(item["delta"], state):
                        yield event
                elif "done" in item:
                    yield ExecutionCompleted(
                        result=self._to_result(target, item["done"] or {})
                    )
                    return
        except asyncio.CancelledError:
            await self._cancel(req_id)
            raise
        finally:
            if req_id is not None:
                self._pending.pop(req_id, None)

    @classmethod
    def _delta_events(cls, delta: dict, state: dict):
        """Translate one companion delta into zero or more canonical events.

        `state["current_tool_id"]` carries the active tool call across
        argument-fragment deltas, which the companion sends without an id.
        """
        dtype = delta.get("type")
        event_cls = _TEXT_DELTAS.get(dtype)
        if event_cls is not None:
            yield event_cls(text=delta.get("text", ""))
        elif dtype == "tool_call_started":
            state["current_tool_id"] = delta.get("id", "")
            yield ToolCallStarted(
                id=state["current_tool_id"],
                name=delta.get("name", ""),
            )
        elif dtype == "tool_call_arguments":
            yield ToolCallArgumentsDelta(
                id=state["current_tool_id"],
                json_fragment=delta.get("json_fragment", ""),
            )

    @staticmethod
    def _to_result(target: ModelTarget, result: dict) -> ModelExecutionResult:
        output = result.get("output") or {}
        usage = result.get("usage")
        tool_calls = tuple(
            ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                arguments=tc.get("arguments") or {},
            )
            for tc in output.get("tool_calls") or []
        )
        token_usage = None
        if usage:
            token_usage = TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens"),
            )
        return ModelExecutionResult(
            target=target,
            output=AssistantOutput(
                text=output.get("text", ""),
                tool_calls=tool_calls,
                thinking=output.get("thinking"),
                finish_reason=output.get("finish_reason", "other"),
            ),
            usage=token_usage,
            response_id=result.get("response_id"),
        )
