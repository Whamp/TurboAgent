"""Model execution admission wraps provider lifetimes without changing results."""

import asyncio
import json
import time

import pytest

from turbo_agent.endpoint_admission import (
    EndpointAdmissionPolicy,
    EndpointAdmissionRegistry,
    EndpointRequestTimeoutError,
)
from turbo_agent.model_execution import (
    AssistantOutput,
    EndpointAdmissionExecutor,
    ModelExecutionRequest,
    ModelExecutionResult,
    ModelTarget,
    TextDelta,
)
from turbo_agent.proxy import Backend
from turbo_agent.proxy.proxy import ProxyServer
from turbo_agent.utils import Config


class RecordingExecutor:
    def __init__(self, registry: EndpointAdmissionRegistry) -> None:
        self.registry = registry
        self.fail = True
        self.active_during_call = []
        self.stream_started = asyncio.Event()
        self.stream_cancelled = False
        self.stream_closed = False

    async def complete(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> ModelExecutionResult:
        self.active_during_call.append(self.registry.snapshot("server60").active)
        if self.fail:
            raise LookupError("provider failed")
        return ModelExecutionResult(
            target=target,
            output=AssistantOutput("ok", (), None, "stop"),
            usage=None,
        )

    async def stream(self, target, request):
        self.active_during_call.append(self.registry.snapshot("server60").active)
        self.stream_started.set()
        try:
            yield TextDelta("started")
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.stream_cancelled = True
            raise
        finally:
            self.stream_closed = True

    async def aclose(self) -> None:
        return None


def admission_registry(
    request_timeout_seconds: float = 900.0,
) -> EndpointAdmissionRegistry:
    return EndpointAdmissionRegistry(
        {
            "server60": EndpointAdmissionPolicy(
                max_concurrency=1,
                max_queue_size=2,
                queue_timeout_seconds=1.0,
                request_timeout_seconds=request_timeout_seconds,
            )
        }
    )


async def test_candidate_provider_error_releases_endpoint_slot():
    registry = admission_registry()
    inner = RecordingExecutor(registry)
    target = ModelTarget("candidate", "openai/local")
    executor = EndpointAdmissionExecutor(
        inner,
        registry,
        {target: "server60"},
    )
    request = ModelExecutionRequest(messages=({"role": "user", "content": "hello"},))

    with pytest.raises(LookupError, match="provider failed"):
        await executor.complete(target, request)

    assert registry.snapshot("server60").active == 0

    inner.fail = False
    result = await executor.complete(target, request)

    assert result.output.text == "ok"
    assert inner.active_during_call == [1, 1]
    assert registry.snapshot("server60").active == 0


async def test_cancelled_candidate_stream_releases_endpoint_slot():
    registry = admission_registry()
    inner = RecordingExecutor(registry)
    target = ModelTarget("candidate", "openai/local")
    executor = EndpointAdmissionExecutor(
        inner,
        registry,
        {target: "server60"},
    )
    request = ModelExecutionRequest(messages=({"role": "user", "content": "hello"},))

    async def consume_stream() -> None:
        async for _event in executor.stream(target, request):
            pass

    task = asyncio.create_task(consume_stream())
    await asyncio.wait_for(inner.stream_started.wait(), timeout=1.0)

    assert registry.snapshot("server60").active == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert inner.stream_cancelled is True
    assert registry.snapshot("server60").active == 0


async def test_closed_candidate_stream_releases_endpoint_slot():
    registry = admission_registry()
    inner = RecordingExecutor(registry)
    target = ModelTarget("candidate", "openai/local")
    executor = EndpointAdmissionExecutor(
        inner,
        registry,
        {target: "server60"},
    )
    request = ModelExecutionRequest(messages=({"role": "user", "content": "hello"},))
    stream = executor.stream(target, request)

    event = await anext(stream)

    assert event == TextDelta("started")
    assert registry.snapshot("server60").active == 1

    await stream.aclose()

    assert registry.snapshot("server60").active == 0


async def test_backend_stream_close_reaches_provider_and_releases_slot(tmp_path):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
endpoints:
  server60:
    base_url: http://server60/v1
    max_concurrency: 1
backend:
  models:
    - name: openai/local
      endpoint: server60
"""
    )
    config = Config(str(path))
    registry = EndpointAdmissionRegistry(
        {name: endpoint.policy for name, endpoint in config.endpoint_configs.items()}
    )
    inner = RecordingExecutor(registry)
    backend = Backend(config, executor=inner)
    stream = backend.stream_openai(
        json.dumps(
            {
                "model": "client-model",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
    )

    first_event = await anext(stream)
    assert "started" in first_event
    assert backend.endpoint_admission_snapshot("server60").active == 1

    await stream.aclose()

    assert inner.stream_closed is True
    assert backend.endpoint_admission_snapshot("server60").active == 0


@pytest.mark.parametrize(
    "proxy_method",
    ["_openai_streaming", "_anthropic_streaming"],
)
async def test_proxy_disconnect_closes_backend_stream(proxy_method):
    class TrackingBackend:
        def __init__(self) -> None:
            self.closed = False

        async def stream_openai(self, _body):
            try:
                yield "data: openai\n\n"
                await asyncio.sleep(60)
            finally:
                self.closed = True

        async def stream_anthropic(self, _body):
            try:
                yield "data: anthropic\n\n"
                await asyncio.sleep(60)
            finally:
                self.closed = True

    tracking_backend = TrackingBackend()
    proxy = ProxyServer.__new__(ProxyServer)
    proxy._backend = tracking_backend
    response = await getattr(proxy, proxy_method)(b"{}", time.monotonic())
    stream = response.body_iterator

    await anext(stream)
    await stream.aclose()

    assert tracking_backend.closed is True


async def test_candidate_deadline_cancels_provider_and_releases_slot():
    registry = admission_registry(request_timeout_seconds=0.01)
    provider_cancelled = False

    class HungExecutor(RecordingExecutor):
        async def complete(self, target, request):
            nonlocal provider_cancelled
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                provider_cancelled = True
                raise

    inner = HungExecutor(registry)
    target = ModelTarget("candidate", "openai/local")
    executor = EndpointAdmissionExecutor(
        inner,
        registry,
        {target: "server60"},
    )
    request = ModelExecutionRequest(messages=({"role": "user", "content": "hello"},))

    with pytest.raises(EndpointRequestTimeoutError, match="server60"):
        await executor.complete(target, request)

    assert provider_cancelled is True
    assert registry.snapshot("server60").active == 0


async def test_candidate_deadline_waits_for_provider_cleanup_before_release():
    registry = admission_registry(request_timeout_seconds=0.01)
    cleanup_finished = False

    class SlowCleanupExecutor(RecordingExecutor):
        async def complete(self, target, request):
            nonlocal cleanup_finished
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await asyncio.sleep(0.03)
                cleanup_finished = True
                raise

    inner = SlowCleanupExecutor(registry)
    target = ModelTarget("candidate", "openai/local")
    executor = EndpointAdmissionExecutor(
        inner,
        registry,
        {target: "server60"},
    )
    request = ModelExecutionRequest(messages=({"role": "user", "content": "hello"},))
    started = time.monotonic()

    with pytest.raises(EndpointRequestTimeoutError, match="server60"):
        await executor.complete(target, request)

    assert time.monotonic() - started >= 0.03
    assert cleanup_finished is True
    assert registry.snapshot("server60").active == 0


async def test_candidate_stream_deadline_cancels_provider_and_releases_slot():
    registry = admission_registry(request_timeout_seconds=0.01)
    inner = RecordingExecutor(registry)
    target = ModelTarget("candidate", "openai/local")
    executor = EndpointAdmissionExecutor(
        inner,
        registry,
        {target: "server60"},
    )
    request = ModelExecutionRequest(messages=({"role": "user", "content": "hello"},))

    with pytest.raises(EndpointRequestTimeoutError, match="server60"):
        async for _event in executor.stream(target, request):
            pass

    assert inner.stream_cancelled is True
    assert registry.snapshot("server60").active == 0
