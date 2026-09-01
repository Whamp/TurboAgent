"""server60 judge request translation for llm-verifier calls."""

import asyncio
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from llm_verifier.fine_grained_reward import call_openai, score_pair_criterion

from turbo_agent.endpoint_admission import (
    EndpointAdmissionPolicy,
    EndpointAdmissionRegistry,
    EndpointQueueTimeoutError,
)
from turbo_agent.model_execution import (
    AssistantOutput,
    ModelExecutionRequest,
    ModelExecutionResult,
)
from turbo_agent.proxy import Backend
from turbo_agent.utils import Config
from turbo_agent.verifier import Server60JudgeClient, Verifier


class FakeCompletions:
    def __init__(self, registry: EndpointAdmissionRegistry) -> None:
        self.registry = registry
        self.calls = []
        self.response = object()
        self.error = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert self.registry.snapshot("server60").active == 1
        if self.error is not None:
            raise self.error
        return self.response


class FakeChat:
    def __init__(self, registry: EndpointAdmissionRegistry) -> None:
        self.completions = FakeCompletions(registry)


class FakeJudgeClient:
    def __init__(self, registry: EndpointAdmissionRegistry) -> None:
        self.chat = FakeChat(registry)
        self.base_url = "http://server60/v1/"


def make_adapter():
    registry = EndpointAdmissionRegistry(
        {
            "server60": EndpointAdmissionPolicy(
                max_concurrency=4,
                max_queue_size=8,
                queue_timeout_seconds=1.0,
            )
        }
    )
    client = FakeJudgeClient(registry)
    adapter = Server60JudgeClient(
        client,
        registry,
        endpoint_name="server60",
        comparison_max_output_tokens=16384,
    )
    return adapter, client, registry


def test_comparison_generation_uses_xhigh_thinking_and_configured_cap():
    adapter, client, registry = make_adapter()

    response = adapter.chat.completions.create(
        model="qwen-local",
        messages=[{"role": "user", "content": "compare A and B"}],
        max_tokens=4096,
        reasoning_effort="low",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    call = client.chat.completions.calls[0]
    assert response is client.chat.completions.response
    assert call["max_tokens"] == 16384
    assert call["reasoning_effort"] == "xhigh"
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert call["timeout"] == 900.0
    assert registry.snapshot("server60").active == 0


def test_score_probe_disables_thinking_and_preserves_prefill_controls():
    adapter, client, registry = make_adapter()
    structured_outputs = {"choice": ["A", "B"]}

    adapter.chat.completions.create(
        model="qwen-local",
        messages=[{"role": "assistant", "content": "analysis\n<score_A>"}],
        max_tokens=1,
        reasoning_effort="xhigh",
        extra_body={
            "add_generation_prompt": False,
            "continue_final_message": True,
            "structured_outputs": structured_outputs,
        },
    )

    call = client.chat.completions.calls[0]
    assert call["max_tokens"] == 1
    assert "reasoning_effort" not in call
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert call["extra_body"]["add_generation_prompt"] is False
    assert call["extra_body"]["continue_final_message"] is True
    assert call["extra_body"]["structured_outputs"] == structured_outputs
    assert registry.snapshot("server60").active == 0


def test_score_probes_produce_discriminating_llm_verifier_rewards():
    registry = EndpointAdmissionRegistry(
        {
            "server60": EndpointAdmissionPolicy(
                max_concurrency=4,
                max_queue_size=8,
                queue_timeout_seconds=1.0,
            )
        }
    )

    class DiscriminatingCompletions:
        def __init__(self) -> None:
            self.score_probe_count = 0

        def create(self, **kwargs):
            assert registry.snapshot("server60").active == 1
            chat_template = kwargs["extra_body"]["chat_template_kwargs"]
            if kwargs["max_tokens"] != 1:
                assert kwargs["reasoning_effort"] == "xhigh"
                assert chat_template["enable_thinking"] is True
                choice = SimpleNamespace(
                    message=SimpleNamespace(content="analysis", reasoning="reasoning"),
                    logprobs=None,
                )
                return SimpleNamespace(choices=[choice], usage=None)

            assert chat_template["enable_thinking"] is False
            probability_a = 0.8 if self.score_probe_count == 0 else 0.2
            self.score_probe_count += 1
            alternatives = [
                SimpleNamespace(token=" A", logprob=math.log(probability_a)),
                SimpleNamespace(token=" T", logprob=math.log(1 - probability_a)),
            ]
            position = SimpleNamespace(
                token=" A" if probability_a > 0.5 else " T",
                logprob=math.log(max(probability_a, 1 - probability_a)),
                top_logprobs=alternatives,
            )
            choice = SimpleNamespace(
                message=SimpleNamespace(content=position.token, reasoning=None),
                logprobs=SimpleNamespace(content=[position]),
            )
            return SimpleNamespace(choices=[choice], usage=None)

    completions = DiscriminatingCompletions()
    raw_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    adapter = Server60JudgeClient(
        raw_client,
        registry,
        endpoint_name="server60",
        comparison_max_output_tokens=16384,
    )

    reward_a, reward_b = score_pair_criterion(
        adapter,
        problem="Name the capital of France.",
        trace_a="Paris",
        trace_b="Berlin",
        criterion={"name": "Correctness", "description": "Prefer the correct answer."},
        ground_truth_note="Paris is correct.",
        model="qwen-local",
    )

    assert reward_a == pytest.approx(0.8)
    assert reward_b == pytest.approx(0.2)
    assert completions.score_probe_count == 2
    assert registry.snapshot("server60").active == 0


def test_llm_verifier_does_not_retry_server60_provider_failure():
    adapter, client, registry = make_adapter()
    client.chat.completions.error = ConnectionError("server unavailable")

    with pytest.raises(ConnectionError, match="server unavailable"):
        call_openai(
            adapter,
            "compare A and B",
            model="qwen-local",
        )

    assert len(client.chat.completions.calls) == 1
    assert registry.snapshot("server60").active == 0


def test_judge_provider_error_releases_endpoint_slot():
    adapter, client, registry = make_adapter()
    client.chat.completions.error = ConnectionError("server unavailable")

    with pytest.raises(ConnectionError, match="server unavailable"):
        adapter.chat.completions.create(
            model="qwen-local",
            messages=[{"role": "user", "content": "compare"}],
            max_tokens=4096,
        )

    assert registry.snapshot("server60").active == 0


def test_verifier_wraps_server60_client_and_limits_llm_verifier_workers(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
endpoints:
  server60:
    base_url: http://server60/v1
    max_concurrency: 4
    max_queue_size: 8
    queue_timeout_seconds: 1
backend:
  models:
    - name: openai/qwen-local
      endpoint: server60
      num_candidates: 2
verifier:
  model:
    name: openai/qwen-local
    endpoint: server60
  execution:
    adapter: server60
    comparison_max_output_tokens: 16384
    max_concurrency: 4
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1}
"""
    )
    config = Config(str(path))
    registry = EndpointAdmissionRegistry(
        {name: endpoint.policy for name, endpoint in config.endpoint_configs.items()}
    )
    raw_client = FakeJudgeClient(registry)
    monkeypatch.setattr(
        "turbo_agent.verifier.verifier.create_openai_client",
        lambda **_kwargs: raw_client,
    )
    select_calls = []

    def fake_select(*args, **kwargs):
        select_calls.append((args, kwargs))
        return SimpleNamespace(index=0, scores=[1.0, 0.0], n_comparisons=1)

    monkeypatch.setattr(
        "turbo_agent.verifier.verifier.llm_verifier.select", fake_select
    )
    verifier_config = config.verifier_config
    assert verifier_config is not None
    verifier = Verifier(verifier_config, admission_registry=registry)

    _, pair_scores = verifier._run_select("history", ["A", "B"])

    assert isinstance(verifier.client, Server60JudgeClient)
    assert raw_client.max_retries == 0
    assert select_calls[0][1]["max_workers"] == 4
    assert select_calls[0][1]["on_error"] == "raise"
    assert select_calls[0][1]["client"] is verifier.client
    assert pair_scores == {}

    provider_prefixed_config = replace(
        verifier_config,
        model=replace(verifier_config.model, name="gemini/qwen-local"),
    )
    provider_prefixed_verifier = Verifier(
        provider_prefixed_config,
        admission_registry=registry,
    )
    assert isinstance(provider_prefixed_verifier.client, Server60JudgeClient)
    assert provider_prefixed_verifier.client._client is raw_client

    default_execution_config = replace(
        verifier_config,
        execution=replace(
            verifier_config.execution,
            adapter="default",
            max_concurrency=None,
        ),
    )
    default_verifier = Verifier(default_execution_config)
    default_verifier._run_select("history", ["A", "B"])
    assert select_calls[-1][1]["on_error"] == "tie"


async def test_server60_admission_failure_uses_backend_first_candidate_fallback(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
endpoints:
  server60:
    base_url: http://server60/v1
    max_concurrency: 4
backend:
  models:
    - name: openai/qwen-local
      endpoint: server60
      num_candidates: 2
verifier:
  model:
    name: openai/qwen-local
    endpoint: server60
  execution:
    adapter: server60
"""
    )
    backend = Backend(Config(str(path)))
    verifier = backend.verifier
    assert verifier is not None

    async def fail_selection(_history, _actions):
        raise EndpointQueueTimeoutError("Endpoint admission timeout: server60")

    monkeypatch.setattr(verifier, "select_best", fail_selection)
    first = {
        "choices": [{"message": {"content": "first"}, "finish_reason": "stop"}]
    }
    second = {
        "choices": [{"message": {"content": "second"}, "finish_reason": "stop"}]
    }
    request_log = {}

    selected = await backend._pick_best(
        [(first, "model-1"), (second, "model-2")],
        [{"role": "user", "content": "choose"}],
        request_log,
    )

    assert selected == (first, "model-1")
    assert "verifier" not in request_log


async def test_candidates_and_judge_share_four_server60_slots(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
endpoints:
  server60:
    base_url: http://server60/v1
    max_concurrency: 4
    max_queue_size: 8
    queue_timeout_seconds: 1
backend:
  models:
    - name: openai/qwen-local
      endpoint: server60
      num_candidates: 4
verifier:
  model:
    name: openai/qwen-local
    endpoint: server60
  execution:
    adapter: server60
    comparison_max_output_tokens: 16384
    max_concurrency: 4
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1}
"""
    )

    class BlockingCandidateExecutor:
        def __init__(self) -> None:
            self.started = 0
            self.all_started = asyncio.Event()
            self.release = asyncio.Event()

        async def complete(self, target, request):
            self.started += 1
            if self.started == 4:
                self.all_started.set()
            await self.release.wait()
            return ModelExecutionResult(
                target=target,
                output=AssistantOutput("candidate", (), None, "stop"),
                usage=None,
            )

        async def stream(self, target, request):
            if False:
                yield None

        async def aclose(self) -> None:
            return None

    candidate_executor = BlockingCandidateExecutor()
    backend = Backend(Config(str(path)), executor=candidate_executor)
    raw_client = FakeJudgeClient(backend._endpoint_admission)
    monkeypatch.setattr(
        "turbo_agent.verifier.verifier.create_openai_client",
        lambda **_kwargs: raw_client,
    )
    request = ModelExecutionRequest(messages=({"role": "user", "content": "solve"},))
    candidate_tasks = [
        asyncio.create_task(backend._execution.executor.complete(target, request))
        for target in backend._execution.candidate_targets
    ]
    await asyncio.wait_for(candidate_executor.all_started.wait(), timeout=1.0)

    verifier = backend.verifier
    assert verifier is not None
    judge_task = asyncio.create_task(
        asyncio.to_thread(
            verifier.client.chat.completions.create,
            model="qwen-local",
            messages=[{"role": "user", "content": "compare"}],
            max_tokens=4096,
        )
    )
    while backend.endpoint_admission_snapshot("server60").queued < 1:
        await asyncio.sleep(0)

    assert backend.endpoint_admission_snapshot("server60").active == 4
    assert raw_client.chat.completions.calls == []

    candidate_executor.release.set()
    await asyncio.gather(*candidate_tasks, judge_task)

    assert len(raw_client.chat.completions.calls) == 1
    assert backend.endpoint_admission_snapshot("server60").active == 0
    assert backend.endpoint_admission_snapshot("server60").queued == 0


def test_eight_judge_calls_enter_provider_at_most_four_at_a_time():
    registry = EndpointAdmissionRegistry(
        {
            "server60": EndpointAdmissionPolicy(
                max_concurrency=4,
                max_queue_size=8,
                queue_timeout_seconds=1.0,
            )
        }
    )
    release = threading.Event()
    first_wave_started = threading.Event()

    class BlockingCompletions:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.active = 0
            self.maximum_active = 0
            self.entered = 0

        def create(self, **_kwargs):
            with self._lock:
                self.active += 1
                self.entered += 1
                self.maximum_active = max(self.maximum_active, self.active)
                if self.entered == 4:
                    first_wave_started.set()
            release.wait(timeout=1.0)
            with self._lock:
                self.active -= 1
            return object()

    class BlockingChat:
        def __init__(self) -> None:
            self.completions = BlockingCompletions()

    class BlockingClient:
        def __init__(self) -> None:
            self.chat = BlockingChat()

    raw_client = BlockingClient()
    adapter = Server60JudgeClient(
        raw_client,
        registry,
        endpoint_name="server60",
        comparison_max_output_tokens=16384,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                adapter.chat.completions.create,
                model="qwen-local",
                messages=[{"role": "user", "content": "compare"}],
                max_tokens=4096,
            )
            for _ in range(8)
        ]
        assert first_wave_started.wait(timeout=1.0)
        assert registry.snapshot("server60").active == 4
        assert registry.snapshot("server60").queued == 4
        release.set()
        for future in futures:
            future.result(timeout=1.0)

    assert raw_client.chat.completions.maximum_active == 4
    assert registry.snapshot("server60").active == 0
    assert registry.snapshot("server60").queued == 0
