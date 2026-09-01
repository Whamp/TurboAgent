"""Configuration contracts for shared endpoint admission and judge execution."""

import asyncio

import pytest

from turbo_agent.endpoint_admission import EndpointAdmissionRegistry
from turbo_agent.model_execution import (
    AssistantOutput,
    ModelExecutionRequest,
    ModelExecutionResult,
    build_candidate_execution,
)
from turbo_agent.utils import Config


def test_unconfigured_paths_preserve_existing_execution(tmp_path):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
backend:
  models:
    - name: openai/default
      base_url: http://other/v1
verifier:
  model:
    name: openai/judge
    base_url: http://judge/v1
"""
    )
    config = Config(str(path))
    inner_executor = object()

    execution = build_candidate_execution(config, executor=inner_executor)
    verifier = config.verifier_config

    assert execution.executor is inner_executor
    assert config.endpoint_configs == {}
    assert verifier is not None
    assert verifier.execution.adapter == "default"
    assert verifier.execution.max_concurrency is None


def test_server60_endpoint_and_judge_execution_parse(tmp_path):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
endpoints:
  server60:
    base_url: http://192.168.0.251:30002/v1
    max_concurrency: 4
    max_queue_size: 32
    queue_timeout_seconds: 120
    request_timeout_seconds: 600

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
  method: {name: pivot_tournament, pivots: 2, n_verifications: 1}
"""
    )

    config = Config(str(path))
    endpoint = config.endpoint_configs["server60"]
    verifier = config.verifier_config
    assert verifier is not None

    assert endpoint.base_url == "http://192.168.0.251:30002/v1"
    assert endpoint.policy.max_concurrency == 4
    assert endpoint.policy.max_queue_size == 32
    assert endpoint.policy.queue_timeout_seconds == 120
    assert endpoint.policy.request_timeout_seconds == 600
    assert config.model_base_url(config.models[0]) == endpoint.base_url
    assert verifier.model.base_url == endpoint.base_url
    assert verifier.model.endpoint == "server60"
    assert verifier.execution.adapter == "server60"
    assert verifier.execution.comparison_max_output_tokens == 16384
    assert verifier.execution.max_concurrency == 4

    execution = build_candidate_execution(config)
    inner_executor = execution.executor._executor
    target_spec = inner_executor.resolve(execution.default_target)
    assert target_spec.max_retries == 0
    assert target_spec.request_timeout_seconds == 600


async def test_configured_endpoint_limits_candidate_execution_to_four(tmp_path):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
endpoints:
  server60:
    base_url: http://192.168.0.251:30002/v1
    max_concurrency: 4
    max_queue_size: 8
    queue_timeout_seconds: 1
backend:
  models:
    - name: openai/qwen-local
      endpoint: server60
      num_candidates: 5
"""
    )
    config = Config(str(path))
    registry = EndpointAdmissionRegistry(
        {name: endpoint.policy for name, endpoint in config.endpoint_configs.items()}
    )

    class CountingExecutor:
        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0

        async def complete(self, target, request):
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return ModelExecutionResult(
                target=target,
                output=AssistantOutput("ok", (), None, "stop"),
                usage=None,
            )

        async def stream(self, target, request):
            if False:
                yield None

        async def aclose(self) -> None:
            return None

    inner = CountingExecutor()
    execution = build_candidate_execution(
        config,
        executor=inner,
        admission_registry=registry,
    )

    request = ModelExecutionRequest(messages=({"role": "user", "content": "hello"},))
    await asyncio.gather(
        *(
            execution.executor.complete(target, request)
            for target in execution.candidate_targets
        )
    )

    assert inner.maximum_active == 4
    assert registry.snapshot("server60").active == 0


@pytest.mark.parametrize(
    ("model_block", "message"),
    [
        ("endpoint: missing", "Unknown model endpoint"),
        (
            "endpoint: server60\n      base_url: http://other/v1",
            "conflicts with its base_url",
        ),
    ],
)
def test_invalid_candidate_endpoint_reference_is_rejected(
    tmp_path,
    model_block,
    message,
):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        f"""
endpoints:
  server60:
    base_url: http://server60/v1
    max_concurrency: 4
backend:
  models:
    - name: openai/qwen-local
      {model_block}
"""
    )

    with pytest.raises(ValueError, match=message):
        build_candidate_execution(Config(str(path)))


def test_server60_judge_rejects_candidate_capacity_bypass(tmp_path):
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
      base_url: http://server60/alternate-api-path
      num_candidates: 2
verifier:
  model:
    name: openai/qwen-local
    endpoint: server60
  execution:
    adapter: server60
"""
    )

    with pytest.raises(
        ValueError,
        match="Candidates sharing the server60 judge origin",
    ):
        _ = Config(str(path)).verifier_config


@pytest.mark.parametrize(
    ("model_name", "env_name"),
    [
        ("openai/qwen-local", "OPENAI_BASE_URL"),
        ("hosted_vllm/qwen-local", "HOSTED_VLLM_API_BASE"),
    ],
)
def test_server60_judge_rejects_environment_capacity_bypass(
    tmp_path,
    monkeypatch,
    model_name,
    env_name,
):
    monkeypatch.setenv(env_name, "http://server60/v1/")
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        f"""
endpoints:
  server60:
    base_url: http://server60/v1
    max_concurrency: 4
backend:
  models:
    - name: {model_name}
verifier:
  model:
    name: openai/qwen-local
    endpoint: server60
  execution:
    adapter: server60
"""
    )

    with pytest.raises(
        ValueError,
        match="Candidates sharing the server60 judge origin",
    ):
        _ = Config(str(path)).verifier_config


def test_named_endpoint_requires_absolute_http_url(tmp_path):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
endpoints:
  server60:
    base_url: server60:30002/v1
backend:
  models:
    - name: openai/qwen-local
      endpoint: server60
"""
    )

    with pytest.raises(ValueError, match="absolute HTTP\\(S\\) base_url"):
        _ = Config(str(path)).endpoint_configs


def test_default_judge_adapter_cannot_ignore_named_endpoint(tmp_path):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
endpoints:
  server60:
    base_url: http://server60/v1
    max_concurrency: 4
backend:
  models:
    - name: openai/remote
      base_url: http://remote/v1
verifier:
  model:
    name: openai/qwen-local
    endpoint: server60
"""
    )

    with pytest.raises(ValueError, match="requires execution.adapter 'server60'"):
        _ = Config(str(path)).verifier_config


def test_default_judge_adapter_cannot_bypass_named_endpoint_by_url(tmp_path):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
endpoints:
  server60:
    base_url: http://server60/v1
    max_concurrency: 4
backend:
  models:
    - name: openai/remote
      base_url: http://remote/v1
verifier:
  model:
    name: openai/qwen-local
    base_url: http://server60/other-path
"""
    )

    with pytest.raises(ValueError, match="base_url matching a named endpoint"):
        _ = Config(str(path)).verifier_config


def test_server60_judge_adapter_requires_named_endpoint(tmp_path):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
backend:
  models:
    - name: openai/qwen-local
      num_candidates: 2
verifier:
  model:
    name: openai/qwen-local
    base_url: http://server60/v1
  execution:
    adapter: server60
"""
    )

    with pytest.raises(ValueError, match="requires model.endpoint"):
        _ = Config(str(path)).verifier_config


def test_server60_endpoint_capacity_cannot_exceed_four(tmp_path):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        """
endpoints:
  server60:
    base_url: http://server60/v1
    max_concurrency: 8
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
    max_concurrency: 4
"""
    )

    with pytest.raises(ValueError, match="endpoint max_concurrency cannot exceed 4"):
        _ = Config(str(path)).verifier_config


@pytest.mark.parametrize(
    ("endpoint_capacity", "comparison_cap", "workers", "message"),
    [
        (4, 1, 4, "comparison_max_output_tokens must be at least 2"),
        (2, 16384, 3, "max_concurrency cannot exceed endpoint capacity"),
    ],
)
def test_server60_judge_execution_bounds_are_validated(
    tmp_path,
    endpoint_capacity,
    comparison_cap,
    workers,
    message,
):
    path = tmp_path / "turbo-agent.yaml"
    path.write_text(
        f"""
endpoints:
  server60:
    base_url: http://server60/v1
    max_concurrency: {endpoint_capacity}
backend:
  models:
    - name: openai/qwen-local
      endpoint: server60
verifier:
  model:
    name: openai/qwen-local
    endpoint: server60
  execution:
    adapter: server60
    comparison_max_output_tokens: {comparison_cap}
    max_concurrency: {workers}
"""
    )

    with pytest.raises(ValueError, match=message):
        _ = Config(str(path)).verifier_config


def test_server60_judge_workers_cannot_exceed_four(tmp_path):
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
    max_concurrency: 5
"""
    )

    with pytest.raises(ValueError, match="between 1 and 4"):
        _ = Config(str(path)).verifier_config
