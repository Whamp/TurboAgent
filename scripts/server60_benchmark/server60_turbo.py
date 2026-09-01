"""Turbo candidate generation, majority, and verifier measurements."""

import asyncio
import statistics
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from turbo_agent.proxy.backend import Backend, _result_to_response_dict
from turbo_agent.utils import Config
from turbo_agent.verifier import Verifier

from .server60_workloads import (
    BOUNDED_ANSWER_PROMPT,
    CORPUS,
    deterministic_task_score,
    normalize_text,
)


def config_yaml(
    base_url: str,
    model: str,
    candidates: int,
    embedding_base_url: str,
    embedding_model: str,
    *,
    majority_voting: bool,
    max_tokens: int = 16384,
) -> str:
    """Render one production-model Turbo benchmark configuration."""
    majority = "true" if majority_voting else "false"
    return f"""endpoints:
  server60:
    base_url: {base_url}
    max_concurrency: 4
    max_queue_size: 64
    queue_timeout_seconds: 600
    request_timeout_seconds: 1800
backend:
  models:
    - name: openai/{model}
      api_key: sk-local
      endpoint: server60
      num_candidates: {candidates}
      temperature: 0.7
      thinking: xhigh
      max_tokens: {max_tokens}
verifier:
  model:
    name: openai/{model}
    api_key: sk-local
    endpoint: server60
  execution:
    adapter: server60
    comparison_max_output_tokens: 16384
    max_concurrency: 4
  majority_voting: {majority}
  majority:
    mode: semantic
    threshold: 0.92
    embedding:
      base_url: {embedding_base_url}
      model: {embedding_model}
  method:
    name: pivot_tournament
    pivots: 1
    n_verifications: 1
    seed: 0
    criteria:
      - name: Task Success
        description: How likely the response correctly and completely satisfies the request.
"""


def extract_response_text(response: dict[str, Any]) -> str:
    """Return final prose plus serialized tool calls from one candidate."""
    return Backend.format_action(response)


async def gather_turbo_results(backend: Backend, request):
    """Run Turbo's configured candidate targets while retaining full usage."""
    results = await asyncio.gather(
        *(
            backend._execution.executor.complete(target, request)
            for target in backend._execution.candidate_targets
        )
    )
    return [
        (_result_to_response_dict(result), result.target, result) for result in results
    ]


class JudgeCallCounter:
    """Record synchronous llm-verifier judge calls without changing them."""

    def __init__(self, verifier: Verifier):
        self.calls = 0
        self.completion_tokens = 0
        self.sum_call_s = 0.0
        self._lock = threading.Lock()
        completions = verifier.client.chat.completions
        original = completions.create

        def counted_create(**kwargs):
            started = time.monotonic()
            response = original(**kwargs)
            elapsed = time.monotonic() - started
            usage = getattr(response, "usage", None)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            with self._lock:
                self.sum_call_s += elapsed
                self.calls += 1
                self.completion_tokens += completion_tokens
            return response

        completions.create = counted_create


def _quality_mode_verifiers(backend: Backend) -> dict[str, Verifier]:
    semantic = backend.verifier
    if semantic is None:
        raise RuntimeError("Quality benchmark requires an enabled verifier")
    return {
        "exact": Verifier(
            replace(
                semantic.cfg,
                majority=replace(semantic.cfg.majority, mode="exact"),
            ),
            admission_registry=backend._endpoint_admission,
        ),
        "normalized": Verifier(
            replace(
                semantic.cfg,
                majority=replace(semantic.cfg.majority, mode="normalized"),
            ),
            admission_registry=backend._endpoint_admission,
        ),
        "semantic": semantic,
    }


def _measure_action_agreement(
    actions: list[str], verifiers: dict[str, Verifier]
) -> dict[str, Any]:
    agreement = {}
    for mode, verifier in verifiers.items():
        started = time.monotonic()
        result = verifier._try_majority_voting(actions)
        agreement[mode] = {
            "strict_majority": result is not None,
            "check_ms": round((time.monotonic() - started) * 1000, 2),
            "clusters": len(set(verifier._agreement_keys(actions))),
        }
    return agreement


async def _measure_quality_task(
    backend: Backend,
    verifiers: dict[str, Verifier],
    judge_counter: JudgeCallCounter,
    task: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": "production-benchmark",
        "max_tokens": max_tokens,
        "reasoning_effort": "xhigh",
        "temperature": 0.7,
        "messages": task["messages"],
        "tools": task.get("tools") or [],
    }
    if task.get("tool_choice"):
        body["tool_choice"] = task["tool_choice"]
    request, _ = backend._build_openai_params(body)
    generation_started = time.monotonic()
    responses = await gather_turbo_results(backend, request)
    generation_s = time.monotonic() - generation_started
    actions = [extract_response_text(response) for response, _, _ in responses]
    agreement = _measure_action_agreement(actions, verifiers)

    calls_before = judge_counter.calls
    tokens_before = judge_counter.completion_tokens
    call_s_before = judge_counter.sum_call_s
    selection_started = time.monotonic()
    selection = await verifiers["semantic"].select_best(
        Backend.format_history(task["messages"]), actions
    )
    selection_s = time.monotonic() - selection_started
    scores = [deterministic_task_score(task, response) for response, _, _ in responses]
    return {
        "task": task["name"],
        "candidate_generation_s": round(generation_s, 4),
        "candidate_normalized_unique": len({normalize_text(a) for a in actions}),
        "candidate_usage": [
            {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "reasoning_tokens": result.usage.reasoning_tokens,
            }
            if result.usage
            else None
            for _, _, result in responses
        ],
        "agreement": agreement,
        "selection_s": round(selection_s, 4),
        "selection_used_tournament": bool(selection.comparisons),
        "tournament_comparisons": len(selection.comparisons),
        "judge_calls": judge_counter.calls - calls_before,
        "judge_completion_tokens": judge_counter.completion_tokens - tokens_before,
        "judge_sum_call_s": round(judge_counter.sum_call_s - call_s_before, 4),
        "best_index": selection.best_index,
        "candidate_scores": scores,
        "first_score": scores[0],
        "selected_score": scores[selection.best_index],
        "actions": actions,
    }


def _summarize_quality_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    deterministic = [record for record in records if record["first_score"] is not None]
    return {
        "tasks": records,
        "exact_majority_tasks": sum(
            record["agreement"]["exact"]["strict_majority"] for record in records
        ),
        "normalized_majority_tasks": sum(
            record["agreement"]["normalized"]["strict_majority"] for record in records
        ),
        "semantic_majority_tasks": sum(
            record["agreement"]["semantic"]["strict_majority"] for record in records
        ),
        "first_candidate_correct": sum(
            record["first_score"] for record in deterministic
        ),
        "selected_candidate_correct": sum(
            record["selected_score"] for record in deterministic
        ),
        "deterministic_tasks": len(deterministic),
    }


async def run_turbo_quality_corpus(
    tmp: Path,
    base_url: str,
    model: str,
    embedding_base_url: str,
    embedding_model: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Generate four candidates once per task, then compare agreement modes."""
    path = tmp / "quality.yaml"
    path.write_text(
        config_yaml(
            base_url,
            model,
            4,
            embedding_base_url,
            embedding_model,
            majority_voting=True,
            max_tokens=max_tokens,
        )
    )
    backend = Backend(Config(str(path)))
    verifiers = _quality_mode_verifiers(backend)
    judge_counter = JudgeCallCounter(verifiers["semantic"])
    records = [
        await _measure_quality_task(backend, verifiers, judge_counter, task, max_tokens)
        for task in CORPUS
    ]
    return _summarize_quality_records(records)


async def run_turbo_selection_scale(
    tmp: Path,
    base_url: str,
    model: str,
    embedding_base_url: str,
    embedding_model: str,
    samples: int,
) -> dict[str, Any]:
    """Measure candidate and forced-tournament phases at N=1, 2, and 4."""
    results = {}
    for candidates in (1, 2, 4):
        path = tmp / f"scale-{candidates}.yaml"
        path.write_text(
            config_yaml(
                base_url,
                model,
                candidates,
                embedding_base_url,
                embedding_model,
                majority_voting=False,
            )
        )
        backend = Backend(Config(str(path)))
        verifier = backend.verifier
        if candidates > 1 and verifier is None:
            raise RuntimeError("Selection benchmark requires an enabled verifier")
        judge_counter = JudgeCallCounter(verifier) if verifier is not None else None
        trials = []
        for _ in range(samples):
            body = {
                "model": "production-benchmark",
                "max_tokens": 16384,
                "reasoning_effort": "xhigh",
                "temperature": 0.7,
                "messages": [
                    {
                        "role": "user",
                        "content": BOUNDED_ANSWER_PROMPT,
                    }
                ],
            }
            request, _ = backend._build_openai_params(body)
            started = time.monotonic()
            responses = await gather_turbo_results(backend, request)
            candidate_s = time.monotonic() - started

            selection_s = 0.0
            comparisons = 0
            judge_calls = 0
            judge_completion_tokens = 0
            judge_sum_call_s = 0.0
            selection_scores = []
            if candidates > 1:
                if judge_counter is None or verifier is None:
                    raise RuntimeError("Selection benchmark lost its verifier")
                actions = [extract_response_text(r) for r, _, _ in responses]
                calls_before = judge_counter.calls
                tokens_before = judge_counter.completion_tokens
                call_s_before = judge_counter.sum_call_s
                started = time.monotonic()
                selected = await verifier.select_best(
                    Backend.format_history(body["messages"]),
                    actions,
                )
                selection_s = time.monotonic() - started
                comparisons = len(selected.comparisons)
                judge_calls = judge_counter.calls - calls_before
                judge_completion_tokens = (
                    judge_counter.completion_tokens - tokens_before
                )
                judge_sum_call_s = judge_counter.sum_call_s - call_s_before
                selection_scores = selected.scores
            trials.append(
                {
                    "candidate_s": round(candidate_s, 4),
                    "selection_s": round(selection_s, 4),
                    "total_s": round(candidate_s + selection_s, 4),
                    "comparisons": comparisons,
                    "judge_calls": judge_calls,
                    "judge_completion_tokens": judge_completion_tokens,
                    "judge_sum_call_s": round(judge_sum_call_s, 4),
                    "selection_scores": selection_scores,
                    "candidate_output_tokens": [
                        result.usage.output_tokens if result.usage else None
                        for _, _, result in responses
                    ],
                    "candidate_reasoning_tokens": [
                        result.usage.reasoning_tokens if result.usage else None
                        for _, _, result in responses
                    ],
                }
            )
        results[str(candidates)] = {
            "trials": trials,
            "candidate_mean_s": round(
                statistics.mean(t["candidate_s"] for t in trials), 4
            ),
            "selection_mean_s": round(
                statistics.mean(t["selection_s"] for t in trials), 4
            ),
            "total_mean_s": round(statistics.mean(t["total_s"] for t in trials), 4),
        }
    return results
