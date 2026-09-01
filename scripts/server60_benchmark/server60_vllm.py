"""Raw vLLM streaming and engine-controlled N=4 measurements."""

import asyncio
import hashlib
import json
import math
import statistics
import time
from typing import Any

import httpx

from .server60_workloads import DIRECT_PROMPT, normalize_text


def percentile(values: list[float], fraction: float) -> float:
    """Return an interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


async def stream_vllm_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    *,
    prompt: str,
    max_tokens: int,
    seed: int | None = None,
    choices: int = 1,
    started_events: dict[int, asyncio.Event] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Measure TTFT and completion time for one streaming vLLM request."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "reasoning_effort": "xhigh",
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "n": choices,
    }
    if seed is not None:
        payload["seed"] = seed

    request_start = time.monotonic()
    first_token_at: dict[int, float] = {}
    reasoning_parts: dict[int, list[str]] = {i: [] for i in range(choices)}
    content_parts: dict[int, list[str]] = {i: [] for i in range(choices)}
    usage: dict[str, Any] = {}
    cancelled = False

    async with client.stream(
        "POST",
        f"{endpoint}/chat/completions",
        json=payload,
        timeout=300,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                index = choice.get("index", 0)
                delta = choice.get("delta") or {}
                reasoning = str(
                    delta.get("reasoning") or delta.get("reasoning_content") or ""
                )
                content = str(delta.get("content") or "")
                if not reasoning and not content:
                    continue
                if index not in first_token_at:
                    first_token_at[index] = time.monotonic()
                    if started_events and index in started_events:
                        started_events[index].set()
                if reasoning:
                    reasoning_parts.setdefault(index, []).append(reasoning)
                if content:
                    content_parts.setdefault(index, []).append(content)

    request_end = time.monotonic()
    ttft = {
        str(i): round(first_token_at.get(i, request_end) - request_start, 4)
        for i in range(choices)
    }
    reasoning_outputs = ["".join(reasoning_parts.get(i, [])) for i in range(choices)]
    content_outputs = ["".join(content_parts.get(i, [])) for i in range(choices)]
    return {
        "ttft_s": ttft,
        "total_s": round(request_end - request_start, 4),
        "usage": usage,
        "reasoning_chars": [len(text) for text in reasoning_outputs],
        "content_chars": [len(text) for text in content_outputs],
        "reasoning_sha256": [
            hashlib.sha256(text.encode()).hexdigest() for text in reasoning_outputs
        ],
        "content_sha256": [
            hashlib.sha256(text.encode()).hexdigest() for text in content_outputs
        ],
        "normalized_unique_outputs": len(
            {normalize_text(text) for text in content_outputs}
        ),
        "cancelled": cancelled,
    }


async def run_direct_concurrency(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    samples: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Measure independent requests at concurrency 1, 2, and 4."""
    await stream_vllm_request(
        client,
        endpoint,
        model,
        prompt="Reply with OK.",
        max_tokens=32,
        seed=1,
    )
    results: dict[str, Any] = {}
    for concurrency in (1, 2, 4):
        trials = []
        for sample in range(samples):
            started = time.monotonic()
            requests = await asyncio.gather(
                *(
                    stream_vllm_request(
                        client,
                        endpoint,
                        model,
                        prompt=DIRECT_PROMPT,
                        max_tokens=max_tokens,
                        seed=1000 + concurrency * 100 + sample * 10 + index,
                    )
                    for index in range(concurrency)
                )
            )
            wall = time.monotonic() - started
            completion_tokens = sum(
                int(item["usage"].get("completion_tokens") or 0) for item in requests
            )
            trials.append(
                {
                    "wall_s": round(wall, 4),
                    "aggregate_output_tokens_per_s": round(
                        completion_tokens / wall if wall else 0.0,
                        2,
                    ),
                    "requests": requests,
                }
            )
        all_ttft = [
            next(iter(request["ttft_s"].values()))
            for trial in trials
            for request in trial["requests"]
        ]
        results[str(concurrency)] = {
            "trials": trials,
            "wall_mean_s": round(statistics.mean(t["wall_s"] for t in trials), 4),
            "ttft_mean_s": round(statistics.mean(all_ttft), 4),
            "ttft_p95_s": round(percentile(all_ttft, 0.95), 4),
            "aggregate_output_tokens_per_s_mean": round(
                statistics.mean(t["aggregate_output_tokens_per_s"] for t in trials), 2
            ),
        }
    return results


async def run_n4_comparison(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    samples: int,
    max_tokens: int,
    prompt: str,
) -> dict[str, Any]:
    """Compare four HTTP requests with one vLLM request using ``n=4``."""
    modes: dict[str, Any] = {"independent_requests": [], "single_request_n4": []}
    for sample in range(samples):
        started = time.monotonic()
        independent = await asyncio.gather(
            *(
                stream_vllm_request(
                    client,
                    endpoint,
                    model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    seed=2000 + sample * 10 + index,
                )
                for index in range(4)
            )
        )
        independent_wall = time.monotonic() - started
        independent_tokens = sum(
            int(item["usage"].get("completion_tokens") or 0) for item in independent
        )
        modes["independent_requests"].append(
            {
                "wall_s": round(independent_wall, 4),
                "aggregate_output_tokens_per_s": round(
                    independent_tokens / independent_wall,
                    2,
                ),
                "ttft_s": [next(iter(item["ttft_s"].values())) for item in independent],
                "requests": independent,
                "normalized_unique_outputs": len(
                    {
                        digest
                        for item in independent
                        for digest in item["content_sha256"]
                    }
                ),
            }
        )

        started = time.monotonic()
        one_request = await stream_vllm_request(
            client,
            endpoint,
            model,
            prompt=prompt,
            max_tokens=max_tokens,
            seed=3000 + sample,
            choices=4,
        )
        one_wall = time.monotonic() - started
        one_tokens = int(one_request["usage"].get("completion_tokens") or 0)
        modes["single_request_n4"].append(
            {
                "wall_s": round(one_wall, 4),
                "aggregate_output_tokens_per_s": round(
                    one_tokens / one_wall,
                    2,
                ),
                "ttft_s": list(one_request["ttft_s"].values()),
                "request": one_request,
                "normalized_unique_outputs": one_request["normalized_unique_outputs"],
            }
        )

    return {
        name: {
            "trials": trials,
            "wall_mean_s": round(statistics.mean(t["wall_s"] for t in trials), 4),
            "unique_outputs_mean": round(
                statistics.mean(t["normalized_unique_outputs"] for t in trials), 2
            ),
            "aggregate_output_tokens_per_s_mean": round(
                statistics.mean(t["aggregate_output_tokens_per_s"] for t in trials), 2
            ),
            "ttft_mean_s": round(
                statistics.mean(value for trial in trials for value in trial["ttft_s"]),
                4,
            ),
            "ttft_p95_s": round(
                percentile(
                    [value for trial in trials for value in trial["ttft_s"]], 0.95
                ),
                4,
            ),
        }
        for name, trials in modes.items()
    }
