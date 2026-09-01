"""server60 queue-capacity and cancellation slot-release measurements."""

import asyncio
import re
import time
from typing import Any

import httpx

from .server60_vllm import stream_vllm_request


async def fetch_vllm_metrics(
    client: httpx.AsyncClient, root_url: str
) -> dict[str, float]:
    """Read current running/waiting gauges from the vLLM metrics endpoint."""
    response = await client.get(f"{root_url}/metrics", timeout=10)
    response.raise_for_status()
    metrics = {}
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        matches = re.findall(
            rf"^{re.escape(name)}\{{[^\n]*\}} ([0-9.eE+-]+)$",
            response.text,
            re.MULTILINE,
        )
        metrics[name] = sum(float(value) for value in matches)
    return metrics


async def _measure_fifth_request_queue(
    client: httpx.AsyncClient,
    endpoint: str,
    root_url: str,
    model: str,
) -> dict[str, Any]:
    active_events = [{0: asyncio.Event()} for _ in range(4)]
    active_tasks = [
        asyncio.create_task(
            stream_vllm_request(
                client,
                endpoint,
                model,
                prompt="Write a detailed technical analysis of continuous batching.",
                max_tokens=2048,
                seed=4000 + index,
                started_events=active_events[index],
            )
        )
        for index in range(4)
    ]
    await asyncio.gather(*(events[0].wait() for events in active_events))
    before_fifth = await fetch_vllm_metrics(client, root_url)
    fifth_task = asyncio.create_task(
        stream_vllm_request(
            client,
            endpoint,
            model,
            prompt="Reply with FIFTH-ADMITTED.",
            max_tokens=96,
            seed=4999,
        )
    )
    max_running = before_fifth["vllm:num_requests_running"]
    max_waiting = before_fifth["vllm:num_requests_waiting"]
    while not fifth_task.done():
        metrics = await fetch_vllm_metrics(client, root_url)
        max_running = max(max_running, metrics["vllm:num_requests_running"])
        max_waiting = max(max_waiting, metrics["vllm:num_requests_waiting"])
        await asyncio.sleep(0.05)
    fifth_result = await fifth_task
    active_results = await asyncio.gather(*active_tasks)
    return {
        "running_before_fifth": before_fifth["vllm:num_requests_running"],
        "max_running": max_running,
        "max_waiting": max_waiting,
        "active_ttft_s": [
            next(iter(result["ttft_s"].values())) for result in active_results
        ],
        "fifth_ttft_s": next(iter(fifth_result["ttft_s"].values())),
        "fifth_total_s": fifth_result["total_s"],
    }


async def _measure_cancellation_replacement(
    client: httpx.AsyncClient,
    endpoint: str,
    root_url: str,
    model: str,
) -> dict[str, Any]:
    cancel_events = [asyncio.Event() for _ in range(4)]
    started_events = [{0: asyncio.Event()} for _ in range(4)]
    active_tasks = [
        asyncio.create_task(
            stream_vllm_request(
                client,
                endpoint,
                model,
                prompt="Write a long technical discussion of GPU inference scheduling.",
                max_tokens=16384,
                seed=5000 + index,
                started_events=started_events[index],
                cancel_event=cancel_events[index],
            )
        )
        for index in range(4)
    ]
    await asyncio.gather(*(events[0].wait() for events in started_events))
    cancelled_at = time.monotonic()
    cancel_events[0].set()
    replacement_task = asyncio.create_task(
        stream_vllm_request(
            client,
            endpoint,
            model,
            prompt="Reply with REPLACEMENT-ADMITTED.",
            max_tokens=96,
            seed=6000,
        )
    )
    cancelled_result = await active_tasks[0]
    cancel_close_s = time.monotonic() - cancelled_at
    replacement = await replacement_task
    for event in cancel_events[1:]:
        event.set()
    await asyncio.gather(*active_tasks[1:])
    idle_started = time.monotonic()
    idle_metrics = await fetch_vllm_metrics(client, root_url)
    while idle_metrics["vllm:num_requests_running"] > 0:
        await asyncio.sleep(0.05)
        idle_metrics = await fetch_vllm_metrics(client, root_url)
    replacement_first = next(iter(replacement["ttft_s"].values()))
    return {
        "cancelled_request_closed": cancelled_result["cancelled"],
        "replacement_ttft_s": replacement_first,
        "replacement_total_s": replacement["total_s"],
        "cancelled_stream_closed_s": round(cancel_close_s, 4),
        "replacement_admission_from_cancel_s": replacement_first,
        "metrics_returned_idle_s": round(time.monotonic() - idle_started, 4),
    }


async def run_capacity_and_cancellation(
    client: httpx.AsyncClient,
    endpoint: str,
    root_url: str,
    model: str,
) -> dict[str, Any]:
    """Measure fifth-request queueing and cancellation slot recovery."""
    return {
        "five_request_test": await _measure_fifth_request_queue(
            client, endpoint, root_url, model
        ),
        "cancellation_test": await _measure_cancellation_replacement(
            client, endpoint, root_url, model
        ),
    }
