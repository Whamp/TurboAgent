"""Endpoint admission behavior shared by candidate and judge calls."""

import asyncio
import threading

import pytest

from turbo_agent.endpoint_admission import (
    EndpointAdmissionPolicy,
    EndpointAdmissionRegistry,
    EndpointQueueFullError,
    EndpointQueueTimeoutError,
)


async def test_async_admission_enforces_capacity_and_fifo_order():
    registry = EndpointAdmissionRegistry(
        {
            "server60": EndpointAdmissionPolicy(
                max_concurrency=2,
                max_queue_size=4,
                queue_timeout_seconds=1.0,
            )
        }
    )
    release_first_wave = asyncio.Event()
    first_wave_started = asyncio.Event()
    active = 0
    maximum_active = 0
    start_order = []

    async def run(index: int) -> None:
        nonlocal active, maximum_active
        async with registry.admit_async("server60"):
            active += 1
            maximum_active = max(maximum_active, active)
            start_order.append(index)
            if len(start_order) == 2:
                first_wave_started.set()
            if index < 2:
                await release_first_wave.wait()
            active -= 1

    tasks = [asyncio.create_task(run(index)) for index in range(4)]
    await asyncio.wait_for(first_wave_started.wait(), timeout=1.0)

    assert start_order == [0, 1]
    assert maximum_active == 2

    release_first_wave.set()
    await asyncio.gather(*tasks)

    assert start_order == [0, 1, 2, 3]
    assert maximum_active == 2


async def test_sync_judge_and_async_candidate_share_fifo_capacity():
    registry = EndpointAdmissionRegistry(
        {
            "server60": EndpointAdmissionPolicy(
                max_concurrency=1,
                max_queue_size=4,
                queue_timeout_seconds=1.0,
            )
        }
    )
    release_judge = threading.Event()
    judge_started = threading.Event()
    candidate_started = asyncio.Event()

    def run_judge() -> None:
        with registry.admit_sync("server60"):
            judge_started.set()
            release_judge.wait(timeout=1.0)

    async def run_candidate() -> None:
        async with registry.admit_async("server60"):
            candidate_started.set()

    async with registry.admit_async("server60"):
        judge_task = asyncio.create_task(asyncio.to_thread(run_judge))
        while registry.snapshot("server60").queued < 1:
            await asyncio.sleep(0)
        candidate_task = asyncio.create_task(run_candidate())
        while registry.snapshot("server60").queued < 2:
            await asyncio.sleep(0)

    assert await asyncio.to_thread(judge_started.wait, 1.0)
    assert not candidate_started.is_set()

    release_judge.set()
    await asyncio.gather(judge_task, candidate_task)

    assert candidate_started.is_set()
    assert registry.snapshot("server60").active == 0


async def test_full_queue_rejects_without_changing_capacity():
    registry = EndpointAdmissionRegistry(
        {
            "server60": EndpointAdmissionPolicy(
                max_concurrency=1,
                max_queue_size=0,
                queue_timeout_seconds=1.0,
            )
        }
    )

    async with registry.admit_async("server60"):
        with pytest.raises(EndpointQueueFullError, match="server60"):
            async with registry.admit_async("server60"):
                pytest.fail("a full queue must not admit another request")
        assert registry.snapshot("server60").active == 1
        assert registry.snapshot("server60").queued == 0


async def test_queue_deadline_removes_waiter_without_leaking_slot():
    registry = EndpointAdmissionRegistry(
        {
            "server60": EndpointAdmissionPolicy(
                max_concurrency=1,
                max_queue_size=1,
                queue_timeout_seconds=0.01,
            )
        }
    )

    async with registry.admit_async("server60"):
        with pytest.raises(EndpointQueueTimeoutError, match="server60"):
            async with registry.admit_async("server60"):
                pytest.fail("a timed-out waiter must not be admitted")
        assert registry.snapshot("server60").active == 1
        assert registry.snapshot("server60").queued == 0

    async with registry.admit_async("server60"):
        assert registry.snapshot("server60").active == 1


async def test_cancelled_waiter_is_removed_and_next_request_can_run():
    registry = EndpointAdmissionRegistry(
        {
            "server60": EndpointAdmissionPolicy(
                max_concurrency=1,
                max_queue_size=2,
                queue_timeout_seconds=1.0,
            )
        }
    )

    async with registry.admit_async("server60"):
        cancelled = asyncio.create_task(registry.admit_async("server60").__aenter__())
        while registry.snapshot("server60").queued < 1:
            await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert registry.snapshot("server60").active == 1
        assert registry.snapshot("server60").queued == 0

    async with registry.admit_async("server60"):
        assert registry.snapshot("server60").active == 1
