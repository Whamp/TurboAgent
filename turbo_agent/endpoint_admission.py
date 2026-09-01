"""Shared endpoint admission control for candidate and judge model calls."""

import asyncio
import threading
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Literal


class EndpointAdmissionError(RuntimeError):
    """Base error for endpoint admission failures before provider dispatch."""


class EndpointQueueFullError(EndpointAdmissionError):
    """Raised when an endpoint's bounded waiting queue is full."""


class EndpointQueueTimeoutError(EndpointAdmissionError):
    """Raised when an endpoint request misses its queue deadline."""


class EndpointRequestTimeoutError(EndpointAdmissionError):
    """Raised when an admitted provider call misses its request deadline."""


@dataclass(frozen=True)
class EndpointAdmissionPolicy:
    """Capacity and queue deadline for one named model endpoint."""

    max_concurrency: int
    max_queue_size: int
    queue_timeout_seconds: float
    request_timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        """Reject policies that cannot provide bounded admission."""
        if self.max_concurrency < 1:
            raise ValueError("Endpoint max_concurrency must be at least 1")
        if self.max_queue_size < 0:
            raise ValueError("Endpoint max_queue_size cannot be negative")
        if self.queue_timeout_seconds <= 0:
            raise ValueError("Endpoint queue_timeout_seconds must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("Endpoint request_timeout_seconds must be positive")


@dataclass(frozen=True)
class EndpointAdmissionSnapshot:
    """Current active and queued request counts for one endpoint."""

    active: int
    queued: int


WaiterState = Literal["waiting", "granted", "cancelled"]


@dataclass(eq=False)
class _AdmissionWaiter:
    notify_granted: Callable[[], None]
    state: WaiterState = "waiting"


class EndpointAdmissionLease:
    """One idempotently releasable endpoint-capacity reservation."""

    def __init__(self, controller: "_EndpointAdmissionController") -> None:
        """Bind the lease to the controller that granted it."""
        self._controller = controller
        self._released = False

    def release(self) -> None:
        """Return this reservation once; repeated releases are harmless."""
        if self._released:
            return
        self._released = True
        self._controller.release()


class _EndpointAdmissionController:
    def __init__(self, name: str, policy: EndpointAdmissionPolicy) -> None:
        self._name = name
        self._policy = policy
        self._lock = threading.Lock()
        self._active = 0
        self._queue: deque[_AdmissionWaiter] = deque()

    @property
    def request_timeout_seconds(self) -> float:
        return self._policy.request_timeout_seconds

    def snapshot(self) -> EndpointAdmissionSnapshot:
        with self._lock:
            return EndpointAdmissionSnapshot(
                active=self._active,
                queued=len(self._queue),
            )

    async def acquire_async(self) -> EndpointAdmissionLease:
        loop = asyncio.get_running_loop()
        granted = loop.create_future()

        def notify_granted() -> None:
            loop.call_soon_threadsafe(self._finish_async_grant, granted)

        waiter = _AdmissionWaiter(notify_granted)
        self._enqueue(waiter)
        try:
            await asyncio.wait_for(
                granted,
                timeout=self._policy.queue_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            self._cancel(waiter)
            raise EndpointQueueTimeoutError(
                f"Endpoint admission timeout: {self._name}"
            ) from exc
        except asyncio.CancelledError:
            self._cancel(waiter)
            raise
        return EndpointAdmissionLease(self)

    def acquire_sync(self) -> EndpointAdmissionLease:
        granted = threading.Event()
        waiter = _AdmissionWaiter(granted.set)
        self._enqueue(waiter)
        if not granted.wait(timeout=self._policy.queue_timeout_seconds):
            self._cancel(waiter)
            raise EndpointQueueTimeoutError(f"Endpoint admission timeout: {self._name}")
        return EndpointAdmissionLease(self)

    @staticmethod
    def _finish_async_grant(granted: asyncio.Future[None]) -> None:
        if not granted.done():
            granted.set_result(None)

    def _enqueue(self, waiter: _AdmissionWaiter) -> None:
        notify: Callable[[], None] | None = None
        with self._lock:
            if self._active < self._policy.max_concurrency and not self._queue:
                self._active += 1
                waiter.state = "granted"
                notify = waiter.notify_granted
            elif len(self._queue) >= self._policy.max_queue_size:
                raise EndpointQueueFullError(
                    f"Endpoint admission queue full: {self._name}"
                )
            else:
                self._queue.append(waiter)
        if notify is not None:
            notify()

    def _cancel(self, waiter: _AdmissionWaiter) -> None:
        notifications: list[Callable[[], None]] = []
        with self._lock:
            if waiter.state == "waiting":
                self._queue.remove(waiter)
            elif waiter.state == "granted":
                self._active -= 1
                notifications = self._dispatch_locked()
            waiter.state = "cancelled"
        for notify in notifications:
            notify()

    def release(self) -> None:
        notifications: list[Callable[[], None]] = []
        with self._lock:
            if self._active <= 0:
                raise RuntimeError(
                    f"Endpoint admission release without lease: {self._name}"
                )
            self._active -= 1
            notifications = self._dispatch_locked()
        for notify in notifications:
            notify()

    def _dispatch_locked(self) -> list[Callable[[], None]]:
        notifications = []
        while self._queue and self._active < self._policy.max_concurrency:
            waiter = self._queue.popleft()
            if waiter.state != "waiting":
                continue
            waiter.state = "granted"
            self._active += 1
            notifications.append(waiter.notify_granted)
        return notifications


class EndpointAdmissionRegistry:
    """Own FIFO admission controllers shared by all users of an endpoint."""

    def __init__(
        self,
        policies: Mapping[str, EndpointAdmissionPolicy],
    ) -> None:
        """Create one shared controller for every configured endpoint."""
        self._controllers = {
            name: _EndpointAdmissionController(name, policy)
            for name, policy in policies.items()
        }

    def _resolve(self, endpoint_name: str) -> _EndpointAdmissionController:
        try:
            return self._controllers[endpoint_name]
        except KeyError:
            raise KeyError(
                f"Unknown endpoint admission policy: {endpoint_name}"
            ) from None

    def snapshot(self, endpoint_name: str) -> EndpointAdmissionSnapshot:
        """Return current endpoint capacity use for diagnostics and tests."""
        return self._resolve(endpoint_name).snapshot()

    def request_timeout_seconds(self, endpoint_name: str) -> float:
        """Return the total provider-call deadline for an endpoint."""
        return self._resolve(endpoint_name).request_timeout_seconds

    async def acquire_async(self, endpoint_name: str) -> EndpointAdmissionLease:
        """Acquire one lease for an explicitly managed async lifetime."""
        return await self._resolve(endpoint_name).acquire_async()

    @asynccontextmanager
    async def admit_async(self, endpoint_name: str) -> AsyncIterator[None]:
        """Hold one endpoint slot for the full lifetime of an async call."""
        lease = await self.acquire_async(endpoint_name)
        try:
            yield
        finally:
            lease.release()

    @contextmanager
    def admit_sync(self, endpoint_name: str) -> Iterator[None]:
        """Hold one endpoint slot for the full lifetime of a blocking call."""
        lease = self._resolve(endpoint_name).acquire_sync()
        try:
            yield
        finally:
            lease.release()
