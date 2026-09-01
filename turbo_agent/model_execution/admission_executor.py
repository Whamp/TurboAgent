"""Endpoint admission decorator for model execution lifetimes."""

import asyncio
from collections.abc import AsyncIterator, Mapping

from ..endpoint_admission import (
    EndpointAdmissionRegistry,
    EndpointRequestTimeoutError,
)
from .types import (
    ModelExecutionEvent,
    ModelExecutionRequest,
    ModelExecutionResult,
    ModelExecutor,
    ModelTarget,
)


class EndpointAdmissionExecutor(ModelExecutor):
    """Apply shared endpoint capacity and deadlines to model targets."""

    def __init__(
        self,
        executor: ModelExecutor,
        registry: EndpointAdmissionRegistry,
        target_endpoints: Mapping[ModelTarget, str],
    ) -> None:
        """Bind selected model targets to shared endpoint policies."""
        self._executor = executor
        self._registry = registry
        self._target_endpoints = dict(target_endpoints)

    async def complete(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> ModelExecutionResult:
        """Complete one request while holding its configured endpoint slot."""
        endpoint_name = self._target_endpoints.get(target)
        if endpoint_name is None:
            return await self._executor.complete(target, request)
        async with self._registry.admit_async(endpoint_name):
            try:
                return await asyncio.wait_for(
                    self._executor.complete(target, request),
                    timeout=self._registry.request_timeout_seconds(endpoint_name),
                )
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise EndpointRequestTimeoutError(
                    f"Endpoint request timeout: {endpoint_name}"
                ) from exc

    async def stream(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> AsyncIterator[ModelExecutionEvent]:
        """Stream one request while holding its slot through final cleanup."""
        endpoint_name = self._target_endpoints.get(target)
        if endpoint_name is None:
            async for event in self._executor.stream(target, request):
                yield event
            return
        # Admission must span yielded events; Backend consumes or explicitly
        # closes this generator on completion, timeout, and cancellation.
        lease = await self._registry.acquire_async(endpoint_name)
        try:
            deadline = (
                asyncio.get_running_loop().time()
                + self._registry.request_timeout_seconds(endpoint_name)
            )
            iterator = self._executor.stream(target, request).__aiter__()
            try:
                try:
                    while True:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        try:
                            event = await asyncio.wait_for(
                                iterator.__anext__(),
                                timeout=remaining,
                            )
                        except StopAsyncIteration:
                            break
                        yield event
                except (asyncio.TimeoutError, TimeoutError) as exc:
                    raise EndpointRequestTimeoutError(
                        f"Endpoint request timeout: {endpoint_name}"
                    ) from exc
            finally:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    await close()
        finally:
            lease.release()

    async def aclose(self) -> None:
        """Close the wrapped executor."""
        await self._executor.aclose()
