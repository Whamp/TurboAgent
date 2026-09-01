---
status: accepted
---

# Share endpoint admission between candidates and judges

Turbo Agent will give each configured model endpoint one admission controller. Candidate execution and verifier judge calls that reference the same endpoint name will share its active count, FIFO queue, queue deadline, and request deadline.

A backend model opts in with `endpoint: <name>`. The endpoint owns the `base_url`; a conflicting model-level `base_url` is a startup error. Models without `endpoint` keep their existing behavior.

The server60 judge uses a dedicated verifier client adapter. Comparison generations use `reasoning_effort: xhigh`, thinking enabled in the Qwen chat template, and a configured output-token cap. One-token score probes disable thinking so their score letter appears in `message.content`, where llm-verifier reads it. The adapter preserves prefill and structured-output controls. Turbo Agent limits llm-verifier to at most four judge workers.

## Why

The production study found two independent failures. llm-verifier returned `0.500 / 0.500` because server60 put one-token score letters in hidden reasoning. Candidate and judge calls also had separate concurrency controls, so verifier work could exceed server60's four-sequence capacity.

A process-wide semaphore would mix unrelated endpoints and would not define queue ownership. Separate candidate and judge semaphores would still oversubscribe one server. Named endpoint admission makes the shared resource explicit.

## Consequences

- The queue is bounded and FIFO across async candidate calls and blocking judge calls.
- Candidate cancellation removes a queued waiter or cancels an active provider call. Complete and stream calls hold their slot until provider teardown.
- Provider errors, queue errors, request deadlines, and stream closure release the slot.
- Endpoint-bound candidate and judge clients disable automatic SDK retries. One admission lease therefore maps to one provider request at a time.
- A blocking llm-verifier HTTP call cannot receive an asyncio cancellation after it enters the synchronous OpenAI client. It keeps its slot until the call returns or reaches the endpoint request deadline. Releasing earlier would admit a fifth server request while the canceled call still runs.
- The server60 adapter is opt-in. Other judges keep llm-verifier's existing request parameters and worker selection.
- Context refinement and progress monitoring do not use endpoint admission in this decision.

The runtime contract and configuration are in [Endpoint admission and server60 judging](../design/endpoint-admission.md).
