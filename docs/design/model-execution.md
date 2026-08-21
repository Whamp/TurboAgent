# Model execution

[ADR-0001](../adr/0001-model-execution-seam.md) selects one deep model execution module for Candidate-producing Backend model calls. This document fixes the seam, interface, ownership, adapter behavior, and migration order before implementation.

## Goal

A caller names a configured Backend model, supplies one prepared Candidate request, and receives one normalized result or stream. The caller does not know how the model is authenticated, where it is hosted, which wire model id it uses, or how its response is translated.

This design fixes the current gap where a Backend model's `base_url` is loaded from YAML but never reaches `llm_completion` or `llm_stream_completion`. It also gives Turbo Agent a path to Pi's `openai-codex-responses` implementation and subscription credentials without copying OAuth behavior into Python.

## Seam and ownership

The seam sits between Concurrent inference and outbound model calls.

| Concern | Owner |
| --- | --- |
| Number of Candidates | Concurrent inference |
| Candidate start order and future capacity limits | Concurrent inference |
| Context refinement | Request pipeline |
| Configured target, endpoint, credentials, wire model id | Model execution |
| Model defaults and client-intent precedence | Model execution |
| Output-cap clamping and thinking compatibility | Model execution |
| Provider-specific request and response translation | Model execution adapter |
| One normalized Candidate result | Model execution |
| Majority voting, Judge calls, Pivot tournament, Fallback | Verification |
| Client response encoding and SSE framing | Anthropic/OpenAI adapter |
| Request trace and Progress monitor | Request pipeline |

The first extraction covers Candidate generation only. Judge, Context refinement, and Progress monitor model calls stay in their current modules until Candidate execution is stable. Moving them later requires a separate decision because their request and failure semantics differ.

## Interface

The production names may change during implementation, but the interface shape must not grow without reopening this design.

```python
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, NewType, Protocol

ModelTargetId = NewType("ModelTargetId", str)
JsonObject = Mapping[str, Any]


@dataclass(frozen=True)
class ModelTarget:
    id: ModelTargetId          # opaque configured-target identity
    name: str                  # safe display name for logs and Request traces


@dataclass(frozen=True)
class ThinkingIntent:
    mode: Literal["default", "disabled", "adaptive", "budget"] = "default"
    effort: str | None = None
    budget_tokens: int | None = None


@dataclass(frozen=True)
class GenerationIntent:
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | tuple[str, ...] | None = None
    tool_choice: str | JsonObject | None = None
    response_format: JsonObject | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: Mapping[str, int] = field(default_factory=dict)
    thinking: ThinkingIntent = field(default_factory=ThinkingIntent)
    sampling_params: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class ModelExecutionRequest:
    # OpenAI-shaped during the first extraction. See "Migration constraint".
    messages: tuple[JsonObject, ...]
    tools: tuple[JsonObject, ...] = ()
    generation: GenerationIntent = field(default_factory=GenerationIntent)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: JsonObject


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class AssistantOutput:
    text: str
    tool_calls: tuple[ToolCall, ...]
    thinking: str | None
    finish_reason: Literal["stop", "length", "tool_use", "content_filter", "other"]


@dataclass(frozen=True)
class ModelExecutionResult:
    target: ModelTarget
    output: AssistantOutput
    usage: TokenUsage | None
    response_id: str | None = None


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    id: str
    name: str


@dataclass(frozen=True)
class ToolCallArgumentsDelta:
    id: str
    json_fragment: str


@dataclass(frozen=True)
class ExecutionCompleted:
    result: ModelExecutionResult


ModelExecutionEvent = (
    TextDelta
    | ThinkingDelta
    | ToolCallStarted
    | ToolCallArgumentsDelta
    | ExecutionCompleted
)


class ModelExecutor(Protocol):
    async def complete(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> ModelExecutionResult: ...

    def stream(
        self,
        target: ModelTarget,
        request: ModelExecutionRequest,
    ) -> AsyncIterator[ModelExecutionEvent]: ...

    async def aclose(self) -> None: ...
```

The runtime interface has three entry points: complete one Candidate, stream one Candidate, and close owned resources. Startup composition creates safe `ModelTarget` handles and gives the module the corresponding private target configuration. Callers never receive API keys, base URLs, provider headers, or Pi credentials.

### Why complete and stream are separate

A single method returning either a result or event stream makes every caller branch on a union and obscures cancellation. A stream-only interface makes the common verified path collect events just to recover one result. Two execution methods keep both call sites direct while sharing all target resolution and translation inside the implementation.

### Migration constraint

The first LiteLLM extraction will keep the current OpenAI-shaped `messages` and `tools` as the internal prompt representation. Replacing that representation while moving model execution would mix two migrations and invalidate the existing Pi interoperability tests. The interface still returns a canonical `ModelExecutionResult`, so provider response dictionaries stop leaking immediately. A later Client request pipeline change may replace the prompt representation without changing adapter selection, credentials, endpoint handling, errors, or result types.

## Invariants

1. One `complete` call produces exactly one Candidate. Client `n` never multiplies choices inside model execution.
2. The module applies configured defaults first, explicit Client request intent second, and the configured output cap last.
3. Explicitly disabled thinking removes configured thinking. Adaptive effort and token-budget thinking never remain active together.
4. A thinking budget greater than or equal to the effective output cap is removed rather than shrinking the Client request's cap.
5. Every call receives an isolated copy of mutable messages and tools.
6. `complete` and `stream` apply the same target, precedence, cap, and reasoning rules.
7. Streaming emits at most one `ExecutionCompleted` event, and it is the final event.
8. Cancellation propagates as `asyncio.CancelledError`; the module cancels the underlying adapter call and does not classify cancellation as a model failure.
9. A model failure never chooses a Fallback or another target. Concurrent inference decides whether other Candidates are sufficient.
10. Secrets, bearer headers, and OAuth tokens never appear in results, exceptions, logs, or the Request trace.
11. Endpoint configuration is target-scoped. Model execution never sets process-wide endpoint or credential environment variables.

## Failures

Adapters translate model-specific failures into one safe exception:

```python
class ModelExecutionError(Exception):
    target: ModelTarget
    kind: Literal[
        "authentication",
        "rate_limited",
        "timeout",
        "invalid_request",
        "unavailable",
        "malformed_response",
        "internal",
    ]
    retryable: bool
    retry_after_seconds: float | None
```

The exception message must be safe for logs. The implementation may chain the original exception for local debugging, but Request traces record only the normalized fields. Configuration errors, unknown targets, unavailable adapter executables, and unknown Pi models fail at startup rather than on the first Client request.

## Target configuration

Existing Backend model entries continue to use LiteLLM by default. A new `adapter` field selects model execution when needed:

```yaml
backend:
  models:
    - name: gemini/gemini-2.5-flash
      adapter: litellm       # optional; this is the default
      api_key: $VERTEX_API_KEY
      num_candidates: 2
      max_tokens: 65536

    - name: openai-codex/<pi-model-id>
      adapter: pi
      num_candidates: 2
      thinking: high
      max_tokens: 32768
```

For `adapter: pi`, the name prefix selects the Pi provider and the remaining text selects the Pi model id. `api_key` and `base_url` are invalid because Pi `ModelRuntime` owns auth and endpoint selection. The configuration loader rejects those combinations.

For `adapter: litellm`, `base_url` is passed explicitly on every call. This fixes local OpenAI-compatible targets without creating a third adapter. Add a dedicated server60 adapter only if a live benchmark proves that engine-specific batching, prefix sharing, or sampling controls cannot remain inside the LiteLLM adapter.

## Adapters

### LiteLLM adapter

The LiteLLM adapter owns:

- model prefix routing;
- explicit `api_key` and `base_url` forwarding;
- conversion from `GenerationIntent` to LiteLLM parameters;
- unsupported-parameter dropping;
- Anthropic budget-thinking translation;
- response and stream normalization;
- LiteLLM exception classification.

It must not mutate global endpoint or credential environment variables. Existing Gemini environment setup moves behind this adapter or is removed when explicit credentials work for the configured route.

### Pi ModelRuntime adapter

Python cannot import Pi's TypeScript `ModelRuntime`. The Pi adapter therefore owns one Node companion process and communicates over versioned newline-delimited JSON on stdio. Stdio ties the companion lifecycle to Turbo Agent and does not expose a new local port.

The companion:

- creates one `ModelRuntime` with Pi's standard auth path;
- resolves targets with `getModel(provider, modelId)`;
- calls `completeSimple` or `streamSimple`;
- lets `ModelRuntime` resolve and refresh subscription credentials;
- multiplexes concurrent calls by request id;
- keeps one `AbortController` per call;
- writes protocol messages to stdout and diagnostics to stderr;
- never returns credentials to Python.

Minimum wire messages:

```json
{"v":1,"id":"req-1","op":"complete","provider":"openai-codex","model":"<pi-model-id>","context":{},"options":{}}
{"v":1,"id":"req-1","op":"stream","provider":"openai-codex","model":"<pi-model-id>","context":{},"options":{}}
{"v":1,"id":"req-1","op":"cancel"}

{"v":1,"id":"req-1","event":"delta","delta":{}}
{"v":1,"id":"req-1","event":"done","result":{}}
{"v":1,"id":"req-1","event":"error","error":{"kind":"rate_limited","retryable":true}}
```

The companion maps the prepared prompt to Pi's `Context` and `SimpleStreamOptions`. It maps Pi `AssistantMessage` content and events back to the canonical result and event types. A companion crash fails all in-flight calls as `unavailable`; the adapter may restart it for a later call but never silently replay an in-flight request.

## Construction and lifecycle

Startup performs these steps:

1. Parse and validate all Backend model entries.
2. Allocate a stable opaque target id for each configured entry.
3. Build the Candidate target list from `num_candidates`.
4. Construct only the adapters required by configured targets.
5. Start and validate the Pi companion when at least one Pi target exists.
6. Inject the composed `ModelExecutor` into `Backend`.
7. Call `aclose` from the FastAPI lifespan hook.

`Backend` must not construct adapters or read credentials. Tests inject a fake `ModelExecutor` through the same seam.

## Migration plan

### 1. Lock current behavior

Add failing tests for the seam before moving code:

- a Backend model `base_url` reaches both complete and stream calls;
- Client token and thinking precedence remains unchanged per configured target;
- concurrent calls receive isolated prompts;
- one failed Candidate does not discard successful Candidates;
- all failed Candidates preserve a normalized first failure;
- cancellation reaches the adapter;
- complete and collected stream output normalize identically.

### 2. Extract LiteLLM execution

Create the model execution module and LiteLLM adapter. Inject it into `Backend`, move `_model_entries`, `_parse_model_params`, request precedence, cap logic, response normalization, and direct `llm_*` calls behind the seam, then delete the replaced helpers. Existing HTTP interoperability tests remain acceptance tests.

### 3. Add the Pi companion

Implement the versioned stdio companion and Pi adapter. Contract-test it with a fake `ModelRuntime`, then run an opt-in smoke test against the user's existing Pi login. The smoke test must confirm that no OAuth token crosses into Python or the Request trace.

### 4. Prove mixed execution

Run one Client request with Candidates from both LiteLLM and Pi. Verify Candidate identity, failure isolation, Verification input, selected response encoding, cancellation, and Request trace redaction.

### 5. Benchmark server60

Restore a live model endpoint, then compare four independent calls against any supported engine-native packing. Record time to first token, total latency, generated tokens per second, GPU memory, and Candidate diversity. Keep local execution in the LiteLLM adapter unless the measurements justify a dedicated adapter.

## Test surface

Tests cross the same `ModelExecutor` seam as `Backend`:

- module contract tests cover precedence, caps, normalization, errors, and cancellation;
- each adapter runs the shared contract suite;
- Pi companion tests cover protocol versioning, multiplexing, cancellation, crashes, and credential redaction;
- existing Anthropic and OpenAI HTTP tests cover client encoding and streaming framing;
- live subscription and server60 checks remain opt-in because they require external state.

Do not retain parallel tests for the old direct LiteLLM helpers after the extraction. The deletion is part of the refactor: duplicated behavior would weaken the seam.
