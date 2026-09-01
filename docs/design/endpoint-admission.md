# Endpoint admission and server60 judging

[ADR-0002](../adr/0002-endpoint-admission-and-server60-judge.md) assigns endpoint capacity to one registry shared by candidate execution and verification.

## Configuration

Declare a named endpoint, then reference it from every backend and verifier model that reaches the same server.

```yaml
endpoints:
  server60:
    base_url: http://192.168.0.251:30002/v1
    max_concurrency: 4
    max_queue_size: 64
    queue_timeout_seconds: 600
    request_timeout_seconds: 1800

backend:
  models:
    - name: openai/qwen3.8-flash-next-intel-autoround-w4a16
      api_key: sk-local
      endpoint: server60
      num_candidates: 4
      thinking: xhigh
      max_tokens: 65536

verifier:
  model:
    name: openai/qwen3.8-flash-next-intel-autoround-w4a16
    api_key: sk-local
    endpoint: server60
  execution:
    adapter: server60
    comparison_max_output_tokens: 16384
    max_concurrency: 4
  majority_voting: true
  majority:
    mode: normalized
  method:
    name: pivot_tournament
    pivots: 1
    n_verifications: 1
```

The endpoint fields have these defaults:

| Field | Default | Meaning |
| --- | ---: | --- |
| `max_concurrency` | `1` | Maximum admitted provider calls for the endpoint. |
| `max_queue_size` | `64` | Maximum waiting calls. Active calls do not count toward this limit. |
| `queue_timeout_seconds` | `300` | Maximum wait before admission. |
| `request_timeout_seconds` | `900` | Maximum provider-call lifetime after admission. |

`base_url` must be an absolute HTTP(S) URL. `max_concurrency` and both deadlines must be positive. `max_queue_size` can be zero to reject requests whenever all slots are active.

A model can still use its existing `base_url` without endpoint admission. To share capacity, replace that field with `endpoint`. If both fields exist, their normalized URLs must match. Unknown endpoint names and conflicting URLs fail during `Backend` startup. When the server60 judge adapter is enabled, a candidate resolving to the same origin (scheme, host, and port)—directly or through LiteLLM's `<PROVIDER>_API_BASE` environment setting—must use the same named endpoint; otherwise startup fails instead of creating a second capacity domain. A verifier that names an endpoint, or uses a direct `base_url` matching one, must also opt into the server60 adapter; the default adapter does not own endpoint admission. Turbo cannot infer that different host aliases or IP addresses reach the same server, so every such route must reference the same endpoint name explicitly.

The server60 verifier adapter requires `verifier.model.endpoint`. The endpoint capacity and the adapter's `max_concurrency` must each be between one and four; judge workers also cannot exceed endpoint capacity. `comparison_max_output_tokens` must be at least two.

## Ownership

`Backend` creates one `EndpointAdmissionRegistry` from `endpoints`. It passes the same object to candidate execution and `Verifier`.

`EndpointAdmissionExecutor` decorates configured model targets. It holds one async lease around a complete call or the full lifetime of a stream. Endpoint-bound LiteLLM targets receive the endpoint request timeout and `max_retries: 0`, so one lease cannot hide overlapping SDK retries. Targets without `endpoint` pass through unchanged.

`Server60JudgeClient` decorates llm-verifier's synchronous OpenAI client. It holds one synchronous lease around each comparison generation and score probe. Async and synchronous waiters enter the same FIFO queue.

`Backend.endpoint_admission_snapshot(name)` returns the current active and queued counts. The snapshot is diagnostic. It does not reserve capacity.

## Server60 judge requests

The adapter recognizes llm-verifier's score probes by both conditions:

- `max_tokens` is `1`;
- `extra_body.continue_final_message` is `true`.

For comparison generations, it sends:

```yaml
reasoning_effort: xhigh
max_tokens: <comparison_max_output_tokens>
extra_body:
  chat_template_kwargs:
    enable_thinking: true
```

For one-token score probes, it removes `reasoning_effort` and sends:

```yaml
max_tokens: 1
extra_body:
  chat_template_kwargs:
    enable_thinking: false
```

The adapter retains `add_generation_prompt`, `continue_final_message`, `structured_outputs`, logprobs, messages, and other llm-verifier fields. It passes `request_timeout_seconds` as the OpenAI client's `timeout` value and disables the OpenAI SDK's automatic retries. Turbo Agent passes `verifier.execution.max_concurrency` to llm-verifier's `max_workers` argument. Because server60 supports `extra_body`, the adapter also prevents llm-verifier's generic compatibility fallback from retrying a failed call.

## Queue and release rules

The controller grants a slot immediately when capacity is free and no waiter is ahead. Otherwise, it appends the request to the FIFO queue.

A request leaves the queue in one of four ways:

1. a slot becomes free;
2. its queue deadline expires;
3. its async task is canceled;
4. the queue rejects it because `max_queue_size` is reached.

An admitted request releases its slot after normal completion, provider error, request deadline, cancellation, or stream closure. Client stream closure propagates through the proxy, Backend, admission executor, and LiteLLM provider iterator before the lease is released. Release grants the oldest waiter before a newer request can enter.

The controller never releases a blocking judge slot only because the outer async request was canceled. llm-verifier uses a synchronous OpenAI client. The provider request can still be active after the async caller stops waiting. The configured request deadline bounds that case.

## Failures

Admission raises searchable errors:

- `EndpointQueueFullError`
- `EndpointQueueTimeoutError`
- `EndpointRequestTimeoutError`

Candidate failures continue through the existing concurrent-inference policy. Other successful candidates can still win. If every candidate fails, the client request fails as before.

Verifier admission or provider failures continue through the existing verification fallback. Turbo Agent returns the first valid candidate and records that no tournament completed.
