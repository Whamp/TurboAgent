# How this fork differs from upstream

This repository is [`Whamp/TurboAgent`](https://github.com/Whamp/TurboAgent), a fork of [`llm-as-a-verifier/TurboAgent`](https://github.com/llm-as-a-verifier/TurboAgent).

The original project was a Gemini-first proxy for Claude Code. It generated several candidates, optionally refined context, selected a response with exact majority voting or a Probabilistic Pivot Tournament, and recorded the request for the visualizer.

This fork keeps that workflow, but changes how models execute and how local capacity is managed. It can use Pi's native Codex subscription support, mix Pi and LiteLLM backends, route local OpenAI-compatible models, and share a fixed endpoint capacity between candidate generation and verification.

## Comparison state

This document records the repository state on 2026-09-01.

| Reference | Commit |
| --- | --- |
| Shared fork point | `eeb61be` |
| This fork's `main` | `da0fa5b` |
| Upstream `main` | `31ea8da` |

At this point, the fork has 28 commits that are not in current upstream. Current upstream has five commits that are not in the fork. Relative to the shared fork point, this fork changes 55 files with about 9,685 insertions and 366 deletions.

The sections below compare this fork with the shared fork point. [Current upstream differences](#current-upstream-differences) are tracked separately because upstream continued development after the split.

## What the original project already had

The original project provided:

- Anthropic `/v1/messages` proxying
- OpenAI `/v1/chat/completions` proxying
- concurrent candidate generation
- optional context refinement
- exact-string majority voting
- Probabilistic Pivot Tournament selection through `llm-verifier`
- progress monitoring after a response
- request logs and a browser visualizer

The fork did not invent this request workflow. Most fork work concerns model execution, Pi compatibility, verification policy, endpoint capacity, protocol correctness, tests, and measurement.

## Changes at a glance

| Area | Original project | This fork |
| --- | --- | --- |
| Primary backend | Gemini through LiteLLM | Codex through Pi by default, plus LiteLLM and local endpoints |
| Model execution | Direct LiteLLM calls in `Backend` | Adapter contract with LiteLLM, Pi, and mixed routing |
| Pi support | None | Pi as a client and as a native Codex backend |
| Local endpoints | Basic `base_url` configuration | Forwarded endpoints plus named admission control |
| Verification | Gemini and Vertex-oriented | Configurable hosted or local judges, plus a server60 adapter |
| Majority matching | Exact strings | Exact, normalized, or semantic modes |
| Capacity control | Unbounded request fan-out | Shared FIFO capacity for candidates and judges |
| Configuration | Working-directory YAML | Explicit path, project config, or XDG global config |
| Tests at fork point | No `tests/` directory | 104 test functions and 108 passing cases at the recorded commit |
| Production measurements | None in the repository | Orchestration and server60 benchmark runners and reports |

## Pi integration

Pi works on both sides of Turbo Agent. These are separate integrations.

### Pi as the client

[`integrations/pi/gen_models_json.py`](integrations/pi/gen_models_json.py) generates a `turbo` provider for Pi's `~/.pi/agent/models.json`. It creates one Pi-visible model for each configured Turbo backend and records context, output-token, reasoning, thinking-level, and image metadata.

Pi then sends requests to Turbo's OpenAI-compatible endpoint. See [the Pi setup in the README](README.md#use-with-pi).

### Pi as the backend

A backend model can declare `executor: pi`. Turbo then executes that model through Pi's native `ModelRuntime` instead of LiteLLM.

The Python adapter and Node companion live in:

- [`turbo_agent/model_execution/pi_adapter.py`](turbo_agent/model_execution/pi_adapter.py)
- [`turbo_agent/model_execution/pi_companion.mjs`](turbo_agent/model_execution/pi_companion.mjs)

The companion owns Pi authentication and model discovery. Python sends complete, stream, cancellation, tool, context, and reasoning requests over versioned NDJSON stdio. OAuth credentials stay in the Node process and use Pi's existing auth store.

A real `openai-codex` completion through this path returned the expected response without exposing credentials to Python.

## Model execution adapters

The original `Backend` called LiteLLM directly. Provider identity, API keys, endpoint URLs, token fields, reasoning controls, and response conversion were mixed into the request pipeline.

[`turbo_agent/model_execution/`](turbo_agent/model_execution/) now owns those concerns behind three operations:

1. complete one candidate;
2. stream one candidate;
3. close owned resources.

The package includes:

- a LiteLLM executor;
- a Pi `ModelRuntime` executor;
- a router for configurations that use both;
- normalized results, stream events, usage, and errors;
- per-candidate request isolation;
- cancellation and provider-stream cleanup.

The request pipeline no longer owns provider credentials, wire model IDs, endpoint URLs, or provider-specific reasoning parameters. See the [model execution design](docs/design/model-execution.md) and [ADR-0001](docs/adr/0001-model-execution-seam.md).

This extraction also fixed a concrete bug. The original configuration loaded `base_url`, but candidate calls did not consistently receive it. Local vLLM, SGLang, llama.cpp, and other OpenAI-compatible endpoints now receive the configured URL.

## Verification changes

### Configurable judges

The verifier can use Gemini, hosted DeepSeek, OpenRouter, hosted OpenAI, or an arbitrary OpenAI-compatible endpoint. Provider prefixes identify the client adapter and are removed from the wire model name.

If an enabled verifier does not name a judge, Turbo uses the first backend model and copies its provider, API key, and endpoint settings.

`verifier.enabled: false` now disables verification. The original parser enabled verification whenever the section existed, even when it explicitly contained `enabled: false`.

### Majority modes

The fork supports three agreement modes:

- `exact` compares the complete formatted response;
- `normalized` ignores prose case, punctuation, and whitespace while keeping tool calls structurally strict;
- `semantic` clusters prose embeddings while still requiring matching tool calls.

Semantic mode falls back to normalized comparison if its embedding endpoint fails.

Use normalized mode for agent work. Server60 testing found that semantic mode can group distinct long technical answers. Four different architecture answers produced pairwise cosine similarities between 0.9168 and 0.9457, enough to create a false majority at threshold 0.92. Tool-call protection does not solve semantic collapse in long prose.

The implementation and tests live in:

- [`turbo_agent/verifier/verifier.py`](turbo_agent/verifier/verifier.py)
- [`tests/test_verifier_majority.py`](tests/test_verifier_majority.py)

### Server60 judge adapter

Server60 exposed logprobs, but one-token score probes placed their score letter in hidden reasoning and left `message.content` empty. Unadapted `llm-verifier` calls therefore returned `0.500 / 0.500` for every pair.

[`turbo_agent/verifier/server60_judge_adapter.py`](turbo_agent/verifier/server60_judge_adapter.py) applies two request policies:

- comparison generations use x-high reasoning and a configurable output ceiling;
- one-token score probes disable thinking so the score letter appears in content.

The adapter also limits judge concurrency, propagates request deadlines, disables hidden compatibility retries, and raises admission or provider failures so `Backend` can use its first-valid-candidate fallback.

A live Paris-versus-Berlin comparison returned scores of about 0.7281 versus 0.2719 and selected Paris.

## Shared endpoint admission

The original project had no process-wide endpoint capacity. Candidate requests and judge calls could oversubscribe the same local server independently.

[`turbo_agent/endpoint_admission.py`](turbo_agent/endpoint_admission.py) adds named endpoints with:

- one shared active count;
- FIFO queueing;
- bounded queue size;
- queue deadlines;
- provider request deadlines;
- cancellation cleanup;
- deterministic slot release;
- active and queued diagnostics.

Candidate executors and synchronous judge calls use the same registry. A configured server60 deployment can therefore enforce four active calls across generation and verification instead of four calls per subsystem.

Endpoint-bound provider streams stay admitted until the proxy, `Backend`, execution adapter, and provider iterator all close. Client disconnects no longer leave a slot waiting for garbage collection.

Startup validation rejects common ways to bypass the capacity domain, including conflicting direct base URLs, matching default-judge URLs, and provider-specific API-base environment variables that point at the named endpoint.

See [the endpoint admission design](docs/design/endpoint-admission.md) and [ADR-0002](docs/adr/0002-endpoint-admission-and-server60-judge.md).

## Client and protocol fixes

The fork includes the following compatibility fixes:

- Client `max_tokens` and `max_completion_tokens` intent survives proxy conversion.
- The configured model cap remains the final output ceiling.
- Client reasoning effort, adaptive thinking, thinking budgets, and explicit thinking disablement override YAML defaults consistently.
- Anthropic thinking budgets are only sent to Anthropic providers.
- Unsupported provider parameters are dropped instead of failing the entire request.
- Per-model temperature and token defaults survive concurrent candidate generation.
- Empty assistant turns with no text or tool calls are removed before provider dispatch.
- URL images and images nested inside Anthropic tool results remain visible to OpenAI-compatible providers.
- Verified response replay preserves tool calls and complete Anthropic SSE framing.
- OpenAI streaming responses echo the model requested by the client.
- Pi model metadata advertises reasoning and thinking-level support.
- Provider streams close when clients disconnect or request deadlines expire.

### Local Anthropic token counting

Turbo handles `POST /v1/messages/count_tokens` locally. It estimates text at roughly four characters per token and assigns a 1,600-token floor to each image, including images inside tool results.

This prevents count requests from falling through to `api.anthropic.com` when Turbo uses another provider for generation.

## Configuration and packaging

Configuration discovery now follows this order:

1. an explicit `--config PATH`;
2. `./turbo-agent.yaml`;
3. `$XDG_CONFIG_HOME/turbo-agent/turbo-agent.yaml`;
4. `~/.config/turbo-agent/turbo-agent.yaml`.

Turbo loads `.env` from the selected config's directory. A project config replaces the global config rather than merging with it. `turbo-agent check` and the Pi model generator follow the same discovery rules.

Packaging changes include:

- development extras with `pytest` and `pytest-asyncio`;
- direct `google-genai` and `openai` dependencies;
- checked-in visualizer build artifacts;
- design documents and accepted ADRs;
- regression tests for Pi, model execution, protocols, verification, and admission.

## Current default behavior

The checked-in [`turbo-agent.yaml`](turbo-agent.yaml) configures:

| Setting | Value |
| --- | --- |
| Model | `openai-codex/gpt-5.6-luna` |
| Executor | `pi` |
| Candidate count | `4` |
| Thinking level | `xhigh` |
| Output ceiling | `65536` |
| Verifier | commented out |
| Progress monitor | commented out |

The candidate count needs explanation. When verification is disabled, the public Anthropic and OpenAI request paths execute only the default target. They do not fan out to four candidates. `num_candidates: 4` takes effect when verification is enabled.

A default request therefore runs one x-high Codex completion through Pi, not four completions.

## Benchmarks and decisions

The repository contains two benchmark entry points:

- [`scripts/bench_orchestration.py`](scripts/bench_orchestration.py) measures Turbo request orchestration against local llama.cpp or Codex through Pi;
- [`scripts/bench_server60_production.py`](scripts/bench_server60_production.py) runs the server60 production study implemented in [`scripts/server60_benchmark/`](scripts/server60_benchmark/).

The full server60 setup, measurements, invalidated experiments, and policy recommendations are in the [server60 production report](docs/benchmarks/server60-production-2026-09-01.md).

The main results were:

- Four independent candidate requests increased aggregate throughput without a fourfold latency increase.
- Pivot verification dominated selected-response latency. It accounted for about 91 percent of the measured four-candidate request time.
- vLLM `n=4` modestly improved bounded-answer throughput and time to first token.
- The same vLLM `n=4` mode took 39.604 seconds on a tool-choice trial, compared with 16.174 seconds for independent requests.
- Four active server60 decodes correctly queued a fifth request.
- Cancellation released a slot and admitted a replacement without exceeding four active calls.
- Long x-high responses can consume tens of thousands of reasoning tokens before producing final content.

These measurements support independent request execution as the current default. The repository benchmarks vLLM engine-controlled `n=4`, but the runtime does not use it.

## Known limits

The fork still has the following limits:

- `num_candidates` does not cause fan-out when verification is disabled.
- Semantic majority is unsafe for long agent or technical responses.
- The runtime does not implement vLLM `n=4`, prefix sharing, or adaptive engine selection.
- Context refinement and progress monitoring do not participate in named endpoint admission.
- Pi client registration uses generated `models.json` metadata rather than live extension registration.
- OpenAI-shaped responses do not expose detailed reasoning-token fields.
- Named endpoint admission is opt-in.
- Synchronous judge calls cannot react to outer async cancellation until the provider returns or its request deadline expires.
- Same-origin endpoint checks cannot identify different hostnames that resolve to the same machine.
- [Issue #7](https://github.com/Whamp/TurboAgent/issues/7) tracks a Python compatibility mismatch. The package advertises Python 3.10, while a fresh installation can resolve a LiteLLM release that imports `typing.NotRequired`, which Python 3.10 lacks.

## Current upstream differences

Current upstream is not the same code as the shared fork point. As of the comparison date, it has five commits absent from this fork.

Some upstream work overlaps changes made independently here. Upstream also added configurable verifier routing for Gemini, DeepSeek, and OpenAI-compatible judges.

The following upstream changes still need reconciliation:

1. Upstream rejects Anthropic models as verifiers because Anthropic does not provide the token logprobs required by `llm-verifier`. This fork does not make that startup rejection explicit.
2. Upstream combines multiple system messages into one before OpenAI-compatible dispatch. That exact compatibility fix is absent here.
3. Upstream declares `llm-verifier>=0.2.0`. This fork uses 0.2 APIs such as `max_workers` and `on_error`, but still declares `llm-verifier>=0.1.0`.
4. Upstream's API-key checker probes DeepSeek generation and logprobs. This fork's runtime supports DeepSeek judging, but `turbo-agent check` lacks that probe.
5. Upstream documents opencode setup. This fork has not incorporated that documentation.

Do not merge upstream blindly. Both branches changed verifier construction, configuration, conversion, dependencies, and README content. Reconcile those areas with tests.

## Where to read next

| Question | Source |
| --- | --- |
| How do I run Turbo? | [`README.md`](README.md) |
| How does Pi connect as a client? | [Pi setup in `README.md`](README.md#use-with-pi) |
| How does backend execution work? | [`docs/design/model-execution.md`](docs/design/model-execution.md) |
| How is local endpoint capacity enforced? | [`docs/design/endpoint-admission.md`](docs/design/endpoint-admission.md) |
| Why were these execution boundaries chosen? | [`docs/adr/0001-model-execution-seam.md`](docs/adr/0001-model-execution-seam.md) and [`docs/adr/0002-endpoint-admission-and-server60-judge.md`](docs/adr/0002-endpoint-admission-and-server60-judge.md) |
| What did server60 testing find? | [`docs/benchmarks/server60-production-2026-09-01.md`](docs/benchmarks/server60-production-2026-09-01.md) |
| What terms does the project use? | [`CONTEXT.md`](CONTEXT.md) |
| Which behaviors have regression coverage? | [`tests/`](tests/) |

## Refreshing this document

Fetch both remotes before updating the comparison:

```bash
git fetch origin main
git fetch upstream main

base=$(git merge-base origin/main upstream/main)
git rev-list --left-right --count origin/main...upstream/main
git diff --stat "$base"..origin/main
```

Then update:

- the comparison date and three commit IDs;
- the fork-only and upstream-only commit counts;
- the diff statistics;
- current defaults from `turbo-agent.yaml`;
- test counts and validation evidence;
- implemented features and known limits;
- the current upstream reconciliation list.

Run the test suite after reconciling any source changes:

```bash
uv run pytest
```
