# server60 production inference study

Date: 2026-09-01

## Executive verdict

Turbo Agent's model-execution seam works against the production server60 endpoint. The server safely supports four active generations, queues excess work, and releases capacity promptly after cancellation.

The product conclusions are narrower:

1. **Keep normalized majority voting. Do not use embedding-based semantic majority for long agent responses.** At the current `0.92` threshold, four materially different 13k–28k-character architecture answers formed a semantic majority and would have skipped verification.
2. **Do not enable the current llm-verifier path against this reasoning endpoint.** Its one-token score probes arrive in `message.reasoning`, while llm-verifier reads `message.content`, producing uninformative `0.500` scores. A benchmark-only compatibility adapter fixed this.
3. **Treat vLLM `n=4` as an opt-in engine strategy, not a universal replacement for four requests.** It improved a short text workload but regressed a tool-call workload.
4. **Add endpoint-aware admission control to Turbo.** server60 enforces four active sequences correctly, but an uncoordinated fifth request waited 43.5 seconds inside vLLM. Turbo should own queue policy, deadlines, and observability.
5. **Do not default every x-high request to four candidates.** The quality corpus found no correctness uplift because every oracle-backed candidate was already correct. One hard four-candidate problem consumed 75,845 output tokens and took 24.8 minutes.

## Production system under test

| Field | Value |
|---|---|
| Endpoint | `http://192.168.0.251:30002/v1` |
| Served model | `qwen3.8-flash-next-intel-autoround-w4a16` |
| Architecture | `Qwen4ExpForConditionalGeneration`, MoE |
| Logical parameters | 123,958,298,771 |
| MoE shape | 512 experts, top 10, 48 layers |
| Approximate active parameters | 5.5B, derived from all non-routed parameters plus 10/512 routed-expert parameters |
| Hidden size | 2,560 |
| Context | 262,144 tokens |
| Weight format | AutoRound 4-bit, group size 128, symmetric; W4A16 |
| PLE | BF16 direct mmap CPU offload |
| KV cache | FP8 E4M3 |
| GPUs | 4× RTX 3090 24GB, tensor parallel 4, expert parallel enabled |
| GPU residency | 23.5GB per GPU during service operation |
| GPU power limit | 230W per GPU |
| vLLM | `0.1.dev20073+g8e685d198` |
| PyTorch | `2.13.0+cu130` |
| Image digest | `sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef` |
| Model revision | `861536dda5bcb208376fc4cd879b2bf76bece9fe` |
| Active-sequence limit | 4 |
| Max batched tokens | 1,024 |
| Prefix caching | Enabled |
| MTP speculative decoding | Disabled |

The production Compose contract is `/home/will/inference/runtime/qwen38-production/compose.yml` on server60.

## Measurement rules

- Quality requests used `reasoning_effort: xhigh`.
- Quality ceilings were 65,536 output tokens where the task could require long reasoning.
- A completion counted only when it produced a normal terminal response. Hidden reasoning was not treated as a usable final answer.
- The useful-response engine comparison used naturally stopping one-sentence answers under a 16,384-token ceiling.
- Fixed lower caps were used only as deliberate decode-load generators for capacity tests, never as quality evidence.
- No phase intentionally exceeded four active generations except the controlled fifth-request queue test.
- Judge HTTP calls were capped at four concurrent calls in the corrected benchmark.
- Latencies are client-observed over the LAN.
- Percentiles over five samples are descriptive, not population estimates.

The repeatable runner is [`scripts/bench_server60_production.py`](../../scripts/bench_server60_production.py).

## Turbo execution seam

The LiteLLM model executor successfully routed target-scoped `base_url`, credentials, x-high reasoning intent, token limits, messages, and tools to server60. Four configured candidate targets executed concurrently through the extracted model-execution seam.

One important API behavior remains: when verification is disabled, Turbo's public OpenAI and Anthropic paths execute only the default target. `num_candidates` fans out through `_gather_completions` only when selection is enabled. Benchmarks that need candidate generation without selection must call the candidate-generation seam directly.

## Engine-controlled `n=4`

### Completed short text workload

Five samples asked for one short sentence. All requests stopped naturally.

| Mode | Mean wall | Mean TTFT | P95 TTFT | Aggregate output tok/s | Mean unique final answers |
|---|---:|---:|---:|---:|---:|
| Four independent HTTP requests | 2.060s | 0.477s | 0.782s | 136.6 | 3.8 |
| One HTTP request with `n=4` | 1.955s | 0.258s | 0.268s | 153.7 | 4.0 |

For this workload, engine-controlled `n=4` reduced wall time by 5%, increased aggregate output throughput by 12.5%, and reduced mean TTFT by 46%. Final-answer diversity was preserved.

### Tool-call workload

Both modes returned four correctly shaped `run_bash` tool calls. Every command used `--force-with-lease`; none used plain `--force`.

| Mode | Wall time |
|---|---:|
| Four independent HTTP requests | 16.174s |
| One HTTP request with `n=4` | 39.604s |

The `n=4` request generated one unusually long shell command and waited for all four choices. This erased the short-text advantage and made `n=4` 2.45× slower.

### Decision

Keep the current independent-request executor as the default. Add `n=4` only behind an explicit, endpoint-specific strategy with:

- text and tool-call equivalence tests;
- per-workload latency and token accounting;
- cancellation behavior for individual choices;
- a fallback to independent requests;
- no claim of universal speedup.

The engine already uses TP=4, expert parallelism, prefix caching, and a four-sequence scheduler. MTP is disabled and remains a separate server-side experiment requiring its own quality-equivalence gate.

## Capacity, queueing, and cancellation

The controlled queue test launched four 2,048-token decode loads, waited until vLLM reported four running requests, and then submitted a fifth.

| Metric | Result |
|---|---:|
| Running before fifth submission | 4 |
| Maximum running | 4 |
| Maximum waiting | 1 |
| First four TTFTs | 0.406s, 0.796s, 0.796s, 0.796s |
| Fifth TTFT | 43.537s |
| Fifth total time | 44.134s |

vLLM did not exceed its declared capacity. It queued the fifth request until a slot became free.

The cancellation test launched four long-running streams, canceled one after all four began, and immediately submitted a replacement.

| Metric | Result |
|---|---:|
| Canceled stream closed | 20.5ms |
| Replacement TTFT after cancellation | 295ms |
| Replacement total time | 1.024s |
| Metrics returned to idle after cleanup | 78ms |

### Decision

The engine's capacity enforcement is sound, but Turbo should not outsource product queue policy to vLLM. A shared endpoint scheduler should own:

- a four-slot capacity model;
- bounded pending work;
- queue discipline and fairness;
- absolute deadlines;
- cancellation and slot-release observability;
- overload errors before requests spend tens of seconds invisibly queued.

## X-high quality and reasoning cost

### Oracle-backed corpus

Five validator-backed tasks covered number theory, constrained lattice paths, Git lease safety, async cleanup, and a tool call. Every one of the four candidates was correct on every task. Exact majority fired in all five cases.

Representative reasoning-token counts:

| Task | Candidate reasoning tokens | Candidate wall time |
|---|---|---:|
| Coprime count | 1,107 / 1,763 / 2,388 / 1,204 | 42.5s |
| Constrained lattice paths | 5,302 / 8,485 / 3,136 / 2,647 | 145.5s |
| Git lease choice | 292 / 721 / 177 / 333 | 12.6s |
| Async cleanup choice | 62 / 120 / 176 / 127 | 4.0s |
| Safe force-push tool | 158 / 206 / 253 / 172 | 6.2s |

The candidates often took different reasoning paths but converged on the same final answer.

### Hard combinatorics task

A locally computed dynamic-programming oracle established the answer `25170000` for:

> How many length-10 decimal digit strings, with leading zero allowed, contain exactly three 0 digits and exactly two 1 digits, if no two adjacent digits may be equal?

All four candidates returned the correct answer.

| Metric | Result |
|---|---:|
| Candidate wall time | 1,490.7s |
| Reasoning tokens | 19,413 / 16,804 / 22,388 / 17,196 |
| Total candidate output tokens | 75,845 |
| Majority | Normalized 4/4 |
| Judge calls | 0 |
| Selection uplift | None; first candidate was already correct |

This is the strongest evidence against unconditional four-way generation for every x-high request. It spent four large reasoning budgets without improving the selected result.

### Long open-ended design task

A single candidate needed 27,027 output tokens—19,214 reasoning and 7,813 final-answer tokens—and 480.4 seconds to stop normally.

Four concurrent candidates all stopped normally under a 65,536-token ceiling:

| Candidate | Output tokens | Reasoning tokens | Final answer characters |
|---:|---:|---:|---:|
| 1 | 27,628 | 21,023 | 27,991 |
| 2 | 22,079 | 16,770 | 22,718 |
| 3 | 26,609 | 23,399 | 13,300 |
| 4 | 25,117 | 20,746 | 18,590 |

Wall time was 543.7 seconds. All four final answers were distinct.

A 16,384-token ceiling had previously truncated all four candidates in reasoning and produced no final answer. Those low-ceiling outputs are invalid quality evidence.

### Quality conclusion

The model is strong enough that this small oracle corpus did not expose a selection uplift. The study proves diversity on open-ended work and substantial reasoning variance, not that best-of-four is better than one candidate.

Before making four candidates the default, add a larger difficult corpus where:

- at least some candidates are wrong;
- correctness has an independent oracle;
- tool trajectories run to completion rather than stopping after one tool call;
- selection uplift can be measured against first-candidate accuracy;
- token cost and wall time are part of the acceptance gate.

## Majority voting

### Exact and normalized agreement

Exact majority was highly effective on the oracle-backed tasks because all four final answers agreed. Normalized matching preserves that safe shortcut while tolerating superficial formatting differences.

Tool-call strings remain structurally exact except for whitespace normalization. Commands such as `rm -rf /build` and `rm -rf /tmp/build` do not become equivalent.

### Semantic agreement

The four long design answers had pairwise octen-embed cosine similarities from approximately 0.919 to 0.945.

- At threshold `0.92`, the implementation found a strict semantic majority and skipped the tournament.
- At `0.94`, no strict majority remained.
- At `0.95`, all four answers separated.

The responses discussed the same subject, so high embedding similarity was expected. That does not prove agreement on cancellation ownership, release timing, queue fairness, shell safety, or other consequential details.

Greedy representative clustering also makes the result sensitive to representative order near the threshold. This is unsuitable as a correctness shortcut for long agent outputs.

### Decision

Use `normalized` mode for production majority voting. Keep semantic mode off for:

- shell commands and tool calls;
- code patches;
- long architectural or operational answers;
- any response where a subtle difference can change correctness or safety.

Embeddings remain useful for analysis and retrieval, not as a proof that complex candidates agree.

## Verifier compatibility and cost

### Current incompatibility

server60 supports logprobs, but its reasoning parser returns one-token score-probe letters in `message.reasoning`. llm-verifier reads `message.content`. The unadapted path therefore produced `0.500 / 0.500` scores, even for obvious comparisons.

llm-verifier also currently:

- disables thinking for its main OpenAI comparison generation;
- hard-codes a 4,096-token comparison ceiling;
- leaves one-token continuation probes in the endpoint's default thinking mode.

That is the inverse of the desired contract for this model.

### Corrected benchmark adapter

The study used a benchmark-only client wrapper:

- main comparison: x-high thinking, 16,384-token ceiling;
- one-token score probes: thinking disabled so the score token lands in `content`;
- at most four active judge calls.

On an obvious Paris-versus-Berlin test, this produced approximately `0.728 / 0.272` and selected Paris. The model and logprobs are capable; the generic judge wire contract is wrong.

### Corrected cost profile

Two samples per candidate count used naturally stopping one-sentence candidates and forced the tournament.

| Candidates | Candidate phase | Selection phase | Total | Judge calls |
|---:|---:|---:|---:|---:|
| 1 | 1.316s | 0s | 1.316s | 0 |
| 2 | 1.578s | 7.691s | 9.270s | 6 |
| 4 | 2.135s | 20.699s | 22.834s | 18 |

For N=4, selection consumed 91% of end-to-end latency and about 1,828 judge output tokens per request. The candidates were equivalent one-sentence answers, so final scores stayed close to 0.5.

### Decision

Do not configure server60 as a Turbo judge until judge execution has an explicit adapter or routes through the model-execution seam. The contract needs to own:

- reasoning mode for comparison generation;
- a separate non-thinking mode for one-token score probes;
- comparison token ceiling;
- maximum judge concurrency;
- response-field normalization;
- logprob compatibility checks;
- fallback when scores are non-discriminating.

Majority voting should remain ahead of the tournament. It is the only measured path that preserves four-candidate quality without adding 7–21 seconds of judge latency on short outputs.

## Recommended product policy

### Now

1. Keep server60 reachable through the LiteLLM model-execution adapter.
2. Use x-high reasoning and a realistic 65,536-token ceiling for quality-sensitive work.
3. Use normalized majority voting.
4. Leave semantic majority off for agent responses.
5. Do not use the unadapted server60 verifier.
6. Let vLLM enforce its hard four-sequence limit until Turbo gains shared endpoint admission control.

### Next implementation milestones

1. **Judge execution adapter**
   - Make the corrected reasoning/logprob contract configurable and tested.
   - Cap judge concurrency at endpoint capacity.
2. **Capacity-aware scheduler**
   - Own queueing, deadlines, cancellation, and slot metrics in Turbo.
3. **Optional engine `n=4` executor**
   - Add behind a per-endpoint strategy flag.
   - Preserve tool-call and usage normalization.
   - Retain independent-request fallback.
4. **Adaptive candidate policy**
   - N=1 for ordinary latency-sensitive requests.
   - N=4 for explicitly high-value work or tasks with a measurable validator.
   - Preserve majority voting when N=4 is used.
5. **Quality benchmark suite**
   - Require demonstrated selection uplift before making N=4 the default.
6. **Separate MTP experiment**
   - MTP is currently disabled. Test it on an alternate deployment with distribution-equivalence and tool-call gates before any production switch.

## Invalidated results

The following measurements must not be cited as quality conclusions:

- 192-, 384-, or 512-token caps on open-ended x-high prompts;
- any result that treated reasoning tokens as final response content;
- earlier N=2/N=4 no-verifier Turbo timings, because the public no-verifier path executes only one target;
- unadapted verifier scores of exactly 0.500, because score letters were read from the wrong response field;
- independent-request diversity counts that summed one-choice uniqueness per request instead of comparing final-output hashes across requests.

They were useful for finding benchmark and integration defects, not for judging the model.
