---
status: accepted
---

# Route Backend model calls through a model execution seam

Turbo Agent will route Candidate-producing Backend model calls through one model execution module. The module owns configured model targets, credentials, endpoints, model-specific request translation, output-cap precedence, response normalization, and failure classification. Concurrent inference will continue to decide how many Candidates to run and when to run them; Verification will continue to choose the response. This seam supports both the existing LiteLLM path and Pi's native `ModelRuntime` without reimplementing Codex subscription auth or rewriting the Python request pipeline.

## Considered options

- Keep calling LiteLLM from `Backend`. Rejected because endpoint, credential, and model-capability rules would remain spread across callers, and Pi's Codex subscription is not a generic API key.
- Reimplement Pi's Codex OAuth flow in Python. Rejected because Turbo Agent would own token storage and refresh behavior already owned by Pi.
- Rewrite Turbo Agent in TypeScript around Pi. Rejected because the migration would mix a language rewrite with a model-routing change and put the existing Verification behavior at risk.

## Consequences

The first adapters will be LiteLLM and a small Node companion that delegates to Pi `ModelRuntime` over versioned stdio. Existing backend configuration will default to LiteLLM. A Pi target must opt in and will not expose OAuth credentials to Python. Judge, Context refinement, Progress monitor, and global capacity scheduling remain outside the first extraction.

The detailed interface and migration plan are in [the model execution design](../design/model-execution.md).
