# Turbo Agent

![Turbo Agent visualizer](screenshot.png)

Turbo Agent is the Claude Code plugin for LLM-as-a-Verifier. It implements an LLM API proxy that improves response quality through concurrent inference, verification, and refinement. It sits between your client (Claude Code, Codex, etc.) and the LLM provider, sending multiple parallel requests and selecting the best response with a **Probabilistic Pivot Tournament (PPT)** scored by a fine-grained logprob verifier.

```
Client request
    │
[Context Refinement]   (optional) rewrite/augment the system prompt for clarity
    │
[Concurrent Inference] send N parallel candidates to the backend model
    │
[Verification]         pivot tournament over the candidates, pick the best one
    │
Best response → Client
```

Verification uses the pivot tournament from the [`llm-verifier`](https://pypi.org/project/llm-verifier/) package to pick the best of `N` candidates.

## Install

```bash
pip install turbo-agent
```

Or from source:

```bash
pip install -e .
```

## Setup

For turbo agent to work, you need a `turbo-agent.yaml`. You can copy the reference file in this repo.

`turbo-agent.yaml` references keys with `$VAR_NAME` syntax. The recommended way to provide them is a `.env` file in the project root (next to `turbo-agent.yaml`) — the proxy loads it automatically on startup. Copy the committed template and fill in your keys:

```bash
cp .env.example .env
# then edit .env
```

```bash
# .env
VERTEX_API_KEY=your-vertex-key     # preferred for Gemini 2.5 logprobs (verifier)
# GEMINI_API_KEY=your-gemini-key     # used by gemini/ models (AI Studio)
# OPENAI_API_KEY=...               # only if you route to openai/ models
# ANTHROPIC_API_KEY=...            # only if you route to anthropic/ models
```

`.env` is gitignored; `.env.example` is committed as the template. Keys already
exported in your shell environment work too and take nothing extra. The verifier
and progress monitor use Gemini **logprobs**, which are best served by a Vertex
AI key (`VERTEX_API_KEY` + `provider: vertex_ai` in the config); a plain
`GEMINI_API_KEY` also works for the `gemini/` backend models.

Verify your keys are valid:

```bash
turbo-agent check
```

It checks every supported provider (Gemini, Vertex AI, OpenAI, Anthropic) and reports each with ✅ / ❌ / ⚠️ / ⚪️, flagging which keys your config actually uses.

## Run

```bash
turbo-agent                   # default port 8888
turbo-agent -p 9000           # custom port
```

### Use with Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8888 claude
```

### Use with OpenAI-compatible clients

```bash
export OPENAI_API_BASE=http://localhost:8888/v1
```

### Use with Pi

[Pi](https://github.com/earendil-works/pi-coding-agent) connects through the
proxy as an OpenAI-compatible client, which works with any backend model
(Gemini, OpenAI, Anthropic, OpenRouter, Zai, Kimi, ...) in your
`turbo-agent.yaml`.

Pi keeps custom providers in `~/.pi/agent/models.json`. Generate the `turbo`
provider block straight from your config:

```bash
python integrations/pi/gen_models_json.py turbo-agent.yaml --merge ~/.pi/agent/models.json
```

That registers one `turbo` model per backend model (metadata such as context
window and max tokens come from your config, so what Pi shows matches what the
proxy runs). Then, with the proxy running:

```bash
turbo-agent               # in a directory containing turbo-agent.yaml
pi                        # select the model with /model: turbo/<backend-model>
```

Notes:

- The proxy ignores the model id a client requests — it always runs the
  backend models from `turbo-agent.yaml` — but it now echoes the requested id
  back in responses, so Pi displays the model you picked.
- The proxy ignores client API keys; the `apiKey` in the generated provider is
  a placeholder. If Pi hides the models until auth is resolved, save any key
  with `/login turbo` or pass `--api-key` when selecting the model.
- Pi always streams. With a verifier configured, Turbo Agent gathers all
  candidates and verifies before replaying the best response as a stream, so
  each turn costs `num_candidates` full responses plus verifier calls. Tune
  `num_candidates` (3 is the reference default) and `majority_voting: true`
  to control cost/latency.
- The verifier judge is configurable — it is not tied to Gemini. Set
  `verifier.model.name` to any litellm-style model: `openrouter/...`
  (defaults to `https://openrouter.ai/api/v1`), `deepseek/...` (hosted DeepSeek),
  or `openai/...` with a `base_url` pointing at a local vLLM/SGLang endpoint
  (full fine-grained logprob reward needs a server that exposes logprobs;
  OpenRouter degrades to parsing the judge's written score when the upstream
  provider does not).
- Pi counts tokens locally; the proxy also answers `/v1/messages/count_tokens`
  with an approximate local count so token-counting clients never leak a
  request to api.anthropic.com.

## Configuration

Edit `turbo-agent.yaml`. API keys can reference environment variables with `$VAR_NAME` syntax. See the reference `turbo-agent.yaml` file for reference and usage.

### Model prefixes

| Prefix | Provider |
|--------|----------|
| `gemini/` | Google Gemini |
| `openai/` | OpenAI |
| `anthropic/` | Anthropic |
| (none) | OpenAI-compatible endpoint |

## API endpoints

| Endpoint | Format |
|----------|--------|
| `POST /v1/messages` | Anthropic |
| `POST /v1/messages/count_tokens` | Anthropic (approximate local count) |
| `POST /v1/chat/completions` | OpenAI |
| `GET /v1/models` | OpenAI |
| `GET /visualizer` | Pipeline visualizer UI |
| `*` | Upstream passthrough to api.anthropic.com |

## Visualizer

A built-in web UI at `http://localhost:8888/visualizer` shows the pipeline DAG for each request — context refinement, all candidate responses, the pairwise tournament comparisons and scores, and the final selection.

To build the frontend (requires Node.js):

```bash
cd frontend
yarn install
yarn build
```

## Publish to PyPI

```bash
cd frontend && yarn build && cd ..
pip install build twine
rm -rf dist
python -m build
twine check dist/*
twine upload dist/*
```