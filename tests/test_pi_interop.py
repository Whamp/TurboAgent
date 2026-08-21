"""Pi-interop integration tests.

These tests exercise the proxy's public HTTP surface with a stub OpenAI
backend, proving the wire behavior a Pi-style client (Anthropic Messages or
OpenAI chat completions, always streaming) depends on:

- full Anthropic SSE stream shape (message_start .. message_stop, tool_use)
- requested-model echo instead of the backend model name
- max_tokens clamped to the config cap
- empty assistant turns dropped (no null-content messages upstream)
- images inside tool_result preserved for the backend
- /v1/messages/count_tokens handled locally (never leaks upstream)
- GET /v1/models carries metadata for client-side model registration
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from turbo_agent.proxy import ProxyServer
from turbo_agent.utils import Config

# ---------------------------------------------------------------------------
# Stub OpenAI backend
# ---------------------------------------------------------------------------

CANNED = {
    "id": "chatcmpl-stub",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "dummy",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "I will inspect the repo first.",
                "tool_calls": [
                    {
                        "id": "call_stub_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": json.dumps({"command": "ls -la"}),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
}


class StubBackend:
    def __init__(self):
        self.requests = []
        self._lock = threading.Lock()

    def _handle(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        try:
            req = json.loads(body)
            with self._lock:
                self.requests.append(req)
        except Exception:
            pass
        resp = json.dumps(CANNED).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(resp)))
        handler.end_headers()
        handler.wfile.write(resp)


_stub = StubBackend()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _stub._handle(self, body)

    def do_GET(self):
        resp = json.dumps({"object": "list", "data": [{"id": "dummy"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


@pytest.fixture(scope="module")
def stub_server():
    _stub.requests.clear()
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture()
def proxy(stub_server, tmp_path, monkeypatch):
    monkeypatch.setenv("DUMMY_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", stub_server)
    config_text = f"""
log_dir: test_logs

backend:
  models:
    - name: openai/dummy
      api_key: $DUMMY_KEY
      num_candidates: 3
      max_tokens: 2048
      thinking: 2048

verifier:
  model:
    name: openai/dummy
    api_key: $DUMMY_KEY
  majority_voting: true
  method:
    name: pivot_tournament
    pivots: 1
    n_verifications: 1
    seed: 0
    note: no ground truth
    criteria:
      - name: Task Success
        description: did it solve the task
"""
    cfg_path = tmp_path / "turbo-agent.yaml"
    cfg_path.write_text(config_text)
    _stub.requests.clear()
    server = ProxyServer(config=Config(str(cfg_path)))
    with TestClient(server.app) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PI_BODY = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 4096,
    "stream": True,
    "system": [
        {"type": "text", "text": "You are a coding agent.",
         "cache_control": {"type": "ephemeral"}}
    ],
    "tools": [
        {"name": "Bash", "description": "run a command",
         "input_schema": {"type": "object",
                          "properties": {"command": {"type": "string"}},
                          "required": ["command"]}}
    ],
    "messages": [
        {"role": "user", "content": "list the directory"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "I should use Bash",
             "signature": "sig123"},
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu1", "name": "Bash",
             "input": {"command": "pwd"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "/home/will"},
        ]},
    ],
}


def parse_sse(text: str):
    """Yield (event, data_dict) pairs from an SSE response body."""
    events = []
    event_name, data_lines = None, []
    for line in text.splitlines():
        if line.startswith("event: "):
            event_name = line[len("event: "):]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: "):])
        elif line == "":
            if event_name is not None:
                events.append((event_name, json.loads("\n".join(data_lines))))
            event_name, data_lines = None, []
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_anthropic_stream_pi_shape(proxy):
    resp = proxy.post(
        "/v1/messages",
        json=PI_BODY,
        headers={"x-api-key": "sk-ant-dummy",
                 "anthropic-version": "2023-06-01"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(resp.text)
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    assert "content_block_start" in names
    assert "message_delta" in names

    start_msg = events[0][1]["message"]
    assert start_msg["model"] == "claude-sonnet-4-5"  # requested model echoed

    tool_starts = [
        e for name, e in events
        if name == "content_block_start" and e["content_block"].get("type") == "tool_use"
    ]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["name"] == "Bash"

    delta = [e for name, e in events if name == "message_delta"]
    assert delta[0]["delta"]["stop_reason"] == "tool_use"


def test_three_concurrent_candidates(proxy):
    proxy.post("/v1/messages", json=PI_BODY,
               headers={"x-api-key": "k"})
    assert len(_stub.requests) == 3
    for req in _stub.requests:
        assert req["model"] == "dummy"
        assert req.get("stream") is None  # candidates gathered non-streaming


def test_upstream_conversion_details(proxy):
    proxy.post("/v1/messages", json=PI_BODY,
               headers={"x-api-key": "k"})
    req = _stub.requests[0]
    messages = req["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool"]
    # cache_control stripped, thinking block dropped
    assert messages[0]["content"] == "You are a coding agent."
    asst = messages[2]
    assert asst["content"] == "Let me check."
    assert asst["tool_calls"][0]["function"]["name"] == "Bash"
    tool = messages[3]
    assert tool["role"] == "tool"
    assert tool["content"] == "/home/will"


def test_empty_assistant_turn_dropped(proxy):
    body = {
        "model": "m", "max_tokens": 100, "stream": True,
        "messages": [
            {"role": "user", "content": "think hard"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hidden reasoning",
                 "signature": "s"},
            ]},
            {"role": "user", "content": "what now?"},
        ],
    }
    resp = proxy.post("/v1/messages", json=body, headers={"x-api-key": "k"})
    assert resp.status_code == 200
    req = _stub.requests[0]
    roles = [m["role"] for m in req["messages"]]
    assert "assistant" not in roles  # null-content assistant never sent


def test_tool_result_image_preserved(proxy):
    body = {
        "model": "m", "max_tokens": 100, "stream": True,
        "messages": [
            {"role": "user", "content": "screenshot?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu9", "name": "Bash",
                 "input": {"command": "screenshot"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu9", "content": [
                    {"type": "text", "text": "here is the screenshot:"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": "aGVsbG8=",
                    }},
                ]},
            ]},
        ],
    }
    resp = proxy.post("/v1/messages", json=body, headers={"x-api-key": "k"})
    assert resp.status_code == 200
    req = _stub.requests[0]
    all_text = json.dumps(req)
    assert "data:image/png;base64,aGVsbG8=" in all_text


def test_max_tokens_clamped(proxy):
    body = dict(PI_BODY)
    body["max_tokens"] = 999999
    proxy.post("/v1/messages", json=body, headers={"x-api-key": "k"})
    for req in _stub.requests:
        assert req.get("max_tokens", 999999) <= 2048


def test_count_tokens_never_leaks(proxy):
    body = {
        "model": "m",
        "system": "abcdefgh" * 100,
        "messages": [{"role": "user", "content": "hello world"}],
    }
    resp = proxy.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 0
    # The stub backend must never see the request.
    assert len(_stub.requests) == 0


def test_models_metadata(proxy):
    resp = proxy.get("/v1/models")
    assert resp.status_code == 200
    models = resp.json()["data"]
    assert len(models) == 1
    m = models[0]
    assert m["id"] == "openai/dummy"
    assert m["max_tokens"] == 2048
    assert m["reasoning"] is True
    assert m["context_window"] > 0
    assert "cost" in m


def test_models_metadata_reasoning_without_yaml_thinking(tmp_path):
    """Pi must be able to send thinking controls even when YAML omits thinking."""
    backend = _make_backend(tmp_path, """
backend:
  models:
    - name: openai/dummy
      api_key: x
      max_tokens: 8192
""")
    meta = backend.get_models_response()["data"][0]
    assert meta["reasoning"] is True


def test_openai_stream_replay_echoes_model(proxy):
    body = {
        "model": "my-client-model",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    resp = proxy.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert "data: [DONE]" in resp.text
    first = json.loads(resp.text.splitlines()[0][len("data: "):])
    assert first["model"] == "my-client-model"



# ---------------------------------------------------------------------------
# Client-limit/thinking survival + judge configurability (Grok review follow-up)
# ---------------------------------------------------------------------------

def _make_backend(tmp_path, text: str):
    """Build a Backend from YAML text in a throwaway dir."""
    from turbo_agent.proxy.backend import Backend

    p = tmp_path / "turbo-agent.yaml"
    p.write_text(text)
    return Backend(Config(str(p)))


def test_candidates_keep_client_limits_and_effort(tmp_path, monkeypatch):
    """H1 regression: the per-candidate YAML overlay must not undo the
    client's clamped max_tokens or its reasoning effort."""
    import asyncio


    backend = _make_backend(tmp_path, """
log_dir: test_logs

backend:
  models:
    - name: openai/dummy
      api_key: x
      num_candidates: 3
      max_tokens: 65536
      thinking: high

verifier:
  model: {name: openai/dummy, api_key: x}
  majority_voting: true
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1, seed: 0}
""")
    seen = []

    async def fake(**kw):
        seen.append({
            k: kw.get(k)
            for k in ("max_tokens", "max_completion_tokens",
                      "reasoning_effort", "thinking_budget")
        })
        return {
            "id": "x", "object": "chat.completion", "created": 1,
            "model": "dummy",
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }

    backend._execution.executor._complete_fn = fake
    asyncio.run(backend.complete_anthropic(json.dumps({
        "model": "claude-x",
        "max_tokens": 32000,
        "thinking": {"type": "adaptive", "effort": "low"},
        "messages": [{"role": "user", "content": "hi"}],
    })))
    assert len(seen) == 3
    for req in seen:
        assert req["max_tokens"] == 32000        # client value, not 65536
        assert req["reasoning_effort"] == "low"  # client, not YAML "high"


def test_no_dual_token_key_and_budget_floor(tmp_path):
    """H4: no max_completion_tokens injected on the Anthropic path; H2: the
    thinking budget is strictly below max_tokens or dropped."""
    backend = _make_backend(tmp_path, """
backend:
  models:
    - name: openai/dummy
      api_key: x
      max_tokens: 65536
      thinking: 2048
""")
    request, _ = backend._build_anthropic_params(
        {"model": "m", "max_tokens": 32000, "messages": []})
    params = backend._execution.executor._build_params(
        backend._execution.default_target, request)
    assert params["max_tokens"] == 32000
    assert "max_completion_tokens" not in params  # one field on the wire

    # tiny client cap: no thinking budget fits strictly below it -> dropped
    tiny = _make_backend(tmp_path, """
backend:
  models:
    - name: openai/dummy
      api_key: x
      max_tokens: 65536
      thinking: 2048
""")
    tiny_request, _ = tiny._build_anthropic_params(
        {"model": "m", "max_tokens": 200, "messages": []})
    tiny_params = tiny._execution.executor._build_params(
        tiny._execution.default_target, tiny_request)
    assert "thinking" not in tiny_params


def test_verifier_judge_is_configurable(tmp_path, monkeypatch):
    """A non-Gemini verifier model builds an OpenAI-compatible judge client
    (OpenRouter by default; local/base_url honored); deepseek builds the
    DeepSeek API client."""
    from turbo_agent.verifier import Verifier

    def cfg(name, base_url="", key="k"):
        p = tmp_path / f"j{name.replace('/', '_')}.yaml"
        p.write_text(f"""
backend:
  models:
    - name: openai/dummy
      api_key: x

verifier:
  model:
    name: {name}
    api_key: {key}
    base_url: {base_url}
  method: {{name: pivot_tournament, pivots: 1, n_verifications: 1}}
""")
        return Config(str(p)).verifier_config

    v = Verifier(cfg("openrouter/deepseek/deepseek-v4-pro-0813"))
    assert v.model_id == "deepseek/deepseek-v4-pro-0813"
    client = v.client
    assert hasattr(client, "chat")                  # OpenAI-compatible
    assert "openrouter.ai" in str(client.base_url)

    v2 = Verifier(cfg("openai/local-judge",
                      "http://100.92.238.117:30000/v1"))
    assert v2.model_id == "local-judge"
    assert str(v2.client.base_url) == "http://100.92.238.117:30000/v1/"

    v3 = Verifier(cfg("deepseek/deepseek-v4-flash"))
    assert getattr(v3.client, "_llm_verifier_deepseek", False)


def test_verifier_judge_defaults_to_backend_model(tmp_path, monkeypatch):
    """An enabled verifier without a named judge falls back to the backend
    candidate model (the active session agent), not to disabled."""
    monkeypatch.setenv("DUMMY_KEY", "dummy")
    p = tmp_path / "turbo-agent.yaml"
    p.write_text("""
backend:
  models:
    - name: openrouter/deepseek/deepseek-v4-flash-0731
      api_key: $DUMMY_KEY
      num_candidates: 3

verifier:
  majority_voting: true
  method: {name: pivot_tournament, pivots: 2, n_verifications: 1}
""")
    vc = Config(str(p)).verifier_config
    assert vc is not None
    assert vc.model.name == "openrouter/deepseek/deepseek-v4-flash-0731"
    assert vc.model.api_key is not None       # key from the backend entry
    assert vc.majority_voting is True

    # verifier section absent -> still disabled
    p2 = tmp_path / "no-verifier.yaml"
    p2.write_text("backend:\n  models:\n    - name: openai/dummy\n      api_key: x\n")
    assert Config(str(p2)).verifier_config is None

    # fallback copies base_url and provider from the backend model (local
    # vLLM / Vertex) so the judge hits the same server the candidates do.
    p3 = tmp_path / "local-backend.yaml"
    p3.write_text("""
backend:
  models:
    - name: openai/local-judge
      api_key: k
      base_url: http://127.0.0.1:30000/v1
      provider: vertex_ai
      num_candidates: 3

verifier:
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1}
""")
    vc3 = Config(str(p3)).verifier_config
    assert vc3 is not None
    assert vc3.model.name == "openai/local-judge"
    assert vc3.model.base_url == "http://127.0.0.1:30000/v1"
    assert vc3.model.provider == "vertex_ai"


def test_config_discovery_precedence(tmp_path, monkeypatch):
    """Project ./turbo-agent.yaml beats the global default; global is used
    when no project file exists; an explicit --config path always wins."""
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    global_cfg = Config.global_dir() / "turbo-agent.yaml"
    global_cfg.parent.mkdir(parents=True)
    global_cfg.write_text(
        "backend:\n  models:\n    - name: openai/global\n      api_key: g\n")

    # no project file in cwd -> global default
    monkeypatch.chdir(tmp_path)
    assert Config().models[0]["name"] == "openai/global"

    # project file exists -> it wins
    (tmp_path / "turbo-agent.yaml").write_text(
        "backend:\n  models:\n    - name: openai/project\n      api_key: p\n")
    assert Config().models[0]["name"] == "openai/project"

    # explicit path beats both
    other = tmp_path / "elsewhere.yaml"
    other.write_text(
        "backend:\n  models:\n    - name: openai/explicit\n      api_key: e\n")
    assert Config(str(other)).models[0]["name"] == "openai/explicit"

    # nothing anywhere -> clear error listing the candidates
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-xdg"))
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(other_dir)
    with pytest.raises(FileNotFoundError, match="No turbo-agent.yaml found"):
        Config()


_CANNED_OK = {
    "id": "x", "object": "chat.completion", "created": 1, "model": "dummy",
    "choices": [{"index": 0,
                 "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def test_openai_client_mct_does_not_keep_yaml_max_tokens(tmp_path, monkeypatch):
    """Pi sends max_completion_tokens only. YAML max_tokens must not stay
    on the candidate payload (a max_tokens-preferring backend would ignore
    the client cap)."""
    import asyncio


    backend = _make_backend(tmp_path, """
backend:
  models:
    - name: openai/dummy
      api_key: x
      num_candidates: 3
      max_tokens: 65536
      thinking: high
verifier:
  model: {name: openai/dummy, api_key: x}
  majority_voting: true
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1, seed: 0}
""")
    seen = []

    async def fake(**kw):
        seen.append({
            k: kw.get(k)
            for k in ("max_tokens", "max_completion_tokens", "reasoning_effort")
        })
        return dict(_CANNED_OK)

    backend._execution.executor._complete_fn = fake
    asyncio.run(backend.complete_openai(json.dumps({
        "model": "pi-label",
        "max_completion_tokens": 100,
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": "hi"}],
    })))
    assert len(seen) == 3
    for req in seen:
        assert req["max_completion_tokens"] == 100
        assert req["max_tokens"] is None
        assert req["reasoning_effort"] == "low"


def test_per_model_temperature_survives_when_client_omits_it(tmp_path, monkeypatch):
    """YAML overlay temperatures must stick when the client did not send
    temperature. Re-asserting default-model sampling flattened both."""
    import asyncio


    backend = _make_backend(tmp_path, """
backend:
  models:
    - name: openai/a
      api_key: x
      num_candidates: 1
      temperature: 0.2
      max_tokens: 1000
    - name: openai/b
      api_key: x
      num_candidates: 1
      temperature: 0.9
      max_tokens: 2000
verifier:
  model: {name: openai/dummy, api_key: x}
  majority_voting: true
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1, seed: 0}
""")
    seen = []

    async def fake(**kw):
        seen.append((kw.get("model"), kw.get("temperature"), kw.get("max_tokens")))
        return dict(_CANNED_OK)

    backend._execution.executor._complete_fn = fake
    asyncio.run(backend.complete_anthropic(json.dumps({
        "model": "m",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": "hi"}],
    })))
    by_model = {name: (temp, mt) for name, temp, mt in seen}
    assert by_model["openai/a"] == (0.2, 500)
    assert by_model["openai/b"] == (0.9, 500)


def test_candidate_messages_are_not_shared(tmp_path, monkeypatch):
    """Each candidate must get its own messages list so a mutating provider
    cannot append into the next candidate's payload."""
    import asyncio


    backend = _make_backend(tmp_path, """
backend:
  models:
    - name: openai/dummy
      api_key: x
      num_candidates: 3
      max_tokens: 128
verifier:
  model: {name: openai/dummy, api_key: x}
  majority_voting: true
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1, seed: 0}
""")
    captured = []

    async def fake(**kw):
        msgs = kw["messages"]
        captured.append(msgs)
        msgs.append({"role": "assistant", "content": "mutated"})
        return dict(_CANNED_OK)

    backend._execution.executor._complete_fn = fake
    asyncio.run(backend.complete_anthropic(json.dumps({
        "model": "m",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}],
    })))
    # Hold the lists so id() is not reused after GC; each candidate must
    # see only its own mutation.
    assert len(captured) == 3
    assert len({id(m) for m in captured}) == 3
    for msgs in captured:
        assert sum(1 for m in msgs if m.get("content") == "mutated") == 1


def test_text_only_preserves_tool_result_images():
    from turbo_agent.utils.conversion import AnthropicToOpenAI

    out = AnthropicToOpenAI.messages(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu9",
                            "content": [
                                {"type": "text", "text": "shot"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "aGVsbG8=",
                                    },
                                },
                            ],
                        }
                    ],
                }
            ]
        },
        text_only=True,
    )
    blob = json.dumps(out)
    assert "data:image/png;base64,aGVsbG8=" in blob
    assert "[tool_result: shot]" in blob


def test_openai_judge_honors_openai_api_base(tmp_path, monkeypatch):
    """openai/* without an explicit base_url must follow OPENAI_API_BASE
    (the stub / local proxy env) instead of hard-coding api.openai.com."""
    from turbo_agent.verifier import Verifier

    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    p = tmp_path / "j.yaml"
    p.write_text("""
backend:
  models:
    - name: openai/dummy
      api_key: x
verifier:
  model: {name: openai/dummy, api_key: k}
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1}
""")
    v = Verifier(Config(str(p)).verifier_config)
    assert "127.0.0.1:9" in str(v.client.base_url)
    assert "api.openai.com" not in str(v.client.base_url)


def test_client_budget_clears_yaml_reasoning_effort(tmp_path, monkeypatch):
    """YAML thinking: high must not survive a client budget_tokens request
    on the verifier gather path."""
    import asyncio


    backend = _make_backend(tmp_path, """
backend:
  models:
    - name: openai/dummy
      api_key: x
      num_candidates: 3
      max_tokens: 65536
      thinking: high
verifier:
  model: {name: openai/dummy, api_key: x}
  majority_voting: true
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1, seed: 0}
""")
    seen = []

    async def fake(**kw):
        seen.append((kw.get("reasoning_effort"), kw.get("thinking_budget")))
        return dict(_CANNED_OK)

    backend._execution.executor._complete_fn = fake
    asyncio.run(backend.complete_anthropic(json.dumps({
        "model": "claude-x",
        "max_tokens": 32000,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "messages": [{"role": "user", "content": "hi"}],
    })))
    # The client budget must clear the YAML effort. On this openai/* target
    # a token budget has no wire representation, so neither key appears;
    # budget translation itself is covered by the model execution tests.
    assert seen == [(None, None)] * 3


def test_pi_adaptive_reads_output_config_effort(tmp_path):
    """Pi anthropic-messages puts adaptive effort on output_config."""
    backend = _make_backend(tmp_path, """
backend:
  models:
    - name: openai/dummy
      api_key: x
      max_tokens: 65536
      thinking: high
""")
    request, _ = backend._build_anthropic_params({
        "model": "claude-opus-4-6",
        "max_tokens": 32000,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "messages": [],
    })
    params = backend._execution.executor._build_params(
        backend._execution.default_target, request)
    assert params.get("reasoning_effort") == "low"
    assert "thinking_budget" not in params


def test_thinking_disabled_clears_yaml_thinking(tmp_path, monkeypatch):
    import asyncio


    backend = _make_backend(tmp_path, """
backend:
  models:
    - name: openai/dummy
      api_key: x
      num_candidates: 2
      max_tokens: 4096
      thinking: high
verifier:
  model: {name: openai/dummy, api_key: x}
  majority_voting: true
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1, seed: 0}
""")
    seen = []

    async def fake(**kw):
        seen.append((kw.get("reasoning_effort"), kw.get("thinking_budget")))
        return dict(_CANNED_OK)

    backend._execution.executor._complete_fn = fake
    asyncio.run(backend.complete_anthropic(json.dumps({
        "model": "m",
        "max_tokens": 2048,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "hi"}],
    })))
    assert seen == [(None, None)] * 2


def test_gemini_keyless_judge_uses_env_fallback(tmp_path, monkeypatch):
    """gemini/* with no api_key must not ask for OPENAI_BASE_URL."""
    from turbo_agent.verifier import Verifier

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    p = tmp_path / "g.yaml"
    p.write_text("""
backend:
  models:
    - name: openai/dummy
      api_key: x
verifier:
  model: {name: gemini/gemini-2.5-flash}
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1}
""")
    v = Verifier(Config(str(p)).verifier_config)
    assert v.client is None


def test_thinking_dropped_when_budget_does_not_fit_cap(tmp_path):
    """YAML thinking that cannot fit under the client's max_tokens is dropped,
    not rewritten to steal output tokens. Uses an anthropic/* target because
    that is where a token budget has a wire representation."""
    backend = _make_backend(tmp_path, """
backend:
  models:
    - name: anthropic/dummy
      api_key: x
      max_tokens: 65536
      thinking: 2048
""")
    tied_request, _ = backend._build_anthropic_params({
        "model": "m", "max_tokens": 2048, "messages": [],
    })
    tied = backend._execution.executor._build_params(
        backend._execution.default_target, tied_request)
    assert "thinking" not in tied

    fits_request, _ = backend._build_anthropic_params({
        "model": "m", "max_tokens": 4096, "messages": [],
    })
    fits = backend._execution.executor._build_params(
        backend._execution.default_target, fits_request)
    assert fits.get("thinking") == {"type": "enabled", "budget_tokens": 2048}


def test_count_tokens_counts_images(proxy):
    body = {
        "model": "m",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "see"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "AA",
                }},
            ],
        }],
    }
    resp = proxy.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] >= 1600


def test_check_api_key_loads_global_env(tmp_path, monkeypatch):
    """turbo-agent check follows Config discovery, not only cwd."""
    from turbo_agent import check_api_key

    xdg = tmp_path / "xdg"
    gdir = xdg / "turbo-agent"
    gdir.mkdir(parents=True)
    (gdir / "turbo-agent.yaml").write_text(
        "backend:\n  models:\n    - name: openai/global\n      api_key: $GK\n")
    (gdir / ".env").write_text("GK=from-global-env\n")
    empty = tmp_path / "emptycwd"
    empty.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.chdir(empty)
    monkeypatch.delenv("GK", raising=False)

    def skip(self):
        return check_api_key.CheckResult(
            self.name, self.env_var, "skip", "skipped", self.roles)

    monkeypatch.setattr(check_api_key.GeminiChecker, "run", skip)
    monkeypatch.setattr(check_api_key.VertexChecker, "run", skip)
    monkeypatch.setattr(check_api_key.OpenAIChecker, "run", skip)
    monkeypatch.setattr(check_api_key.AnthropicChecker, "run", skip)

    rc = check_api_key.main()
    assert rc == 0
    assert os.environ.get("GK") == "from-global-env"


def test_gen_models_json_discovers_global(tmp_path, monkeypatch):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "gen_models_json",
        root / "integrations" / "pi" / "gen_models_json.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    xdg = tmp_path / "xdg"
    gdir = xdg / "turbo-agent"
    gdir.mkdir(parents=True)
    (gdir / "turbo-agent.yaml").write_text(
        "backend:\n  models:\n    - name: openai/from-global\n      api_key: g\n")
    empty = tmp_path / "emptycwd"
    empty.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.chdir(empty)
    assert mod.discover_turbo_agent_yaml() == str(gdir / "turbo-agent.yaml")


def test_gen_models_json_reasoning_is_always_on(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "gen_models_json",
        root / "integrations" / "pi" / "gen_models_json.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    entry = mod.model_entry({"name": "openai/no-thinking", "max_tokens": 8192})
    assert entry["reasoning"] is True
    assert entry["thinkingLevelMap"]["high"] == "high"
    block = mod.provider_block([{"name": "openai/no-thinking"}])
    assert block["providers"]["turbo"]["compat"]["supportsReasoningEffort"] is True
