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
