"""Orchestration benchmark for Turbo Agent candidate/verifier configurations.

Spawns one proxy per scenario against an OpenAI-compatible backend, sends a
fixed prompt set, and records per-request latency plus token usage. Answers
two questions with numbers instead of guesses:

- how wall-clock latency scales with ``num_candidates``;
- what verification (pivot-tournament judging) adds on top.

Usage:
    python scripts/bench_orchestration.py --backend local --out results.json
    python scripts/bench_orchestration.py --backend codex --codex-sample

Results are written to --out (JSON) and summarized to stdout (markdown).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error as urlerror
from urllib import request

REPO = Path(__file__).resolve().parent.parent
LLAMA_LOCAL_YAML = """
backend:
  models:
    - name: openai/qwen3-1.7b
      api_key: sk-local
      base_url: http://127.0.0.1:8097/v1
      num_candidates: {candidates}
      max_tokens: 512
{verifier}"""

CODEX_YAML = """
backend:
  models:
    - name: openai-codex/gpt-5.6-sol
      executor: pi
      num_candidates: {candidates}
      max_tokens: 200
{verifier}"""

MAJORITY_VERIFIER = """verifier:
  model: {name: openai/qwen3-1.7b, api_key: sk-local, base_url: 'http://127.0.0.1:8097/v1'}
  majority_voting: true
  method: {name: pivot_tournament, pivots: 1, n_verifications: 1, seed: 0}"""

PROMPTS = [
    "Name three primary colors.",
    "What is the capital of France? One sentence.",
    "Write a haiku about rain.",
    "Summarize why the sky is blue in two sentences.",
    "Count from 1 to 5, comma-separated.",
]


@dataclass(frozen=True)
class Scenario:
    name: str
    candidates: int
    verifier: bool


LOCAL_SCENARIOS = [
    Scenario("local n=1 no-verifier", 1, False),
    Scenario("local n=2 no-verifier", 2, False),
    Scenario("local n=4 no-verifier", 4, False),
    Scenario("local n=4 majority+pivot", 4, True),
]

CODEX_SCENARIOS = [
    Scenario("codex n=1 no-verifier", 1, False),
    Scenario("codex n=4 no-verifier", 4, False),
]


def render_yaml(scenario: Scenario, backend: str) -> str:
    template = LLAMA_LOCAL_YAML if backend == "local" else CODEX_YAML
    verifier = MAJORITY_VERIFIER if scenario.verifier else ""
    return template.format(candidates=scenario.candidates, verifier=verifier)


def wait_ready(port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=1
            ) as resp:
                if resp.status == 200:
                    return
        except (urlerror.URLError, TimeoutError, ConnectionError):
            time.sleep(0.2)
    raise RuntimeError(f"proxy on port {port} never became ready")


_DEFAULT_PI_MODULE = (
    Path.home() / ".local/share/mise/installs/node/24.16.0/lib/node_modules"
    "/@earendil-works/pi-coding-agent/dist/index.js"
)


def run_proxy(config_path: Path, port: int, backend: str) -> subprocess.Popen:
    env = dict(os.environ)
    if backend == "codex" and "TURBO_PI_MODULE_PATH" not in env:
        # The companion must resolve Pi outside the repo tree; point it at
        # the global install unless the caller already chose a location.
        env["TURBO_PI_MODULE_PATH"] = str(_DEFAULT_PI_MODULE)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "turbo_agent.cli",
            "-p",
            str(port),
            "-c",
            str(config_path),
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        wait_ready(port)
    except RuntimeError:
        proc.terminate()
        raise
    return proc


def send_prompt(port: int, prompt: str) -> dict:
    body = json.dumps(
        {
            "model": "bench",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read())
    latency = time.monotonic() - start
    usage = payload.get("usage") or {}
    choices = payload.get("choices") or [{}]
    message = choices[0].get("message") or {}
    text = message.get("content", "")
    return {
        "latency_s": round(latency, 3),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "chars": len(text or ""),
    }


def run_scenario(scenario: Scenario, backend: str, port: int, tmp: Path) -> dict:
    config_path = (
        tmp / f"turbo-{backend}-{scenario.candidates}-{int(scenario.verifier)}.yaml"
    )
    config_path.write_text(render_yaml(scenario, backend))
    proxy = run_proxy(config_path, port, backend)
    try:
        samples = [send_prompt(port, p) for p in PROMPTS]
    finally:
        proxy.terminate()
        proxy.wait(timeout=10)
    latencies = [s["latency_s"] for s in samples]
    tokens = [s["completion_tokens"] or 0 for s in samples]
    return {
        "scenario": scenario.name,
        "candidates": scenario.candidates,
        "verifier": scenario.verifier,
        "samples": samples,
        "latency_mean_s": round(sum(latencies) / len(latencies), 3),
        "latency_min_s": min(latencies),
        "latency_max_s": max(latencies),
        "completion_tokens_total": sum(tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "codex"], default="local")
    parser.add_argument("--port", type=int, default=8893)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    scenarios = LOCAL_SCENARIOS if args.backend == "local" else CODEX_SCENARIOS
    tmp = Path(__file__).resolve().parent / ".bench-configs"
    tmp.mkdir(exist_ok=True)

    results = []
    for index, scenario in enumerate(scenarios):
        port = args.port + index
        print(f"### {scenario.name} (port {port})", flush=True)
        results.append(run_scenario(scenario, args.backend, port, tmp))

    report = {"backend": args.backend, "prompts": len(PROMPTS), "results": results}
    print("\n| scenario | mean s | min s | max s | completion tokens |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['scenario']} | {r['latency_mean_s']} | "
            f"{r['latency_min_s']} | {r['latency_max_s']} | "
            f"{r['completion_tokens_total']} |"
        )

    out = args.out or REPO / f"bench-{args.backend}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
