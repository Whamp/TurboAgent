"""Prompts and independent correctness oracles for server60 benchmarks."""

import json
import re
from collections.abc import Callable
from typing import Any

DEFAULT_BASE_URL = "http://192.168.0.251:30002/v1"
DEFAULT_MODEL = "qwen3.8-flash-next-intel-autoround-w4a16"
DEFAULT_EMBEDDING_BASE_URL = "http://endurance:8090/v1"
DEFAULT_EMBEDDING_MODEL = "octen-embed"
DIRECT_PROMPT = (
    "Explain in about 120 words why prefix caching helps batched LLM "
    "inference. Be technically precise."
)
BOUNDED_ANSWER_PROMPT = (
    "Write exactly one short sentence about GPU inference. No preamble."
)


def normalize_text(text: str) -> str:
    """Normalize prose for diversity counts, not correctness decisions."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


TaskValidator = Callable[[dict[str, Any], dict[str, Any], str], int]


def _score_exact_text(
    task: dict[str, Any], _message: dict[str, Any], content: str
) -> int:
    return int(content == task["expected"])


def _score_integer_text(
    task: dict[str, Any], _message: dict[str, Any], content: str
) -> int:
    digits = re.findall(r"\d+", content.replace(",", ""))
    return int(bool(digits) and digits[-1] == task["expected"])


def _score_json_text(
    task: dict[str, Any], _message: dict[str, Any], content: str
) -> int:
    try:
        return int(json.loads(content) == task["expected"])
    except json.JSONDecodeError:
        return 0


def _score_safe_force_push_tool(
    _task: dict[str, Any], message: dict[str, Any], _content: str
) -> int:
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != "run_bash":
            continue
        arguments = function.get("arguments") or "{}"
        try:
            command = json.loads(arguments).get("command", "")
        except json.JSONDecodeError:
            command = arguments
        safe = "--force-with-lease" in command
        unsafe = bool(re.search(r"--force(?:\s|$)", command))
        return int(safe and not unsafe)
    return 0


def _score_multiple_choice(
    task: dict[str, Any], _message: dict[str, Any], content: str
) -> int:
    return int(content.upper().strip(" .`*") == task["expected"])


_TASK_VALIDATORS: dict[str, TaskValidator] = {
    "exact": _score_exact_text,
    "integer": _score_integer_text,
    "json": _score_json_text,
    "safe_force_push_tool": _score_safe_force_push_tool,
    "choice": _score_multiple_choice,
}


def deterministic_task_score(
    task: dict[str, Any], response: dict[str, Any]
) -> int | None:
    """Score deterministic corpus tasks; return None for diversity-only tasks."""
    validator = _TASK_VALIDATORS.get(task["validator"])
    if validator is None:
        return None
    message = response["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    return validator(task, message, content)


CORPUS = [
    {
        "name": "coprime_count",
        "messages": [
            {
                "role": "user",
                "content": (
                    "How many integers n from 1 through 1,000,000 inclusive "
                    "are coprime to 840? Reply only with the integer."
                ),
            }
        ],
        "validator": "integer",
        "expected": "228571",
    },
    {
        "name": "constrained_lattice_paths",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A robot moves only right or up from (0,0) to (12,12), "
                    "never visits a point with y>x, and must pass through (6,4). "
                    "How many paths are possible? Reply only with the integer."
                ),
            }
        ],
        "validator": "integer",
        "expected": "90090",
    },
    {
        "name": "git_lease_choice",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A deploy must update remote branch release to the current "
                    "HEAD without overwriting remote work that appeared since "
                    "the last fetch. Choose one: A) git push --force origin "
                    "HEAD:release B) git push --force-with-lease=release origin "
                    "HEAD:release C) git reset --hard origin/release D) git push "
                    "origin +HEAD:release. Reply only A, B, C, or D."
                ),
            }
        ],
        "validator": "choice",
        "expected": "B",
    },
    {
        "name": "async_cleanup_choice",
        "messages": [
            {
                "role": "user",
                "content": (
                    "An async function starts four tasks, waits with "
                    "FIRST_COMPLETED, and returns the winner. It must always "
                    "cancel and await the three pending tasks, including when "
                    "the winner raises. Which design is correct? A) return the "
                    "winner immediately B) use try/finally; cancel pending in "
                    "finally and await gather(*pending, return_exceptions=True) "
                    "C) shield every task D) call task.cancel() without awaiting. "
                    "Reply only A, B, C, or D."
                ),
            }
        ],
        "validator": "choice",
        "expected": "B",
    },
    {
        "name": "safe_force_push_tool",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Make exactly one run_bash tool call that safely force-pushes "
                    "the current branch to origin using a lease. Do not inspect "
                    "state first and do not use plain --force."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "run_bash",
                    "description": "Run one bash command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        "tool_choice": "required",
        "validator": "safe_force_push_tool",
    },
    {
        "name": "open_ended_orchestration_design",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Design a cancellation-safe capacity-aware scheduler for an "
                    "LLM proxy that fans one request into four candidates against "
                    "an endpoint with exactly four generation slots. Explain the "
                    "ownership seam, queue discipline, deadlines, slot release, "
                    "and the tests needed to prove no capacity leaks."
                ),
            }
        ],
        "validator": "none",
    },
]
