"""
Verifier: best-of-N selection over candidate agent responses, delegated to
the `llm-verifier` package.

Given the conversation history and N candidate responses, `llm_verifier.select`
scores directed pairs (candidate `a` in slot A, `b` in slot B) with the
fine-grained logprob reward and aggregates them through a Probabilistic Pivot
Tournament (PPT) to pick the best candidate in O(N·k) comparisons rather than
the O(N^2) of full round-robin. This module wraps that call with TurboAgent's
config, majority-voting shortcut, and the per-comparison records the
visualizer displays.
"""

import asyncio
import json
import os
import re
import tempfile
from collections.abc import Callable
from typing import List

import llm_verifier
from llm_verifier.fine_grained_reward import (
    build_prompt,
    create_deepseek_client,
    create_openai_client,
    directed_reward,
)
from llm_verifier.prompts import normalize_criteria

from ..utils import VerifierConfig, create_logger

_logger = create_logger("verifier")

# Matches the "[tool_call: name(args)]" lines Backend.format_action emits.
_TOOL_CALL_LINE = re.compile(r"^\[tool_call: (\S+)\((.*)\)\]$")


def split_action(action: str):
    """Split a formatted action into (prose, tool_calls).

    ``tool_calls`` preserves order; entries are the raw "name(args)" text.
    Lines that are not tool-call markers are prose.
    """
    prose, tools = [], []
    for line in action.splitlines():
        m = _TOOL_CALL_LINE.match(line.strip())
        if m:
            tools.append(f"{m.group(1)}({m.group(2)})")
        elif line.strip():
            prose.append(line.strip())
    return "\n".join(prose), tools


def _normalize_prose(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _normalize_tool_calls(tools: List[str]) -> tuple:
    # Whitespace only. Case and punctuation are significant in commands;
    # `rm -rf /build` must never equal `rm -rf /tmp/build`.
    return tuple(re.sub(r"\s+", " ", t).strip() for t in tools)


class Comparison:
    """A single directed comparison (candidate i in slot A, j in slot B)."""

    def __init__(self, i: int, j: int, reward_a: float, reward_b: float,
                 prompt: str, text: str):
        self.i = i
        self.j = j
        self.reward_a = reward_a
        self.reward_b = reward_b
        self.text = text
        self.prompt = prompt

    def to_dict(self) -> dict:
        if self.reward_a > self.reward_b:
            winner = "A"
        elif self.reward_b > self.reward_a:
            winner = "B"
        else:
            winner = "tie"
        return {
            "i": self.i,
            "j": self.j,
            "rating_A": self.reward_a,
            "rating_B": self.reward_b,
            "winner": winner,
            "request": [{"role": "user", "content": self.prompt}],
            "text": self.text,
        }


class SelectionResult:
    def __init__(self, best_index: int, scores: List[float],
                 comparisons: List[Comparison]):
        self.best_index = best_index
        self.scores = scores
        self.comparisons = comparisons


class Verifier:
    # Distinct from None: None means "let llm-verifier build from the env"
    # (keyless gemini/*). _UNSET means the client has not been resolved yet.
    _UNSET = object()

    def __init__(self, cfg: VerifierConfig):
        self.cfg = cfg
        self.method = cfg.method
        self.model_id = self._wire_model_id(cfg.model.name)
        self.criteria = normalize_criteria(
            [{"name": c.name, "description": c.description}
             for c in self.method.criteria]
        )
        self._client = self._UNSET  # created lazily on first scoring call
        # Injectable for tests; built from config on first use otherwise.
        self._embed_fn: Callable | None = None
        _logger.info(
            f"Verifier: model={cfg.model.name}, method=pivot_tournament, "
            f"pivots={self.method.pivots}, K={self.method.n_verifications}, "
            f"criteria={[c['name'] for c in self.criteria]}, "
            f"majority_mode={cfg.majority.mode}"
        )

    @staticmethod
    def _wire_model_id(name: str) -> str:
        """The model id actually sent to the judge API: strip the litellm
        provider prefix and keep everything after it
        (gemini/gemini-2.5-flash -> gemini-2.5-flash,
        openrouter/deepseek/deepseek-v4-flash-0731 ->
        deepseek/deepseek-v4-flash-0731, openai/dummy -> dummy)."""
        return name.split("/", 1)[1] if "/" in name else name

    @property
    def client(self):
        """The llm-verifier judge client, built from the verifier model
        config. The model name selects the backend:
          gemini/*      -> google-genai (Vertex AI for full logprobs)
          deepseek/*    -> DeepSeek hosted API
          openrouter/*  -> OpenRouter (OpenAI-compatible, logprobs where
                           the upstream provider exposes them)
          anything else -> any OpenAI-compatible server; ``base_url``
                           selects a local vLLM/SGLang/OpenAI endpoint
        ``None`` (gemini with no key) lets llm-verifier create a client
        from the environment.
        """
        if self._client is not self._UNSET:
            return self._client

        name = self.cfg.model.name
        api_key = self.cfg.model.api_key
        base_url = self.cfg.model.base_url

        if name.startswith("gemini/"):
            if api_key:
                from google import genai
                if self.cfg.model.provider == "vertex_ai":
                    self._client = genai.Client(vertexai=True, api_key=api_key)
                else:
                    self._client = genai.Client(api_key=api_key)
            else:
                # No key in the config: llm-verifier.create_gemini_client
                # reads VERTEX_API_KEY from the environment.
                self._client = None
        elif name.startswith("deepseek/"):
            self._client = create_deepseek_client(api_key=api_key)
        else:
            if base_url is None and name.startswith("openrouter/"):
                base_url = "https://openrouter.ai/api/v1"
            elif base_url is None and name.startswith("openai/"):
                # Official SDK env first, then the litellm/stub alias, then
                # hosted OpenAI. Hard-coding api.openai.com first would send
                # openai/dummy judges off-box.
                base_url = (
                    os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("OPENAI_API_BASE")
                    or "https://api.openai.com/v1"
                )
            self._client = create_openai_client(base_url=base_url,
                                                api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def select_best(
        self, history: str, actions: list[str],
    ) -> SelectionResult:
        n = len(actions)
        if n == 0:
            return SelectionResult(0, [], [])
        if n == 1:
            return SelectionResult(0, [1.0], [])

        if self.cfg.majority.mode == "semantic":
            await asyncio.to_thread(self._ensure_embed_fn)
        majority = self._try_majority_voting(actions)
        if majority is not None:
            return majority

        result, pair_scores = await asyncio.to_thread(
            self._run_select, history, actions)
        comparisons = self._build_comparisons(history, actions, pair_scores)

        _logger.info(
            f"PPT: N={n} comparisons={result.n_comparisons} "
            f"scores=[{', '.join(f'{s:.3f}' for s in result.scores)}] "
            f"best={result.index}"
        )
        return SelectionResult(result.index, result.scores, comparisons)

    def select_best_sync(self, actions: list[str]) -> SelectionResult:
        """Synchronous selection over actions alone (no history). Exercises
        the same agreement shortcut and tournament dispatch as the async
        path."""
        if not actions:
            return SelectionResult(0, [], [])
        if len(actions) == 1:
            return SelectionResult(0, [1.0], [])
        majority = self._try_majority_voting(actions)
        if majority is not None:
            return majority
        result, pair_scores = self._run_select("", actions)
        return SelectionResult(result.index, result.scores,
                               self._build_comparisons("", actions,
                                                       pair_scores))

    # ------------------------------------------------------------------
    # llm-verifier tournament
    # ------------------------------------------------------------------

    def _run_select(self, history: str, actions: List[str]):
        """Run llm_verifier.select with a per-request score cache, returning
        (VerifierResult, raw pair scores) so the comparisons the tournament
        actually ran can be reconstructed for the request log."""
        m = self.method
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "scores.json")
            result = llm_verifier.select(
                history,
                actions,
                criteria=self.criteria,
                ground_truth_note=m.note,
                n_evaluations=m.n_verifications,
                pivots=m.pivots,
                seed=m.seed,
                model=self.model_id,
                client=self.client,
                cache=cache,
                progress=False,
            )
            pair_scores = {}
            if os.path.exists(cache):
                with open(cache) as f:
                    pair_scores = json.load(f)
        return result, pair_scores

    def _build_comparisons(
        self, history: str, actions: List[str], pair_scores: dict,
    ) -> List[Comparison]:
        """One Comparison per directed pair in the score cache, with rewards
        averaged over criteria and repeats (matching the tournament's
        aggregation) and the slot-A/slot-B prompt of the first criterion."""
        m = self.method
        criteria_ids = [c["id"] for c in self.criteria]
        pairs = {}
        for key in pair_scores:
            _, task, pair, _ = key.split("|")
            a, b = (int(x) for x in pair.split(","))
            pairs[(a, b)] = task
        comparisons = []
        for (a, b), task in sorted(pairs.items()):
            ra, rb = directed_reward(
                pair_scores, task, a, b, criteria_ids, m.n_verifications)
            prompt = build_prompt(
                history, actions[a], actions[b], self.criteria[0], m.note)
            comparisons.append(Comparison(a, b, ra, rb, prompt, ""))
        return comparisons

    # ------------------------------------------------------------------
    # Majority voting
    # ------------------------------------------------------------------

    def _agreement_keys(self, actions: list[str]) -> list:
        """Per-action grouping key under the configured mode."""
        mode = self.cfg.majority.mode
        if mode == "exact":
            return list(actions)
        split = [split_action(a) for a in actions]
        norm_prose = [_normalize_prose(p) for p, _ in split]
        tools = [_normalize_tool_calls(t) for _, t in split]
        if mode == "normalized":
            return [(t, p) for t, p in zip(tools, norm_prose)]
        # semantic: tool calls stay exact-ish; prose groups by embedding.
        unique = sorted({p for p, _ in split})
        cluster_ids = dict(zip(unique, self._cluster_ids(unique)))
        return [(tools[i], cluster_ids[split[i][0]]) for i in range(len(actions))]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = sum(x * x for x in a) ** 0.5
        db = sum(y * y for y in b) ** 0.5
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)

    def _ensure_embed_fn(self) -> Callable:
        if self._embed_fn is not None:
            return self._embed_fn
        embed_cfg = self.cfg.majority.embedding
        from openai import OpenAI
        client = OpenAI(base_url=embed_cfg.base_url,
                        api_key=embed_cfg.api_key or "none")
        model = embed_cfg.name

        def embed(texts: list[str]) -> list[list[float]]:
            response = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in response.data]

        self._embed_fn = embed
        return embed

    def _cluster_ids(self, texts: list[str]) -> list[int]:
        """Cluster ids for texts: greedy join to the first representative at
        or above the configured cosine threshold. Empty prose always groups
        alone; an embedding-endpoint failure degrades to normalized equality
        so selection never crashes."""
        non_empty = [t for t in texts if t]
        ids = {}
        if non_empty:
            try:
                vectors = self._ensure_embed_fn()(non_empty)
            except Exception as exc:  # noqa: BLE001 — degrade, never crash selection
                _logger.warning(
                    f"Embedding endpoint failed ({exc}); degrading majority "
                    f"comparison to normalized"
                )
                return [hash(_normalize_prose(t)) for t in texts]
            threshold = self.cfg.majority.threshold
            reps: list[list[float]] = []
            for text, vec in zip(non_empty, vectors):
                for cid, rep in enumerate(reps):
                    if self._cosine(vec, rep) >= threshold:
                        ids[text] = cid
                        break
                else:
                    reps.append(vec)
                    ids[text] = len(reps) - 1
        return [ids[t] if t else object() for t in texts]

    def _try_majority_voting(
        self, actions: List[str],
    ) -> SelectionResult | None:
        if not self.cfg.majority_voting:
            return None
        keys = self._agreement_keys(actions)
        counts = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        majority_key, majority_count = None, 0
        for key, count in counts.items():
            if count > majority_count:
                majority_count, majority_key = count, key
        if majority_key is None or majority_count <= len(actions) / 2:
            return None

        best = next(i for i, k in enumerate(keys) if k == majority_key)
        scores = [1.0 if k == majority_key else 0.0 for k in keys]
        _logger.info(
            f"Majority voting ({self.cfg.majority.mode}): "
            f"{majority_count}/{len(actions)} responses agree, "
            f"skipping tournament"
        )
        return SelectionResult(best, scores, [])
