"""Majority-voting agreement modes for the verifier shortcut.

exact       — current behavior: raw string equality.
normalized  — equality after lowercasing, collapsing whitespace, and
              stripping punctuation.
semantic    — tool calls must still match exactly (whitespace-collapsed);
              prose may agree via cosine similarity from an OpenAI-compatible
              embedding endpoint. A failed embedding call degrades to the
              normalized comparison rather than crashing selection.
"""

import pytest

from turbo_agent.utils import Config
from turbo_agent.verifier import Verifier


def make_verifier(tmp_path, extra=""):
    p = tmp_path / "cfg.yaml"
    p.write_text(f"""
backend:
  models:
    - name: openai/dummy
      api_key: x
verifier:
  enabled: false
{extra}
""")
    return Config(str(p))


def verifier_with_mode(tmp_path, majority_block):
    p = tmp_path / "cfg.yaml"
    p.write_text(f"""
backend:
  models:
    - name: openai/dummy
      api_key: x
verifier:
  model:
    name: openai/judge
    api_key: k
  method: {{name: pivot_tournament, pivots: 1, n_verifications: 1}}
{majority_block}
""")
    return Verifier(Config(str(p)).verifier_config)


class TournamentRan(Exception):
    pass


def guard_tournament(verifier):
    def boom(*args, **kwargs):
        raise TournamentRan("tournament should have been skipped")
    verifier._run_select = boom


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def test_majority_defaults_to_exact_mode(tmp_path):
    v = verifier_with_mode(tmp_path, "  majority_voting: true")
    assert v.cfg.majority.mode == "exact"
    assert v.cfg.majority.embedding is None


def test_majority_semantic_requires_embedding_endpoint(tmp_path):
    with pytest.raises(ValueError, match="embedding"):
        verifier_with_mode(
            tmp_path,
            "  majority_voting: true\n  majority: {mode: semantic}",
        )


def test_majority_semantic_parses_embedding_and_threshold(tmp_path):
    v = verifier_with_mode(tmp_path, """
  majority_voting: true
  majority:
    mode: semantic
    threshold: 0.95
    embedding:
      base_url: http://endurance:8090/v1
      model: octen-embed
""")
    assert v.cfg.majority.mode == "semantic"
    assert v.cfg.majority.threshold == pytest.approx(0.95)
    assert v.cfg.majority.embedding.base_url == "http://endurance:8090/v1"
    assert v.cfg.majority.embedding.name == "octen-embed"


def test_majority_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        verifier_with_mode(
            tmp_path,
            "  majority_voting: true\n  majority: {mode: vibes}",
        )


# ---------------------------------------------------------------------------
# exact mode (current behavior preserved)
# ---------------------------------------------------------------------------

def test_exact_mode_ignores_near_duplicates(tmp_path):
    v = verifier_with_mode(tmp_path, "  majority_voting: true")
    guard_tournament(v)
    with pytest.raises(TournamentRan):
        v.select_best_sync([
            "Use systemctl restart nginx",
            "Use systemctl restart nginx.",
            "use SYSTEMCTL RESTART NGINX",
            "Something entirely different",
        ])  # No strict majority of identical strings -> tournament path.


def test_exact_mode_still_fires_on_identical_strings(tmp_path):
    v = verifier_with_mode(tmp_path, "  majority_voting: true")
    actions = ["same", "same", "other"]
    result = v.select_best_sync(actions)
    assert result.best_index in (0, 1)
    assert len(result.comparisons) == 0


# ---------------------------------------------------------------------------
# normalized mode
# ---------------------------------------------------------------------------

def test_normalized_mode_fires_across_case_punct_whitespace(tmp_path):
    v = verifier_with_mode(
        tmp_path,
        "  majority_voting: true\n  majority: {mode: normalized}",
    )
    guard_tournament(v)
    result = v.select_best_sync([
        "The capital is Paris.",
        "the capital  is paris",
        "THE CAPITAL IS PARIS!",
        "Berlin is the capital",
    ])
    assert result.best_index in (0, 1, 2)


# ---------------------------------------------------------------------------
# semantic mode
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Deterministic stand-in for the embedding endpoint."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [self.vectors[t] for t in texts]


def semantic_verifier(tmp_path, vectors, threshold=0.9):
    v = verifier_with_mode(tmp_path, f"""
  majority_voting: true
  majority:
    mode: semantic
    threshold: {threshold}
    embedding:
      base_url: http://endurance:8090/v1
      model: octen-embed
""")
    v._embed_fn = FakeEmbedder(vectors)
    return v


def test_semantic_fires_on_paraphrase_with_same_tool_calls(tmp_path):
    a = 'Deploy now\n[tool_call: bash({"command": "systemctl restart nginx"})]'
    b = 'Deploying service\n[tool_call: bash({"command": "systemctl restart nginx"})]'
    c = "Completely different plan"
    v = semantic_verifier(tmp_path, {
        "Deploy now": [1.0, 0.0],
        "Deploying service": [0.95, 0.31],   # cos ~0.95 vs "Deploy now"
        "Completely different plan": [0.0, 1.0],
    })
    guard_tournament(v)
    result = v.select_best_sync([a, b, c])
    assert result.best_index in (0, 1)


def test_semantic_never_relaxes_tool_call_differences(tmp_path):
    """The calibrated failure mode: near-miss commands embed as almost
    identical. Tool calls must match exactly regardless of cosine."""
    a = '[tool_call: bash({"command": "git push --force-with-lease origin main"})] t'
    b = '[tool_call: bash({"command": "git push --force origin main"})] t'
    v = semantic_verifier(
        tmp_path, {"t": [1.0, 0.0],
                   "git push --force-with-lease origin main": [1.0, 0.0]})
    guard_tournament(v)
    with pytest.raises(TournamentRan):
        v.select_best_sync([a, b])


def test_semantic_low_similarity_runs_tournament(tmp_path):
    v = semantic_verifier(tmp_path, {
        "Answer one": [1.0, 0.0],
        "Answer two": [0.0, 1.0],
        "Answer one again": [0.707, 0.707],   # cos ~0.707 to both: below 0.9
    })
    guard_tournament(v)
    with pytest.raises(TournamentRan):
        v.select_best_sync(["Answer one", "Answer two", "Answer one again"])


def test_semantic_embedding_failure_degrades_to_normalized(tmp_path):
    class BrokenEmbedder:
        def __call__(self, texts):
            raise ConnectionError("endpoint down")

    v = verifier_with_mode(tmp_path, """
  majority_voting: true
  majority:
    mode: semantic
    embedding:
      base_url: http://endurance:8090/v1
      model: octen-embed
""")
    v._embed_fn = BrokenEmbedder()
    guard_tournament(v)
    result = v.select_best_sync([
        "The answer is forty-two.",
        "the answer  is FORTY-TWO",
        "nope",
    ])
    assert result.best_index in (0, 1)


def test_semantic_batches_all_texts_in_one_call(tmp_path):
    texts = ["alpha one", "alpha  one!", "beta"]
    v = semantic_verifier(tmp_path, {
        "alpha one": [1.0, 0.0],
        "beta": [0.0, 1.0],
    })
    guard_tournament(v)
    v.select_best_sync([
        "alpha one",
        "alpha  one!",
        "beta",
    ])
    assert len(v._embed_fn.calls) == 1
    # Distinct texts only, deduplicated.
    assert sorted(v._embed_fn.calls[0]) == sorted(set(texts)) or \
        set(v._embed_fn.calls[0]) <= set(texts)
