import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv

from ..endpoint_admission import EndpointAdmissionPolicy


@dataclass
class ModelConfig:
    name: str
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    endpoint: str | None = None


@dataclass(frozen=True)
class EndpointConfig:
    """Named endpoint URL and its shared admission policy."""

    name: str
    base_url: str
    policy: EndpointAdmissionPolicy


@dataclass(frozen=True)
class JudgeExecutionConfig:
    """Provider-specific execution rules for verifier judge calls."""

    adapter: str = "default"
    comparison_max_output_tokens: int = 4096
    max_concurrency: int | None = None


@dataclass
class CriterionConfig:
    name: str
    description: str = ""


@dataclass
class PivotTournamentConfig:
    """Parameters for the Probabilistic Pivot Tournament selection method."""
    pivots: int = 2            # k: number of pivot (empirical-leader) candidates
    n_verifications: int = 4   # K: repeated verifications per directed pair
    seed: int = 0             # seed for the random ring pass (reproducible)
    note: str = ""             # ground-truth note injected into the prompt
    criteria: List[CriterionConfig] = field(default_factory=list)


@dataclass
class MajorityConfig:
    """How the majority-voting shortcut decides that candidates agree.

    exact      — raw string equality (historical behavior).
    normalized — equality after lowercasing, collapsing whitespace, and
                 stripping punctuation.
    semantic   — tool calls must match exactly (whitespace-collapsed); prose
                 may agree via cosine similarity at or above ``threshold``
                 from an OpenAI-compatible embedding endpoint. Embedding
                 failures degrade to the normalized comparison.
    """
    mode: str = "exact"
    threshold: float = 0.92
    embedding: ModelConfig | None = None


@dataclass
class VerifierConfig:
    model: ModelConfig
    method: PivotTournamentConfig
    majority_voting: bool = False
    majority: MajorityConfig = field(default_factory=MajorityConfig)
    execution: JudgeExecutionConfig = field(default_factory=JudgeExecutionConfig)


@dataclass
class ContextConfig:
    model_name: str
    api_key: str
    refinement_prompt: str


@dataclass
class ProgressMonitorConfig:
    """A post-hoc progress score for the selected trajectory, computed with
    `llm_verifier.track` (K repeated verifications, averaged). Observability
    only — it never changes the response."""
    model: ModelConfig
    n_verifications: int = 4   # K: repeated verifications


# Default holistic criterion used when the config declares none.
_DEFAULT_CRITERIA = [
    CriterionConfig(
        name="Task Success",
        description=(
            "How likely the agent correctly and completely solved the task. "
            "The strongest signal is the agent verifying its solution against "
            "the task's specific requirements. Trajectory length, number of "
            "steps, and apparent confidence do not predict correctness."
        ),
    ),
]


class Config:
    # Global fallback config, used when the current directory has no
    # turbo-agent.yaml (same idea as pi's ~/.pi/agent/settings.json global
    # default). Explicit --config PATH and a project-level file both win
    # over this.
    @staticmethod
    def global_dir() -> Path:
        """~/.config/turbo-agent (or $XDG_CONFIG_HOME/turbo-agent)."""
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        return Path(base) / "turbo-agent"

    @staticmethod
    def _discover() -> Path:
        """Find the config to use: project-level ./turbo-agent.yaml first,
        then the global ~/.config/turbo-agent/turbo-agent.yaml default."""
        candidates = [
            Path.cwd() / "turbo-agent.yaml",
            Config.global_dir() / "turbo-agent.yaml",
        ]
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(
            "No turbo-agent.yaml found. Looked for:\n  "
            + "\n  ".join(str(c) for c in candidates)
            + "\nCreate one, or pass --config PATH."
        )

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(self._discover())
        else:
            config_path = str(Path(config_path))

        # Load .env from the same directory as the config file.
        env_path = Path(config_path).parent / ".env"
        if env_path.exists():
            load_dotenv(str(env_path), override=True)

        with open(config_path, "r") as f:
            self._raw: Dict[str, Any] = yaml.safe_load(f) or {}

        self._expand_env_vars()

    @staticmethod
    def _resolve_env(value: str) -> str:
        if isinstance(value, str) and value.startswith("$"):
            return os.environ.get(value[1:], "")
        return value

    def _expand_env_vars(self) -> None:
        for model in self.models:
            for field_name in ("api_key", "base_url"):
                val = model.get(field_name, "")
                if isinstance(val, str) and val.startswith("$"):
                    model[field_name] = self._resolve_env(val)
        for endpoint in (self._raw.get("endpoints") or {}).values():
            base_url = endpoint.get("base_url", "")
            if isinstance(base_url, str) and base_url.startswith("$"):
                endpoint["base_url"] = self._resolve_env(base_url)

    # ------------------------------------------------------------------
    # Shared model endpoints
    # ------------------------------------------------------------------

    @property
    def endpoint_configs(self) -> dict[str, EndpointConfig]:
        """Parse named endpoint URLs and bounded admission policies."""
        configs = {}
        for name, raw in (self._raw.get("endpoints") or {}).items():
            base_url = raw.get("base_url")
            if not base_url:
                raise ValueError(f"Endpoint '{name}' requires base_url")
            scheme, host, _port = self.endpoint_origin(base_url)
            if scheme not in ("http", "https") or not host:
                raise ValueError(
                    f"Endpoint '{name}' requires an absolute HTTP(S) base_url"
                )
            configs[name] = EndpointConfig(
                name=name,
                base_url=base_url,
                policy=EndpointAdmissionPolicy(
                    max_concurrency=int(raw.get("max_concurrency", 1)),
                    max_queue_size=int(raw.get("max_queue_size", 64)),
                    queue_timeout_seconds=float(
                        raw.get("queue_timeout_seconds", 300)
                    ),
                    request_timeout_seconds=float(
                        raw.get("request_timeout_seconds", 900)
                    ),
                ),
            )
        return configs

    def model_base_url(self, model: dict[str, Any]) -> str | None:
        """Resolve a model's direct or named endpoint URL without merging them."""
        direct_url = model.get("base_url") or None
        endpoint_name = model.get("endpoint") or None
        if endpoint_name is None:
            return direct_url
        try:
            endpoint = self.endpoint_configs[endpoint_name]
        except KeyError:
            raise ValueError(f"Unknown model endpoint '{endpoint_name}'") from None
        if direct_url and direct_url.rstrip("/") != endpoint.base_url.rstrip("/"):
            raise ValueError(
                f"Model endpoint '{endpoint_name}' conflicts with its base_url"
            )
        return endpoint.base_url

    @staticmethod
    def endpoint_origin(base_url: str) -> tuple[str, str, int | None]:
        """Identify the server capacity domain independent of API path."""
        parsed = urlsplit(base_url)
        default_port = 443 if parsed.scheme.lower() == "https" else 80
        return (
            parsed.scheme.lower(),
            (parsed.hostname or "").lower(),
            parsed.port or default_port,
        )

    @staticmethod
    def provider_environment_base_url(model: dict[str, Any]) -> str | None:
        """Resolve LiteLLM's provider-specific implicit API-base variables."""
        if model.get("executor") == "pi":
            return None
        provider = str(model.get("name", "")).split("/", 1)[0]
        if not provider:
            return None
        env_prefix = provider.replace("-", "_").upper()
        keys = [f"{env_prefix}_API_BASE"]
        if provider == "openai":
            keys.insert(0, "OPENAI_BASE_URL")
        for key in keys:
            value = os.environ.get(key)
            if value:
                return value
        return None

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    @property
    def models(self) -> List[Dict[str, Any]]:
        return self._raw.get("backend", {}).get("models", [])

    @property
    def default_model(self) -> Dict[str, Any]:
        if not self.models:
            raise ValueError("No models configured under backend.models")
        return self.models[0]

    @property
    def total_candidates(self) -> int:
        return sum(m.get("num_candidates", 1) for m in self.models)

    # ------------------------------------------------------------------
    # Context refinement (optional)
    # ------------------------------------------------------------------

    @property
    def context_config(self) -> Optional[ContextConfig]:
        raw_ctx = self._raw.get("context")
        if not raw_ctx:
            return None
        raw_model = raw_ctx.get("refinement_model")
        prompt = raw_ctx.get("refinement_prompt")
        if not raw_model or not raw_model.get("name") or not prompt:
            return None
        return ContextConfig(
            model_name=raw_model["name"],
            api_key=self._resolve_env(raw_model.get("api_key", "")),
            refinement_prompt=prompt,
        )

    # ------------------------------------------------------------------
    # Verifier
    # ------------------------------------------------------------------

    @property
    def verifier_config(self) -> Optional[VerifierConfig]:
        raw_v = self._raw.get("verifier")
        if not raw_v or raw_v.get("enabled") is False:
            return None

        raw_model = raw_v.get("model") or {}
        if not raw_model.get("name"):
            # The judge defaults to the active session's backend model when
            # the verifier is enabled but no judge is named.
            if not self.models:
                return None
            first = self.models[0]
            raw_model = {
                "name": first.get("name"),
                "api_key": first.get("api_key"),
                "base_url": first.get("base_url"),
                "provider": first.get("provider"),
                "endpoint": first.get("endpoint"),
            }
        model_name = raw_model.get("name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Verifier model name must be a non-empty string")
        raw_api_key = raw_model.get("api_key", "")
        raw_base_url = raw_model.get("base_url", "")
        if raw_base_url:
            raw_model = {
                **raw_model,
                "base_url": self._resolve_env(raw_base_url),
            }
        model_cfg = ModelConfig(
            name=model_name,
            provider=raw_model.get("provider"),
            api_key=self._resolve_env(raw_api_key) if raw_api_key else None,
            base_url=self.model_base_url(raw_model),
            endpoint=raw_model.get("endpoint") or None,
        )

        raw_method = raw_v.get("method", {})
        method_name = raw_method.get("name", "pivot_tournament")
        if method_name != "pivot_tournament":
            raise ValueError(
                f"Unknown verifier method '{method_name}'. "
                f"Only 'pivot_tournament' is supported."
            )

        criteria = [
            CriterionConfig(
                name=c.get("name", ""),
                description=c.get("description", ""),
            )
            for c in raw_method.get("criteria", [])
        ] or list(_DEFAULT_CRITERIA)

        method_cfg = PivotTournamentConfig(
            pivots=raw_method.get("pivots", 2),
            n_verifications=raw_method.get("n_verifications", 4),
            seed=raw_method.get("seed", 0),
            note=raw_method.get("note", ""),
            criteria=criteria,
        )

        raw_majority = raw_v.get("majority", {}) or {}
        majority_cfg = self._majority_config(raw_majority)
        execution_cfg = self._judge_execution_config(
            raw_v.get("execution") or {}, model_cfg
        )

        return VerifierConfig(
            model=model_cfg,
            method=method_cfg,
            majority_voting=raw_v.get("majority_voting", False),
            majority=majority_cfg,
            execution=execution_cfg,
        )

    def _judge_execution_config(
        self,
        raw: dict[str, Any],
        model: ModelConfig,
    ) -> JudgeExecutionConfig:
        adapter = raw.get("adapter", "default")
        if adapter not in ("default", "server60"):
            raise ValueError(
                f"Unknown verifier execution adapter '{adapter}'. "
                "Expected default or server60."
            )
        if adapter == "default":
            if model.endpoint is not None:
                raise ValueError(
                    "verifier model.endpoint requires execution.adapter 'server60'"
                )
            if model.base_url is not None:
                judge_origin = self.endpoint_origin(model.base_url)
                if any(
                    self.endpoint_origin(endpoint.base_url) == judge_origin
                    for endpoint in self.endpoint_configs.values()
                ):
                    raise ValueError(
                        "verifier base_url matching a named endpoint requires "
                        "execution.adapter 'server60' and model.endpoint"
                    )
            return JudgeExecutionConfig()
        if model.endpoint is None:
            raise ValueError(
                "verifier.execution.adapter 'server60' requires model.endpoint"
            )
        endpoint = self.endpoint_configs[model.endpoint]
        if endpoint.policy.max_concurrency > 4:
            raise ValueError(
                "server60 endpoint max_concurrency cannot exceed 4"
            )
        endpoint_origin = self.endpoint_origin(endpoint.base_url)
        for candidate in self.models:
            candidate_url = self.model_base_url(candidate)
            if candidate_url is None:
                candidate_url = self.provider_environment_base_url(candidate)
            if (
                candidate_url
                and self.endpoint_origin(candidate_url) == endpoint_origin
                and candidate.get("endpoint") != model.endpoint
            ):
                raise ValueError(
                    "Candidates sharing the server60 judge origin must use "
                    f"endpoint '{model.endpoint}'"
                )
        comparison_cap = int(raw.get("comparison_max_output_tokens", 16384))
        max_concurrency = int(raw.get("max_concurrency", 4))
        if comparison_cap < 2:
            raise ValueError("server60 comparison_max_output_tokens must be at least 2")
        if max_concurrency < 1 or max_concurrency > 4:
            raise ValueError("server60 judge max_concurrency must be between 1 and 4")
        if max_concurrency > endpoint.policy.max_concurrency:
            raise ValueError(
                "server60 judge max_concurrency cannot exceed endpoint capacity"
            )
        return JudgeExecutionConfig(
            adapter=adapter,
            comparison_max_output_tokens=comparison_cap,
            max_concurrency=max_concurrency,
        )

    def _majority_config(self, raw: dict) -> MajorityConfig:
        mode = raw.get("mode", "exact")
        if mode not in ("exact", "normalized", "semantic"):
            raise ValueError(
                f"Unknown verifier majority mode '{mode}'. "
                f"Expected exact, normalized, or semantic."
            )
        embedding = None
        if mode == "semantic":
            raw_embed = raw.get("embedding") or {}
            if not raw_embed.get("base_url") or not raw_embed.get("model"):
                raise ValueError(
                    "verifier.majority.mode 'semantic' requires an "
                    "embedding endpoint (base_url and model)."
                )
            api_key = raw_embed.get("api_key") or None
            embedding = ModelConfig(
                name=raw_embed["model"],
                base_url=raw_embed["base_url"],
                api_key=self._resolve_env(api_key) if api_key else None,
            )
        return MajorityConfig(
            mode=mode,
            threshold=float(raw.get("threshold", 0.92)),
            embedding=embedding,
        )

    # ------------------------------------------------------------------
    # Progress monitor (optional, post-hoc observability)
    # ------------------------------------------------------------------

    @property
    def progress_monitor_config(self) -> Optional[ProgressMonitorConfig]:
        raw_pm = self._raw.get("progress_monitor")
        if not raw_pm:
            return None
        raw_model = raw_pm.get("model")
        if not raw_model or not raw_model.get("name"):
            return None
        raw_api_key = raw_model.get("api_key", "")
        model_cfg = ModelConfig(
            name=raw_model["name"],
            provider=raw_model.get("provider"),
            api_key=self._resolve_env(raw_api_key) if raw_api_key else None,
        )
        return ProgressMonitorConfig(
            model=model_cfg,
            n_verifications=raw_pm.get("n_verifications", 4),
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def log_dir(self) -> str:
        dir_name = self._raw.get("log_dir", "default")
        return str(Path(".turbo-agent") / dir_name)

    @property
    def raw_config(self) -> Dict[str, Any]:
        return self._raw
