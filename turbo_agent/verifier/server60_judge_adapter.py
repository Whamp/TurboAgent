"""server60 request adapter for synchronous llm-verifier judge calls."""

import copy
from typing import NoReturn, Protocol, cast

from ..endpoint_admission import EndpointAdmissionRegistry


class _JudgeCompletions(Protocol):
    def create(self, **kwargs: object) -> object:
        """Create one OpenAI-compatible chat completion."""
        ...


class _JudgeChat(Protocol):
    @property
    def completions(self) -> _JudgeCompletions:
        """The OpenAI-compatible completion resource."""
        ...


class _DeferredJudgeFailure:
    """Move a server60 call failure past llm-verifier's retry catch."""

    usage: None = None
    usage_metadata: None = None

    def __init__(self, error: Exception) -> None:
        self._error = error

    @property
    def choices(self) -> NoReturn:
        """Re-raise the original call failure during response decoding."""
        raise self._error


class _Server60JudgeCompletions:
    def __init__(
        self,
        completions: _JudgeCompletions,
        registry: EndpointAdmissionRegistry,
        endpoint_name: str,
        comparison_max_output_tokens: int,
    ) -> None:
        self._completions = completions
        self._registry = registry
        self._endpoint_name = endpoint_name
        self._comparison_max_output_tokens = comparison_max_output_tokens

    def create(self, **kwargs: object) -> object:
        defer_failure = "extra_body" in kwargs
        params = self._adapt_request(kwargs)
        try:
            with self._registry.admit_sync(self._endpoint_name):
                return self._completions.create(**params)
        except Exception as exc:
            if defer_failure:
                return _DeferredJudgeFailure(exc)
            raise

    def _adapt_request(self, kwargs: dict[str, object]) -> dict[str, object]:
        params = copy.deepcopy(kwargs)
        raw_extra_body = params.get("extra_body")
        extra_body = dict(raw_extra_body) if isinstance(raw_extra_body, dict) else {}
        raw_chat_template = extra_body.get("chat_template_kwargs")
        chat_template_kwargs = (
            dict(raw_chat_template) if isinstance(raw_chat_template, dict) else {}
        )
        is_score_probe = (
            params.get("max_tokens") == 1
            and extra_body.get("continue_final_message") is True
        )
        if is_score_probe:
            params.pop("reasoning_effort", None)
            chat_template_kwargs["enable_thinking"] = False
        else:
            params.pop("max_completion_tokens", None)
            params["max_tokens"] = self._comparison_max_output_tokens
            params["reasoning_effort"] = "xhigh"
            chat_template_kwargs["enable_thinking"] = True
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        params["extra_body"] = extra_body
        params["timeout"] = self._registry.request_timeout_seconds(self._endpoint_name)
        return params


class _Server60JudgeChat:
    def __init__(
        self,
        chat: _JudgeChat,
        registry: EndpointAdmissionRegistry,
        endpoint_name: str,
        comparison_max_output_tokens: int,
    ) -> None:
        self.completions = _Server60JudgeCompletions(
            chat.completions,
            registry,
            endpoint_name,
            comparison_max_output_tokens,
        )


class Server60JudgeClient:
    """Adapt llm-verifier's OpenAI calls to server60 reasoning semantics."""

    def __init__(
        self,
        client: object,
        registry: EndpointAdmissionRegistry,
        endpoint_name: str,
        comparison_max_output_tokens: int,
    ) -> None:
        """Wrap the chat-completion surface used by llm-verifier."""
        raw_chat = getattr(client, "chat", None)
        if raw_chat is None:
            raise TypeError("server60 judge client requires chat completions")
        self._client = client
        self.chat = _Server60JudgeChat(
            cast(_JudgeChat, raw_chat),
            registry,
            endpoint_name,
            comparison_max_output_tokens,
        )
