import asyncio
import json
import time
import uuid
from dataclasses import replace as dataclass_replace
from typing import AsyncIterator, Dict, List, Optional, Tuple

from ..model_execution import (
    ExecutionCompleted,
    GenerationIntent,
    ModelExecutionRequest,
    ModelExecutionResult,
    ModelExecutor,
    ModelTarget,
    TextDelta,
    ThinkingIntent,
    ToolCallArgumentsDelta,
    ToolCallStarted,
    build_candidate_execution,
)
from ..utils import (
    AnthropicToOpenAI,
    Config,
    OpenAIToAnthropic,
    STOP_REASON_MAP,
    SSEFormatter,
    create_logger,
    create_request_log,
    save_request_log,
)
from ..context import ContextRefiner
from ..progress_monitor import ProgressMonitor
from ..verifier import Verifier

logger = create_logger("backend")

# Canonical finish reasons -> OpenAI wire values.
_FINISH_TO_OPENAI = {
    "stop": "stop",
    "length": "length",
    "tool_use": "tool_calls",
    "content_filter": "content_filter",
    "other": "stop",
}

# Canonical finish reasons -> Anthropic stop reasons.
_FINISH_TO_ANTHROPIC = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_use": "tool_use",
    "content_filter": "end_turn",
    "other": "end_turn",
}


def _generation_intent(fields: dict) -> GenerationIntent:
    """Build the client's explicit generation intent from request fields.

    Only fields the client actually sent appear here; configured target
    defaults are applied inside model execution, and the target's output cap
    clamps last.
    """
    kwargs: dict = {}
    if fields.get("max_completion_tokens") is not None:
        kwargs["max_output_tokens"] = fields["max_completion_tokens"]
        kwargs["max_output_tokens_field"] = "max_completion_tokens"
    elif fields.get("max_tokens") is not None:
        kwargs["max_output_tokens"] = fields["max_tokens"]
    for key in (
        "temperature", "top_p", "stop", "tool_choice", "response_format",
        "seed", "presence_penalty", "frequency_penalty", "logit_bias",
        "stream_options",
    ):
        if key in fields:
            kwargs[key] = fields[key]
    thinking = fields.get("thinking")
    if isinstance(thinking, ThinkingIntent):
        kwargs["thinking"] = thinking
    elif fields.get("reasoning_effort") is not None:
        kwargs["thinking"] = ThinkingIntent(
            mode="adaptive", effort=fields["reasoning_effort"],
        )
    elif fields.get("thinking_budget") is not None:
        kwargs["thinking"] = ThinkingIntent(
            mode="budget", budget_tokens=int(fields["thinking_budget"]),
        )
    return GenerationIntent(**kwargs)


def _result_to_response_dict(result: ModelExecutionResult) -> dict:
    """Render a canonical result as an OpenAI-shaped response dict for the
    downstream pipeline (verification, request trace, SSE replay)."""
    message: dict = {"role": "assistant"}
    message["content"] = result.output.text or None
    if result.output.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in result.output.tool_calls
        ]
    response: dict = {
        "id": result.response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": 0,
        "model": result.target.name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _FINISH_TO_OPENAI[
                    result.output.finish_reason
                ],
            }
        ],
    }
    if result.usage:
        response["usage"] = {
            "prompt_tokens": result.usage.input_tokens,
            "completion_tokens": result.usage.output_tokens,
            "total_tokens": (
                result.usage.input_tokens + result.usage.output_tokens
            ),
        }
    return response


class Backend:
    """Request pipeline: (optional) context refinement -> concurrent inference
    -> pivot-tournament verification -> best response."""

    def __init__(self, config: Config, executor: Optional[ModelExecutor] = None):
        self.config = config

        execution = build_candidate_execution(config)
        if executor is not None:
            execution = dataclass_replace(execution, executor=executor)
        self._execution = execution

        self.refiner: Optional[ContextRefiner] = None
        self.verifier: Optional[Verifier] = None
        self.progress_monitor: Optional[ProgressMonitor] = None
        self._bg_tasks: set = set()  # strong refs so background tasks aren't GC'd

        ctx_cfg = config.context_config
        if ctx_cfg:
            self.refiner = ContextRefiner(ctx_cfg)
            logger.info(f"Context refinement enabled (model={ctx_cfg.model_name})")

        ver_cfg = config.verifier_config
        if ver_cfg and config.total_candidates > 1:
            self.verifier = Verifier(ver_cfg)
            logger.info(
                f"Verifier enabled (total_candidates={config.total_candidates})"
            )

        pm_cfg = config.progress_monitor_config
        if pm_cfg:
            self.progress_monitor = ProgressMonitor(pm_cfg)
            logger.info(f"Progress monitor enabled (model={pm_cfg.model.name})")

    @property
    def model_name(self) -> str:
        """Display name of the default backend model (startup log line)."""
        return self._execution.default_target.name

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _sanitized_config(self) -> dict:
        raw = dict(self.config.raw_config)
        if raw.get("backend", {}).get("models"):
            raw["backend"] = {
                **raw["backend"],
                "models": [
                    {**m, "api_key": "***"} for m in raw["backend"]["models"]
                ],
            }
        if raw.get("context", {}).get("refinement_model"):
            raw["context"] = {
                **raw["context"],
                "refinement_model": {
                    **raw["context"]["refinement_model"],
                    "api_key": "***",
                },
            }
        if raw.get("verifier", {}).get("model"):
            raw["verifier"] = {
                **raw["verifier"],
                "model": {**raw["verifier"]["model"], "api_key": "***"},
            }
        if raw.get("progress_monitor", {}).get("model"):
            raw["progress_monitor"] = {
                **raw["progress_monitor"],
                "model": {**raw["progress_monitor"]["model"], "api_key": "***"},
            }
        return raw

    async def _refine_messages(
        self, request: ModelExecutionRequest, req_log: Optional[dict] = None,
    ) -> ModelExecutionRequest:
        if self.refiner:
            original_messages = list(request.messages)
            refined = await self.refiner.refine(original_messages)
            if req_log:
                req_log["contextRefinement"] = {
                    "enabled": True,
                    "originalMessages": original_messages,
                    "refinedMessages": refined,
                }
            return dataclass_replace(request, messages=tuple(refined))
        return request

    async def _gather_completions(
        self, request: ModelExecutionRequest,
    ) -> List[Tuple[dict, str]]:
        executor = self._execution.executor

        async def run(target: ModelTarget) -> Tuple[dict, str]:
            result = await executor.complete(target, request)
            return _result_to_response_dict(result), target.name

        tasks = [run(target) for target in self._execution.candidate_targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes: List[Tuple[dict, str]] = []
        errors: List[Exception] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"CONCURRENT REQUEST FAILED: {r}")
                errors.append(r)
            else:
                successes.append(r)

        if not successes:
            raise RuntimeError(
                f"All {len(errors)} concurrent requests failed. "
                f"First error: {errors[0]}"
            ) from errors[0]
        return successes

    async def _pick_best(
        self,
        responses: List[Tuple[dict, str]],
        messages: list,
        req_log: Optional[dict] = None,
    ) -> Tuple[dict, str]:
        if len(responses) == 1 or not self.verifier:
            return responses[0]

        history_str = Backend.format_history(messages)

        # Drop responses with empty choices.
        valid_responses: List[Tuple[dict, str]] = []
        for resp, model_name in responses:
            if resp.get("choices"):
                valid_responses.append((resp, model_name))
            else:
                logger.warn(
                    f"RESPONSE model={model_name} returned empty choices, skipping"
                )

        if not valid_responses:
            logger.error("All responses had empty choices, falling back to first")
            return responses[0]
        if len(valid_responses) == 1:
            return valid_responses[0]

        actions = [Backend.format_action(resp) for resp, _ in valid_responses]
        for (_, model_name), action in zip(valid_responses, actions):
            logger.info(f"RESPONSE model={model_name} text='{action[:50]}'")

        try:
            result = await self.verifier.select_best(history_str, actions)
        except Exception as e:
            logger.error(
                f"Verifier failed ({type(e).__name__}: {e}); "
                f"falling back to first response"
            )
            return valid_responses[0]

        best_idx = result.best_index
        verifier_scores = [
            {
                "index": i,
                "model": valid_responses[i][1],
                "score": result.scores[i] if i < len(result.scores) else 0.0,
                "details": {
                    "score": result.scores[i] if i < len(result.scores) else 0.0,
                    "criterionScores": [],
                },
            }
            for i in range(len(valid_responses))
        ]
        for s in verifier_scores:
            logger.info(f"VERIFY model={s['model']} score={s['score']:.3f}")

        best_resp, best_model = valid_responses[best_idx]
        best_score = result.scores[best_idx] if best_idx < len(result.scores) else 0.0
        logger.info(f"BEST model={best_model} score={best_score:.3f}")

        if req_log:
            req_log["verifier"] = {
                "enabled": True,
                "scores": verifier_scores,
                "comparisons": [c.to_dict() for c in result.comparisons],
                "bestIndex": best_idx,
                "bestModel": best_model,
                "bestScore": best_score,
            }

        return best_resp, best_model

    def _spawn_progress(
        self, messages: list, final_response: Optional[dict], req_log: dict,
    ) -> None:
        """Kick off progress evaluation in the background so it never delays the
        client's response. When it finishes it updates req_log and re-saves the
        log file (already written once without progress)."""
        if not self.progress_monitor:
            return
        log_dir = self.config.log_dir

        async def _run() -> None:
            await self._evaluate_progress(messages, final_response, req_log)
            save_request_log(req_log, log_dir)

        task = asyncio.create_task(_run())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _evaluate_progress(
        self, messages: list, final_response: Optional[dict],
        req_log: Optional[dict] = None,
    ) -> None:
        """Post-hoc progress estimate. Runs after the response is selected and
        never changes it — observability only. The score lands in the request
        log and the visualizer's progress node."""
        if not self.progress_monitor:
            return
        problem = Backend.format_history(messages)
        response_text = (
            Backend.format_action(final_response)
            if final_response and final_response.get("choices")
            else "(empty response)"
        )
        try:
            result = await self.progress_monitor.evaluate(problem, response_text)
            if req_log is not None:
                req_log["progressMonitor"] = {
                    "enabled": True,
                    "score": result.score,
                    "details": result.to_dict(),
                }
        except Exception as e:
            logger.error(f"Progress monitor failed: {type(e).__name__}: {e}")
            if req_log is not None:
                req_log["progressMonitor"] = {"enabled": True, "error": str(e)}

    @staticmethod
    def format_history(messages: list) -> str:
        parts: List[str] = []
        for msg in messages:
            role = (msg.get("role") or "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                texts: List[str] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            texts.append(
                                f"[tool_result: {block.get('content', '')}]"
                            )
                        elif block.get("type") == "tool_use":
                            texts.append(
                                f"[tool_use: {block.get('name', '')}"
                                f"({json.dumps(block.get('input', {}))})]"
                            )
                    else:
                        texts.append(str(block))
                content = "\n".join(texts)
            if content:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    @staticmethod
    def format_action(response: dict) -> str:
        message = response["choices"][0]["message"]
        parts: List[str] = []
        if message.get("content"):
            parts.append(message["content"])
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                parts.append(
                    f"[tool_call: {tc['function']['name']}"
                    f"({tc['function']['arguments']})]"
                )
        return "\n".join(parts) if parts else "(empty response)"

    # ------------------------------------------------------------------
    # Anthropic-format API
    # ------------------------------------------------------------------

    def _build_anthropic_params(
        self, anthropic_body: dict,
    ) -> Tuple[ModelExecutionRequest, str]:
        """Build a ModelExecutionRequest from an Anthropic-format request.

        Returns (request, requested_model). The requested model id is echoed
        back to the client in responses; target selection happens inside
        model execution.
        """
        requested_model = anthropic_body.get(
            "model", self._execution.default_target.name,
        )
        fields: dict = {}
        for key in ("max_tokens", "temperature", "top_p"):
            if key in anthropic_body:
                fields[key] = anthropic_body[key]
        if anthropic_body.get("stop_sequences"):
            fields["stop"] = anthropic_body["stop_sequences"]
        if anthropic_body.get("tool_choice"):
            fields["tool_choice"] = AnthropicToOpenAI.tool_choice(
                anthropic_body["tool_choice"]
            )

        # Honor the client's thinking request when present; configured target
        # thinking remains the default for requests that omit it. Client
        # thinking replaces the config's thinking settings wholesale: a budget
        # request must not leave a YAML reasoning_effort in force, and vice
        # versa.
        thinking = anthropic_body.get("thinking")
        if isinstance(thinking, dict):
            # Pi adaptive thinking puts effort on output_config, not
            # thinking.effort. Honor either, plus a top-level effort field.
            output_config = anthropic_body.get("output_config")
            effort = thinking.get("effort")
            if not effort and isinstance(output_config, dict):
                effort = output_config.get("effort")
            if not effort:
                effort = anthropic_body.get("effort")
            if effort or thinking.get("type") == "adaptive":
                fields["thinking"] = ThinkingIntent(
                    mode="adaptive", effort=effort or "high",
                )
            elif thinking.get("budget_tokens"):
                fields["thinking"] = ThinkingIntent(
                    mode="budget", budget_tokens=int(thinking["budget_tokens"]),
                )
            else:
                fields["thinking"] = ThinkingIntent(mode="disabled")

        tools = (
            AnthropicToOpenAI.tools(anthropic_body["tools"])
            if anthropic_body.get("tools") else []
        )
        request = ModelExecutionRequest(
            messages=tuple(AnthropicToOpenAI.messages(anthropic_body)),
            tools=tuple(tools),
            generation=_generation_intent(fields),
        )
        return request, requested_model

    async def complete_anthropic(
        self, body: bytes | str,
    ) -> Tuple[Optional[dict], Optional[str]]:
        start = time.monotonic()
        try:
            anthropic_body = json.loads(
                body if isinstance(body, str) else body.decode()
            )
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e}"

        req_log = create_request_log("anthropic", self._sanitized_config())
        req_log["request"] = anthropic_body

        request, requested_model = self._build_anthropic_params(anthropic_body)
        request = await self._refine_messages(request, req_log)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} "
                f"concurrent requests (anthropic)"
            )
            responses = await self._gather_completions(request)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            response, model_name = await self._pick_best(
                responses, list(request.messages), req_log,
            )
            final_result = OpenAIToAnthropic.response(response, requested_model)
        else:
            target = self._execution.default_target
            logger.info(f"BACKEND calling {target.name} (anthropic)")
            result = await self._execution.executor.complete(target, request)
            response = _result_to_response_dict(result)
            req_log["responses"] = [
                {"model": target.name, "response": response}
            ]
            final_result = OpenAIToAnthropic.response(response, requested_model)

        req_log["finalResponse"] = final_result
        req_log["elapsedMs"] = (time.monotonic() - start) * 1000
        save_request_log(req_log, self.config.log_dir)
        self._spawn_progress(list(request.messages), response, req_log)
        return final_result, None

    async def stream_anthropic(
        self, body: bytes | str,
    ) -> AsyncIterator[str]:
        start = time.monotonic()
        anthropic_body = json.loads(
            body if isinstance(body, str) else body.decode()
        )
        req_log = create_request_log(
            "anthropic_stream", self._sanitized_config(),
        )
        req_log["request"] = anthropic_body

        request, requested_model = self._build_anthropic_params(anthropic_body)
        request = await self._refine_messages(request, req_log)

        # When the verifier is active, collect all responses, verify, replay.
        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} concurrent "
                f"requests for verification (anthropic stream)"
            )
            responses = await self._gather_completions(request)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            best_resp, best_model = await self._pick_best(
                responses, list(request.messages), req_log,
            )
            req_log["finalResponse"] = best_resp
            req_log["elapsedMs"] = (time.monotonic() - start) * 1000
            save_request_log(req_log, self.config.log_dir)
            self._spawn_progress(list(request.messages), best_resp, req_log)
            async for event in self._replay_anthropic_sse(
                best_resp, requested_model,
            ):
                yield event
            return

        logger.info(
            f"BACKEND streaming {self._execution.default_target.name} (anthropic)"
        )

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield SSEFormatter.message_start(requested_model, msg_id)

        block_index = 0
        text_block_open = False
        tool_blocks: dict[str, dict] = {}
        output_tokens = 0

        async for event in self._execution.executor.stream(
            self._execution.default_target, request,
        ):
            if isinstance(event, TextDelta):
                if not text_block_open:
                    yield SSEFormatter.content_block_start(block_index, "text")
                    text_block_open = True
                yield SSEFormatter.text_delta(block_index, event.text)
            elif isinstance(event, ToolCallStarted):
                if text_block_open:
                    yield SSEFormatter.content_block_stop(block_index)
                    block_index += 1
                    text_block_open = False

                tool_blocks[event.id] = {
                    "index": block_index,
                    "name": event.name,
                }
                yield SSEFormatter.content_block_start(
                    block_index,
                    "tool_use",
                    tool_id=event.id,
                    tool_name=event.name,
                )
            elif isinstance(event, ToolCallArgumentsDelta):
                target_block = tool_blocks.get(event.id)
                if target_block and event.json_fragment:
                    yield SSEFormatter.input_json_delta(
                        target_block["index"],
                        event.json_fragment,
                    )
            elif isinstance(event, ExecutionCompleted):
                if text_block_open:
                    yield SSEFormatter.content_block_stop(block_index)
                    text_block_open = False

                for tinfo in tool_blocks.values():
                    yield SSEFormatter.content_block_stop(tinfo["index"])

                if event.result.usage:
                    output_tokens = event.result.usage.output_tokens
                stop_reason = _FINISH_TO_ANTHROPIC[
                    event.result.output.finish_reason
                ]
                yield SSEFormatter.message_delta(stop_reason, output_tokens)
                yield SSEFormatter.message_stop()

    async def _replay_anthropic_sse(
        self, response: dict, model_name: str,
    ) -> AsyncIterator[str]:
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield SSEFormatter.message_start(model_name, msg_id)

        choice = response["choices"][0]
        message = choice["message"]
        block_index = 0

        if message.get("content"):
            yield SSEFormatter.content_block_start(block_index, "text")
            yield SSEFormatter.text_delta(block_index, message["content"])
            yield SSEFormatter.content_block_stop(block_index)
            block_index += 1

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                yield SSEFormatter.content_block_start(
                    block_index,
                    "tool_use",
                    tool_id=tc.get("id", ""),
                    tool_name=tc["function"]["name"],
                )
                if tc["function"].get("arguments"):
                    yield SSEFormatter.input_json_delta(
                        block_index, tc["function"]["arguments"],
                    )
                yield SSEFormatter.content_block_stop(block_index)
                block_index += 1

        stop_reason = STOP_REASON_MAP.get(
            choice.get("finish_reason", "stop"), "end_turn",
        )
        output_tokens = response.get("usage", {}).get("completion_tokens", 0)
        yield SSEFormatter.message_delta(stop_reason, output_tokens)
        yield SSEFormatter.message_stop()

    # ------------------------------------------------------------------
    # OpenAI-format API
    # ------------------------------------------------------------------

    def _build_openai_params(self, openai_body: dict) -> Tuple[ModelExecutionRequest, str]:
        requested_model = openai_body.get(
            "model", self._execution.default_target.name,
        )
        fields: dict = {}

        direct_keys = [
            "temperature", "top_p", "stop", "tool_choice",
            "response_format", "seed", "presence_penalty",
            "frequency_penalty", "logit_bias", "stream_options",
        ]
        for key in direct_keys:
            if key in openai_body:
                fields[key] = openai_body[key]

        # Pi sends only max_completion_tokens. The client's token field
        # choice must survive to the wire so a backend that maps only one
        # key cannot ignore the client cap.
        if openai_body.get("max_completion_tokens"):
            fields["max_completion_tokens"] = openai_body["max_completion_tokens"]
        elif openai_body.get("max_tokens"):
            fields["max_tokens"] = openai_body["max_tokens"]
        if openai_body.get("reasoning_effort"):
            fields["reasoning_effort"] = openai_body["reasoning_effort"]

        request = ModelExecutionRequest(
            messages=tuple(openai_body.get("messages", [])),
            tools=tuple(openai_body.get("tools") or []),
            generation=_generation_intent(fields),
        )
        return request, requested_model

    async def complete_openai(
        self, body: bytes | str,
    ) -> Tuple[Optional[dict], Optional[str]]:
        start = time.monotonic()
        try:
            openai_body = json.loads(
                body if isinstance(body, str) else body.decode()
            )
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e}"

        req_log = create_request_log("openai", self._sanitized_config())
        req_log["request"] = openai_body

        request, requested_model = self._build_openai_params(openai_body)
        request = await self._refine_messages(request, req_log)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} "
                f"concurrent requests (openai)"
            )
            responses = await self._gather_completions(request)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            response, _ = await self._pick_best(
                responses, list(request.messages), req_log,
            )
            final_result = response
        else:
            target = self._execution.default_target
            logger.info(f"BACKEND calling {target.name} (openai)")
            result = await self._execution.executor.complete(target, request)
            final_result = _result_to_response_dict(result)
            req_log["responses"] = [
                {"model": target.name, "response": final_result}
            ]

        # Echo the model the client asked for; the backend model stays in the
        # request log for debugging.
        final_result["model"] = requested_model

        req_log["finalResponse"] = final_result
        req_log["elapsedMs"] = (time.monotonic() - start) * 1000
        save_request_log(req_log, self.config.log_dir)
        self._spawn_progress(list(request.messages), final_result, req_log)
        return final_result, None

    async def stream_openai(
        self, body: bytes | str,
    ) -> AsyncIterator[str]:
        start = time.monotonic()
        openai_body = json.loads(
            body if isinstance(body, str) else body.decode()
        )
        req_log = create_request_log("openai_stream", self._sanitized_config())
        req_log["request"] = openai_body

        request, requested_model = self._build_openai_params(openai_body)
        request = await self._refine_messages(request, req_log)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} concurrent "
                f"requests for verification (openai stream)"
            )
            responses = await self._gather_completions(request)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            best_resp, _ = await self._pick_best(
                responses, list(request.messages), req_log,
            )
            req_log["finalResponse"] = best_resp
            req_log["elapsedMs"] = (time.monotonic() - start) * 1000
            save_request_log(req_log, self.config.log_dir)
            self._spawn_progress(list(request.messages), best_resp, req_log)
            async for event in self._replay_openai_sse(best_resp, requested_model):
                yield event
            return

        logger.info(
            f"BACKEND streaming {self._execution.default_target.name} (openai)"
        )

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        first_delta = True
        tool_indexes: Dict[str, int] = {}
        next_tool_index = 0

        async for event in self._execution.executor.stream(
            self._execution.default_target, request,
        ):
            delta: dict = {}
            if isinstance(event, TextDelta):
                delta["content"] = event.text
                if first_delta:
                    delta["role"] = "assistant"
            elif isinstance(event, ToolCallStarted):
                tool_indexes[event.id] = next_tool_index
                next_tool_index += 1
                delta["tool_calls"] = [
                    {
                        "index": tool_indexes[event.id],
                        "id": event.id,
                        "type": "function",
                        "function": {"name": event.name, "arguments": ""},
                    }
                ]
            elif isinstance(event, ToolCallArgumentsDelta):
                delta["tool_calls"] = [
                    {
                        "index": tool_indexes.get(event.id, 0),
                        "function": {"arguments": event.json_fragment},
                    }
                ]
            elif isinstance(event, ExecutionCompleted):
                result = event.result
                chunk: dict = {
                    "id": result.response_id or chunk_id,
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": requested_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": _FINISH_TO_OPENAI[
                                result.output.finish_reason
                            ],
                        }
                    ],
                }
                if result.usage:
                    chunk["usage"] = {
                        "prompt_tokens": result.usage.input_tokens,
                        "completion_tokens": result.usage.output_tokens,
                        "total_tokens": (
                            result.usage.input_tokens
                            + result.usage.output_tokens
                        ),
                    }
                yield f"data: {json.dumps(chunk, default=str)}\n\n"
                continue
            else:
                continue

            first_delta = False
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': 0, 'model': requested_model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}, default=str)}\n\n"

        yield "data: [DONE]\n\n"

    async def _replay_openai_sse(
        self, response: dict, model_name: str,
    ) -> AsyncIterator[str]:
        choices = response.get("choices", [])
        choice = choices[0] if choices else {}

        chunk = {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion.chunk",
            "created": response.get("created", 0),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": choice.get("message", {}),
                    "finish_reason": choice.get("finish_reason"),
                },
            ],
        }
        yield f"data: {json.dumps(chunk, default=str)}\n\n"
        yield "data: [DONE]\n\n"

    @staticmethod
    def _model_metadata(model: dict) -> dict:
        """Rich metadata for GET /v1/models. Pi and other clients use these
        fields to register models with sane context/cost defaults."""
        name = model["name"]
        cfg_max = model.get("max_tokens")
        prefix = name.split("/", 1)[0] if "/" in name else ""
        context_defaults = {
            "gemini": 1_000_000,
            "openai": 200_000,
            "anthropic": 200_000,
            "openrouter": 200_000,
            "kimi": 128_000,
            "zai": 128_000,
        }
        return {
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "turbo-agent",
            "name": name,
            "context_window": model.get("context_window")
            or context_defaults.get(prefix, 128_000),
            "max_tokens": cfg_max or 8192,
            "reasoning": True,
            "input": (["text", "image"] if prefix == "gemini" else ["text"]),
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        }

    def get_models_response(self) -> dict:
        return {
            "object": "list",
            "data": [self._model_metadata(m) for m in self.config.models],
        }
