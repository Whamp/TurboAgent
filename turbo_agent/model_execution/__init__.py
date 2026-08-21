"""Model execution seam: one deep module for candidate-producing model calls.

See docs/design/model-execution.md and docs/adr/0001-model-execution-seam.md.
"""

from .errors import FailureKind, ModelExecutionError
from .factory import ExecutionTargets, build_candidate_execution
from .litellm_adapter import LiteLLMExecutor
from .types import (
    AssistantOutput,
    ExecutionCompleted,
    GenerationIntent,
    ModelExecutionEvent,
    ModelExecutionRequest,
    ModelExecutionResult,
    ModelExecutor,
    ModelTarget,
    TargetSpec,
    TextDelta,
    ThinkingDelta,
    ThinkingIntent,
    TokenUsage,
    ToolCall,
    ToolCallArgumentsDelta,
    ToolCallStarted,
)

__all__ = [
    "AssistantOutput",
    "ExecutionCompleted",
    "ExecutionTargets",
    "FailureKind",
    "GenerationIntent",
    "LiteLLMExecutor",
    "ModelExecutionError",
    "ModelExecutionEvent",
    "ModelExecutionRequest",
    "ModelExecutionResult",
    "ModelExecutor",
    "ModelTarget",
    "TargetSpec",
    "TextDelta",
    "ThinkingDelta",
    "ThinkingIntent",
    "TokenUsage",
    "ToolCall",
    "ToolCallArgumentsDelta",
    "ToolCallStarted",
    "build_candidate_execution",
]
