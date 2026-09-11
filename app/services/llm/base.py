
from typing import Any, AsyncIterator, Optional
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class LLMMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMConfig:
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    finish_reason: FinishReason
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    raw_response: Any = None

    @property
    def was_truncated(self) -> bool:
        return self.finish_reason == FinishReason.LENGTH

    @property
    def was_filtered(self) -> bool:
        return self.finish_reason == FinishReason.CONTENT_FILTER


@dataclass
class StreamChunk:
    content: str
    is_final: bool = False
    finish_reason: Optional[FinishReason] = None


class LLMProvider(ABC):
    """
    Every provider implements this interface.
    Nothing else in the codebase imports provider-specific code.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def supported_models(self) -> list[str]: ...

    @abstractmethod
    async def complete(
        self,
        LLMMessages: list[LLMMessage],
        model: str,
        config: LLMConfig,
    ) -> LLMResponse: ...

    @abstractmethod
    async def stream(
        self,
        LLMMessages: list[LLMMessage],
        model: str,
        config: LLMConfig,
    ) -> AsyncIterator[StreamChunk]: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    def supports_model(self, model: str) -> bool:
        return model in self.supported_models
