import time
import logging
from typing import AsyncIterator

import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.services.llm.base import (
    FinishReason,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    LLMMessage,
    StreamChunk,
)
from app.config import settings

logger = logging.getLogger(__name__)

_FINISH_REASON_MAP = {
    "end_turn": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "stop_sequence": FinishReason.STOP,
    "tool_use": FinishReason.STOP,
}


def _split_system(messages: list[LLMMessage]) -> tuple[str, list[LLMMessage]]:
    """Anthropic requires system prompt as a separate argument."""
    system_parts = [m.content for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]
    return "\n\n".join(system_parts), non_system


class AnthropicProvider(LLMProvider):
    def __init__(self):
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.ANTHROPIC_TIMEOUT,
            max_retries=0,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def supported_models(self) -> list[str]:
        return [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(anthropic.RateLimitError),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        config: LLMConfig,
    ) -> LLMResponse:
        t0 = time.monotonic()
        system_prompt, user_messages = _split_system(messages)

        try:
            kwargs: dict = {
                "model": model,
                "max_tokens": config.max_tokens,
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in user_messages
                ],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            if config.temperature is not None:
                kwargs["temperature"] = config.temperature
            if config.stop_sequences:
                kwargs["stop_sequences"] = config.stop_sequences

            resp = await self._client.messages.create(**kwargs)
            latency = (time.monotonic() - t0) * 1000

            content = ""
            for block in resp.content:
                if block.type == "text":
                    content += block.text

            return LLMResponse(
                content=content,
                model=resp.model,
                provider=self.provider_name,
                finish_reason=_FINISH_REASON_MAP.get(
                    resp.stop_reason, FinishReason.UNKNOWN
                ),
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
                latency_ms=latency,
                raw_response=resp,
            )

        except anthropic.AuthenticationError as e:
            logger.error("Anthropic auth failed: %s", e)
            raise
        except anthropic.APIConnectionError as e:
            logger.error("Anthropic connection error: %s", e)
            raise

    async def stream(
        self,
        messages: list[LLMMessage],
        model: str,
        config: LLMConfig,
    ) -> AsyncIterator[StreamChunk]:
        system_prompt, user_messages = _split_system(messages)

        async with self._client.messages.stream(
            model=model,
            max_tokens=config.max_tokens,
            messages=[
                {"role": m.role, "content": m.content} for m in user_messages
            ],
            system=system_prompt or anthropic.NOT_GIVEN,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield StreamChunk(content=event.delta.text)
                elif event.type == "message_stop":
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        finish_reason=FinishReason.STOP,
                    )

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
