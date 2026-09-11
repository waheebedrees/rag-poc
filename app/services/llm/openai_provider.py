import time
import logging
from typing import AsyncIterator

import openai
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
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self._client = openai.AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            timeout=settings.OPENAI_TIMEOUT,
            max_retries=0,  
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o3-mini",
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(openai.RateLimitError),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        config: LLMConfig,
    ) -> LLMResponse:
        t0 = time.monotonic()
        model = model or self.supported_models[0]

        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": m.role, "content": m.content} for m in messages
                ],
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
            # o1 models don't support temperature/system messages
            if model.startswith("o1") or model.startswith("o3"):
                kwargs.pop("temperature", None)
                kwargs.pop("top_p", None)
                kwargs["messages"] = [
                    m for m in kwargs["messages"] if m["role"] != "system"
                ]

            if config.stop_sequences:
                kwargs["stop"] = config.stop_sequences
            if config.frequency_penalty:
                kwargs["frequency_penalty"] = config.frequency_penalty
            if config.presence_penalty:
                kwargs["presence_penalty"] = config.presence_penalty

            resp = await self._client.chat.completions.create(**kwargs)
            latency = (time.monotonic() - t0) * 1000

            choice = resp.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                model=resp.model,
                provider=self.provider_name,
                finish_reason=_FINISH_REASON_MAP.get(
                    choice.finish_reason, FinishReason.UNKNOWN
                ),
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                latency_ms=latency,
                raw_response=resp,
            )

        except openai.AuthenticationError as e:
            logger.error("OpenAI auth failed: %s", e)
            raise
        except openai.APIConnectionError as e:
            logger.error("OpenAI connection error: %s", e)
            raise
        except openai.BadRequestError as e:
            logger.error("OpenAI bad request: %s", e)
            raise

    async def stream(
        self,
        messages: list[LLMMessage],
        model: str,
        config: LLMConfig,
    ) -> AsyncIterator[StreamChunk]:
        model = model or self.supported_models[0]

        async with self._client.chat.completions.stream(
            model=model,
            messages=[{"role": m.role, "content": m.content}
                      for m in messages],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                is_final = chunk.choices[0].finish_reason is not None
                yield StreamChunk(
                    content=delta.content or "",
                    is_final=is_final,
                    finish_reason=_FINISH_REASON_MAP.get(
                        chunk.choices[0].finish_reason, FinishReason.UNKNOWN
                    ) if is_final else None,
                )

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
