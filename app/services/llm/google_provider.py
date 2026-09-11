import time
import logging
from typing import AsyncIterator

from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
from google.genai.client import AsyncClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
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
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
}


def _to_gemini_messages(
    messages: list[LLMMessage],
) -> tuple[str, list[genai_types.Content]]:
    """
    Gemini's API wants:
      - system prompt as a separate `system_instruction` field
      - conversation as a list of Content objects with roles 'user' / 'model'
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    system_instruction = "\n\n".join(system_parts).strip()

    contents: list[genai_types.Content] = []
    for m in messages:
        if m.role == "system":
            continue
        role = "model" if m.role == "assistant" else "user"
        contents.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=m.content)],
            )
        )
    return system_instruction, contents


class GoogleProvider(LLMProvider):
    def __init__(self):
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def supported_models(self) -> list[str]:
        return [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-8b",
            "gemini-2.5-pro",
        ]

    @retry(
        retry=retry_if_exception_type(
            (genai_errors.APIError, ConnectionError, TimeoutError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
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

        system_instruction, contents = _to_gemini_messages(messages)
        if not contents:
            contents = [
                genai_types.Content(
                    role="user", parts=[genai_types.Part(text="")]
                )
            ]

        gen_config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
            stop_sequences=config.stop_sequences or None,
        )

        try:
            chat = self._client.aio.chats.create(
                model=model,
                config=gen_config,
                # list[Content] is fine for history
                history=contents[:-1],
            )

            last = contents[-1]                  # Content
            resp = await chat.send_message(
                message=[genai_types.Part(text=p.text or "") for p in last.parts],
                config=gen_config,
            )

        except genai_errors.APIError as e:
            logger.error("Google Gemini API error: %s", e)
            raise

        latency = (time.monotonic() - t0) * 1000

        finish_reason = FinishReason.UNKNOWN
        if resp.candidates:
            fr = resp.candidates[0].finish_reason
            fr_name = getattr(fr, "name", None) or str(fr)
            finish_reason = _FINISH_REASON_MAP.get(
                fr_name, FinishReason.UNKNOWN)

        usage = resp.usage_metadata
        return LLMResponse(
            content=resp.text or "",
            model=model,
            provider=self.provider_name,
            finish_reason=finish_reason,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
            latency_ms=latency,
            raw_response=resp,
        )

    async def health_check(self) -> bool:
        try:
            # paged iterator — just consume one page
            await self._client.aio.models.list()
            return True
        except Exception as e:
            logger.warning("Google health check failed: %s", e)
            return False


    async def stream(
        self,
        messages: list[LLMMessage],
        model: str,
        config: LLMConfig,
    ) -> AsyncIterator[StreamChunk]:
        model = model or self.supported_models[0]

        system_instruction, contents = _to_gemini_messages(messages)
        if not contents:
            contents = [genai_types.Content(
                role="user", parts=[genai_types.Part(text="")])]

        gen_config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
            tools=[],
        )

        chat = self._client.aio.chats.create(
            model=model,
            config=gen_config,
            history=contents[:-1],
        )

        last_parts = [genai_types.Part(text=p.text or "")
                    for p in contents[-1].parts]

        async for chunk in await chat.send_message_stream(last_parts):
            text = chunk.text or ""
            if not text:
                continue
            is_final = bool(chunk.candidates) and (
                chunk.candidates[0].finish_reason is not None
            )
            finish_reason = None
            if is_final:
                fr = chunk.candidates[0].finish_reason
                fr_name = getattr(fr, "name", None) or str(fr)
                finish_reason = _FINISH_REASON_MAP.get(
                    fr_name, FinishReason.UNKNOWN)

            yield StreamChunk(
                content=text,
                is_final=is_final,
                finish_reason=finish_reason,
            )
