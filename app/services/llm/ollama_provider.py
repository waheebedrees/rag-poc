import time
import logging
import json
from typing import AsyncIterator

import httpx

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


class OllamaProvider(LLMProvider):
    """
    Local Ollama instance — no API key needed.
    Models are whatever you have pulled locally.
    """

    def __init__(self):
        self._base = settings.OLLAMA_BASE_URL.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=settings.OLLAMA_TIMEOUT,
        )
        self._supported: list[str] = []  # Populated lazily

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def supported_models(self) -> list[str]:
        return self._supported

    async def _fetch_available_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning("Could not fetch Ollama models: %s", e)
            return []

    async def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        config: LLMConfig,
    ) -> LLMResponse:
        model = model or self.supported_models[0]

        t0 = time.monotonic()
        payload = {
            "model": model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "top_p": config.top_p,
            },
        }

        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = (time.monotonic() - t0) * 1000

            return LLMResponse(
                content=data["message"]["content"],
                model=model,
                provider=self.provider_name,
                finish_reason=(
                    FinishReason.STOP
                    if data.get("done")
                    else FinishReason.LENGTH
                ),
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                total_tokens=(
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
                latency_ms=latency,
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            logger.error("Ollama request failed: %s", e)
            raise
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s", self._base)
            raise

    async def stream(
        self,
        messages: list[LLMMessage],
        model: str,
        config: LLMConfig,
    ) -> AsyncIterator[StreamChunk]:
        model = model or self.supported_models[0]

        payload = {
            "model": model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }

        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = data.get("message", {}).get("content", "")
                done = data.get("done", False)

                yield StreamChunk(
                    content=content,
                    is_final=done,
                    finish_reason=FinishReason.STOP if done else None,
                )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    def supports_model(self, model: str) -> bool:
        # Ollama: we trust the caller to know their pulled models
        return True