from functools import lru_cache
import logging
from typing import AsyncIterator, Optional

from app.services.llm.base import LLMConfig, LLMResponse, LLMMessage, StreamChunk
from app.services.llm.registry import registry
from app.services.llm.router import LLMRouter, RoutingRequest, RoutingResult
logger = logging.getLogger(__name__)

from app.services.llm.errors import LLMContentFilteredError

class LLMService:

    def __init__(self):
        self._router = LLMRouter(registry)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        allow_fallback: bool = True,
    ) -> RoutingResult:
        """
        Complete a conversation. Returns a RoutingResult that includes
        the response plus metadata about which provider/model was used.
        """
        config = LLMConfig(temperature=temperature, max_tokens=max_tokens)
        request = RoutingRequest(
            messages=messages,
            config=config,
            provider=provider,
            model=model,
            allow_fallback=allow_fallback,
        )
        result = await self._router.route(request)

        if result.response.was_truncated:
            logger.warning(
                "Response truncated [%s/%s] — consider increasing max_tokens",
                result.provider_used, result.model_used,
            )
        if result.response.was_filtered:
            raise LLMContentFilteredError("Response blocked by content policy")
        if result.was_fallback:
            logger.info(
                "Used fallback provider: %s/%s",
                result.provider_used, result.model_used,
            )

        return result

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming version — yields chunks as they arrive."""
        config = LLMConfig(temperature=temperature, max_tokens=max_tokens)

        # Resolve provider/model same way as complete()
        request = RoutingRequest(
            messages=messages,
            config=config,
            provider=provider,
            model=model,
            allow_fallback=False,  # Streaming can't fallback mid-stream
        )
        llm_provider, llm_model = self._router._resolve(request)

        async for chunk in llm_provider.stream(messages, llm_model, config):
            yield chunk

    async def available_models(self) -> dict[str, list[str]]:
        return registry.available_models()

    async def health(self) -> dict[str, bool]:
        return await registry.health_status()


@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService()
