import logging
from dataclasses import dataclass
from typing import Optional

from app.services.llm.base import LLMConfig, LLMProvider, LLMResponse, LLMMessage
from app.services.llm.registry import ProviderRegistry
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RoutingRequest:
    messages: list[LLMMessage]
    config: LLMConfig
    provider: Optional[str] = None   # Explicit provider override
    model: Optional[str] = None      # Explicit model override
    allow_fallback: bool = True


@dataclass
class RoutingResult:
    response: LLMResponse
    provider_used: str
    model_used: str
    was_fallback: bool = False


class LLMRouter:
    """
    Decides which provider + model to use for a request.

    Priority:
      1. Explicit provider + model from request
      2. Explicit model (auto-detect provider)
      3. Default provider + model from config
      4. Fallback provider if primary fails and allow_fallback=True
    """

    def __init__(self, registry: ProviderRegistry):
        self._registry = registry

    async def route(self, request: RoutingRequest) -> RoutingResult:
        provider, model = self._resolve(request)

        try:
            response = await provider.complete(
                messages=request.messages,
                model=model,
                config=request.config,
            )
            return RoutingResult(
                response=response,
                provider_used=provider.provider_name,
                model_used=model,
            )

        except Exception as primary_error:
            if not request.allow_fallback:
                raise

            logger.warning(
                "Primary provider %s/%s failed: %s. Trying fallback.",
                provider.provider_name, model, primary_error,
            )
            return await self._fallback(request, primary_error)

    def _resolve(self, request: RoutingRequest) -> tuple[LLMProvider, str]:
        # Case 1: both provider and model specified
        if request.provider and request.model:
            provider = self._registry.get(request.provider)
            if not provider:
                raise ValueError(
                    f"Provider '{request.provider}' not registered. "
                    f"Available: {list(self._registry._providers.keys())}"
                )
            return provider, request.model

        # Case 2: model specified, auto-detect provider
        if request.model:
            provider = self._registry.find_provider_for_model(request.model)
            if not provider:
                raise ValueError(
                    f"No provider found for model '{request.model}'"
                )
            return provider, request.model

        # Case 3: provider specified, use its default model
        if request.provider:
            provider = self._registry.get(request.provider)
            if not provider:
                raise ValueError(f"Provider '{request.provider}' not registered")
            model = self._default_model_for(request.provider)
            return provider, model

        # Case 4: use configured defaults
        provider = self._registry.get(settings.DEFAULT_LLM_PROVIDER)
        if not provider:
            raise ValueError(
                f"Default provider '{settings.DEFAULT_LLM_PROVIDER}' not registered"
            )
        return provider, settings.DEFAULT_LLM_MODEL

    def _default_model_for(self, provider_name: str) -> str:
        defaults = {
            "openai": settings.DEFAULT_LLM_MODEL,
            "anthropic": settings.FALLBACK_LLM_MODEL,
            "google": "gemini-2.5-flash",
            "ollama": "llama3.2",
        }
        return defaults.get(provider_name, "")

    async def _fallback(
        self, request: RoutingRequest, original_error: Exception
    ) -> RoutingResult:
        fallback_provider = self._registry.get(settings.FALLBACK_LLM_PROVIDER)
        if not fallback_provider:
            logger.error("Fallback provider not available either")
            raise original_error

        fallback_model = settings.FALLBACK_LLM_MODEL

        try:
            response = await fallback_provider.complete(
                messages=request.messages,
                model=fallback_model,
                config=request.config,
            )
            logger.info(
                "Fallback succeeded: %s/%s",
                fallback_provider.provider_name, fallback_model,
            )
            return RoutingResult(
                response=response,
                provider_used=fallback_provider.provider_name,
                model_used=fallback_model,
                was_fallback=True,
            )

        except Exception as fallback_error:
            logger.error(
                "Fallback also failed: %s", fallback_error
            )
            raise original_error 