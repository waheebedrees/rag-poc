import logging
from typing import Optional

from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Central registry of all available providers.
    Providers are registered once at startup and reused.
    """

    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider):
        name = provider.provider_name
        self._providers[name] = provider
        logger.info("Registered LLM provider: %s", name)

    def get(self, name: str) -> Optional[LLMProvider]:
        return self._providers.get(name)

    def all_providers(self) -> list[LLMProvider]:
        return list(self._providers.values())

    def available_models(self) -> dict[str, list[str]]:
        return {
            name: p.supported_models
            for name, p in self._providers.items()
        }

    def find_provider_for_model(self, model: str) -> Optional[LLMProvider]:
        """Find which provider handles a given model name."""
        for provider in self._providers.values():
            if provider.supports_model(model):
                return provider
        return None

    async def health_status(self) -> dict[str, bool]:
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception:
                results[name] = False
        return results


registry = ProviderRegistry()

def setup_registry() -> ProviderRegistry:
    """
    Called once at startup.
    Only registers providers whose API keys are configured.
    """
    from app.config import settings

    if settings.OPENAI_API_KEY:
        from app.services.llm.openai_provider import OpenAIProvider
        registry.register(OpenAIProvider())

    if settings.ANTHROPIC_API_KEY:
        from app.services.llm.anthropic_provider import AnthropicProvider
        registry.register(AnthropicProvider())

    if settings.GOOGLE_API_KEY:
        from app.services.llm.google_provider import GoogleProvider
        registry.register(GoogleProvider())

    # Ollama is always registered — it's local and has no key
    from app.services.llm.ollama_provider import OllamaProvider
    registry.register(OllamaProvider())

    logger.info(
        "LLM registry ready. Providers: %s",
        list(registry._providers.keys()),
    )
    return registry




setup_registry()