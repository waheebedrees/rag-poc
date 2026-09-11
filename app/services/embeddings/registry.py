from app.config import settings
import logging

from app.services.embeddings.base import EmbeddingProvider
logger = logging.getLogger(__name__)


class EmbeddingRegistry:

    def __init__(self):
        self._providers: dict[str, EmbeddingProvider] = {}

    def register(self, provider: EmbeddingProvider) -> None:
        if not provider.provider_name:
            raise ValueError("Provider must define a provider_name")
        self._providers[provider.provider_name] = provider
        logger.info("Registered embedding provider: %s",
                    provider.provider_name)

    def get(self, name: str) -> EmbeddingProvider | None:
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def find_provider_for_model(self, model: str) -> EmbeddingProvider | None:
        for p in self._providers.values():
            if p.supports_model(model):
                return p
        return None

    async def health_status(self) -> dict[str, bool]:
        return {name: await p.health() for name, p in self._providers.items()}



registry = EmbeddingRegistry()


def setup_registry() -> EmbeddingRegistry:

    from app.config import settings
    from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider


    from app.services.embeddings.huggingface_provider import HuggingFaceProvider
    registry.register(
        HuggingFaceProvider(
            default_model=settings.EMBEDDING_MODEL,
            default_dimensions=settings.VECTOR_DIM,
        )
    )
    # Ollama is always registered — it's local and has no key
    from app.services.embeddings.ollama_provider import OllamaEmbeddingProvider
    registry.register(
        OllamaEmbeddingProvider(base_url=settings.OLLAMA_BASE_URL)
    )

    if settings.OPENAI_API_KEY:
        registry.register(
            OpenAIEmbeddingProvider(
                api_key=settings.OPENAI_API_KEY,
                default_model=settings.EMBEDDING_MODEL,
                default_dimensions=settings.VECTOR_DIM,
                timeout=settings.OPENAI_TIMEOUT,
                max_retries=settings.OPENAI_MAX_RETRIES,
            )
        )

    logger.info(
        "embeddings registry ready. Providers: %s",
        list(registry._providers.keys()),
    )
    return registry


setup_registry()