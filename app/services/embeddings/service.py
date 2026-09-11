from functools import lru_cache
import logging

from app.config import settings
from app.services.embeddings.base import (
    EmbeddingConfig,
    EmbeddingResult,
)
from app.services.embeddings.errors import (
    EmbeddingNoProviderAvailableError,
    EmbeddingProviderError,
)
from app.services.embeddings.registry import registry



logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Single entry point for all embedding calls.
    Provider details stay behind this interface.
    """

    def __init__(self):
        self._registry = registry

    async def embed(
        self,
        texts: list[str],
        *,
        provider: str | None = None,
        model: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> EmbeddingResult:
        """Embed a batch of texts. Preserves input order."""
        if not texts:
            raise ValueError("texts must be non-empty")

        config = EmbeddingConfig(normalize=normalize, batch_size=batch_size)
        selected, selected_model = self._resolve(provider, model)
    
        try:
            response = await selected.embed(texts, selected_model, config)
        except EmbeddingProviderError:
            raise
        except Exception as e:
            raise EmbeddingProviderError(
                f"{selected.provider_name} failed: {type(e).__name__}: {e}"
            ) from e

        # Sanity check: dims must match configured VECTOR_DIM.
        if response.dimensions and response.dimensions != settings.VECTOR_DIM:
            logger.warning(
                "Embedding dim mismatch: provider=%s dim=%d config.VECTOR_DIM=%d",
                selected.provider_name, response.dimensions, settings.VECTOR_DIM,
            )

        return EmbeddingResult(
            response=response,
            provider_used=response.provider_used,
            model_used=response.model_used,
            dimensions=response.dimensions,
        )

    async def embed_one(
        self,
        text: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[float]:
        result = await self.embed([text], provider=provider, model=model)
        return result.response.vectors[0]

    async def embed_batch(
        self,
        texts: list[str],
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[list[float]]:
        result = await self.embed(texts, provider=provider, model=model)
        return result.response.vectors

    def dimensions(self, provider: str | None = None) -> int:
        p, _ = self._resolve(provider, None)
        return p.default_dimensions or settings.VECTOR_DIM

    async def health(self) -> dict[str, bool]:
        return await self._registry.health_status()

    def list_providers(self) -> list[str]:
        return self._registry.list_providers()

    # ------------------------------------------------------------------

    def _resolve(
        self, provider: str | None, model: str | None
    ):
        """Pick a provider + model. Priority: explicit > model-match > default."""
        if provider:
            p = self._registry.get(provider)
            if not p:
                raise EmbeddingNoProviderAvailableError(
                    f"Provider '{provider}' not registered. "
                    f"Available: {self._registry.list_providers()}"
                )
            return p, (model or p.default_model)

        if model:
            p = self._registry.find_provider_for_model(model)
            if not p:
                raise EmbeddingNoProviderAvailableError(
                    f"No provider found for model '{model}'"
                )
            return p, model

        default_name = settings.EMBEDDING_PROVIDER  # add this to config
        p = self._registry.get(default_name)
        if not p:
            raise EmbeddingNoProviderAvailableError(
                f"Default embedding provider '{default_name}' not registered"
            )
        return p, (model or settings.EMBEDDING_MODEL)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """App-level singleton — model loads once, reused across all requests."""
    return EmbeddingService()
