import logging
from functools import lru_cache

import httpx

from app.services.embeddings.base import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResponse,
)
from app.services.embeddings.errors import EmbeddingProviderError

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    provider_name = "ollama"
    default_model = "nomic-embed-text"
    default_dimensions = 768

    def __init__(self, base_url: str, default_model: str | None = None):
        self.base_url = base_url.rstrip("/")
        if default_model:
            self.default_model = default_model

    @lru_cache(maxsize=1)
    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    async def embed(
        self,
        texts: list[str],
        model: str,
        config: EmbeddingConfig,
    ) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(
                vectors=[], provider_used=self.provider_name,
                model_used=model, dimensions=self.default_dimensions,
            )
        try:
            client = self._client()
            resp = await client.post(
                "/api/embed",
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            vectors = data["embeddings"]
        except Exception as e:
            raise EmbeddingProviderError(
                f"OllamaEmbeddingProvider failed: {type(e).__name__}: {e}"
            ) from e

        dims = len(vectors[0]) if vectors else self.default_dimensions
        return EmbeddingResponse(
            vectors=vectors,
            provider_used=self.provider_name,
            model_used=model,
            dimensions=dims,
        )

    async def health(self) -> bool:
        try:
            client = self._client()
            r = await client.get("/api/tags")
            return r.status_code == 200
        except Exception:
            return False
