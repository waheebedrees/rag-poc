from functools import lru_cache
import asyncio
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.utils.logger import get_logger
from app.services.embeddings.errors import EmbeddingProviderError
from app.services.embeddings.base import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResponse,
)


logger = get_logger(__name__)

def _get_dims(model) -> int:
    # ST 6.x renamed this; fall back for ST < 6.
    getter = getattr(model, "get_embedding_dimension", None) \
        or getattr(model, "get_sentence_embedding_dimension", None)
    if getter is None:
        raise RuntimeError(
            "SentenceTransformer has no dimension method — check your version"
        )
    return getter()

class HuggingFaceProvider(EmbeddingProvider):
    """
    Local embeddings via sentence-transformers.
    Model is loaded once (see _model_cache) and reused across calls.
    """

    provider_name = "huggingface"

    # Registry of already-loaded models, keyed by model name.
    # Shared across all provider instances in this process.
    _model_cache: dict[str, SentenceTransformer] = {}

    def __init__(self, default_model: str, default_dimensions: int = 0):
        self.default_model = default_model
        self.default_dimensions = default_dimensions


    def _get_model(self, model_name: str) -> SentenceTransformer:
        if model_name not in self._model_cache:
            logger.info("Loading embedding model: %s", model_name)
            try:
                self._model_cache[model_name] = SentenceTransformer(model_name)
            except Exception as e:
                raise EmbeddingProviderError(
                    f"Failed to load model '{model_name}': {e}"
                ) from e
            dims = _get_dims(self._model_cache[model_name])
            logger.info("Loaded '%s' (dim=%d)", model_name, dims)
        return self._model_cache[model_name]


    def _embed_sync(
        self, model_name: str, texts: list[str], config: EmbeddingConfig
    ) -> list[list[float]]:
        model = self._get_model(model_name)
        if config.truncate_chars:
            texts = [t[: config.truncate_chars] for t in texts]
        vecs = model.encode(
            texts,
            normalize_embeddings=config.normalize,
            batch_size=config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    async def embed(
        self,
        texts: list[str],
        model: str,
        config: EmbeddingConfig,
    ) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(
                vectors=[],
                provider_used=self.provider_name,
                model_used=model,
                dimensions=self.default_dimensions,
            )
        try:
            # Offload CPU-bound work to a thread so we don't block the loop.
            vectors = await asyncio.to_thread(
                self._embed_sync, model, texts, config
            )
        except EmbeddingProviderError:
            raise
        except Exception as e:
            raise EmbeddingProviderError(
                f"HuggingFaceProvider failed: {type(e).__name__}: {e}"
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
            # Cheap check — is the model loadable / already loaded?
            self._get_model(self.default_model)
            return True
        except Exception:
            return False
