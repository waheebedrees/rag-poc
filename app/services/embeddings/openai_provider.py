import asyncio
import logging
from functools import lru_cache

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.services.embeddings.base import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResponse,
)
from app.services.embeddings.errors import EmbeddingProviderError

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embeddings provider.
    Handles batching, retries, and truncation internally.
    """

    provider_name = "openai"

    # OpenAI's model → dimensions map (the "small"/"large" variants
    # support custom output dims via the `dimensions` param).
    _MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # Max input chars per request. OpenAI caps at 8191 tokens
    # (~32K chars for English). We truncate conservatively.
    MAX_CHARS = 8191 * 4

    def __init__(
        self,
        api_key: str,
        default_model: str = "text-embedding-3-small",
        default_dimensions: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        if not api_key:
            raise EmbeddingProviderError("OPENAI_API_KEY is required")
        self.default_model = default_model
        self.default_dimensions = (
            default_dimensions or self._MODEL_DIMS.get(default_model, 1536)
        )
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=0,      # we handle retries via tenacity
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.MAX_CHARS:
            return text
        cut = text[: self.MAX_CHARS]
        last_period = cut.rfind(".")
        if last_period > self.MAX_CHARS * 0.8:
            return cut[: last_period + 1]
        return cut

    async def _create_embeddings(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
    ):
        kwargs = {"model": model, "input": texts}
        if dimensions and model != "text-embedding-ada-002":
            # ada-002 doesn't support the dimensions param
            kwargs["dimensions"] = dimensions
        return await self._client.embeddings.create(**kwargs)

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

        truncated = [self._truncate(t) for t in texts]
        all_vectors: list[list[float]] = []
        total_tokens = 0
        dims = self.default_dimensions

        # OpenAI caps batch size at 2048 inputs, but smaller batches
        # keep per-request latency low and reduce blast radius on retry.
        batch_size = max(1, min(config.batch_size, 100))

        for i in range(0, len(truncated), batch_size):
            batch = truncated[i: i + batch_size]
            resp = await self._call_with_retry(batch, model, self.default_dimensions)
            # OpenAI doesn't guarantee order across the `data` list —
            # re-sort by the `index` field to match input order.
            ordered = sorted(resp.data, key=lambda d: d.index)
            all_vectors.extend(item.embedding for item in ordered)
            if resp.usage:
                total_tokens += resp.usage.total_tokens or 0

            # Gentle pacing between batches
            if i + batch_size < len(truncated):
                await asyncio.sleep(0.1)

        if all_vectors:
            dims = len(all_vectors[0])

        return EmbeddingResponse(
            vectors=all_vectors,
            provider_used=self.provider_name,
            model_used=model,
            dimensions=dims,
            tokens_used=total_tokens or None,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)
        ),
        reraise=True,
    )
    async def _call_with_retry(self, batch: list[str], model: str, dims: int | None):
        try:
            return await self._create_embeddings(batch, model, dims)
        except (
            openai.AuthenticationError,
            openai.BadRequestError,
            openai.NotFoundError,
            openai.PermissionDeniedError,
        ):
            raise
        except openai.APIError as e:
            # Retryable server-side error
            logger.warning("OpenAI transient error: %s", e)
            raise

    async def health(self) -> bool:
        try:
            resp = await self._client.embeddings.create(
                model=self.default_model,
                input="ping",
                dimensions=self.default_dimensions,
            )
            return bool(resp.data)
        except Exception as e:
            logger.warning("OpenAI embedding health check failed: %s", e)
            return False

    def supports_model(self, model: str) -> bool:
        return model in self._MODEL_DIMS
