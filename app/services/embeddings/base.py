from abc import ABC, abstractmethod
from typing import List

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Per-call options."""
    normalize: bool = True
    batch_size: int = 32
    truncate_chars: Optional[int] = None    # None → provider default


@dataclass
class EmbeddingResponse:
    """What every provider returns."""
    vectors: list[list[float]]
    provider_used: str
    model_used: str
    dimensions: int
    tokens_used: Optional[int] = None       # provider-reported if available


@dataclass
class EmbeddingResult:
    """Service-level wrapper with routing metadata."""
    response: EmbeddingResponse
    provider_used: str
    model_used: str
    dimensions: int
    was_fallback: bool = False


class EmbeddingProvider(ABC):
    """
    Every embedding backend implements this.
    Providers are stateless after __init__ — safe to share across requests.
    """

    provider_name: str = ""
    default_model: str = ""
    default_dimensions: int = 0

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str,
        config: EmbeddingConfig,
    ) -> EmbeddingResponse:
        """Embed a batch of texts. Must preserve input order."""

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the provider is usable right now."""

    def supports_model(self, model: str) -> bool:
        """Override if the provider hosts multiple models."""
        return model == self.default_model
