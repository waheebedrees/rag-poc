class EmbeddingError(Exception):
    """Base for all embedding-layer failures."""


class EmbeddingProviderError(EmbeddingError):
    """A provider failed (load error, network, bad response)."""


class EmbeddingNoProviderAvailableError(EmbeddingError):
    """Registry has no healthy provider for this request."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Provider returned vectors of an unexpected size."""
