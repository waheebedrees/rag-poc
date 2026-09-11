# app/services/llm/errors.py
"""
Typed exceptions for the LLM service layer.
Callers catch these — never catch bare Exception from LLM calls.
"""


class LLMError(Exception):
    """Base for all LLM service errors."""


class LLMContentFilteredError(LLMError):
    """Response was blocked by the provider's content policy."""


class LLMProviderNotFoundError(LLMError):
    """Requested provider is not registered."""


class LLMModelNotFoundError(LLMError):
    """No registered provider supports the requested model."""


class LLMRateLimitError(LLMError):
    """Provider rate limit hit and retries exhausted."""


class LLMTimeoutError(LLMError):
    """Provider did not respond within the timeout."""


class LLMAuthError(LLMError):
    """Provider rejected the API key."""


class LLMFallbackExhaustedError(LLMError):
    """Primary and fallback providers both failed."""

    def __init__(self, primary_error: Exception, fallback_error: Exception):
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        super().__init__(
            f"Primary: {primary_error!r} | Fallback: {fallback_error!r}"
        )
