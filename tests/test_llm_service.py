import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.llm.llm_service import LLMService
from app.services.llm.base import (
    FinishReason, LLMResponse, LLMMessage, StreamChunk, LLMConfig, LLMMessage as Message
)
from app.services.llm.router import LLMRouter, RoutingRequest, RoutingResult

from app.services.llm.registry import ProviderRegistry
from app.services.llm.router import LLMRouter
from app.services.llm.errors import (
    LLMContentFilteredError,
    LLMProviderNotFoundError,
)

def _make_response(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    finish_reason: FinishReason = FinishReason.STOP,
) -> LLMResponse:
    return LLMResponse(
        content="Test response",
        model=model,
        provider=provider,
        finish_reason=finish_reason,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        latency_ms=200.0,
    )


def _make_provider(
    name: str,
    models: list[str],
    healthy: bool = True,
    side_effect=None,
):
    p = MagicMock()
    p.provider_name = name
    p.supported_models = models
    p.supports_model = lambda m: m in models

    if side_effect:
        p.complete = AsyncMock(side_effect=side_effect)
    else:
        p.complete = AsyncMock(return_value=_make_response(name))

    p.health_check = AsyncMock(return_value=healthy)
    return p


@pytest.fixture
def reg() -> ProviderRegistry:
    r = ProviderRegistry()
    r.register(_make_provider("openai", ["gpt-4o", "gpt-4o-mini"]))
    r.register(_make_provider("anthropic", ["claude-3-5-sonnet-20241022"]))
    return r


class TestRouterResolution:
    def test_explicit_provider_and_model(self, reg):
        router = LLMRouter(reg)
        provider, model = router._resolve(
            MagicMock(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
            )
        )
        assert provider.provider_name == "anthropic"
        assert model == "claude-3-5-sonnet-20241022"

    def test_auto_detect_provider_from_model(self, reg):
        router = LLMRouter(reg)
        provider, model = router._resolve(
            MagicMock(provider=None, model="gpt-4o")
        )
        assert provider.provider_name == "openai"
        assert model == "gpt-4o"

    def test_unknown_provider_raises(self, reg):
        router = LLMRouter(reg)
        with pytest.raises(ValueError, match="not registered"):
            router._resolve(
                MagicMock(provider="nonexistent", model="some-model")
            )

    def test_unknown_model_raises(self, reg):
        router = LLMRouter(reg)
        with pytest.raises(ValueError, match="No provider found"):
            router._resolve(
                MagicMock(provider=None, model="totally-unknown-model")
            )



class TestRouterFallback:
    @pytest.mark.asyncio
    async def test_fallback_used_when_primary_fails(self, reg):
        # Make openai fail
        reg.get("openai").complete = AsyncMock(
            side_effect=Exception("OpenAI down")
        )
        router = LLMRouter(reg)

        fallback_response = _make_response(
            "anthropic", "claude-3-5-sonnet-20241022")

        with patch.object(router, "_fallback", new_callable=AsyncMock) as mock_fallback:
            mock_fallback.return_value = RoutingResult(
                response=fallback_response,
                provider_used="anthropic",
                model_used="claude-3-5-sonnet-20241022",
                was_fallback=True,
            )
            result = await router.route(
                RoutingRequest(
                    messages=[Message(role="user", content="hello")],
                    config=LLMConfig(),
                    allow_fallback=True,
                )
            )

        assert result.was_fallback is True
        assert result.provider_used == "anthropic"
        mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_fallback_when_disabled(self, reg):
        reg.get("openai").complete = AsyncMock(
            side_effect=Exception("OpenAI down")
        )
        router = LLMRouter(reg)

        with pytest.raises(Exception, match="OpenAI down"):
            await router.route(
                RoutingRequest(
                    messages=[Message(role="user", content="hello")],
                    config=LLMConfig(),
                    allow_fallback=False,
                )
            )



class TestRegistry:
    @pytest.mark.asyncio
    async def test_health_status_all_healthy(self, reg):
        status = await reg.health_status()
        assert status == {"openai": True, "anthropic": True}

    @pytest.mark.asyncio
    async def test_health_status_one_unhealthy(self, reg):
        reg.get("openai").health_check = AsyncMock(return_value=False)
        status = await reg.health_status()
        assert status["openai"] is False
        assert status["anthropic"] is True

    def test_available_models(self, reg):
        models = reg.available_models()
        assert "openai" in models
        assert "gpt-4o" in models["openai"]

    def test_find_provider_for_known_model(self, reg):
        p = reg.find_provider_for_model("gpt-4o-mini")
        assert p is not None
        assert p.provider_name == "openai"

    def test_find_provider_for_unknown_model(self, reg):
        assert reg.find_provider_for_model("does-not-exist") is None



class TestLLMService:
    @pytest.mark.asyncio
    async def test_content_filter_raises_typed_error(self):
        """
        LLMService.complete() must raise LLMContentFilteredError
        when the provider returns FinishReason.CONTENT_FILTER.
        Not a bare ValueError — callers need to catch a typed exception.
        """
        filtered_response = _make_response(
            finish_reason=FinishReason.CONTENT_FILTER
        )
        routing_result = RoutingResult(
            response=filtered_response,
            provider_used="openai",
            model_used="gpt-4o-mini",
        )

        with patch(
            "app.services.llm.llm_service.LLMRouter.route",
            new_callable=AsyncMock,
            return_value=routing_result,
        ):
            svc = LLMService()
            with pytest.raises(LLMContentFilteredError, match="content policy"):
                await svc.complete(
                    messages=[Message(role="user", content="test")]
                )

    @pytest.mark.asyncio
    async def test_successful_completion_returns_routing_result(self):
        good_response = _make_response()
        routing_result = RoutingResult(
            response=good_response,
            provider_used="openai",
            model_used="gpt-4o-mini",
        )

        with patch(
            "app.services.llm.llm_service.LLMRouter.route",
            new_callable=AsyncMock,
            return_value=routing_result,
        ):
            svc = LLMService()
            result = await svc.complete(
                messages=[Message(role="user", content="test")]
            )

        assert result.provider_used == "openai"
        assert result.response.content == "Test response"

    @pytest.mark.asyncio
    async def test_truncated_response_does_not_raise(self):
        """Truncated responses are logged but not raised — caller decides."""
        truncated = _make_response(finish_reason=FinishReason.LENGTH)
        routing_result = RoutingResult(
            response=truncated,
            provider_used="openai",
            model_used="gpt-4o-mini",
        )

        with patch(
            "app.services.llm.llm_service.LLMRouter.route",
            new_callable=AsyncMock,
            return_value=routing_result,
        ):
            svc = LLMService()
            result = await svc.complete(
                messages=[Message(role="user", content="test")]
            )

        # Returns normally — caller can check result.response.was_truncated
        assert result.response.was_truncated is True

    @pytest.mark.asyncio
    async def test_fallback_result_returned_normally(self):
        """Fallback results are returned normally — just logged."""
        fallback_response = _make_response(
            "anthropic", "claude-3-haiku-20240307")
        routing_result = RoutingResult(
            response=fallback_response,
            provider_used="anthropic",
            model_used="claude-3-haiku-20240307",
            was_fallback=True,
        )

        with patch(
            "app.services.llm.llm_service.LLMRouter.route",
            new_callable=AsyncMock,
            return_value=routing_result,
        ):
            svc = LLMService()
            result = await svc.complete(
                messages=[Message(role="user", content="test")]
            )

        assert result.was_fallback is True
        assert result.provider_used == "anthropic"
