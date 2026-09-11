
from unittest.mock import AsyncMock, patch
import asyncio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from app.services.embeddings import get_embedding_service
from app.services.vector_service import get_vector_service
from app.services.llm import LLMService, setup_registry, LLMMessage

from app.services.llm.errors import LLMContentFilteredError
from app.services.llm.llm_service import LLMService
from app.services.llm.router import RoutingResult
from app.services.llm.base import LLMResponse, LLMMessage, FinishReason
setup_registry()


async def main():
    svc = get_vector_service()
    print('dims:', svc._dims)
    await svc.qdrant_client.delete_collection(svc._collection)
    await svc.ensure_collection()
    print('collection recreated at', svc._dims, 'dims')

    llm = LLMService()
    msg = LLMMessage(
        content="hello",
        role="user"
    )
    result = await llm.complete(messages=[msg], provider="google")
    print(result)



async def debug_filter():
    with patch(
        "app.services.llm.llm_service.LLMRouter.route",
        new_callable=AsyncMock,
    ) as mock_route:
        filtered = LLMResponse(
            content="",
            model="gpt-4o-mini",
            provider="openai",
            finish_reason=FinishReason.CONTENT_FILTER,
            input_tokens=10, output_tokens=0, total_tokens=10, latency_ms=1.0,
        )
        print("constructed was_filtered:", filtered.was_filtered)

        mock_route.return_value = RoutingResult(
            response=filtered,
            provider_used="openai",
            model_used="gpt-4o-mini",
        )

        svc = LLMService()
        try:
            result = await svc.complete(
                messages=[LLMMessage(role="user", content="test")]
            )
            print("NO RAISE")
            print("  result is:", type(result).__name__)
            print("  result.response is filtered:",
                  result.response is filtered)
            print("  result.response.was_filtered:",
                  result.response.was_filtered)
            print("  result.response.finish_reason:",
                  result.response.finish_reason)
        except LLMContentFilteredError as e:
            print("RAISED correctly:", e)


asyncio.run(debug_filter())