from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json

from app.api.deps import get_current_user
from app.models import User
from app.services.llm import LLMService, LLMMessage

router = APIRouter(prefix="/llm", tags=["LLM"])


class CompletionRequest(BaseModel):
    messages: list[dict] 
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False


class CompletionResponse(BaseModel):
    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    was_fallback: bool


@router.post("/complete", response_model=CompletionResponse)
async def complete(
    body: CompletionRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Direct LLM completion endpoint.
    Use provider and model to override defaults.

    Examples:
      {"provider": "openai",    "model": "gpt-4o"}
      {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"}
      {"provider": "google",    "model": "gemini-2.0-flash"}
      {"provider": "ollama",    "model": "llama3.2"}
      {"model": "gpt-4o"}          # auto-detects openai
      {}                           # uses configured defaults
    """
    llm = LLMService()
    messages = [
        LLMMessage(role=m["role"], content=m["content"])
        for m in body.messages
    ]

    try:
        result = await llm.complete(
            messages=messages,
            provider=body.provider,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return CompletionResponse(
        content=result.response.content,
        provider=result.provider_used,
        model=result.model_used,
        input_tokens=result.response.input_tokens,
        output_tokens=result.response.output_tokens,
        total_tokens=result.response.total_tokens,
        latency_ms=result.response.latency_ms,
        was_fallback=result.was_fallback,
    )


@router.post("/stream")
async def stream_complete(
    body: CompletionRequest,
    current_user: User = Depends(get_current_user),
):
    """Server-sent events streaming endpoint."""
    llm = LLMService()
    messages = [
        LLMMessage(role=m["role"], content=m["content"])
        for m in body.messages
    ]

    async def generate():
        try:
            async for chunk in llm.stream(
                messages=messages,
                provider=body.provider,
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            ):
                event = {"content": chunk.content, "done": chunk.is_final}
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/models")
async def list_models(current_user: User = Depends(get_current_user)):
    """List all available models per provider."""
    llm = LLMService()
    return await llm.available_models()


@router.get("/health")
async def provider_health(current_user: User = Depends(get_current_user)):
    """Check which providers are reachable."""
    llm = LLMService()
    return await llm.health()