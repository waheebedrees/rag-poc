import uuid
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_delete_conversation_not_found(authed: AsyncClient):
    """Deleting a non-existent conversation returns 404."""
    res = await authed.delete(f"/api/v1/chat/conversations/{uuid.uuid4()}")
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_list_conversations_empty(authed: AsyncClient):
    res = await authed.get("/api/v1/chat/conversations")
    assert res.status_code == 200
    assert isinstance(res.json(), list)   
    
@pytest.mark.asyncio
async def test_create_and_delete_conversation(authed: AsyncClient, mock_llm):
    """Full lifecycle: create via /chat, list, delete, confirm it's gone."""
    # 1. Create a conversation by sending a message
    r = await authed.post("/api/v1/chat/", json={"message": "hello"})
    assert r.status_code == 200
    conv_id = r.json()["conversation_id"]

    # 2. It shows up in the list
    r = await authed.get("/api/v1/chat/conversations")
    assert r.status_code == 200
    ids = [c["id"] for c in r.json()]
    assert conv_id in ids

    # 3. Delete it
    r = await authed.delete(f"/api/v1/chat/conversations/{conv_id}")
    assert r.status_code == 204

    # 4. List no longer contains it
    r = await authed.get("/api/v1/chat/conversations")
    ids = [c["id"] for c in r.json()]
    assert conv_id not in ids

    # 5. Fetching it returns 404
    r = await authed.get(f"/api/v1/chat/conversations/{conv_id}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_cannot_delete_other_users_conversation(authed: AsyncClient, mock_llm):
    """User B can't delete User A's conversation."""
    from app.main import app

    # User A creates a conversation
    r = await authed.post("/api/v1/chat/", json={"message": "secret"})
    conv_id = r.json()["conversation_id"]

    # Now log in as a different user
    await authed.post("/api/v1/auth/register", json={
        "email": "other@test.com",
        "username": "otheruser",
        "password": "StrongPass1",
    })
    login = await authed.post("/api/v1/auth/login", json={
        "email": "other@test.com",
        "password": "StrongPass1",
    })
    authed.headers["Authorization"] = f"Bearer {login.json()['access_token']}"

    # User B tries to delete User A's conversation
    r = await authed.delete(f"/api/v1/chat/conversations/{conv_id}")
    assert r.status_code == 404      # not 403 — don't leak existence