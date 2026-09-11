import pytest
from httpx import AsyncClient
import os
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

@pytest.mark.asyncio
async def test_register_and_login(client: AsyncClient):
    reg = await client.post("/api/v1/auth/register", json={
        "email": "a@b.com",
        "username": "alice",
        "password": "StrongPass1",
    })
    assert reg.status_code == 201
    body = reg.json()
    assert body["email"] == "a@b.com"
    assert "hashed_password" not in body

    login = await client.post("/api/v1/auth/login", json={
        "email": "a@b.com",
        "password": "StrongPass1",
    })
    assert login.status_code == 200
    tokens = login.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    payload = {"email": "dup@test.com", "username": "user2", "password": "StrongPass1"}
    await client.post("/api/v1/auth/register", json=payload)
    resp = await client.post("/api/v1/auth/register", json={**payload, "username": "user2"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_weak_password_rejected(client: AsyncClient):
    resp = await client.post("/api/v1/auth/register", json={
        "email": "w@b.com",
        "username": "weak",
        "password": "short",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_me(authed: AsyncClient):
    # Simpler:
    r = await authed.get("/api/v1/auth/me")
    assert r.status_code == 200
    assert r.json()["email"] == "test@test.com"


@pytest.mark.asyncio
async def test_unauthenticated_blocked(client: AsyncClient):
    r = await client.get("/api/v1/auth/me")
    assert r.status_code in (401, 403)
    
    
@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    base = {"email": "dup@test.com",
            "username": "user1", "password": "StrongPass1"}
    r1 = await client.post("/api/v1/auth/register", json=base)
    assert r1.status_code == 201

    r2 = await client.post(
        "/api/v1/auth/register",
        # different username, same email
        json={**base, "username": "user2"},
    )
    assert r2.status_code == 409
    assert "email" in r2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_logout_blacklists_token(authed):
    # Verify token works
    assert (await authed.get("/api/v1/auth/me")).status_code == 200

    # Logout — this calls add() on the stateful fake
    assert (await authed.post("/api/v1/auth/logout")).status_code == 204

    # Same token is now rejected — contains() returns True
    assert (await authed.get("/api/v1/auth/me")).status_code == 401


@pytest.mark.asyncio
async def test_refresh_returns_new_tokens(client: AsyncClient):
    await client.post("/api/v1/auth/register", json={
        "email": "r@test.com", "username": "refresher", "password": "StrongPass1",
    })
    login = await client.post("/api/v1/auth/login", json={
        "email": "r@test.com", "password": "StrongPass1",
    })
    refresh_token = login.json()["refresh_token"]

    r = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert body["access_token"] != login.json()["access_token"]   # fresh token


@pytest.mark.asyncio
async def test_refresh_rejects_access_token(client: AsyncClient):
    """An access token must not be accepted by /refresh."""
    await client.post("/api/v1/auth/register", json={
        "email": "r2@test.com", "username": "refresher2", "password": "StrongPass1",
    })
    login = await client.post("/api/v1/auth/login", json={
        "email": "r2@test.com", "password": "StrongPass1",
    })
    access = login.json()["access_token"]

    r = await client.post("/api/v1/auth/refresh", json={"refresh_token": access})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_register_duplicate_username(client: AsyncClient):
    base = {"email": "u1@test.com",
            "username": "taken", "password": "StrongPass1"}
    await client.post("/api/v1/auth/register", json=base)

    r = await client.post(
        "/api/v1/auth/register",
        json={**base, "email": "u2@test.com"},   # same username, new email
    )
    assert r.status_code == 409
    assert "username" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    base = {"email": "dup@test.com",
            "username": "user1", "password": "StrongPass1"}
    r1 = await client.post("/api/v1/auth/register", json=base)
    assert r1.status_code == 201

    # Same email, different username → must be rejected on email
    r2 = await client.post(
        "/api/v1/auth/register",
        json={**base, "username": "user2"},
    )
    assert r2.status_code == 409
    assert "email" in r2.json()["detail"].lower()
