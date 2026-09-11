import io 
import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_upload_accepted(authed: AsyncClient):
    fake_pdf = b"%PDF-1.4 fake content for testing"
    with patch(
        "app.api.v1.documents.DocumentService.process",
        new_callable=AsyncMock
    ):
        resp = await authed.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(
                fake_pdf), "application/pdf")},
            data={"title": "My PDF"},

        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["title"] == "My PDF"
    assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_reject_unsupported_type(authed: AsyncClient):
    res = await authed.post(
        "api/v1/documents/upload",
        files={'file': ("malware.exe", b"MZ", "application/octet-stream")}
    )
    assert res.status_code == 400
    
    
    
@pytest.mark.asyncio
async def test_list_documents(authed: AsyncClient):
    res = await authed.get("/api/v1/documents/") 
       
    assert res.status_code == 200
    body = res.json()
    assert "documents" in body
    assert "total" in body
    
    
@pytest.mark.asyncio
async def test_document_not_found(authed: AsyncClient):
    resp = await authed.get(
        "/api/v1/documents/00000000-0000-0000-0000-000000000000"
    )
    assert resp.status_code == 404
