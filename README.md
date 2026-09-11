# rag-poc

A proof-of-concept RAG (Retrieval-Augmented Generation) pipeline. Upload documents, ask questions, get answers grounded in your content with source citations.

---

## What It Does

```
Upload PDF/DOCX/TXT/MD/CSV
        ↓
Extract & chunk text
        ↓
Generate embeddings (HuggingFace / OpenAI / Ollama)
        ↓
Store chunks in PostgreSQL + vectors in Qdrant
        ↓
User asks a question
        ↓
Hybrid retrieval (vector + BM25, fused with RRF)
        ↓
Send context + question to LLM
        ↓
Return answer with source citations
```

---

## Features

| Feature | Detail |
|---|---|
| **Document ingestion** | PDF, DOCX, TXT, MD, CSV |
| **Hybrid retrieval** | Vector similarity (Qdrant) + BM25 full-text (PostgreSQL), fused with Reciprocal Rank Fusion |
| **Embedding providers** | HuggingFace (default), OpenAI, Ollama — pluggable via registry |
| **Multiple LLM providers** | OpenAI, Anthropic, Google Gemini, Ollama (local) |
| **LLM fallback** | Automatic failover to secondary provider |
| **Conversation history** | Multi-turn chat with context-aware query rewriting |
| **Source citations** | Every answer links back to the exact document chunk |
| **Chunk deduplication** | Jaccard token-set similarity drops near-duplicate chunks before the LLM |
| **Authentication** | JWT access + refresh tokens, token blacklist on logout |
| **Security filter** | Prompt injection detection, secret redaction, input sanitization |
| **Rate limiting** | Per-endpoint, per-IP limits |
| **Async throughout** | FastAPI + asyncpg + async clients for Qdrant and LLM providers |
| **Streaming** | SSE streaming for LLM responses |
| **API documentation** | Auto-generated Swagger UI and ReDoc |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI |
| Language | Python 3.11 |
| Package manager | uv |
| Database | PostgreSQL 16 + asyncpg |
| ORM | SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| Vector DB | Qdrant |
| Full-text search | PostgreSQL `tsvector` + GIN index |
| Cache / Blacklist | Redis 7 |
| Embeddings (default) | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Embeddings (alt) | OpenAI text-embedding-3-small, Ollama |
| LLM (default) | Configurable — OpenAI, Google Gemini, Anthropic, Ollama |
| Auth | JWT (python-jose) + bcrypt |
| Token counting | tiktoken |
| Containerization | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio + httpx |

---

## Project Structure

```
rag-poc/
│
├── app/
│   ├── main.py                     # FastAPI app, lifespan, middleware
│   ├── config.py                   # Settings (pydantic-settings)
│   ├── database.py                 # Async engine, session factory
│   ├── prompts.py                  # System prompts
│   ├── models.py                   # SQLAlchemy ORM models
│   ├── schemas.py                  # Pydantic models (request/response)
│   │
│   ├── api/
│   │   ├── deps.py                 # Auth dependency (get_current_user)
│   │   ├── limiter.py              # Rate limiting
│   │   └── v1/
│   │       ├── auth.py             # /auth — register, login, logout, refresh
│   │       ├── documents.py        # /documents — upload, list, get, delete
│   │       ├── chat.py             # /chat — message, conversations
│   │       └── llm.py              # /llm — direct completion, stream, models
│   │
│   ├── services/
│   │   ├── auth_service.py         # Registration, authentication, token ops
│   │   ├── document_service.py     # Upload, processing pipeline, delete
│   │   ├── vector_service.py       # Qdrant upsert, search, delete
│   │   ├── bm25_service.py         # PostgreSQL FTS + RRF fusion
│   │   ├── redis_service.py        # Redis-backed token revocation
│   │   ├── chat_service.py         # RAG pipeline, conversation management
│   │   ├── conversation_service.py # Conversation CRUD, history, sources
│   │   ├── embeddings/
│   │   │   ├── base.py             # Embedding provider ABC
│   │   │   ├── registry.py         # Embedding provider registry
│   │   │   ├── huggingface_embeddings.py
│   │   │   ├── openai_embeddings.py
│   │   │   └── ollama_embeddings.py
│   │   └── llm/
│   │       ├── base.py             # LLMProvider ABC, Message, LLMResponse
│   │       ├── errors.py           # Typed LLM exceptions
│   │       ├── registry.py         # Provider registry, health checks
│   │       ├── router.py           # Provider selection, fallback logic
│   │       ├── llm_service.py      # Public API (injectable for testing)
│   │       └── providers/
│   │           ├── openai_provider.py
│   │           ├── anthropic_provider.py
│   │           ├── google_provider.py
│   │           └── ollama_provider.py
│   │
│   ├── security/
│   │   ├── filters.py              # Input validation, injection detection
│   │   └── jwt.py                  # JWT handling
│   │
│   └── utils/
│       ├── security.py             # Password hashing, JWT encode/decode
│       ├── text_extractor.py       # PDF/DOCX/TXT/CSV extraction (threaded)
│       ├── text_splitter.py        # Token-aware semantic chunking
│       ├── file_utils.py           # Safe filenames, upload paths
│       └── logger.py               # Structured logging
│
├── tests/
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_documents.py
│   ├── test_conversation_service.py
│   ├── test_llm_service.py
│   ├── test_text_extractor.py
│   ├── test_text_splitter.py
│   └── security/
│       └── test_filters.py
│
├── alembic/
│   ├── env.py
│   └── versions/
│       └── a92ef9f710fb_initial_schema.py
│
├── script.sh                         # End-to-end smoke test
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── alembic.ini
└── .env.example
```

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Docker and Docker Compose
- At least one LLM provider API key (OpenAI, Anthropic, or Google)
- Optionally: a running Ollama instance for fully local operation

Install uv once:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/your-username/rag-poc.git
cd rag-poc

cp .env.example .env
```

Open `.env` and fill in at minimum:

```bash
SECRET_KEY=<32+ random characters>
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge_db

# Pick a provider and supply its key
DEFAULT_LLM_PROVIDER=google
DEFAULT_LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=...
```

### 2. Start infrastructure

```bash
docker compose up -d db qdrant redis
```

### 3. Install Python dependencies

```bash
uv sync
```

### 4. Apply database migrations

```bash
uv run alembic upgrade head
```

### 5. Run the API

```bash
uv run uvicorn app.main:app --reload --port 8000
```

### 6. Verify

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "1.0.0"}
```

### 7. Run the smoke test (in another terminal)

```bash
./script.sh
```

This registers a user, uploads a test document, polls until processing completes,
checks Qdrant state, and runs a two-turn chat. It exercises the whole pipeline.

To skip the DB reset and just re-run the checks:

```bash
SKIP_RESET=1 ./script.sh
```

### 8. API documentation

```
http://localhost:8000/docs      ← Swagger UI
http://localhost:8000/redoc     ← ReDoc
```

---

## API Usage

### Authentication

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "myuser",
    "password": "MyPass123",
    "full_name": "My Name"
  }'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "MyPass123"}'

TOKEN="eyJhbGci..."
```

### Upload a Document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/your/document.pdf" \
  -F "title=My Document"

# Response: {"id": "...", "status": "pending", ...}

# Poll status
curl http://localhost:8000/api/v1/documents/{document_id} \
  -H "Authorization: Bearer $TOKEN"
# status: pending → processing → completed (or failed)
```

### Ask a Question

```bash
# First turn
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Who created Python?"}'

# Follow-up in the same conversation
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "When was it released?",
    "conversation_id": "<uuid-from-previous-response>"
  }'

# Restrict to specific documents
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Summarize the key findings.",
    "document_ids": ["uuid-1", "uuid-2"]
  }'
```

### Response Shape

```json
{
  "conversation_id": "3fa85f64-...",
  "conversation_title": "Who created Python?",
  "message": {
    "id": "7cb12d44-...",
    "role": "assistant",
    "content": "Python was created by Guido van Rossum [Source 1].",
    "source": [
      {
        "chunk_id": "a1b2c3-...",
        "document_id": "d4e5f6-...",
        "document_title": "Python history",
        "content_preview": "Python was created by Guido van Rossum and first released in 1991.",
        "similarity_score": 0.82
      }
    ],
    "tokens_used": 236,
    "model_used": "google/gemini-2.5-flash",
    "created_at": "2026-09-11T20:40:25Z"
  }
}
```

### Select LLM Provider

```bash
# Explicit provider
curl -X POST http://localhost:8000/api/v1/llm/complete \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "provider": "openai",
    "model": "gpt-4o-mini"
  }'

# List available providers and models
curl http://localhost:8000/api/v1/llm/models \
  -H "Authorization: Bearer $TOKEN"

# Health check
curl http://localhost:8000/api/v1/llm/health \
  -H "Authorization: Bearer $TOKEN"

# Streaming
curl -X POST http://localhost:8000/api/v1/llm/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

---

## Configuration Reference

```bash
# .env.example

# ---- Application ----
SECRET_KEY=change-me-minimum-32-characters-long
DEBUG=false
APP_NAME="rag-poc"
APP_VERSION="1.0.0"

# ---- Database ----
# Use @localhost for host tools (alembic, uvicorn, tests)
# Use @db for containers running inside the docker network
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge_db
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# ---- Redis ----
REDIS_URL=redis://localhost:6379/0

# ---- Qdrant ----
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_collection

# ---- Embeddings ----
# Active provider: huggingface | openai | ollama
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384

# Alternative: OpenAI embeddings (1536-dim, requires OPENAI_API_KEY)
# EMBEDDING_PROVIDER=openai
# EMBEDDING_MODEL=text-embedding-3-small
# EMBEDDING_DIMENSIONS=1536

# Offline mode — skip HuggingFace HEAD requests once model is cached
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# ---- LLM providers ----
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434

# ---- LLM routing ----
DEFAULT_LLM_PROVIDER=google
DEFAULT_LLM_MODEL=gemini-2.5-flash
FALLBACK_LLM_PROVIDER=ollama
FALLBACK_LLM_MODEL=llama3.2
LLM_MAX_TOKENS=2048

# ---- Document processing ----
CHUNK_SIZE_TOKENS=500
CHUNK_OVERLAP_TOKENS=100
MAX_FILE_SIZE_BYTES=52428800            # 50MB
ALLOWED_EXTENSIONS=["pdf","txt","docx","md","csv"]

# ---- RAG retrieval ----
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
MAX_CONTEXT_TOKENS=3000
MAX_HISTORY_MESSAGES=10

# Hybrid retrieval weights (RRF)
VECTOR_WEIGHT=0.6
BM25_WEIGHT=0.4
RRF_K=60

# ---- Rate limiting ----
CHAT_RATE_LIMIT=30/minute
UPLOAD_RATE_LIMIT=10/minute
AUTH_RATE_LIMIT=5/minute

# ---- CORS ----
ALLOWED_ORIGINS=["http://localhost:3000"]
```

---

## Document Processing Pipeline

```
File uploaded by user
        │
        ▼
┌───────────────────┐
│  Validation       │  Size · Extension whitelist · Content hash dedup
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Text Extraction  │  PDF (pdfplumber + PyPDF2 fallback)
│  (thread pool)    │  DOCX (python-docx)
│                   │  TXT / MD (encoding detection)
│                   │  CSV (structured → text)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Semantic         │  Token-aware splitting (tiktoken)
│  Chunking         │  Respects paragraph and sentence boundaries
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Embedding        │  Batched embedding calls
│  Generation       │  Retry on rate limit
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Storage          │  Chunks → PostgreSQL (text + metadata)
│                   │  Embeddings → Qdrant (vectors + payload)
│                   │  Payload: {user_id, document_id, content}
└───────────────────┘
```

---

## RAG Query Pipeline

```
User sends message
        │
        ▼
┌───────────────────┐
│  Security Filter  │  Injection detection · Secret redaction
│                   │  Homoglyph normalization · Length check
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Query Rewriting  │  Resolves pronouns and references
│                   │  using conversation history
└────────┬──────────┘
         │
         ├───────────────────────────┐
         ▼                           ▼
┌───────────────────┐     ┌────────────────────┐
│  Query Embedding  │     │  BM25 (tsvector)   │
│  (MiniLM 384)     │     │  PostgreSQL FTS    │
└────────┬──────────┘     └─────────┬──────────┘
         │                          │
         ▼                          ▼
┌───────────────────┐     ┌────────────────────┐
│  Qdrant search    │     │  ts_rank_cd        │
│  filter: user_id  │     │  filter: owner_id  │
└────────┬──────────┘     └─────────┬──────────┘
         └────────────┬─────────────┘
                      ▼
          ┌───────────────────────┐
          │  RRF Fusion           │  score = Σ weight / (k + rank)
          │  + Jaccard dedup      │  drop near-identical chunks
          └──────────┬────────────┘
                     ▼
          ┌───────────────────────┐
          │  Context Builder      │  Token budget enforcement
          │                       │  [Source N] labels
          └──────────┬────────────┘
                     ▼
          ┌───────────────────────┐
          │  LLM Call             │  System prompt + context
          │                       │  Last N turns + user question
          └──────────┬────────────┘
                     ▼
          ┌───────────────────────┐
          │  Response             │  Answer text
          │                       │  Sources with document titles
          │                       │  Token usage · Model used
          └───────────────────────┘
```

---

## Hybrid Retrieval

Two independent retrieval paths run concurrently and are fused with
Reciprocal Rank Fusion:

| Path | Backend | Strength |
|---|---|---|
| Vector | Qdrant HNSW cosine | Semantic similarity, paraphrases |
| BM25 | PostgreSQL `ts_rank_cd` over GIN FTS index | Exact tokens, names, numbers, acronyms |

RRF combines ranks (not scores) using:

```
score(chunk) = VECTOR_WEIGHT / (RRF_K + rank_in_vector)
             + BM25_WEIGHT  / (RRF_K + rank_in_bm25)
```

Chunks appearing in both lists accumulate contributions from both paths.

After fusion, near-duplicate chunks are removed using Jaccard similarity over
lowercased word tokens (default threshold 0.70). This prevents the LLM from
seeing two copies of the same sentence from different documents.

---

## LLM Provider System

All providers implement the same interface:

```python
class LLMProvider(ABC):
    @property
    def provider_name(self) -> str: ...
    @property
    def supported_models(self) -> list[str]: ...
    async def complete(self, messages, model, config) -> LLMResponse: ...
    async def stream(self, messages, model, config) -> AsyncIterator[StreamChunk]: ...
    async def health_check(self) -> bool: ...
```

| Provider | Example Models | Requires |
|---|---|---|
| OpenAI | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| Anthropic | claude-3-5-sonnet, claude-3-haiku | `ANTHROPIC_API_KEY` |
| Google | gemini-2.5-flash, gemini-2.5-pro | `GOOGLE_API_KEY` |
| Ollama | llama3.2, mistral, any local model | Running Ollama instance |

Fallback: when the primary provider raises a retryable error, the router
automatically retries with the configured fallback provider. If both fail,
a typed error propagates to the API layer.

---

## Security

### Authentication
- Passwords hashed with bcrypt
- JWT access tokens (30 min expiry)
- JWT refresh tokens (7 day expiry)
- Token blacklist in Redis — logout immediately invalidates tokens

### Input Validation
- Every user query passes through a security filter before touching the LLM
- Prompt injection detection with Unicode normalization (defeats homoglyph attacks)
- Secret detection blocks queries containing API keys, JWTs, AWS credentials
- Retrieved document chunks are scanned separately with narrower rules

### Data Isolation
- All vector searches filter by `user_id` (Qdrant payload)
- All BM25 searches filter by `owner_id` (SQL WHERE)
- Users can only access their own documents and conversations
- Document content is never returned to other users

### Infrastructure
- CORS restricted to configured origins
- Rate limiting on all endpoints
- File uploads validated by extension whitelist and content hash
- User-supplied filenames never used in filesystem paths

---

## Database

### Migrations

```bash
# Apply
uv run alembic upgrade head

# Check current revision
uv run alembic current

# Show history
uv run alembic history

# Roll back one revision
uv run alembic downgrade -1
```

### Making schema changes

```bash
# 1. Edit app/models.py
# 2. Generate a migration
uv run alembic revision --autogenerate -m "describe the change"
# 3. REVIEW the generated file — always
cat alembic/versions/<new>.py
# 4. Apply
uv run alembic upgrade head
```

Do not skip step 3. Alembic autogenerate is a draft, not a final answer.

### Schema

```
users
  id · email · username · hashed_password
  is_active · is_verified · created_at · updated_at

documents
  id · owner_id → users · title · filename · file_path
  file_size · file_type · content_hash · status
  chunk_count · page_count · word_count
  metadata_ · error_message · created_at · updated_at

document_chunk
  id · document_id → documents · chunk_index · content
  token_count · vector_id · metadata_ · created_at

conversations
  id · user_id → users · title · created_at · updated_at

conversation_documents
  conversation_id · document_id    (many-to-many link)

messages
  id · conversation_id → conversations · role
  content · source · model_used · tokens_used
  confidence_score · metadata_ · created_at
```

### Performance indexes

| Index | Purpose |
|---|---|
| `ix_users_email`, `ix_users_username` | Login, uniqueness |
| `ix_conversations_user_id` | List user's conversations |
| `ix_documents_owner_id` | List user's documents |
| `ix_documents_content_hash` | Duplicate detection |
| `ix_documents_owner_hash` | Per-user duplicate detection |
| `ix_documents_owner_status` | Startup recovery, status filters |
| `ix_documents_owner_created` | Paginated list, ordered fetch |
| `ix_document_chunk_document_id` | Joins, cascade delete |
| `ix_document_chunk_content_fts` (GIN) | BM25 full-text search |
| `ix_messages_conversation_created` | Recent history fetch |

---

## Reset & Smoke Test

```bash
# Full reset + smoke test
./script.sh

# Smoke test only (keeps existing data)
SKIP_RESET=1 ./script.sh
```

The script:

1. Drops and recreates the PostgreSQL database
2. Deletes the Qdrant collection
3. Flushes Redis
4. Clears uploaded files
5. Applies Alembic migrations
6. Registers a user, uploads a test document
7. Polls until the document reaches `completed`
8. Runs a two-turn chat and prints sources

Requires uvicorn running on port 8000 in another terminal.

---

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific module
uv run pytest tests/test_llm_service.py -v
uv run pytest tests/security/test_filters.py -v

# With coverage
uv run pytest tests/ --cov=app --cov-report=term-missing
```

| Module | What is tested |
|---|---|
| `test_auth.py` | Register, login, logout, token blacklist, auth guard |
| `test_documents.py` | Upload validation, list, get, not-found |
| `test_conversation_service.py` | Create, delete, isolation between users |
| `test_llm_service.py` | Provider routing, fallback, registry, content filter |
| `test_text_extractor.py` | PDF, DOCX, TXT, CSV extraction, encoding fallback |
| `test_text_splitter.py` | Chunk size, empty input, token counting |
| `test_filters.py` | Injection detection, homoglyph bypass, secret blocking |

---

## Development

### Run locally without Docker for the app

```bash
# 1. Start infrastructure only
docker compose up -d db qdrant redis

# 2. Install dependencies
uv sync

# 3. Configure
cp .env.example .env
# ensure DATABASE_URL uses @localhost, not @db

# 4. Migrate
uv run alembic upgrade head

# 5. Run
uv run uvicorn app.main:app --reload --port 8000
```

### Dependency management

```bash
uv add httpx                # runtime dependency
uv add --dev pytest         # dev dependency
uv remove httpx
uv lock                     # update lockfile
uv sync                     # sync environment
```

### Add a new LLM provider

```python
# 1. app/services/llm/providers/myprovider_provider.py
class MyProvider(LLMProvider):
    @property
    def provider_name(self) -> str: return "myprovider"

    @property
    def supported_models(self) -> list[str]: return ["my-model-v1"]

    async def complete(self, messages, model, config) -> LLMResponse: ...
    async def stream(self, messages, model, config) -> AsyncIterator[StreamChunk]: ...
    async def health_check(self) -> bool: ...

# 2. Register in app/services/llm/registry.py
if settings.MY_PROVIDER_API_KEY:
    from app.services.llm.providers.myprovider_provider import MyProvider
    registry.register(MyProvider())

# 3. Add MY_PROVIDER_API_KEY to config.py and .env.example
```

### Add a new document format

```python
# app/utils/text_extractor.py
def _extract_xlsx(file_bytes: bytes) -> dict[str, Any]:
    return {"text": full_text, "metadata": {...}}

_EXTRACTORS = {
    ...
    "xlsx": _extract_xlsx,
}
# Then add "xlsx" to ALLOWED_EXTENSIONS in config.py
```

### Add a new embedding provider

```python
# 1. app/services/embeddings/myprovider_embeddings.py
class MyEmbeddingProvider(EmbeddingProvider):
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_one(self, text: str) -> list[float]: ...

# 2. Register in app/services/embeddings/registry.py
# 3. Set EMBEDDING_PROVIDER=myprovider in .env
# 4. Ensure EMBEDDING_DIMENSIONS matches the new model
# 5. Delete and recreate the Qdrant collection
```

---

## Deployment Notes

- Set `DEBUG=false`
- Use a strong random `SECRET_KEY` (minimum 32 characters)
- Set `ALLOWED_ORIGINS` to your frontend domain
- Use `docker compose up -d` with the app service included
- Data persists in named Docker volumes (`postgres_data`, `qdrant_data`, `redis_data`)
- Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` once the embedding model
  is cached to skip remote verification on every process start

---

## License

MIT
