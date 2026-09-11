from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App ----
    APP_NAME: str = "Knowledge Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False               # env: DEBUG
    HOST: str = "0.0.0.0"             # env: HOST
    PORT: int = 8000                  # env: PORT

    # ---- Auth ----
    SECRET_KEY: str = "test-super-secret-key-change-in-production-min-32-chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ---- CORS ----
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    # ---- Database ----
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # ---- Redis ----
    REDIS_URL: str = "redis://localhost:6379"

    # ---- Neo4j ----
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "testpassword"

    # ---- Embeddings / Vector store ----
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_PATH: str = "./data/vectordb"
    VECTOR_DIM: int =  384

    EMBEDDING_PROVIDER: str = "huggingface"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "document_chunks"

    QDRANT_COLLECTION_NAME: str = "rag_collection"
    QDRANT_LOCAL_PATH: str = "./qdrant_local_data"

    #  Document processing 
    CHUNK_SIZE_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 100
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS: set[str] = {"pdf", "txt", "docx", "md", "csv"}
    UPLOAD_DIR: str = "/tmp/uploads"

    #  RAG
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_CONTEXT_TOKENS: int = 3000
    MAX_HISTORY_MESSAGES: int = 10

    #  OpenAI 
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_LLM_MAX_TOKENS: int = 2000
    OPENAI_LLM_TEMPERATURE: float = 0.7
    OPENAI_TIMEOUT: int = 60
    OPENAI_MAX_RETRIES: int = 3

    #  Anthropic 
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_TIMEOUT: int = 60
    ANTHROPIC_MAX_RETRIES: int = 3

    # Google Gemini ----
    GOOGLE_API_KEY: str = ""
    GOOGLE_TIMEOUT: int = 60
    LLM_MAX_TOKENS: int = 2000

    # ---- Ollama ----
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 120

    # ---- Rate limiting ----
    CHAT_RATE_LIMIT: str = "30/minute"
    UPLOAD_RATE_LIMIT: str = "10/minute"
    AUTH_RATE_LIMIT: str = "10/minute"

    # ---- Logging ----
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/agent.log"

    # ---- LLM routing ----


    DEFAULT_LLM_PROVIDER: str = "google"
    DEFAULT_API_BASE: str = "https://api.openai.com/v1"
    DEFAULT_LLM_MODEL: str = "gemini-2.5-flash"
    FALLBACK_LLM_PROVIDER: str = "anthropic"
    FALLBACK_LLM_MODEL: str = "claude-3-haiku-20240307"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
