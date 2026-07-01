"""
Central configuration — extended with Elasticsearch and ingestion worker settings.
"""
from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", env_file=".env", extra="ignore")
    host:         str = Field(default="localhost")
    port:         int = Field(default=5432)
    name:         str = Field(default="nyaya_ai")
    user:         str = Field(default="nyaya_user")
    password:     str = Field(default="nyaya_pass")
    # 512GB RAM machine — a large pool is fine; keep headroom for PG's own
    # shared_buffers (128GB) and work_mem per parallel query worker.
    pool_size:    int = Field(default=64)
    max_overflow: int = Field(default=128)
    pool_timeout: int = Field(default=30)
    pool_recycle: int = Field(default=1800)
    echo:         bool = Field(default=False)

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class ElasticsearchSettings(BaseSettings):
    """
    Elasticsearch settings for TB-scale BM25 retrieval.
    Falls back gracefully to PostgreSQL FTS when ES is not configured.
    """
    model_config = SettingsConfigDict(env_prefix="ES_", env_file=".env", extra="ignore")
    enabled:      bool = Field(default=False)          # set True when ES is running
    hosts:        str  = Field(default="http://localhost:9200")
    index_name:   str  = Field(default="nyaya_legal_chunks")
    username:     Optional[str] = Field(default=None)
    password:     Optional[str] = Field(default=None)
    timeout:      int  = Field(default=30)
    max_retries:  int  = Field(default=3)
    # Shard/replica config for production cluster
    number_of_shards:   int = Field(default=3)
    number_of_replicas: int = Field(default=1)


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")
    host:              str  = Field(default="localhost")
    port:              int  = Field(default=6333)
    grpc_port:         int  = Field(default=6334)
    api_key:           Optional[str] = Field(default=None)
    collection_name:   str  = Field(default="nyaya_legal_chunks")
    vector_size:       int  = Field(default=1024)
    distance:          str  = Field(default="Cosine")
    # m=32 (was 16): more edges per node = better recall at million-scale.
    # ef_construct=400 (was 200): more candidates during index build = better
    # index quality. One-time cost at ingestion, pays off on every query.
    hnsw_m:            int  = Field(default=32)
    hnsw_ef_construct: int  = Field(default=400)
    # ef=256 at query time: larger beam = higher recall, 512GB RAM means no
    # reason to compromise here — latency stays sub-10ms with RAM-resident index.
    hnsw_ef:           int  = Field(default=256)
    # 512GB RAM — keep everything in memory. on_disk_payload=False means
    # payloads (chunk text, metadata) are RAM-resident, not mmap'd from disk.
    # on_disk_vectors=False means the vector index itself is in RAM.
    # At 50M chunks × 1024 dim × 4 bytes = ~200GB vectors + ~100GB payloads
    # = ~300GB total, well within 512GB with room for PG + app overhead.
    on_disk_payload:   bool = Field(default=False)
    on_disk_vectors:   bool = Field(default=False)
    # Scalar quantization: compress vectors from float32 → int8 in RAM
    # (4× memory reduction with <1% recall loss). Keeps the full corpus
    # in memory even as it grows past 200M chunks.
    scalar_quantization: bool = Field(default=True)


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")
    host:            str  = Field(default="localhost")
    port:            int  = Field(default=6379)
    password:        Optional[str] = Field(default=None)
    db:              int  = Field(default=0)
    max_connections: int  = Field(default=50)
    socket_timeout:  int  = Field(default=5)
    ttl_search:      int  = Field(default=3600)
    ttl_embedding:   int  = Field(default=86400)
    ttl_session:     int  = Field(default=86400 * 7)

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")
    provider:       str  = Field(default="groq")
    groq_api_key:   Optional[str] = Field(default=None)
    groq_model:     str  = Field(default="llama-3.3-70b-versatile")
    groq_base_url:  str  = Field(default="https://api.groq.com/openai/v1")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model:   str  = Field(default="llama3.1:8b")
    openai_api_key: Optional[str] = Field(default=None)
    openai_model:   str  = Field(default="gpt-4o-mini")
    temperature:    float = Field(default=0.1)
    max_tokens:     int   = Field(default=4096)
    timeout:        int   = Field(default=60)
    max_retries:    int   = Field(default=3)


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")
    model:      str  = Field(default="BAAI/bge-large-en-v1.5")
    device:     str  = Field(default="cpu")
    batch_size: int  = Field(default=32)
    # Ingestion-time batch size — larger on GPU workers
    ingest_batch_size: int = Field(default=512)
    max_length: int  = Field(default=512)
    normalize:  bool = Field(default=True)
    cache_dir:  str  = Field(default="./data/embeddings/models")


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RERANKER_", env_file=".env", extra="ignore")
    model:      str  = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    device:     str  = Field(default="cpu")
    top_k:      int  = Field(default=10)
    batch_size: int  = Field(default=32)
    cache_dir:  str  = Field(default="./data/embeddings/models")


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", env_file=".env", extra="ignore")
    # Candidate pool sizes
    exact_top_k:   int   = Field(default=5)    # path 1: exact lookup
    bm25_top_k:    int   = Field(default=50)   # path 2: lexical (up from 20)
    vector_top_k:  int   = Field(default=30)   # path 3: ANN (conditional)
    hybrid_top_k:  int   = Field(default=60)   # pool before reranking
    reranker_top_k: int  = Field(default=10)
    final_context_k: int = Field(default=6)
    # Fusion weights — BM25 weighted higher for legal precision
    bm25_weight:   float = Field(default=0.55)
    vector_weight: float = Field(default=0.45)
    min_score_threshold: float = Field(default=0.25)
    # Vector retrieval is conditional — only fires when these are all true:
    # 1. exact lookup returns 0 results
    # 2. no section numbers detected in query
    # 3. query word count > this threshold
    vector_min_query_words: int = Field(default=6)


class IngestionSettings(BaseSettings):
    """Distributed ingestion pipeline settings."""
    model_config = SettingsConfigDict(env_prefix="INGEST_", env_file=".env", extra="ignore")
    # Celery broker
    celery_broker:  str = Field(default="redis://localhost:6379/1")
    celery_backend: str = Field(default="redis://localhost:6379/2")
    # Worker concurrency
    parser_concurrency:   int = Field(default=8)   # CPU workers for PDF parsing
    embedder_concurrency: int = Field(default=2)   # GPU workers for embedding
    # Staging flush
    flush_batch_size:  int = Field(default=10000)  # chunks per Qdrant batch upsert
    flush_interval_s:  int = Field(default=60)     # seconds between flush cycles
    # Chunked HTTP upload
    chunk_size_mb:     int = Field(default=25)     # per upload chunk
    max_file_size_gb:  float = Field(default=10.0) # max single file — raised for
                                                     # large bundled judgment PDFs and
                                                     # multi-volume scanned case files
    # OCR settings
    ocr_dpi:           int  = Field(default=300)
    ocr_threshold:     float = Field(default=0.6)
    # Hierarchical summarisation window
    summary_window_chars:   int = Field(default=12000)
    summary_overlap_chars:  int = Field(default=500)


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUTH_", env_file=".env", extra="ignore")
    secret_key:                    str = Field(default="CHANGE_ME_IN_PRODUCTION_USE_256_BIT_KEY")
    algorithm:                     str = Field(default="HS256")
    access_token_expire_minutes:   int = Field(default=60)
    refresh_token_expire_days:     int = Field(default=30)


class RateLimitSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RATE_", env_file=".env", extra="ignore")
    search_per_minute: int = Field(default=30)
    chat_per_minute:   int = Field(default=20)
    upload_per_minute: int = Field(default=5)
    draft_per_minute:  int = Field(default=10)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name:        str       = Field(default="NyayaAI")
    app_version:     str       = Field(default="2.0.0")
    environment:     str       = Field(default="development")
    debug:           bool      = Field(default=False)
    log_level:       str       = Field(default="INFO")
    allowed_origins: List[str] = Field(default=["http://localhost:3000"])
    max_upload_size_mb: int    = Field(default=200)  # single-shot path; anything
                                                        # larger auto-routes to chunked upload
    upload_dir:      str       = Field(default="./data/raw/uploads")
    workers:         int       = Field(default=4)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


class Settings:
    def __init__(self):
        self.app         = AppSettings()
        self.db          = DatabaseSettings()
        self.es          = ElasticsearchSettings()
        self.qdrant      = QdrantSettings()
        self.redis       = RedisSettings()
        self.llm         = LLMSettings()
        self.embedding   = EmbeddingSettings()
        self.reranker    = RerankerSettings()
        self.retrieval   = RetrievalSettings()
        self.ingestion   = IngestionSettings()
        self.auth        = AuthSettings()
        self.rate_limit  = RateLimitSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
