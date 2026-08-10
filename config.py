"""
Application configuration.

All settings are read from environment variables (or a .env file).
pydantic-settings handles this automatically — you never need to call
os.getenv() manually anywhere in the codebase.

How to use:
    from config import Settings
    settings = Settings()  # reads env vars + .env file automatically
    print(settings.openai_api_key)

To override a value locally, either:
    - Set it in your shell: export OPENAI_API_KEY=sk-...
    - Add it to your .env file: OPENAI_API_KEY=sk-...
    - Pass it directly in tests: Settings(openai_api_key="fake-key")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration object.

    Every field corresponds to one environment variable of the same name
    (uppercased). pydantic-settings reads them automatically.
    """

    model_config = SettingsConfigDict(
        # Look for a .env file in the current working directory.
        # This is only used locally — in Docker/AWS real env vars are injected.
        env_file=".env",
        env_file_encoding="utf-8",
        # Don't crash on extra env vars that appear in the environment.
        extra="ignore",
    )

    # ── Data directories ──────────────────────────────────────────────────────
    # These defaults work locally. In Docker they are overridden via ENV
    # instructions in the Dockerfile (set to /app/data/...).
    raw_documents_dir: Path = Path("data/raw")
    processed_documents_dir: Path = Path("data/processed")
    vectorstore_dir: Path = Path("data/embedded")

    # Where PaddleOCR/PaddleX downloads and caches its ML models.
    # In Docker this points to a mounted volume so models persist across
    # container restarts and do not need re-downloading each time.
    paddle_cache_dir: Path = Path("data/.paddlex_cache")

    # ── OpenAI ────────────────────────────────────────────────────────────────
    # Required for --index and --ask (embedding + generation).
    # In AWS this is injected from Secrets Manager — never hardcode it.
    openai_api_key: str | None = None
    # Leave empty to use the default OpenAI endpoint.
    # Set this to point at a compatible alternative (e.g. Azure OpenAI, LM Studio).
    openai_base_url: str | None = None
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"

    # ── Pipeline tuning ───────────────────────────────────────────────────────
    preprocess_chunk_size: int = 1800
    preprocess_chunk_overlap: int = 200
    pdf_render_scale: float = 2.0
    # Maximum pages processed per batch during OCR + layout. Lower values use
    # less peak RAM; higher values reduce per-batch model warm-up overhead.
    # 25 keeps peak memory well under 4 GB headroom on the 8 GB Fargate task.
    preprocess_page_batch_size: int = 25
    default_top_k: int = 4
    # Set to True if you have chromadb installed and prefer it over the
    # built-in JSON vector store.
    prefer_chroma: bool = False

    # ── Weaviate vector store ─────────────────────────────────────────────────
    # Set PREFER_WEAVIATE=true to use Weaviate instead of the JSON vector store.
    # Requires weaviate-client to be installed and a running Weaviate instance.
    prefer_weaviate: bool = False
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    weaviate_grpc_port: int = 50051
    weaviate_collection: str = "RagChunk"
    # Set for Weaviate Cloud — leave empty for local/self-hosted.
    weaviate_api_key: str = ""

    # ── Logging ───────────────────────────────────────────────────────────────
    # LOG_LEVEL: standard Python log level (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"
    # LOG_FORMAT: "text" for human-readable (local dev), "json" for CloudWatch
    log_format: Literal["text", "json"] = "text"

    # ── Application mode ─────────────────────────────────────────────────────
    # "cli"  → use the argparse CLI (python main.py --preprocess ...)
    # "api"  → start the FastAPI server (python main.py --serve)
    app_mode: Literal["cli", "api"] = "cli"

    # ── Qwen VLM endpoint (fine-tuned Qwen3-VL-4B) ───────────────────────────
    # When set, VLM summaries and document-intelligence descriptor calls are
    # routed to this endpoint instead of OpenAI.  Point it at a vLLM server
    # running your SFT checkpoint:
    #   vllm serve /path/to/qwen3-vl-4b-sft --dtype auto --port 8001
    # Then set:
    #   QWEN_BASE_URL=http://localhost:8001/v1
    #   QWEN_MODEL=your-checkpoint-name   (as registered in vLLM)
    #   QWEN_API_KEY=not-needed           (vLLM accepts any non-empty string)
    qwen_base_url: str | None = None
    qwen_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    qwen_api_key: str = "not-needed"

    # ── VLM visual summaries ──────────────────────────────────────────────────
    # When enabled, each cropped table/figure is sent to the vision model and
    # the returned description replaces the OCR-text fallback in
    # visual_summaries.json. This is the recommended setting for production
    # because figures and charts carry zero useful OCR text.
    #
    # Cost note: one API call per detected table/figure region per document.
    # A 20-page report with 5 tables and 3 figures = 8 vision calls.
    use_vlm_summaries: bool = False
    vlm_model: str = "gpt-4o"

    # Self-hosted VLM via Modal (optional).
    # When set, the pipeline calls the Modal endpoint first and falls back to
    # vlm_model (gpt-4o) on any connection error or timeout.
    # Leave unset to use gpt-4o exclusively (existing behaviour).
    vlm_base_url: str | None = None  # e.g. https://lzhuang971--qwen3-vl-rag-model-fastapi-app.modal.run/v1
    vlm_self_hosted_model: str = "qwen3-vl-rag"  # model name Modal endpoint exposes

    # ── Document intelligence ─────────────────────────────────────────────────
    # Enables title propagation, section grouping, and a document descriptor.
    # The descriptor's summary embedding is stored in document.json and used
    # for doc-level pre-filtering at query time.
    use_document_intelligence: bool = False
    # When enabled, the chunking strategy (size, overlap, table handling) is
    # chosen per document by the descriptor model rather than using the fixed
    # PREPROCESS_CHUNK_SIZE / PREPROCESS_CHUNK_OVERLAP defaults.
    use_adaptive_chunking: bool = False
    # Model for section and document summarization (lightweight is fine).
    descriptor_model: str = "gpt-4o-mini"
    # Cosine similarity threshold for doc-level pre-filtering at query time.
    # Queries are only searched against documents whose summary embedding
    # exceeds this threshold.
    doc_filter_threshold: float = 0.20

    # ── Reading order ────────────────────────────────────────────────────────
    # When enabled, uses LayoutLMv3 (hantian/layoutreader) to determine reading
    # order instead of the default geometric bbox sort. Handles multi-column
    # layouts, sidebars, and other complex page structures correctly.
    # Requires torch and transformers to be installed.
    use_layout_reader: bool = False
    # HuggingFace model ID for LayoutReader. Override to use a local checkpoint.
    layout_reader_model: str = "hantian/layoutreader"

    # ── Query enhancement ─────────────────────────────────────────────────────
    # When enabled, simple queries are enhanced with HyDE (hypothetical document
    # embeddings) and complex queries are decomposed into sub-queries before
    # retrieval. Disable in tests to avoid extra LLM calls.
    use_query_enhancement: bool = True

    # ── Hybrid retrieval ──────────────────────────────────────────────────────
    # When enabled, retrieval fuses dense vector search with BM25 sparse search
    # via Reciprocal Rank Fusion (RRF), then applies region-type boosting and
    # parent-context expansion.  Set USE_HYBRID_RETRIEVAL=true to activate.
    use_hybrid_retrieval: bool = False

    # ── LLM reranker ──────────────────────────────────────────────────────────
    # When enabled, retrieved chunks are re-scored by the LLM in a single batch
    # call before being passed to the answer synthesiser.  Chunks scoring below
    # 0.3 are dropped entirely.  Set USE_LLM_RERANKER=true to activate.
    # Adds one LLM API call per query; disable in tests.
    use_llm_reranker: bool = False

    # ── Context compression ───────────────────────────────────────────────────
    # When enabled, retrieved passages are compressed by the LLM before being
    # fed to the synthesis agent.  OCR artifacts, boilerplate, and sentences
    # that do not bear on the query are removed.  Target: ≥40% token reduction.
    # Adds one LLM API call per query; disable in tests.
    use_context_compression: bool = False
    # Passages whose rerank score falls below this threshold AND that contain no
    # unique information not found in higher-scoring passages are dropped entirely
    # during compression.
    compression_threshold: float = 0.5

    # ── Faithfulness check ────────────────────────────────────────────────────
    # When enabled, the generated answer is verified claim-by-claim against the
    # retrieved source passages.  Answers with UNSUPPORTED or INFERRED claims
    # are automatically rewritten to remove or hedge those claims.
    # Adds two LLM API calls per query when issues are found; disable in tests.
    use_faithfulness_check: bool = False

    # ── LangGraph agentic pipeline ────────────────────────────────────────────
    # When enabled, POST /query and the eval harness route through rag/graph.py
    # (LangGraph) instead of the linear rag/qa.py pipeline. The graph adds two
    # things the linear pipeline cannot express: conditional routing on query
    # type, and a self-correcting retrieval loop that re-queries when the
    # retrieved passages are graded insufficient.
    # Both paths are kept so they can be A/B compared on the same eval set.
    use_langgraph_agent: bool = False
    # Maximum number of query rewrites after the initial retrieval attempt.
    # Total retrieval attempts = graph_max_retrieval_loops + 1.
    # Each extra loop costs one grade call plus one rewrite call.
    graph_max_retrieval_loops: int = 2
    # Chunks scoring at or above this grade are considered sufficient and the
    # graph proceeds to synthesis instead of rewriting the query.
    graph_grade_threshold: float = 0.6

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret_key: str | None = None
    jwt_algorithm: str = "HS256"
    database_url: str = "sqlite:///data/auth.db"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    https_only: bool = False
    google_client_id: str | None = None
    google_client_secret: str | None = None
    github_client_id: str | None = None
    github_client_secret: str | None = None

    # ── AWS / S3 ─────────────────────────────────────────────────────────────
    # Set S3_BUCKET_NAME when running on AWS to persist processed artifacts
    # and the vector store across container restarts (ECS tasks are ephemeral).
    # Leave empty for local development where the filesystem is persistent.
    s3_bucket_name: str | None = None
    aws_region: str = "us-east-1"


def ensure_data_dirs(settings: Settings) -> None:
    """
    Create data directories if they do not already exist.

    This is called explicitly at application startup (main.py, api/app.py),
    NOT inside Settings itself. Keeping side effects out of the config object
    makes Settings safe to instantiate in tests without touching the filesystem.
    """
    settings.raw_documents_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_documents_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    settings.paddle_cache_dir.mkdir(parents=True, exist_ok=True)
    (settings.paddle_cache_dir / "temp").mkdir(parents=True, exist_ok=True)


def user_scoped_settings(settings: Settings, user_id: str) -> Settings:
    """Return a copy of settings with all data dirs scoped under user_id/."""
    return settings.model_copy(
        update={
            "raw_documents_dir": settings.raw_documents_dir / user_id,
            "processed_documents_dir": settings.processed_documents_dir / user_id,
            "vectorstore_dir": settings.vectorstore_dir / user_id,
        }
    )
