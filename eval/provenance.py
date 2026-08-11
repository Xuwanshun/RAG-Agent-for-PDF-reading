"""
Run provenance, shaped for LangSmith experiment metadata.

LangSmith stores an arbitrary metadata dict per experiment and lets you compare
experiments in its UI, so there is no reason to build a run-record format here.
What it cannot know is what is specific to this project: which pipeline flags
were set, and what the indexed corpus actually contained.

Why it matters: the deleted runs (attention-paper, -v2 ... -v4c) stored scores
and nothing else, so no difference between them could be attributed to a flag, a
model, or a re-ingest of the corpus. Attaching this dict to every experiment is
what stops that from recurring.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from config import Settings

# Pipeline behaviour is governed by these. Anything here changes what a run means.
_FLAG_FIELDS = (
    "use_query_enhancement",
    "use_hybrid_retrieval",
    "use_llm_reranker",
    "use_context_compression",
    "use_faithfulness_check",
    "use_document_intelligence",
    "use_adaptive_chunking",
    "use_layout_reader",
    "use_vlm_summaries",
    "use_langgraph_agent",
    "prefer_weaviate",
    "prefer_chroma",
)

# Tuning knobs. A run at top_k=4 is not comparable to one at top_k=10, so these
# belong in the fingerprint just as much as the booleans do.
_TUNING_FIELDS = (
    "default_top_k",
    "compression_threshold",
    "doc_filter_threshold",
    "preprocess_chunk_size",
    "preprocess_chunk_overlap",
    "graph_max_retrieval_loops",
    "graph_grade_threshold",
)

_MODEL_FIELDS = (
    "openai_model",
    "embedding_model",
    "descriptor_model",
    "vlm_model",
)


def collect_flags(settings: Settings) -> dict[str, Any]:
    """Capture every setting that changes pipeline behaviour. Never secrets."""
    return {name: getattr(settings, name) for name in (*_FLAG_FIELDS, *_TUNING_FIELDS) if hasattr(settings, name)}


def collect_models(settings: Settings) -> dict[str, str]:
    return {name: str(getattr(settings, name)) for name in _MODEL_FIELDS if hasattr(settings, name)}


def config_fingerprint(flags: dict[str, Any], models: dict[str, str]) -> str:
    """Stable short hash of a configuration, insensitive to key ordering."""
    payload = json.dumps({"flags": flags, "models": models}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def corpus_fingerprint(processed_documents_dir: Path) -> dict[str, Any]:
    """
    Fingerprint the indexed corpus: which documents, how many chunks, what text.

    The content hash covers chunk text, not just ids, because a re-ingest can
    change the text under stable ids — which is exactly what the PDF text-layer
    fix did. Two experiments with different corpus hashes are not comparable,
    however similar their configs look.
    """
    document_ids: list[str] = []
    chunk_count = 0
    digest = hashlib.sha256()

    if processed_documents_dir.exists():
        for doc_dir in sorted(p for p in processed_documents_dir.iterdir() if p.is_dir()):
            chunks_path = doc_dir / "chunks.json"
            if not chunks_path.exists():
                continue
            try:
                chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            document_ids.append(doc_dir.name)
            chunk_count += len(chunks)
            for chunk in chunks:
                digest.update(str(chunk.get("chunk_id", "")).encode("utf-8"))
                digest.update(str(chunk.get("text", "")).encode("utf-8"))

    return {
        "document_ids": document_ids,
        "document_count": len(document_ids),
        "chunk_count": chunk_count,
        "content_hash": digest.hexdigest()[:16],
    }


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def run_metadata(settings: Settings) -> dict[str, Any]:
    """
    Everything needed to reproduce or invalidate an experiment, as a flat dict.

    Pass straight to LangSmith's ``metadata=`` so the comparison UI can group and
    filter experiments by config_hash without any bookkeeping of our own.
    """
    flags = collect_flags(settings)
    models = collect_models(settings)
    corpus = corpus_fingerprint(settings.processed_documents_dir)
    return {
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        # A dirty tree means the recorded SHA does not fully describe the code.
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_hash": config_fingerprint(flags, models),
        "corpus_hash": corpus["content_hash"],
        "corpus_documents": corpus["document_count"],
        "corpus_chunks": corpus["chunk_count"],
        **{f"flag.{name}": value for name, value in flags.items()},
        **{f"model.{name}": value for name, value in models.items()},
    }
