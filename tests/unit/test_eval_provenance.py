"""
Tests for run provenance.

Six historical runs (attention-paper, -v2, -v3, -v4, -v4b, -v4c) recorded scores
but not the configuration that produced them, so no delta between them can be
attributed to anything. A run is only evidence if you can say what it ran.
"""

from __future__ import annotations

import json

from eval.provenance import (
    collect_flags,
    config_fingerprint,
    corpus_fingerprint,
)

# ── flags ─────────────────────────────────────────────────────────────────────


def test_collect_flags_captures_every_pipeline_toggle(tmp_settings):
    flags = collect_flags(tmp_settings)

    for name in (
        "use_query_enhancement",
        "use_hybrid_retrieval",
        "use_llm_reranker",
        "use_context_compression",
        "use_faithfulness_check",
        "use_langgraph_agent",
    ):
        assert name in flags, f"{name} missing from run provenance"


def test_collect_flags_captures_tuning_values_not_just_booleans(tmp_settings):
    """A run at top_k=4 is not comparable to one at top_k=10."""
    flags = collect_flags(tmp_settings)

    assert "default_top_k" in flags
    assert "compression_threshold" in flags


def test_collect_flags_excludes_secrets(tmp_settings):
    """Manifests are committed with results; they must not carry credentials."""
    flags = collect_flags(tmp_settings)

    serialised = json.dumps(flags).lower()
    assert "sk-test-fake-key-for-unit-tests" not in serialised
    assert "openai_api_key" not in flags
    assert "jwt_secret_key" not in flags
    assert "weaviate_api_key" not in flags


# ── config fingerprint ────────────────────────────────────────────────────────


def test_identical_configs_share_a_fingerprint(tmp_settings):
    a = config_fingerprint(collect_flags(tmp_settings), models={"llm": "gpt-4.1-mini"})
    b = config_fingerprint(collect_flags(tmp_settings), models={"llm": "gpt-4.1-mini"})

    assert a == b


def test_flipping_one_flag_changes_the_fingerprint(tmp_settings):
    """This is what would have distinguished v4b from v4c."""
    baseline = config_fingerprint(collect_flags(tmp_settings), models={"llm": "gpt-4.1-mini"})
    variant = config_fingerprint(
        collect_flags(tmp_settings(use_llm_reranker=True)),
        models={"llm": "gpt-4.1-mini"},
    )

    assert baseline != variant


def test_changing_the_model_changes_the_fingerprint(tmp_settings):
    flags = collect_flags(tmp_settings)

    assert config_fingerprint(flags, models={"llm": "gpt-4.1-mini"}) != config_fingerprint(
        flags, models={"llm": "gpt-4o"}
    )


def test_fingerprint_is_insensitive_to_key_order(tmp_settings):
    assert config_fingerprint({"a": 1, "b": 2}, models={"x": "y"}) == config_fingerprint(
        {"b": 2, "a": 1}, models={"x": "y"}
    )


# ── corpus fingerprint ────────────────────────────────────────────────────────


def _write_corpus(root, document_id, chunk_texts):
    doc_dir = root / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    chunks = [{"chunk_id": f"{document_id}:chunk:{i}", "text": text} for i, text in enumerate(chunk_texts)]
    (doc_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


def test_corpus_fingerprint_records_documents_and_chunk_count(tmp_path):
    _write_corpus(tmp_path, "doc_a", ["alpha", "beta"])

    result = corpus_fingerprint(tmp_path)

    assert result["document_ids"] == ["doc_a"]
    assert result["chunk_count"] == 2


def test_corpus_fingerprint_changes_when_chunk_text_changes(tmp_path):
    """Re-ingesting after the OCR fix produced different text for identical ids."""
    _write_corpus(tmp_path, "doc_a", ["Microsoft Cloudrevenuewas$51.5billion"])
    before = corpus_fingerprint(tmp_path)["content_hash"]

    _write_corpus(tmp_path, "doc_a", ["Microsoft Cloud revenue was $51.5 billion"])
    after = corpus_fingerprint(tmp_path)["content_hash"]

    assert before != after, "a run's scores are not comparable across a corpus change"


def test_corpus_fingerprint_is_stable_for_unchanged_content(tmp_path):
    _write_corpus(tmp_path, "doc_a", ["alpha", "beta"])

    assert corpus_fingerprint(tmp_path) == corpus_fingerprint(tmp_path)


def test_corpus_fingerprint_handles_an_empty_corpus(tmp_path):
    result = corpus_fingerprint(tmp_path)

    assert result["chunk_count"] == 0
    assert result["document_ids"] == []
