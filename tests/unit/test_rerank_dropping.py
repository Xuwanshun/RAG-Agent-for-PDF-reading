"""
Tests for the LLM reranker's drop logic and candidate labelling.

Two defects made the reranker silently discard every chunk on many queries:

1. When the model's ``scores`` map was missing a chunk, the code substituted the
   chunk's own incoming score and compared THAT against the 0.3 relevance
   threshold. With hybrid retrieval those incoming scores are RRF values
   (~0.03-0.10), which can never reach 0.3 — so every chunk was dropped and the
   reranker fell back to the unranked order.
2. Candidates were labelled with full 70-character chunk ids, so the model had
   to echo a 64-hex-digit hash back exactly for its scores to bind.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from rag.chunk import RetrievedChunk
from rag.rerank import LLMReranker


def _chunk(index: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"b2f46851450f744d072fb4511187c0a80a1cbe708f3e7e402f72d9338a122b22:chunk:{index}",
        text=f"Passage number {index} with enough text to count as prose rather than a header line.",
        metadata={},
        score=score,
    )


def _reranker_returning(payload: dict, tmp_settings):
    """Patch the LLM to return `payload` and capture the prompt it was sent."""
    mock_client = MagicMock()
    mock_client.generate_text.return_value = json.dumps(payload)
    return patch("rag.rerank.build_openai_client", return_value=mock_client), mock_client


# ── unscored chunks must not be measured on the RRF scale ─────────────────────


def test_chunks_the_model_did_not_score_are_kept(tmp_settings):
    """
    The exact production failure: RRF-scored chunks and no usable scores map.

    The ranking is deliberately a different order from the input so this cannot
    pass via the "dropped everything, fall back to original order" path — that
    fallback would return 3,1,2 order unchanged.
    """
    chunks = [_chunk(1, 0.099), _chunk(2, 0.08), _chunk(3, 0.07)]
    patcher, _client = _reranker_returning({"ranking": ["3", "1", "2"], "scores": {}}, tmp_settings)

    with patcher:
        result = LLMReranker(tmp_settings).rerank("What was revenue?", chunks)

    assert [c.chunk_id.rsplit(":", 1)[1] for c in result] == ["3", "1", "2"], (
        "unscored chunks were dropped by an RRF-vs-0.3 comparison"
    )


def test_explicitly_low_scored_chunks_are_still_dropped(tmp_settings):
    """The threshold must keep working for scores the model actually assigned."""
    chunks = [_chunk(1, 0.099), _chunk(2, 0.08)]
    patcher, _client = _reranker_returning(
        {"ranking": ["1", "2"], "scores": {"1": 0.9, "2": 0.05}},
        tmp_settings,
    )

    with patcher:
        result = LLMReranker(tmp_settings).rerank("What was revenue?", chunks)

    assert [c.chunk_id.rsplit(":", 1)[1] for c in result] == ["1"]


def test_explicitly_dropped_chunks_are_removed(tmp_settings):
    chunks = [_chunk(1, 0.099), _chunk(2, 0.08)]
    patcher, _client = _reranker_returning(
        {"ranking": ["1", "2"], "scores": {"1": 0.9, "2": 0.8}, "dropped": ["2"]},
        tmp_settings,
    )

    with patcher:
        result = LLMReranker(tmp_settings).rerank("What was revenue?", chunks)

    assert [c.chunk_id.rsplit(":", 1)[1] for c in result] == ["1"]


def test_assigned_scores_are_carried_onto_the_returned_chunks(tmp_settings):
    chunks = [_chunk(1, 0.099), _chunk(2, 0.08)]
    patcher, _client = _reranker_returning(
        {"ranking": ["2", "1"], "scores": {"1": 0.5, "2": 0.95}},
        tmp_settings,
    )

    with patcher:
        result = LLMReranker(tmp_settings).rerank("What was revenue?", chunks)

    assert [c.score for c in result] == [0.95, 0.5]


# ── short candidate labels ────────────────────────────────────────────────────


def test_candidates_are_labelled_with_short_indices_not_hash_ids(tmp_settings):
    """A 64-hex-digit hash is easy to mis-transcribe; an index is not."""
    chunks = [_chunk(1, 0.099), _chunk(2, 0.08)]
    patcher, client = _reranker_returning({"ranking": ["1", "2"], "scores": {"1": 0.9, "2": 0.8}}, tmp_settings)

    with patcher:
        LLMReranker(tmp_settings).rerank("What was revenue?", chunks)

    prompt = client.generate_text.call_args.kwargs["user_prompt"]
    assert "[ID: 1]" in prompt
    assert "b2f46851450f744d072fb4511187c0a80a1cbe708f3e7e402f72d9338a122b22" not in prompt


def test_index_labels_map_back_to_the_correct_chunks(tmp_settings):
    chunks = [_chunk(7, 0.099), _chunk(8, 0.08), _chunk(9, 0.07)]
    patcher, _client = _reranker_returning(
        {"ranking": ["3", "1"], "scores": {"3": 0.95, "1": 0.6}, "dropped": ["2"]},
        tmp_settings,
    )

    with patcher:
        result = LLMReranker(tmp_settings).rerank("What was revenue?", chunks)

    # label "3" is the third candidate, which is chunk:9
    assert [c.chunk_id.rsplit(":", 1)[1] for c in result] == ["9", "7"]


def test_malformed_response_still_falls_back_to_original_order(tmp_settings):
    chunks = [_chunk(1, 0.099), _chunk(2, 0.08)]
    mock_client = MagicMock()
    mock_client.generate_text.return_value = "not json at all"

    with patch("rag.rerank.build_openai_client", return_value=mock_client):
        result = LLMReranker(tmp_settings).rerank("What was revenue?", chunks)

    assert len(result) == 2
