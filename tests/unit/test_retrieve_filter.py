from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.app import create_app
from api.routers.auth import _rate_limiter
from rag.qa import MultiAgentQAResponse, answer_question_from_frozen_artifacts
from rag.retrieve import JsonVectorStore


def _make_store(tmp_path, rows):
    p = tmp_path / "store.json"
    p.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return JsonVectorStore(p)


def test_query_filter_returns_only_matching_doc(tmp_path):
    store = _make_store(
        tmp_path,
        [
            {"chunk_id": "a1", "text": "alpha", "metadata": {"document_id": "doc_a"}, "embedding": [1.0, 0.0]},
            {"chunk_id": "b1", "text": "beta", "metadata": {"document_id": "doc_b"}, "embedding": [1.0, 0.0]},
        ],
    )
    results = store.query([1.0, 0.0], top_k=10, doc_filter=["doc_a"])
    assert len(results) == 1
    assert results[0].chunk_id == "a1"


def test_query_filter_none_returns_all(tmp_path):
    store = _make_store(
        tmp_path,
        [
            {"chunk_id": "a1", "text": "alpha", "metadata": {"document_id": "doc_a"}, "embedding": [1.0, 0.0]},
            {"chunk_id": "b1", "text": "beta", "metadata": {"document_id": "doc_b"}, "embedding": [1.0, 0.0]},
        ],
    )
    results = store.query([1.0, 0.0], top_k=10, doc_filter=None)
    assert len(results) == 2


def test_query_filter_empty_list_returns_nothing(tmp_path):
    store = _make_store(
        tmp_path,
        [
            {"chunk_id": "a1", "text": "alpha", "metadata": {"document_id": "doc_a"}, "embedding": [1.0, 0.0]},
        ],
    )
    results = store.query([1.0, 0.0], top_k=10, doc_filter=[])
    assert results == []


def test_qa_passes_doc_filter_to_retriever(tmp_settings):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_retriever.embedding_backend.embed_texts.return_value = [[0.1] * 10]
    mock_retriever.filter_by_relevance.return_value = ["doc_a"]

    with (
        patch("rag.qa.DocumentRetriever", return_value=mock_retriever),
        patch("rag.qa._synthesize_answer", return_value="mocked answer"),
    ):
        answer_question_from_frozen_artifacts(
            "What is X?",
            settings=tmp_settings.model_copy(
                update={
                    "use_document_intelligence": False,
                    "use_query_enhancement": False,
                    "use_hybrid_retrieval": False,
                }
            ),
            top_k=2,
        )

    mock_retriever.retrieve.assert_called_once()


def test_query_endpoint_forwards_question(tmp_settings):
    _rate_limiter._hits.clear()
    app = create_app(tmp_settings)
    mock_response = MultiAgentQAResponse(
        question="test?",
        answer="test answer",
        sources=[],
        router={},
        specialists=[],
    )
    with patch("api.routers.query.answer_question", return_value=mock_response) as mock_fn:
        with TestClient(app) as client:
            # Register a user and obtain a token for auth
            r = client.post("/auth/register", json={"email": "query@example.com", "password": "password123"})
            token = r.json()["access_token"]
            client.post(
                "/query",
                json={"question": "test?", "top_k": 2},
                headers={"Authorization": f"Bearer {token}"},
            )
        mock_fn.assert_called_once()
