from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.chunk import RetrievedChunk
from rag.graph import answer_question_with_graph, grade_chunks, rewrite_query


def _chunk(chunk_id: str = "doc:chunk:1", text: str = "Some evidence.") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, metadata={}, score=0.5)


# ── grade_chunks ──────────────────────────────────────────────────────────────


def test_grade_chunks_returns_parsed_score(tmp_settings):
    with patch("rag.graph.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "0.8"
        mock_build.return_value = mock_client

        result = grade_chunks("What is X?", [_chunk()], tmp_settings)

    assert result == 0.8


def test_grade_chunks_assumes_sufficient_when_output_unparseable(tmp_settings):
    """A broken judge must not be able to trigger an endless retrieval loop."""
    with patch("rag.graph.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "I cannot determine this"
        mock_build.return_value = mock_client

        result = grade_chunks("What is X?", [_chunk()], tmp_settings)

    assert result == 1.0


def test_grade_chunks_clamps_scores_above_one(tmp_settings):
    with patch("rag.graph.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "1.5"
        mock_build.return_value = mock_client

        result = grade_chunks("What is X?", [_chunk()], tmp_settings)

    assert result == 1.0


def test_grade_chunks_returns_zero_for_empty_chunks_without_calling_llm(tmp_settings):
    with patch("rag.graph.build_openai_client") as mock_build:
        result = grade_chunks("What is X?", [], tmp_settings)

    assert result == 0.0
    mock_build.assert_not_called()


# ── rewrite_query ─────────────────────────────────────────────────────────────


def test_rewrite_query_returns_stripped_rewrite(tmp_settings):
    with patch("rag.graph.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "  vocabulary size WSJ constituency parsing  "
        mock_build.return_value = mock_client

        result = rewrite_query("How big was the vocab?", [_chunk()], tmp_settings)

    assert result == "vocabulary size WSJ constituency parsing"


def test_rewrite_query_falls_back_to_original_on_blank_response(tmp_settings):
    with patch("rag.graph.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "   "
        mock_build.return_value = mock_client

        original = "How big was the vocab?"
        result = rewrite_query(original, [_chunk()], tmp_settings)

    assert result == original


def test_rewrite_query_shows_failed_chunks_to_the_model(tmp_settings):
    """The rewrite is informed by what the last attempt wrongly retrieved."""
    with patch("rag.graph.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "better query"
        mock_build.return_value = mock_client

        rewrite_query("What is X?", [_chunk(text="Irrelevant passage about Y.")], tmp_settings)

    user_prompt = mock_client.generate_text.call_args.kwargs["user_prompt"]
    assert "Irrelevant passage about Y." in user_prompt


# ── graph control flow ────────────────────────────────────────────────────────


@pytest.fixture
def graph_settings(tmp_settings):
    """Settings for the routing tests — classification/HyDE are the paths under test."""
    return tmp_settings(use_query_enhancement=True)


class _FakeRetriever:
    """Records every query the graph actually sends to retrieval."""

    def __init__(self, chunks=None):
        self.queries: list[str] = []
        self._chunks = chunks if chunks is not None else [_chunk()]
        self.embedding_backend = MagicMock()

    def retrieve(self, question, top_k=None, *, doc_filter=None):
        self.queries.append(question)
        return list(self._chunks)

    def hybrid_retrieve(self, question, top_k=None, *, doc_filter=None):
        self.queries.append(question)
        return list(self._chunks)

    def filter_by_relevance(self, query_embedding, threshold):
        return []

    def __exit__(self, *_):
        return None


def _run_graph(settings, retriever, *, query_type="simple", grades=(1.0,), question="What is X?"):
    """Drive the graph with scripted classification and grading."""
    remaining = list(grades)

    def _fake_grade(_question, _chunks, _settings):
        return remaining.pop(0) if remaining else 1.0

    with (
        patch("rag.graph.classify_query", return_value=query_type),
        patch("rag.graph.hyde_enhance", side_effect=lambda q, s: f"hyde::{q}"),
        patch("rag.graph.decompose_query", return_value=["sub one", "sub two"]),
        patch("rag.graph.grade_chunks", side_effect=_fake_grade),
        patch("rag.graph.rewrite_query", side_effect=lambda q, c, s: f"rewritten::{q}"),
        patch("rag.graph._synthesize_answer", return_value="The answer."),
    ):
        return answer_question_with_graph(question, settings=settings, retriever=retriever)


def test_graph_routes_simple_question_straight_to_retrieval(graph_settings):
    """A simple query is HyDE-enhanced, never decomposed."""
    retriever = _FakeRetriever()

    _run_graph(graph_settings, retriever, query_type="simple")

    assert retriever.queries == ["hyde::What is X?"]


def test_graph_routes_complex_question_through_decomposition(graph_settings):
    """A complex query is split, and every sub-query reaches retrieval."""
    retriever = _FakeRetriever()

    _run_graph(graph_settings, retriever, query_type="complex")

    assert retriever.queries == ["sub one", "sub two"]


def test_graph_does_not_retry_when_chunks_grade_sufficient(graph_settings):
    retriever = _FakeRetriever()

    _run_graph(graph_settings, retriever, grades=(1.0,))

    assert len(retriever.queries) == 1


def test_graph_rewrites_and_retries_when_chunks_grade_insufficient(graph_settings):
    """A weak first attempt is followed by a rewritten second attempt."""
    retriever = _FakeRetriever()

    _run_graph(graph_settings, retriever, grades=(0.1, 1.0))

    assert retriever.queries == [
        "hyde::What is X?",
        "rewritten::What is X?",
    ]


def test_graph_stops_retrying_at_the_configured_loop_cap(tmp_settings):
    """Persistently bad grades must terminate, not loop forever."""
    settings = tmp_settings(use_query_enhancement=True, graph_max_retrieval_loops=2)
    retriever = _FakeRetriever()

    _run_graph(settings, retriever, grades=(0.0, 0.0, 0.0, 0.0, 0.0))

    assert len(retriever.queries) == 3  # initial attempt + 2 rewrites


def test_graph_honours_a_high_loop_cap_without_hitting_langgraphs_recursion_budget(tmp_settings):
    """
    Regression guard, not a TDD-derived test: this passed the moment it was
    written. LangGraph aborts a run that exceeds its recursion budget, and a
    large loop cap is the only way this graph could approach it. Pinned here so
    a future LangGraph upgrade that tightens the budget fails loudly.
    """
    settings = tmp_settings(use_query_enhancement=True, graph_max_retrieval_loops=10)
    retriever = _FakeRetriever()

    _run_graph(settings, retriever, grades=tuple([0.0] * 20))

    assert len(retriever.queries) == 11  # initial attempt + 10 rewrites


def test_graph_returns_answer_and_sources(graph_settings):
    retriever = _FakeRetriever(chunks=[_chunk(chunk_id="doc:chunk:7")])

    response = _run_graph(graph_settings, retriever)

    assert response.answer == "The answer."
    assert [source["chunk_id"] for source in response.sources] == ["doc:chunk:7"]
