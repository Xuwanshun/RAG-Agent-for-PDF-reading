from __future__ import annotations

from unittest.mock import patch

from rag.dispatch import answer_question
from rag.qa import MultiAgentQAResponse


def _response(answer: str) -> MultiAgentQAResponse:
    return MultiAgentQAResponse(question="q", answer=answer, sources=[], router={}, specialists=[])


def test_dispatch_uses_linear_pipeline_by_default(tmp_settings):
    with (
        patch("rag.dispatch.answer_question_from_frozen_artifacts", return_value=_response("linear")),
        patch("rag.dispatch.answer_question_with_graph", return_value=_response("graph")),
    ):
        result = answer_question("What is X?", settings=tmp_settings)

    assert result.answer == "linear"


def test_dispatch_uses_graph_when_flag_enabled(tmp_settings):
    settings = tmp_settings(use_langgraph_agent=True)

    with (
        patch("rag.dispatch.answer_question_from_frozen_artifacts", return_value=_response("linear")),
        patch("rag.dispatch.answer_question_with_graph", return_value=_response("graph")),
    ):
        result = answer_question("What is X?", settings=settings)

    assert result.answer == "graph"


def test_dispatch_forwards_top_k_and_doc_filter(tmp_settings):
    settings = tmp_settings(use_langgraph_agent=True)

    with patch("rag.dispatch.answer_question_with_graph", return_value=_response("graph")) as mock_graph:
        answer_question("What is X?", settings=settings, top_k=7, doc_filter=["doc-a"])

    assert mock_graph.call_args.kwargs["top_k"] == 7
    assert mock_graph.call_args.kwargs["doc_filter"] == ["doc-a"]
