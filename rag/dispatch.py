"""
Single entry point that selects the QA pipeline implementation.

Two pipelines answer questions and both return ``MultiAgentQAResponse``:

  * ``rag.qa``    — linear: enhance -> retrieve -> rerank -> synthesize -> verify
  * ``rag.graph`` — LangGraph: adds conditional routing and a self-correcting
                    retrieval loop (``USE_LANGGRAPH_AGENT=true``)

Callers (the API router, the CLI, the eval harness) go through here so the two
can be swapped with one env var and compared on the same eval set.
"""

from __future__ import annotations

from config import Settings
from rag.graph import answer_question_with_graph
from rag.qa import MultiAgentQAResponse, answer_question_from_frozen_artifacts


def answer_question(
    question: str,
    *,
    settings: Settings | None = None,
    top_k: int | None = None,
    doc_filter: list[str] | None = None,
) -> MultiAgentQAResponse:
    """Answer a question using whichever pipeline the settings select."""
    resolved_settings = settings or Settings()
    pipeline = (
        answer_question_with_graph if resolved_settings.use_langgraph_agent else answer_question_from_frozen_artifacts
    )
    return pipeline(
        question,
        settings=resolved_settings,
        top_k=top_k,
        doc_filter=doc_filter,
    )
