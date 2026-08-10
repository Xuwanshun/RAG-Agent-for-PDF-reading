"""
Agentic RAG pipeline built as a LangGraph state machine.

This is an alternative front end to the linear pipeline in ``rag/qa.py``. Both
produce a ``MultiAgentQAResponse`` and both reuse the same retrieval, rerank,
compression and faithfulness components — the difference is control flow:

                        ┌── complex ──> decompose ──┐
    START ──> classify ─┤                           ├──> retrieve ──> grade
                        └── simple ───> enhance ────┘        ▲          │
                                                             │          │
                                              rewrite <──────┼──────────┤ insufficient
                                                             │          │
                                                  (loop cap) └          ▼ sufficient
                                                              synthesize ──> verify ──> END

``rag/qa.py`` cannot express the ``grade -> rewrite -> retrieve`` cycle without
an ad-hoc while loop, which is the reason this module exists. Selected with
``USE_LANGGRAPH_AGENT=true``; the linear pipeline stays the default so the two
can be compared on the same eval set.

Cost note: each extra loop adds one grade call and one rewrite call. The
expensive stages (LLM rerank, compression, synthesis, faithfulness) run once,
after the loop settles.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from config import Settings
from document_process.clients import build_openai_client
from rag.chunk import RetrievedChunk
from rag.compress import ContextCompressor
from rag.faithfulness import FaithfulnessChecker
from rag.qa import (
    MultiAgentQAResponse,
    SpecialistResult,
    _filter_by_region_type,
    _load_visual_summaries,
    _rerank_chunks,
    _route_question,
    _run_specialist,
    _source_payload,
    _synthesize_answer,
)
from rag.query_enhancement import classify_query, decompose_query, hyde_enhance
from rag.rerank import LLMReranker
from rag.retrieve import DocumentRetriever

logger = logging.getLogger(__name__)

_GRADE_SYSTEM_PROMPT = (
    "You grade whether a set of retrieved passages contains enough information to "
    "fully answer the user's question.\n"
    "Respond with a single number between 0.0 and 1.0 and nothing else:\n"
    "  1.0 — the passages fully answer the question\n"
    "  0.5 — the passages are on topic but miss key facts\n"
    "  0.0 — the passages are irrelevant\n"
    "Respond with only the number."
)

_REWRITE_SYSTEM_PROMPT = (
    "You rewrite a search query that failed to retrieve useful passages from a "
    "technical document corpus.\n"
    "Use domain vocabulary the document itself would use, expand abbreviations, and "
    "target the specific fact the question is after. Do not restate the question in "
    "different words — change what is being searched for.\n"
    "Respond with only the rewritten query, no explanation."
)


def grade_chunks(question: str, chunks: list[RetrievedChunk], settings: Settings) -> float:
    """
    Score how well the retrieved chunks support answering the question (0.0-1.0).

    Retrieval that returns nothing scores 0.0 without spending an LLM call.
    An unparseable judge response scores 1.0 ("sufficient") on purpose: a flaky
    grader must never be able to drive the retrieval loop round again forever.
    """
    if not chunks:
        return 0.0
    client = build_openai_client(settings)
    evidence = "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.text}" for chunk in chunks)
    raw = client.generate_text(
        system_prompt=_GRADE_SYSTEM_PROMPT,
        user_prompt=f"Question: {question}\n\nRetrieved passages:\n{evidence}",
    )
    match = re.search(r"\d+(?:\.\d+)?", str(raw))
    if not match:
        logger.warning("Chunk grader returned unparseable output; assuming sufficient")
        return 1.0
    return max(0.0, min(1.0, float(match.group())))


def rewrite_query(question: str, chunks: list[RetrievedChunk], settings: Settings) -> str:
    """
    Produce a new retrieval query after a graded-insufficient attempt.

    The passages that were just retrieved are shown to the model as negative
    evidence so the rewrite moves away from them rather than restating the
    original question in different words.
    """
    client = build_openai_client(settings)
    evidence = "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.text}" for chunk in chunks) or "None"
    rewritten = client.generate_text(
        system_prompt=_REWRITE_SYSTEM_PROMPT,
        user_prompt=(
            f"Original question: {question}\n\n"
            f"Passages retrieved so far (insufficient):\n{evidence}\n\n"
            "Write the improved search query."
        ),
    ).strip()
    return rewritten or question


# ── Graph ─────────────────────────────────────────────────────────────────────


class GraphState(TypedDict, total=False):
    """State threaded through the graph. Each node returns a partial update."""

    question: str
    query_type: str
    # The queries actually sent to the vector store this round — one HyDE
    # passage, several sub-queries, or one rewrite.
    queries: list[str]
    chunks: list[RetrievedChunk]
    grade: float
    # Number of completed retrieval attempts, used to enforce the loop cap.
    iterations: int
    answer: str
    router: dict[str, Any]
    specialists: list[SpecialistResult]


def build_rag_graph(
    settings: Settings,
    retriever: DocumentRetriever,
    *,
    top_k: int | None = None,
    doc_filter: list[str] | None = None,
):
    """Compile the agentic RAG graph for a single question."""
    k = top_k or settings.default_top_k
    fetch_k = k * 3

    def classify(state: GraphState) -> GraphState:
        if not settings.use_query_enhancement:
            return {"query_type": "simple"}
        return {"query_type": classify_query(state["question"], settings)}

    def enhance(state: GraphState) -> GraphState:
        if not settings.use_query_enhancement:
            return {"queries": [state["question"]]}
        return {"queries": [hyde_enhance(state["question"], settings)]}

    def decompose(state: GraphState) -> GraphState:
        return {"queries": decompose_query(state["question"], settings)}

    def retrieve(state: GraphState) -> GraphState:
        question = state["question"]
        seen: set[str] = set()
        collected: list[RetrievedChunk] = []
        for query in state["queries"]:
            if settings.use_hybrid_retrieval:
                results = retriever.hybrid_retrieve(query, top_k=fetch_k, doc_filter=doc_filter)
            else:
                results = retriever.retrieve(query, top_k=fetch_k, doc_filter=doc_filter)
            for chunk in results:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    collected.append(chunk)
        collected = _rerank_chunks(question, collected)
        collected = _filter_by_region_type(question, collected)
        return {
            "chunks": collected[:k],
            "iterations": state.get("iterations", 0) + 1,
        }

    def grade(state: GraphState) -> GraphState:
        return {"grade": grade_chunks(state["question"], state["chunks"], settings)}

    def rewrite(state: GraphState) -> GraphState:
        return {"queries": [rewrite_query(state["question"], state["chunks"], settings)]}

    def synthesize(state: GraphState) -> GraphState:
        question = state["question"]
        chunks = state["chunks"]

        # Expensive precision stages run once, after the retrieval loop settles.
        visual_summaries = _load_visual_summaries(settings, chunks)
        if settings.use_llm_reranker:
            chunks = LLMReranker(settings).rerank(question, chunks, visual_summaries=visual_summaries)

        router = _route_question(question, chunks, visual_summaries)
        specialists: list[SpecialistResult] = []
        if router["use_table_agent"] and router["table_regions"]:
            specialists.append(_run_specialist("table", question, router["table_regions"], visual_summaries, settings))
        if router["use_figure_agent"] and router["figure_regions"]:
            specialists.append(
                _run_specialist("figure", question, router["figure_regions"], visual_summaries, settings)
            )

        compressed_context: str | None = None
        if settings.use_context_compression:
            compressed_context = ContextCompressor(settings).compress(
                question,
                chunks,
                visual_summaries=visual_summaries,
                compression_threshold=settings.compression_threshold,
            )

        answer = _synthesize_answer(question, chunks, specialists, settings, compressed_context)
        return {"answer": answer, "chunks": chunks, "router": router, "specialists": specialists}

    def verify(state: GraphState) -> GraphState:
        if not settings.use_faithfulness_check:
            return {}
        checker = FaithfulnessChecker(settings)
        result = checker.check(state["question"], state["chunks"], state["answer"])
        if result.recommended_action == "return_as_is":
            return {}
        return {"answer": checker.correct(state["question"], state["answer"], result, state["chunks"])}

    def route_after_classify(state: GraphState) -> str:
        return "decompose" if state["query_type"] == "complex" else "enhance"

    def route_after_grade(state: GraphState) -> str:
        if state["grade"] >= settings.graph_grade_threshold:
            return "synthesize"
        if state["iterations"] > settings.graph_max_retrieval_loops:
            logger.info(
                "Retrieval loop cap reached (%d attempts, grade %.2f); synthesizing anyway",
                state["iterations"],
                state["grade"],
            )
            return "synthesize"
        return "rewrite"

    builder = StateGraph(GraphState)
    builder.add_node("classify", classify)
    builder.add_node("enhance", enhance)
    builder.add_node("decompose", decompose)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("synthesize", synthesize)
    builder.add_node("verify", verify)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify, ["decompose", "enhance"])
    builder.add_edge("enhance", "retrieve")
    builder.add_edge("decompose", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade, ["rewrite", "synthesize"])
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("synthesize", "verify")
    builder.add_edge("verify", END)

    return builder.compile()


def answer_question_with_graph(
    question: str,
    *,
    settings: Settings | None = None,
    top_k: int | None = None,
    doc_filter: list[str] | None = None,
    retriever: DocumentRetriever | None = None,
) -> MultiAgentQAResponse:
    """
    Answer a question through the LangGraph pipeline.

    Drop-in replacement for ``rag.qa.answer_question_from_frozen_artifacts`` —
    same signature and same return type, so the API layer and the eval harness
    can switch between them on a flag.

    ``retriever`` is injectable so the graph can be driven in tests without a
    vector store; when omitted one is built and closed here.
    """
    resolved_settings = settings or Settings()
    owns_retriever = retriever is None
    active_retriever = retriever or DocumentRetriever(resolved_settings)

    try:
        if doc_filter is None and resolved_settings.use_document_intelligence:
            query_embedding = active_retriever.embedding_backend.embed_texts([question])[0]
            matched_docs = active_retriever.filter_by_relevance(query_embedding, resolved_settings.doc_filter_threshold)
            if not matched_docs:
                return MultiAgentQAResponse(
                    question=question,
                    answer="No relevant documents found in the corpus for this question.",
                    sources=[],
                    router={
                        "use_table_agent": False,
                        "use_figure_agent": False,
                        "table_regions": [],
                        "figure_regions": [],
                    },
                    specialists=[],
                )
            doc_filter = matched_docs

        graph = build_rag_graph(
            resolved_settings,
            active_retriever,
            top_k=top_k,
            doc_filter=doc_filter,
        )
        final_state = graph.invoke({"question": question, "iterations": 0})

        return MultiAgentQAResponse(
            question=question,
            answer=final_state.get("answer", ""),
            sources=[_source_payload(chunk) for chunk in final_state.get("chunks", [])],
            router=final_state.get("router", {}),
            specialists=final_state.get("specialists", []),
        )
    finally:
        if owns_retriever:
            active_retriever.__exit__(None, None, None)
