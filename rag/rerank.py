"""
LLM-based reranking for retrieved chunks.

Two modes:
- Batch (preferred): all candidates scored in one LLM call via RERANK_BATCH_PROMPT.
  Returns ranking + scores + dropped chunk ids.
- Single (utility): score one passage at a time via RERANK_SCORING_PROMPT.
  Useful for debugging or when the batch is too large to fit in context.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from config import Settings
from document_process.clients import build_openai_client
from rag.chunk import RetrievedChunk
from rag.hybrid import detect_query_type

logger = logging.getLogger(__name__)

# ── Prompt templates (verbatim from spec) ─────────────────────────────────────

RERANK_SYSTEM_PROMPT = """You are a precision relevance scorer for a document QA system.

You will receive a query and a candidate passage. The passage may be one of:
- PROSE: regular paragraph text from the document
- TABLE_DESC: a structured description of a table (column names, row summaries, key values)
- FIGURE_DESC: a structured description of a figure (chart type, axes, key data points, caption)
- SECTION_HEADER: a section or subsection title with surrounding context

Your task: score how well the passage answers or supports answering the query.

Scoring rules:
- Score 0.0–1.0 (float)
- PROSE: score on direct semantic relevance to query intent
- TABLE_DESC: score high if the table's described columns/values contain the answer; \
a description that mentions the right metrics beats prose that vaguely references them
- FIGURE_DESC: score high if the figure's described trend/data directly addresses the query; \
penalize if the description only mentions the figure exists without describing its content
- SECTION_HEADER: score 0.1–0.3 only; useful as tiebreaker context, not primary answer

Never inflate scores. A passage that is topically related but does not help answer \
the specific query should score < 0.4."""


RERANK_SCORING_PROMPT = """Score the relevance of this passage to the query.

Query: {query}

Passage metadata:
  region_type: {region_type}
  parent_title: {parent_title}
  parent_subtitle: {parent_subtitle}
  source_document: {document_id}

Passage content:
{passage_content}

{description_block}

Instructions:
1. Read the query carefully — identify whether it seeks a fact, comparison, procedure, or explanation
2. Read the passage — identify what information it actually contains
3. For TABLE_DESC / FIGURE_DESC: check if the described data directly answers the query, \
not just whether the topic matches
4. Output a single float score between 0.0 and 1.0
5. Output one sentence explaining the score

Output format (strict):
score: <float>
reason: <one sentence>"""


RERANK_BATCH_PROMPT = """You are re-ranking {n_candidates} retrieved passages for the query below.
Each passage has already passed a first-stage vector similarity filter.
Your job is precision: identify which passages genuinely answer the query vs. which are topically adjacent but unhelpful.

Query: {query}
Query intent: {query_intent}

Candidates:
{candidates_block}

Rank all candidates from most to least relevant.
For table/figure candidates: a good description that names relevant columns, values, \
or trends ranks above prose that only mentions the topic in passing.

Refer to each candidate by its [ID: n] number. Score every candidate you rank.

Output format (strict JSON):
{{
  "ranking": ["2", "1", ...],
  "scores": {{"2": 0.95, "1": 0.82, ...}},
  "dropped": ["3"],
  "top_region_type": "prose|table|figure"
}}"""

# Chunks below this score are dropped from context entirely (per spec).
_DROP_THRESHOLD = 0.3


# ── Internal helpers ──────────────────────────────────────────────────────────


@dataclass
class SingleScoreResult:
    score: float
    reason: str


@dataclass
class BatchRerankResult:
    ranking: list[str]
    scores: dict[str, float]
    dropped: list[str]
    top_region_type: str


def _primary_region_type(chunk: RetrievedChunk) -> str:
    """Return a single canonical region type label for display."""
    types: list[str] = chunk.metadata.get("region_types") or []
    if "table" in types:
        return "TABLE_DESC"
    if "figure" in types:
        return "FIGURE_DESC"
    # A very short chunk with no body text is likely a header
    if len(chunk.text.strip()) < 80:
        return "SECTION_HEADER"
    return "PROSE"


def _format_candidate_block(chunks: list[RetrievedChunk], visual_summaries: dict[str, Any]) -> str:
    """
    Format all candidates as the multi-line block expected by RERANK_BATCH_PROMPT.

    Candidates are labelled with their 1-based position, not their chunk_id. A
    chunk_id is a 64-hex-digit document hash plus a suffix; asking the model to
    echo that back exactly for every passage meant a single mis-transcribed
    character silently detached that passage's score. Positions are mapped back
    to chunks by the caller.
    """
    lines: list[str] = []
    for label, chunk in enumerate(chunks, start=1):
        region_type = _primary_region_type(chunk)
        parent_title = chunk.metadata.get("parent_title") or ""
        parent_subtitle = chunk.metadata.get("parent_subtitle") or ""
        section = f"{parent_title} > {parent_subtitle}".strip(" >")

        # Prefer visual summary description for table/figure
        content = _best_content(chunk, visual_summaries)

        lines.append(
            f"[ID: {label}]\n"
            f"Type: {region_type}\n"
            f"Section: {section}\n"
            f"Content: {content[:600]}"  # cap to avoid oversized prompts
        )
    return "\n\n".join(lines)


def _best_content(chunk: RetrievedChunk, visual_summaries: dict[str, Any]) -> str:
    """
    Return the most informative content string for a chunk.

    Figures have no OCR text, so the VLM summary is the only signal — always use it.
    Tables contain exact numbers in OCR text; the VLM summary paraphrases and can
    drop specific values, so we keep the OCR text for tables.
    """
    region_ids: list[str] = chunk.metadata.get("source_region_ids") or chunk.metadata.get("region_ids") or []
    for rid in region_ids:
        summary = visual_summaries.get(str(rid))
        if summary and summary.get("summary_text") and summary.get("region_type") == "figure":
            return summary["summary_text"]
    return chunk.text


def _parse_single_score(raw: str) -> SingleScoreResult:
    """Parse the strict `score: <float>\\nreason: <sentence>` format."""
    score = 0.0
    reason = ""
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("score:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    return SingleScoreResult(score=max(0.0, min(1.0, score)), reason=reason)


def _parse_batch_result(raw: str, candidate_ids: list[str]) -> BatchRerankResult:
    """
    Parse the JSON response from RERANK_BATCH_PROMPT.
    Falls back gracefully when the model returns malformed JSON.
    """
    cleaned = raw.strip()
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting the first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}

    ranking: list[str] = payload.get("ranking") or candidate_ids
    scores: dict[str, float] = {k: float(v) for k, v in (payload.get("scores") or {}).items()}
    dropped: list[str] = payload.get("dropped") or []
    top_region_type: str = payload.get("top_region_type") or "prose"

    # Ensure all candidate ids appear in ranking (defensive)
    seen = set(ranking)
    for cid in candidate_ids:
        if cid not in seen:
            ranking.append(cid)

    return BatchRerankResult(
        ranking=ranking,
        scores=scores,
        dropped=dropped,
        top_region_type=top_region_type,
    )


# ── Public API ─────────────────────────────────────────────────────────────────


class LLMReranker:
    """
    LLM-powered reranker.  Accepts a list of retrieved chunks, calls the model
    to batch-score them, and returns a filtered, reranked list.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        visual_summaries: dict[str, Any] | None = None,
        drop_threshold: float = _DROP_THRESHOLD,
    ) -> list[RetrievedChunk]:
        """
        Batch-rerank ``chunks`` for ``query`` and return the filtered result.

        Chunks whose score falls below ``drop_threshold`` (default 0.3) are
        excluded entirely, matching the spec's "dropped" behaviour.

        Falls back to the original order if the LLM call fails.
        """
        if not chunks:
            return chunks

        summaries = visual_summaries or {}
        query_intent = detect_query_type(query)
        candidates_block = _format_candidate_block(chunks, summaries)
        # Positional labels, matching the [ID: n] markers in the candidate block.
        candidate_ids = [str(index) for index in range(1, len(chunks) + 1)]

        user_prompt = RERANK_BATCH_PROMPT.format(
            n_candidates=len(chunks),
            query=query,
            query_intent=query_intent,
            candidates_block=candidates_block,
        )

        try:
            client = build_openai_client(self.settings)
            raw = client.generate_text(
                system_prompt="You are a retrieval reranker. Respond with valid JSON only.",
                user_prompt=user_prompt,
            )
            result = _parse_batch_result(raw, candidate_ids)
        except Exception as exc:
            logger.warning("LLM reranker failed, returning original order: %s", exc)
            return chunks

        dropped_set = set(result.dropped)
        # Candidates were labelled by position, so map the labels back to chunks.
        chunk_map = {label: chunk for label, chunk in zip(candidate_ids, chunks, strict=False)}

        if not result.scores:
            # Distinct from "the model judged these irrelevant": the reply had no
            # usable scores at all (unparseable JSON, or the field was omitted).
            logger.warning("LLM reranker returned no usable scores — keeping retrieval order")

        reranked: list[RetrievedChunk] = []
        for label in result.ranking:
            if label in dropped_set:
                continue
            chunk = chunk_map.get(label)
            if chunk is None:
                continue
            llm_score = result.scores.get(label)
            if llm_score is None:
                # The model did not score this chunk. Its existing score is a
                # retrieval score (RRF values top out around 0.03) and cannot be
                # compared against a 0-1 relevance threshold, so keep the chunk:
                # "not scored" is not the same as "judged irrelevant".
                reranked.append(chunk)
                continue
            if llm_score < drop_threshold:
                continue
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=llm_score,
                )
            )

        # If the model genuinely rated everything below the threshold, fall back
        # rather than hand the synthesiser nothing.
        if not reranked:
            logger.warning(
                "LLM reranker scored all chunks below %.2f — falling back to retrieval order", drop_threshold
            )
            return chunks

        return reranked

    def score_single(
        self,
        query: str,
        chunk: RetrievedChunk,
        *,
        visual_summaries: dict[str, Any] | None = None,
    ) -> SingleScoreResult:
        """
        Score a single passage.  Useful for debugging or fine-grained inspection.
        """
        summaries = visual_summaries or {}
        region_type = _primary_region_type(chunk)
        content = _best_content(chunk, summaries)

        description_block = ""
        if region_type in ("TABLE_DESC", "FIGURE_DESC"):
            description_block = f"Visual description:\n{content}"
            passage_content = chunk.text  # raw OCR text as the "passage"
        else:
            passage_content = content

        user_prompt = RERANK_SCORING_PROMPT.format(
            query=query,
            region_type=region_type,
            parent_title=chunk.metadata.get("parent_title") or "N/A",
            parent_subtitle=chunk.metadata.get("parent_subtitle") or "N/A",
            document_id=chunk.metadata.get("document_id") or "unknown",
            passage_content=passage_content[:800],
            description_block=description_block,
        )

        try:
            client = build_openai_client(self.settings)
            raw = client.generate_text(
                system_prompt=RERANK_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            return _parse_single_score(raw)
        except Exception as exc:
            logger.warning("Single-passage scorer failed: %s", exc)
            return SingleScoreResult(score=chunk.score, reason="LLM scorer unavailable")
