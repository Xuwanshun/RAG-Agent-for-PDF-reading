from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.clients import build_openai_client
from rag.retrieve import DocumentRetriever, RetrievedChunk


@dataclass(frozen=True)
class SpecialistResult:
    agent_name: str
    output: str
    region_ids: list[str]


@dataclass(frozen=True)
class MultiAgentQAResponse:
    question: str
    answer: str
    sources: list[dict[str, Any]]
    router: dict[str, Any]
    specialists: list[SpecialistResult]


def answer_question_from_frozen_artifacts(
    question: str,
    *,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> MultiAgentQAResponse:
    resolved_settings = settings or Settings()
    retriever = DocumentRetriever(resolved_settings)
    retrieved = _rerank_chunks(question, retriever.retrieve(question, top_k=(top_k or resolved_settings.default_top_k) * 2))
    retrieved = retrieved[: top_k or resolved_settings.default_top_k]
    visual_summaries = _load_visual_summaries(resolved_settings, retrieved)
    router = _route_question(question, retrieved, visual_summaries)
    specialists: list[SpecialistResult] = []
    if router["use_table_agent"] and router["table_regions"]:
        specialists.append(_run_specialist("table", question, router["table_regions"], visual_summaries, resolved_settings))
    if router["use_figure_agent"] and router["figure_regions"]:
        specialists.append(_run_specialist("figure", question, router["figure_regions"], visual_summaries, resolved_settings))
    answer = _synthesize_answer(question, retrieved, specialists, resolved_settings)
    return MultiAgentQAResponse(
        question=question,
        answer=answer,
        sources=[_source_payload(chunk) for chunk in retrieved],
        router=router,
        specialists=specialists,
    )


def _rerank_chunks(question: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    question_terms = _token_set(question)
    rescored: list[RetrievedChunk] = []
    for chunk in retrieved:
        overlap = len(question_terms & _token_set(chunk.text))
        rescored.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=chunk.score + (overlap * 0.01),
            )
        )
    return sorted(rescored, key=lambda item: item.score, reverse=True)


def _route_question(
    question: str,
    retrieved: list[RetrievedChunk],
    visual_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    lowered = question.lower()
    region_ids = {
        region_id
        for chunk in retrieved
        for region_id in chunk.metadata.get("source_region_ids", []) or chunk.metadata.get("region_ids", [])
        if region_id in visual_summaries
    }
    table_regions = [region_id for region_id in region_ids if visual_summaries[region_id]["region_type"] == "table"]
    figure_regions = [region_id for region_id in region_ids if visual_summaries[region_id]["region_type"] == "figure"]
    return {
        "use_table_agent": bool(table_regions) and any(term in lowered for term in ("table", "row", "column")),
        "use_figure_agent": bool(figure_regions) and any(term in lowered for term in ("figure", "chart", "image", "diagram", "shown")),
        "table_regions": table_regions,
        "figure_regions": figure_regions,
    }


def _run_specialist(
    agent_name: str,
    question: str,
    region_ids: list[str],
    visual_summaries: dict[str, dict[str, Any]],
    settings: Settings,
) -> SpecialistResult:
    client = build_openai_client(settings)
    evidence = []
    for region_id in region_ids:
        summary = visual_summaries[region_id]
        evidence.append(
            f"[region={region_id} page={summary['page_number']} type={summary['region_type']}]\n"
            f"{summary['summary_text']}"
        )
    output = client.generate_text(
        system_prompt=(
            f"You are a grounded {agent_name} specialist. "
            "Answer only from the provided frozen preprocessing summaries. "
            "If the evidence is insufficient, say so."
        ),
        user_prompt=f"Question: {question}\n\nEvidence:\n" + "\n\n".join(evidence),
    ).strip()
    return SpecialistResult(agent_name=agent_name, output=output, region_ids=region_ids)


def _synthesize_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    specialists: list[SpecialistResult],
    settings: Settings,
) -> str:
    client = build_openai_client(settings)
    sources = []
    for index, chunk in enumerate(retrieved, start=1):
        sources.append(
            f"[Source {index} chunk={chunk.chunk_id} page={chunk.metadata.get('page_number')} "
            f"regions={chunk.metadata.get('source_region_ids') or chunk.metadata.get('region_ids', [])}]\n{chunk.text}"
        )
    specialist_sections = [
        f"[{item.agent_name} agent regions={item.region_ids}]\n{item.output}"
        for item in specialists
    ]
    evidence_text = "\n\n".join(sources)
    specialist_text = "\n\n".join(specialist_sections) if specialist_sections else "None"
    return client.generate_text(
        system_prompt=(
            "You are a synthesis agent for document-grounded QA. "
            "Answer only from the retrieved chunk evidence and specialist outputs. "
            "Cite chunk ids and page numbers in the answer."
        ),
        user_prompt=(
            f"Question: {question}\n\n"
            f"Retrieved evidence:\n{evidence_text}\n\n"
            f"Specialist outputs:\n{specialist_text}"
        ),
    ).strip()


def _load_visual_summaries(settings: Settings, retrieved: list[RetrievedChunk]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for chunk in retrieved:
        document_id = chunk.metadata.get("document_id")
        if not document_id:
            continue
        path = settings.processed_documents_dir / str(document_id) / "visual_summaries.json"
        if not path.exists():
            continue
        for item in _load_json(path) or []:
            if isinstance(item, dict):
                summaries[str(item["region_id"])] = item
    return summaries


def _source_payload(chunk: RetrievedChunk) -> dict[str, Any]:
    metadata = chunk.metadata
    return {
        "chunk_id": chunk.chunk_id,
        "page_number": metadata.get("page_number"),
        "document_id": metadata.get("document_id"),
        "source_filename": metadata.get("source_filename") or metadata.get("source_file"),
        "region_ids": metadata.get("source_region_ids") or metadata.get("region_ids", []),
        "crop_asset_ids": metadata.get("crop_asset_ids", []),
        "score": round(chunk.score, 4),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _token_set(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", value.lower())}
