"""
Build a labelled question set from an indexed corpus.

Gold labels come from generation rather than from labelling after the fact: a
question written *from* chunk N has chunk N as its answer by construction, so
the label is free and cannot drift from the corpus.

Two honest limits, both of which is why every record keeps ``generated_from``
and the output is meant to be reviewed before use:

  * The label is a floor, not a ceiling. Another chunk may also answer the
    question, which makes recall@k pessimistic rather than wrong.
  * A model writing questions from a passage tends to write questions that
    passage answers well. Human review is what keeps the set honest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config import Settings
from document_process.clients import build_openai_client

logger = logging.getLogger(__name__)

# Kept small and meaningful: each maps to a different retrieval path in the
# pipeline, so per-category scores say something actionable.
QUESTION_TYPES = {"factual", "table_lookup", "multi_hop"}
_DEFAULT_TYPE = "factual"

GENERATION_PROMPT = """You are building an evaluation set for a document retrieval system.

Write {n} questions that this passage — and only this passage — can answer.

Rules:
- Each question must be answerable purely from the passage below. Do not use outside knowledge.
- Ask for specific facts: figures, percentages, named segments, dates.
- Do not refer to "the passage" or "the table" in the question. Ask as a user would.
- The answer must be short and quoted from the passage.
- Label each question:
    factual       answer is stated in prose
    table_lookup  answer must be read out of tabular data
    multi_hop     answer requires combining two or more facts in the passage

Passage:
{passage}

Output strict JSON:
{{"questions": [{{"question": "...", "answer": "...", "type": "factual"}}]}}"""


@dataclass(frozen=True)
class QuestionRecord:
    question: str
    ground_truth: str
    # Gold chunks for retrieval scoring. Seeded with the chunk the question was
    # generated from; a reviewer may add more.
    relevant_chunk_ids: list[str]
    question_type: str
    source_page: int | None
    # Audit trail: which chunk produced this question.
    generated_from: str


def parse_generated_questions(payload: Any, *, chunk_id: str, page: int | None) -> list[QuestionRecord]:
    """Turn one generation response into records, discarding unusable entries."""
    if not isinstance(payload, dict):
        return []
    items = payload.get("questions")
    if not isinstance(items, list):
        return []

    records: list[QuestionRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        # A record without ground truth cannot score answer correctness, and one
        # without a question is not a record at all.
        if not question or not answer:
            continue
        question_type = str(item.get("type") or "").strip().lower()
        records.append(
            QuestionRecord(
                question=question,
                ground_truth=answer,
                relevant_chunk_ids=[chunk_id],
                question_type=question_type if question_type in QUESTION_TYPES else _DEFAULT_TYPE,
                source_page=page,
                generated_from=chunk_id,
            )
        )
    return records


def dedupe_questions(records: list[QuestionRecord]) -> list[QuestionRecord]:
    """
    Drop repeated questions, keeping the first occurrence.

    Chunks overlap by design, so neighbouring chunks routinely yield the same
    question. Left in, duplicates would silently weight those questions twice in
    every aggregate score.
    """
    seen: set[str] = set()
    unique: list[QuestionRecord] = []
    for record in records:
        key = " ".join(record.question.lower().split())
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def to_jsonl(records: list[QuestionRecord]) -> str:
    return "".join(json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in records)


def load_chunks(processed_documents_dir: Path) -> list[dict[str, Any]]:
    """Read every indexed chunk across all processed documents."""
    chunks: list[dict[str, Any]] = []
    if not processed_documents_dir.exists():
        return chunks
    for doc_dir in sorted(p for p in processed_documents_dir.iterdir() if p.is_dir()):
        chunks_path = doc_dir / "chunks.json"
        if chunks_path.exists():
            chunks.extend(json.loads(chunks_path.read_text(encoding="utf-8")))
    return chunks


def generate_dataset(
    settings: Settings,
    *,
    questions_per_chunk: int = 2,
    min_chunk_chars: int = 200,
) -> list[QuestionRecord]:
    """
    Generate a labelled question set covering every substantial chunk.

    Short chunks (headers, page furniture) are skipped: they carry no fact worth
    asking about and would produce questions whose gold label is meaningless.
    """
    client = build_openai_client(settings)
    records: list[QuestionRecord] = []

    for chunk in load_chunks(settings.processed_documents_dir):
        text = chunk.get("text") or ""
        if len(text.strip()) < min_chunk_chars:
            continue
        chunk_id = str(chunk.get("chunk_id"))
        try:
            raw = client.generate_text(
                system_prompt="You write evaluation questions. Respond with valid JSON only.",
                user_prompt=GENERATION_PROMPT.format(n=questions_per_chunk, passage=text),
            )
            payload = json.loads(_strip_fences(raw))
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the run
            logger.warning("Question generation failed for %s: %s", chunk_id, exc)
            continue
        records.extend(parse_generated_questions(payload, chunk_id=chunk_id, page=chunk.get("page_number")))

    return dedupe_questions(records)


def _strip_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()
