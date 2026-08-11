"""
Tests for building a labelled question set from an indexed corpus.

Gold labels come from generation, not from labelling after the fact: a question
written *from* a specific chunk has that chunk as its answer by construction.
Retrofitting labels onto pre-written questions is where the previous question set
became unverifiable.

The labels are still a floor, not a ceiling — another chunk may also answer the
question — so every record carries the chunk it came from for human review.
"""

from __future__ import annotations

from eval.dataset import QuestionRecord, dedupe_questions, parse_generated_questions, to_jsonl


def _payload(*items):
    return {"questions": list(items)}


# ── parsing ───────────────────────────────────────────────────────────────────


def test_generated_question_is_labelled_with_the_chunk_it_came_from():
    payload = _payload({"question": "What was total revenue?", "answer": "$81.3 billion", "type": "factual"})

    records = parse_generated_questions(payload, chunk_id="doc:chunk:1", page=1)

    assert records[0].relevant_chunk_ids == ["doc:chunk:1"]
    assert records[0].generated_from == "doc:chunk:1"


def test_question_and_answer_are_carried_through():
    payload = _payload({"question": "What was total revenue?", "answer": "$81.3 billion", "type": "factual"})

    record = parse_generated_questions(payload, chunk_id="doc:chunk:1", page=1)[0]

    assert record.question == "What was total revenue?"
    assert record.ground_truth == "$81.3 billion"
    assert record.source_page == 1


def test_question_type_is_preserved_for_per_category_analysis():
    """Table questions exercise a different path than prose questions."""
    payload = _payload({"question": "What was diluted EPS?", "answer": "$5.16", "type": "table_lookup"})

    record = parse_generated_questions(payload, chunk_id="doc:chunk:2", page=1)[0]

    assert record.question_type == "table_lookup"


def test_unknown_question_type_falls_back_to_factual():
    payload = _payload({"question": "Q?", "answer": "A", "type": "wildly-invented-category"})

    assert parse_generated_questions(payload, chunk_id="c", page=1)[0].question_type == "factual"


def test_entries_missing_a_question_are_skipped():
    payload = _payload({"answer": "$81.3 billion"}, {"question": "Valid?", "answer": "Yes"})

    records = parse_generated_questions(payload, chunk_id="c", page=1)

    assert [r.question for r in records] == ["Valid?"]


def test_entries_missing_an_answer_are_skipped():
    """Without ground truth the record cannot score answer correctness."""
    payload = _payload({"question": "Unanswerable?"}, {"question": "Valid?", "answer": "Yes"})

    records = parse_generated_questions(payload, chunk_id="c", page=1)

    assert [r.question for r in records] == ["Valid?"]


def test_blank_strings_are_treated_as_missing():
    payload = _payload({"question": "   ", "answer": "A"}, {"question": "Valid?", "answer": "   "})

    assert parse_generated_questions(payload, chunk_id="c", page=1) == []


def test_malformed_payload_yields_no_records():
    assert parse_generated_questions({}, chunk_id="c", page=1) == []
    assert parse_generated_questions({"questions": "not a list"}, chunk_id="c", page=1) == []


# ── dedupe ────────────────────────────────────────────────────────────────────


def test_duplicate_questions_across_chunks_are_removed():
    """Adjacent chunks overlap, so the same question gets generated twice."""
    a = QuestionRecord("What was total revenue?", "$81.3B", ["doc:chunk:1"], "factual", 1, "doc:chunk:1")
    b = QuestionRecord("What was total revenue?", "$81.3B", ["doc:chunk:2"], "factual", 1, "doc:chunk:2")

    assert len(dedupe_questions([a, b])) == 1


def test_dedupe_ignores_case_and_surrounding_whitespace():
    a = QuestionRecord("What was total revenue?", "x", ["c1"], "factual", 1, "c1")
    b = QuestionRecord("  what was TOTAL revenue?  ", "x", ["c2"], "factual", 1, "c2")

    assert len(dedupe_questions([a, b])) == 1


def test_dedupe_keeps_genuinely_different_questions():
    a = QuestionRecord("What was total revenue?", "x", ["c1"], "factual", 1, "c1")
    b = QuestionRecord("What was operating income?", "y", ["c1"], "factual", 1, "c1")

    assert len(dedupe_questions([a, b])) == 2


def test_dedupe_preserves_order():
    a = QuestionRecord("First?", "x", ["c1"], "factual", 1, "c1")
    b = QuestionRecord("Second?", "y", ["c1"], "factual", 1, "c1")

    assert [r.question for r in dedupe_questions([b, a])] == ["Second?", "First?"]


# ── serialisation ─────────────────────────────────────────────────────────────


def test_jsonl_round_trips_every_field():
    import json

    record = QuestionRecord("Q?", "A", ["doc:chunk:1"], "table_lookup", 3, "doc:chunk:1")

    line = to_jsonl([record]).strip()
    restored = json.loads(line)

    assert restored["question"] == "Q?"
    assert restored["ground_truth"] == "A"
    assert restored["relevant_chunk_ids"] == ["doc:chunk:1"]
    assert restored["question_type"] == "table_lookup"
    assert restored["source_page"] == 3
    assert restored["generated_from"] == "doc:chunk:1"


def test_jsonl_writes_one_record_per_line():
    records = [
        QuestionRecord("A?", "1", ["c"], "factual", 1, "c"),
        QuestionRecord("B?", "2", ["c"], "factual", 1, "c"),
    ]

    assert len(to_jsonl(records).strip().splitlines()) == 2
