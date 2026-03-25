from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from config import Settings
from document_Process.clients import build_agent_llm
from document_Process.models import ProcessedChunk, ProcessedDocument
from rag.chunk import ChunkRecord, chunk_records_from_processed_chunks
from rag.embed import EmbeddingBackend, build_embedding_backend


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class QAResponse:
    question: str
    answer: str
    sources: list[dict[str, Any]]


class VectorStore(Protocol):
    def upsert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        ...

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        ...


class JsonVectorStore:
    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_rows(self) -> list[dict[str, Any]]:
        if not self.store_path.exists():
            return []
        payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        return payload.get("rows", [])

    def _save_rows(self, rows: list[dict[str, Any]]) -> None:
        self.store_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def upsert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        existing = {row["chunk_id"]: row for row in self._load_rows()}
        for chunk, embedding in zip(chunks, embeddings):
            existing[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": embedding,
            }
        self._save_rows(list(existing.values()))

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        scored: list[RetrievedChunk] = []
        for row in self._load_rows():
            score = _cosine_similarity(embedding, row.get("embedding", []))
            scored.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    text=row.get("text", ""),
                    metadata=row.get("metadata", {}),
                    score=score,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "rag_agent_pdf") -> None:
        try:
            import chromadb  # type: ignore
        except Exception as exc:
            raise RuntimeError("chromadb is not installed.") from exc

        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_or_create_collection(name=collection_name)

    def upsert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=embeddings,
        )

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        response = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text or "",
                metadata=metadata or {},
                score=1.0 - float(distance),
            )
            for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
        ]


class DocumentRetriever:
    def __init__(
        self,
        settings: Settings,
        *,
        embedding_backend: EmbeddingBackend | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_backend = embedding_backend or build_embedding_backend(settings)
        self.vector_store = vector_store or build_vector_store(settings)

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        embeddings = self.embedding_backend.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.upsert(chunks, embeddings)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query_embedding = self.embedding_backend.embed_texts([question])[0]
        return self.vector_store.query(query_embedding, top_k or self.settings.default_top_k)

    def index_processed_chunks(
        self,
        chunks: list[ProcessedChunk],
        *,
        document_id: str | None = None,
        source_filename: str | None = None,
    ) -> int:
        records = chunk_records_from_processed_chunks(
            chunks,
            document_id=document_id,
            source_filename=source_filename,
        )
        if not records:
            return 0
        self.upsert_chunks(records)
        return len(records)

    def answer_question(self, question: str, *, top_k: int | None = None) -> QAResponse:
        retrieved = self.retrieve(question, top_k=top_k)
        return answer_question(question, retrieved, settings=self.settings)


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.prefer_chroma:
        try:
            return ChromaVectorStore(settings.vectorstore_dir)
        except RuntimeError:
            pass
    return JsonVectorStore(settings.vectorstore_dir / "store.json")


def load_processed_document_bundle(document_dir: Path) -> tuple[ProcessedDocument | None, list[ProcessedChunk]]:
    document_payload = _load_json(_artifact_path(document_dir, "document.json"))
    chunks_payload = _load_json(_artifact_path(document_dir, "chunks.json")) or []
    document = ProcessedDocument.model_validate(document_payload) if isinstance(document_payload, dict) else None
    chunks = [ProcessedChunk.model_validate(item) for item in chunks_payload if isinstance(item, dict)]
    return document, chunks


def index_processed_document(
    document_id_or_path: str | Path,
    *,
    settings: Settings | None = None,
    retriever: DocumentRetriever | None = None,
) -> int:
    resolved_settings = settings or Settings()
    active_retriever = retriever or DocumentRetriever(resolved_settings)
    document_dir = _resolve_processed_document_dir(document_id_or_path, resolved_settings)
    document, chunks = load_processed_document_bundle(document_dir)
    return active_retriever.index_processed_chunks(
        chunks,
        document_id=document.document_id if document else document_dir.name,
        source_filename=document.source_filename if document else None,
    )


def index_all_processed_documents(
    *,
    settings: Settings | None = None,
    retriever: DocumentRetriever | None = None,
) -> dict[str, int]:
    resolved_settings = settings or Settings()
    active_retriever = retriever or DocumentRetriever(resolved_settings)
    indexed: dict[str, int] = {}
    for document_dir in sorted(path for path in resolved_settings.processed_documents_dir.iterdir() if path.is_dir()):
        document, chunks = load_processed_document_bundle(document_dir)
        if not chunks:
            continue
        document_id = document.document_id if document else document_dir.name
        indexed[document_id] = active_retriever.index_processed_chunks(
            chunks,
            document_id=document_id,
            source_filename=document.source_filename if document else None,
        )
    return indexed


def answer_corpus_question(
    question: str,
    *,
    settings: Settings | None = None,
    retriever: DocumentRetriever | None = None,
    top_k: int | None = None,
) -> QAResponse:
    resolved_settings = settings or Settings()
    active_retriever = retriever or DocumentRetriever(resolved_settings)
    return active_retriever.answer_question(question, top_k=top_k)


def answer_question(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    *,
    settings: Settings,
) -> QAResponse:
    if not retrieved_chunks:
        return QAResponse(
            question=question,
            answer="I cannot answer from the indexed documents because no relevant context was retrieved.",
            sources=[],
        )
    prompt = _build_qa_prompt(question, retrieved_chunks)
    answer = _generate_qa_answer(prompt=prompt, settings=settings)
    return QAResponse(
        question=question,
        answer=answer,
        sources=[_source_payload(chunk) for chunk in retrieved_chunks],
    )


def _build_qa_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context_sections = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        page_number = chunk.metadata.get("page_number")
        label = f"Source {index} | chunk={chunk.chunk_id}"
        if page_number is not None:
            label += f" | page={page_number}"
        context_sections.append(f"[{label}]\n{chunk.text.strip()}")
    context = "\n\n".join(section for section in context_sections if section.strip())
    return (
        "Answer the question using only the provided context.\n"
        "Do not invent facts.\n"
        "If the answer is not in the context, say: I cannot answer from the provided context.\n"
        "Keep the answer concise and cite chunk/page identifiers when available.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}"
    )


def _generate_qa_answer(*, prompt: str, settings: Settings) -> str:
    client = build_agent_llm(settings)
    return client.generate_text(
        system_prompt="You are a grounded QA assistant.",
        user_prompt=prompt,
    ).strip()


def _source_payload(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "page_number": chunk.metadata.get("page_number"),
        "document_id": chunk.metadata.get("document_id"),
        "source_filename": chunk.metadata.get("source_filename") or chunk.metadata.get("source_file"),
        "score": round(chunk.score, 4),
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size])) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right[:size])) or 1.0
    return dot / (left_norm * right_norm)


def _artifact_path(document_dir: Path, filename: str) -> Path:
    direct_path = document_dir / filename
    if direct_path.exists():
        return direct_path
    return document_dir / "structured" / filename


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_processed_document_dir(document_id_or_path: str | Path, settings: Settings) -> Path:
    candidate = Path(document_id_or_path)
    if candidate.exists():
        return candidate
    return settings.processed_documents_dir / candidate
