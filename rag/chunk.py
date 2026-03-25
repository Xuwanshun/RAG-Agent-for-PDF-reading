from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from document_Process.models import ProcessedChunk


@dataclass(frozen=True)
class ChunkRecord:
    """A vector-store ready record derived from a processed chunk."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


def chunk_record_from_processed_chunk(
    chunk: ProcessedChunk,
    *,
    document_id: str | None = None,
    source_filename: str | None = None,
) -> ChunkRecord:
    metadata = dict(chunk.metadata)
    metadata.setdefault("document_id", document_id)
    metadata.setdefault("source_filename", source_filename)
    metadata.setdefault("page_number", chunk.page_number)
    metadata.setdefault("chunk_id", chunk.chunk_id)
    metadata.setdefault("ordered_block_ids", chunk.ordered_block_ids)
    metadata.setdefault("item_ids", chunk.item_ids)
    metadata.setdefault("source_region_ids", chunk.source_region_ids)
    metadata.setdefault("region_types", chunk.region_types)
    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        text=chunk.page_content or chunk.text,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def chunk_records_from_processed_chunks(
    chunks: list[ProcessedChunk],
    *,
    document_id: str | None = None,
    source_filename: str | None = None,
) -> list[ChunkRecord]:
    return [
        chunk_record_from_processed_chunk(
            chunk,
            document_id=document_id,
            source_filename=source_filename,
        )
        for chunk in chunks
        if (chunk.page_content or chunk.text).strip()
    ]
