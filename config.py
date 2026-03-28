from __future__ import annotations

import os
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    raw_documents_dir: Path = field(default_factory=lambda: Path("Data/Raw"))
    processed_documents_dir: Path = field(default_factory=lambda: Path("Data/Processed"))
    vectorstore_dir: Path = field(default_factory=lambda: Path("Data/Embedded"))
    preprocess_chunk_size: int = 1800
    preprocess_chunk_overlap: int = 200
    pdf_render_scale: float = 3.0
    embedding_model: str = "text-embedding-3-small"
    openai_model: str = "gpt-4.1-mini"
    default_top_k: int = 4
    prefer_chroma: bool = False
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_documents_dir", Path(self.raw_documents_dir))
        object.__setattr__(self, "processed_documents_dir", Path(self.processed_documents_dir))
        object.__setattr__(self, "vectorstore_dir", Path(self.vectorstore_dir))
        object.__setattr__(self, "openai_api_key", self.openai_api_key or resolve_env_value("OPENAI_API_KEY"))
        object.__setattr__(self, "openai_base_url", self.openai_base_url or resolve_env_value("OPENAI_BASE_URL"))
        self.raw_documents_dir.mkdir(parents=True, exist_ok=True)
        self.processed_documents_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


def resolve_env_value(key: str) -> str | None:
    return os.getenv(key) or _dotenv_values().get(key)


@lru_cache(maxsize=1)
def _dotenv_values() -> dict[str, str]:
    values = dotenv_values(".env")
    return {key: value for key, value in values.items() if key and value}
