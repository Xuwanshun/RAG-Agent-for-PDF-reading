from __future__ import annotations

import os
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    raw_documents_dir: Path = field(default_factory=lambda: Path("Data/Raw"))
    processed_documents_dir: Path = field(default_factory=lambda: Path("Data/Processed"))
    vectorstore_dir: Path = field(default_factory=lambda: Path("Data/Embedded"))
    preprocess_chunk_size: int = 1800
    preprocess_chunk_overlap: int = 200
    min_pdf_text_chars: int = 200
    embedding_model: str = "text-embedding-3-small"
    qa_model: str = "gpt-4.1-mini"
    default_top_k: int = 4
    prefer_chroma: bool = False
    agent_llm_provider: str = "openai"
    agent_llm_model: str = "gpt-4.1-mini"
    agent_llm_api_key: str | None = None
    agent_llm_base_url: str | None = None
    vlm_provider: str = "openai"
    vlm_model: str = "gpt-4.1-mini"
    vlm_api_key: str | None = None
    vlm_base_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_documents_dir", Path(self.raw_documents_dir))
        object.__setattr__(self, "processed_documents_dir", Path(self.processed_documents_dir))
        object.__setattr__(self, "vectorstore_dir", Path(self.vectorstore_dir))
        object.__setattr__(self, "agent_llm_api_key", self.agent_llm_api_key or resolve_env_value("OPENAI_API_KEY"))
        object.__setattr__(self, "agent_llm_base_url", self.agent_llm_base_url or resolve_env_value("OPENAI_BASE_URL"))
        object.__setattr__(self, "vlm_api_key", self.vlm_api_key or resolve_env_value("OPENAI_API_KEY"))
        object.__setattr__(self, "vlm_base_url", self.vlm_base_url or resolve_env_value("OPENAI_BASE_URL"))
        if not self.agent_llm_api_key:
            raise RuntimeError("OPENAI_API_KEY is required. Set it in the environment or in a local .env file.")
        self.raw_documents_dir.mkdir(parents=True, exist_ok=True)
        self.processed_documents_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


def resolve_env_value(key: str) -> str | None:
    return os.getenv(key) or _dotenv_values().get(key)


@lru_cache(maxsize=1)
def _dotenv_values() -> dict[str, str]:
    try:
        from dotenv import dotenv_values  # type: ignore
    except Exception:
        return _read_simple_dotenv(Path(".env"))
    values = dotenv_values(".env")
    return {key: value for key, value in values.items() if key and value}


def _read_simple_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values
