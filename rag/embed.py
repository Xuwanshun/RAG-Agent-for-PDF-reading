from __future__ import annotations

from typing import Protocol

from config import Settings
from document_Process.clients import request_openai_embeddings


class EmbeddingBackend(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingBackend:
    def __init__(self, model_name: str, *, api_key: str, base_url: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return request_openai_embeddings(
            model=self.model_name,
            texts=texts,
            api_key=self.api_key,
            base_url=self.base_url,
        )


def build_embedding_backend(settings: Settings) -> EmbeddingBackend:
    return OpenAIEmbeddingBackend(
        settings.embedding_model,
        api_key=settings.agent_llm_api_key or "",
        base_url=settings.agent_llm_base_url,
    )
