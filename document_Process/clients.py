from __future__ import annotations

import json
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from config import Settings


ModelT = TypeVar("ModelT", bound=BaseModel)


class OpenAIJSONModelClient:
    def __init__(self, *, model: str, api_key: str, base_url: str | None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[ModelT]) -> ModelT:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = str((response.choices[0].message.content or "").strip())
        return _validate_response_model(response_model, _extract_json_from_text(content))

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return str(response.choices[0].message.content or "")


def build_openai_client(settings: Settings) -> OpenAIJSONModelClient:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI-backed model calls.")
    return OpenAIJSONModelClient(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def request_openai_embeddings(*, model: str, texts: list[str], api_key: str, base_url: str | None) -> list[list[float]]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _extract_json_from_text(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise RuntimeError("OpenAI model did not return valid JSON.")


def _validate_response_model(response_model: type[ModelT], payload: dict[str, Any]) -> ModelT:
    try:
        return response_model.model_validate(payload)
    except Exception:
        normalized = _normalize_model_payload(payload)
        return response_model.model_validate(normalized)


def _normalize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    relevant_region_ids = normalized.get("relevant_region_ids")
    if relevant_region_ids is None:
        normalized["relevant_region_ids"] = []
    elif isinstance(relevant_region_ids, str):
        normalized["relevant_region_ids"] = [relevant_region_ids]
    return normalized
