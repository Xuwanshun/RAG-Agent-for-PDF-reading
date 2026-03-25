from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib import error, request

from pydantic import BaseModel

from config import Settings


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ModelClientConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None


class JSONModelClient:
    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[ModelT]) -> ModelT:
        raise NotImplementedError

    def analyze_image(
        self,
        *,
        image_path: str | Path,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
    ) -> ModelT:
        raise NotImplementedError


class OpenAIJSONModelClient(JSONModelClient):
    def __init__(self, config: ModelClientConfig) -> None:
        self.config = config

    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[ModelT]) -> ModelT:
        payload = self._chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _validate_response_model(response_model, payload)

    def analyze_image(
        self,
        *,
        image_path: str | Path,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
    ) -> ModelT:
        payload = self._chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_image_to_base64(image_path)}"}},
                    ],
                },
            ]
        )
        return _validate_response_model(response_model, payload)

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return self._chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=None,
        )

    def _chat_json(self, *, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return _extract_json_from_text(self._chat(messages=messages, response_format={"type": "json_object"}))

    def _chat(self, *, messages: list[dict[str, Any]], response_format: dict[str, str] | None) -> str:
        body: dict[str, Any] = {
            "model": self.config.model,
            "temperature": 0,
            "messages": messages,
        }
        if response_format is not None:
            body["response_format"] = response_format
        payload = _openai_request(
            url=_chat_completions_url(self.config.base_url),
            api_key=self.config.api_key,
            payload=body,
        )
        return str((((payload.get("choices") or [{}])[0]).get("message") or {}).get("content") or "")


def build_json_model_client(*, provider: str, model: str, api_key: str | None, base_url: str | None) -> JSONModelClient:
    if provider != "openai":
        raise RuntimeError(f"Unsupported provider: {provider}")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI-backed model calls.")
    return OpenAIJSONModelClient(ModelClientConfig(provider=provider, model=model, api_key=api_key, base_url=base_url))


def build_agent_llm(settings: Settings) -> OpenAIJSONModelClient:
    return _build_openai_text_client(
        provider=settings.agent_llm_provider,
        model=settings.agent_llm_model,
        api_key=settings.agent_llm_api_key,
        base_url=settings.agent_llm_base_url,
    )


def build_agent_repair_client(settings: Settings) -> JSONModelClient:
    return _build_openai_text_client(
        provider=settings.agent_llm_provider,
        model=settings.agent_llm_model,
        api_key=settings.agent_llm_api_key,
        base_url=settings.agent_llm_base_url,
    )


def build_vlm_client(settings: Settings) -> JSONModelClient:
    return build_json_model_client(
        provider=settings.vlm_provider,
        model=settings.vlm_model,
        api_key=settings.vlm_api_key,
        base_url=settings.vlm_base_url,
    )


def request_openai_embeddings(*, model: str, texts: list[str], api_key: str, base_url: str | None) -> list[list[float]]:
    payload = _openai_request(
        url=_embeddings_url(base_url),
        api_key=api_key,
        payload={"model": model, "input": texts},
    )
    return [item["embedding"] for item in payload.get("data", [])]


def _build_openai_text_client(*, provider: str, model: str, api_key: str | None, base_url: str | None) -> OpenAIJSONModelClient:
    client = build_json_model_client(provider=provider, model=model, api_key=api_key, base_url=base_url)
    if not isinstance(client, OpenAIJSONModelClient):
        raise RuntimeError("Only OpenAI-backed execution is supported.")
    return client


def _openai_request(*, url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI request failed with status {exc.code}: {details or exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc


def _chat_completions_url(base_url: str | None) -> str:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if root.endswith("/v1"):
        return root + "/chat/completions"
    return root + "/v1/chat/completions"


def _embeddings_url(base_url: str | None) -> str:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    if root.endswith("/embeddings"):
        return root
    if root.endswith("/v1"):
        return root + "/embeddings"
    return root + "/v1/embeddings"


def _image_to_base64(image_path: str | Path) -> str:
    helper = _load_openai_vision_helper("image_to_base64")
    if helper is not None:
        return helper(image_path)
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _extract_json_from_text(content: str) -> dict[str, Any]:
    helper = _load_openai_vision_helper("extract_json_from_text")
    if helper is not None:
        return helper(content)
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
    for field_name in ("notes", "headers", "rows", "values", "axes", "data_points", "relevant_region_ids", "tool_calls", "tables_analyzed", "charts_analyzed"):
        value = normalized.get(field_name)
        if value is None:
            normalized[field_name] = []
            continue
        if isinstance(value, str):
            normalized[field_name] = [value]
    headers = normalized.get("headers")
    rows = normalized.get("rows")
    values = normalized.get("values")
    region_type = normalized.get("region_type")
    if isinstance(region_type, str):
        lowered = region_type.lower()
        if "table" in lowered:
            normalized["region_type"] = "table"
        elif "chart" in lowered or "graph" in lowered or "figure" in lowered:
            normalized["region_type"] = "chart"
    if isinstance(rows, list) and rows and all(isinstance(item, dict) for item in rows):
        header_order = list(headers) if isinstance(headers, list) and headers else list(rows[0].keys())
        normalized["headers"] = header_order
        normalized["rows"] = [[row.get(header) for header in header_order] for row in rows]
        normalized["values"] = rows
    elif isinstance(values, list) and values and all(isinstance(item, list) for item in values):
        header_order = list(headers) if isinstance(headers, list) and headers else []
        if header_order:
            normalized["values"] = [
                {header_order[index]: item[index] for index in range(min(len(header_order), len(item)))}
                for item in values
            ]
    return normalized


def _load_openai_vision_helper(name: str):
    try:
        from tools import openai_vision  # type: ignore
    except Exception:
        return None
    return getattr(openai_vision, name, None)
