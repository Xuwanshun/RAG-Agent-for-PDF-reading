"""
OpenAI API client wrappers.

WHY the error handling was added
---------------------------------
The original code made OpenAI API calls with no try/except. Any network
hiccup, rate-limit response, or invalid API key would crash the entire
pipeline with a raw openai exception — no helpful message, no context.

Now every OpenAI call is wrapped to catch the most common failure modes
and re-raise them as RuntimeError with a clear, actionable message.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from config import Settings

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)


def configure_openai_env(settings: Settings) -> None:
    """
    Export OpenAI credentials into os.environ for SDKs that read them there.

    Our own client is constructed with an explicit api_key, so this is only
    needed for third-party libraries that look the key up themselves — RAGAS
    being the case in hand, which failed with "Missing credentials" while a valid
    key sat in Settings.

    Same root cause as configure_langsmith_env: pydantic-settings loads .env into
    an object, not into the process environment. Any SDK added later will hit it,
    so call this before handing work to one.

    Uses setdefault so an explicitly exported variable always wins.
    """
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.openai_base_url:
        os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)


def configure_langsmith_env(settings: Settings) -> None:
    """
    Export LangSmith configuration into os.environ.

    pydantic-settings loads .env into Settings, but the langsmith SDK and
    LangGraph's tracing callbacks read os.environ directly — nothing bridges the
    two. Without this, a perfectly valid key sitting in .env never reaches the
    SDK and the service answers 401, which looks indistinguishable from a
    revoked key.

    Uses setdefault so an explicitly exported variable always beats .env.
    """
    if not getattr(settings, "langsmith_tracing", False):
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    api_key = getattr(settings, "langsmith_api_key", None)
    if api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    else:
        logger.warning("LANGSMITH_TRACING is on but no LANGSMITH_API_KEY is set; traces will not be recorded")


def maybe_trace_client(client: Any, settings: Settings) -> Any:
    """
    Wrap an OpenAI client for LangSmith tracing when it is switched on.

    Returns the client untouched when tracing is disabled, when the SDK is
    absent, or when wrapping fails for any reason: observability must never be
    the reason a query stops working. Failures are logged, not raised.
    """
    if not getattr(settings, "langsmith_tracing", False):
        return client
    try:
        from langsmith.wrappers import wrap_openai

        configure_langsmith_env(settings)
        return wrap_openai(client)
    except Exception as exc:  # noqa: BLE001 - tracing is strictly best-effort
        logger.warning("LangSmith tracing unavailable, continuing untraced: %s", exc)
        return client


class OpenAIJSONModelClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None,
        settings: Settings | None = None,
    ) -> None:
        self.model = model
        client = OpenAI(api_key=api_key, base_url=base_url)
        # Traced only when explicitly enabled; a no-op otherwise.
        self.client = maybe_trace_client(client, settings) if settings is not None else client

    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[ModelT]) -> ModelT:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except RateLimitError as exc:
            raise RuntimeError(
                "OpenAI rate limit reached. Wait a moment and retry, or reduce request frequency."
            ) from exc
        except APITimeoutError as exc:
            raise RuntimeError("OpenAI API request timed out. Check your network connection.") from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"Could not connect to OpenAI API: {exc}") from exc
        except APIStatusError as exc:
            raise RuntimeError(f"OpenAI API returned an error (HTTP {exc.status_code}): {exc.message}") from exc

        content = str((response.choices[0].message.content or "").strip())
        return _validate_response_model(response_model, _extract_json_from_text(content))

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except RateLimitError as exc:
            raise RuntimeError(
                "OpenAI rate limit reached. Wait a moment and retry, or reduce request frequency."
            ) from exc
        except APITimeoutError as exc:
            raise RuntimeError("OpenAI API request timed out. Check your network connection.") from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"Could not connect to OpenAI API: {exc}") from exc
        except APIStatusError as exc:
            raise RuntimeError(f"OpenAI API returned an error (HTTP {exc.status_code}): {exc.message}") from exc

        return str(response.choices[0].message.content or "")


def build_openai_client(settings: Settings) -> OpenAIJSONModelClient:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file or set it as an environment variable.")
    return OpenAIJSONModelClient(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        settings=settings,
    )


def request_openai_embeddings(*, model: str, texts: list[str], api_key: str, base_url: str | None) -> list[list[float]]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.embeddings.create(model=model, input=texts)
    except RateLimitError as exc:
        raise RuntimeError(
            "OpenAI rate limit reached while generating embeddings. "
            "Wait a moment and retry, or reduce the number of texts per batch."
        ) from exc
    except APITimeoutError as exc:
        raise RuntimeError("OpenAI embedding request timed out. Check your network connection.") from exc
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to OpenAI API for embeddings: {exc}") from exc
    except APIStatusError as exc:
        raise RuntimeError(
            f"OpenAI API returned an error (HTTP {exc.status_code}) during embedding: {exc.message}"
        ) from exc

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
        raise RuntimeError("OpenAI model did not return valid JSON.") from None


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
