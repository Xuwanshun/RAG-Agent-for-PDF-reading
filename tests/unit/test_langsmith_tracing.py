"""
Tests for optional LangSmith tracing of OpenAI clients.

Tracing sends prompts and document content to a third-party service, so it must
be opt-in and must never be the reason a request fails. A missing SDK, a bad key
or an unreachable service has to degrade to an untraced client rather than break
answering.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from document_process.clients import configure_langsmith_env, configure_openai_env, maybe_trace_client


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch):
    """
    Give each test its own os.environ.

    These tests exercise functions whose whole job is to WRITE to the
    environment, and monkeypatch.delenv registers no restore for a variable that
    was already absent — so a value set during the test outlived it and leaked
    into unrelated tests (test_vlm.py started failing only when run together).
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))


def test_client_is_returned_unchanged_when_tracing_is_disabled(tmp_settings):
    client = MagicMock(name="raw-client")

    assert maybe_trace_client(client, tmp_settings) is client


def test_client_is_wrapped_when_tracing_is_enabled(tmp_settings):
    settings = tmp_settings(langsmith_tracing=True)
    client = MagicMock(name="raw-client")
    wrapped = MagicMock(name="wrapped-client")

    with patch("langsmith.wrappers.wrap_openai", return_value=wrapped) as mock_wrap:
        result = maybe_trace_client(client, settings)

    mock_wrap.assert_called_once_with(client)
    assert result is wrapped


def test_a_broken_tracing_setup_falls_back_to_the_raw_client(tmp_settings):
    """Observability must never take down the thing it observes."""
    settings = tmp_settings(langsmith_tracing=True)
    client = MagicMock(name="raw-client")

    with patch("langsmith.wrappers.wrap_openai", side_effect=RuntimeError("no api key")):
        result = maybe_trace_client(client, settings)

    assert result is client


def test_a_missing_langsmith_sdk_falls_back_to_the_raw_client(tmp_settings):
    settings = tmp_settings(langsmith_tracing=True)
    client = MagicMock(name="raw-client")

    with patch.dict("sys.modules", {"langsmith.wrappers": None}):
        result = maybe_trace_client(client, settings)

    assert result is client


# ── bridging .env into the SDK's environment ──────────────────────────────────
# pydantic-settings reads .env into Settings; the langsmith SDK reads os.environ.
# Nothing connects the two on its own, so a key present in .env never reaches the
# SDK and the service answers 401 as though the key were invalid.


def test_settings_are_exported_to_the_environment_the_sdk_reads(tmp_settings, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    settings = tmp_settings(
        langsmith_tracing=True,
        langsmith_api_key="lsv2_pt_testkey",
        langsmith_project="pagesense",
    )

    configure_langsmith_env(settings)

    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_pt_testkey"
    assert os.environ["LANGSMITH_PROJECT"] == "pagesense"
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_nothing_is_exported_when_tracing_is_disabled(tmp_settings, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    settings = tmp_settings(langsmith_tracing=False, langsmith_api_key="lsv2_pt_testkey")

    configure_langsmith_env(settings)

    assert "LANGSMITH_API_KEY" not in os.environ


def test_an_existing_environment_variable_wins(tmp_settings, monkeypatch):
    """An explicitly exported key must beat whatever .env happens to hold."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_from_shell")
    settings = tmp_settings(langsmith_tracing=True, langsmith_api_key="lsv2_pt_from_dotenv")

    configure_langsmith_env(settings)

    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_pt_from_shell"


def test_tracing_without_a_key_does_not_export_a_blank_key(tmp_settings, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    settings = tmp_settings(langsmith_tracing=True, langsmith_api_key=None)

    configure_langsmith_env(settings)

    assert "LANGSMITH_API_KEY" not in os.environ


# ── the same bridge, for any SDK that reads os.environ ────────────────────────
# RAGAS reads OPENAI_API_KEY from the environment and failed with "Missing
# credentials" while a valid key sat in Settings. Identical cause to the
# LangSmith 401: pydantic-settings loads .env into an object, not into the
# process environment. Any third-party SDK we add will hit this.


def test_openai_key_is_exported_for_sdks_that_read_the_environment(tmp_settings, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    configure_openai_env(tmp_settings)

    assert os.environ["OPENAI_API_KEY"] == tmp_settings.openai_api_key


def test_an_existing_openai_key_in_the_environment_wins(tmp_settings, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")

    configure_openai_env(tmp_settings)

    assert os.environ["OPENAI_API_KEY"] == "sk-from-shell"


def test_no_key_exports_nothing_rather_than_a_blank(tmp_settings, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    configure_openai_env(tmp_settings(openai_api_key=None))

    assert "OPENAI_API_KEY" not in os.environ


def test_base_url_is_exported_when_set(tmp_settings, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    configure_openai_env(tmp_settings(openai_base_url="https://proxy.example/v1"))

    assert os.environ["OPENAI_BASE_URL"] == "https://proxy.example/v1"
