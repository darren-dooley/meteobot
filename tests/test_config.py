"""Settings loading: env precedence, defaults, and the fail-fast key check."""

from __future__ import annotations

from pathlib import Path

import pytest

from meteobot.config import (
    DEFAULT_CODE_EXEC_ENABLED,
    DEFAULT_CODE_EXEC_TIMEOUT_SECONDS,
    DEFAULT_EXECUTOR,
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_LANGSMITH_ENDPOINT,
    DEFAULT_LANGSMITH_PROJECT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL,
    DEFAULT_REQUEST_LIMIT,
    DEFAULT_SANDBOX_IMAGE,
    DEFAULT_TOOL_SEARCH_ENABLED,
    DEFAULT_TOTAL_TOKENS_LIMIT,
    ConfigError,
    MissingAPIKeyError,
    load_settings,
)

MANAGED_VARS = (
    "OPENAI_API_KEY",
    "METEOBOT_MODEL",
    "METEOBOT_HTTP_TIMEOUT",
    "METEOBOT_REQUEST_LIMIT",
    "METEOBOT_TOTAL_TOKENS_LIMIT",
    "METEOBOT_LOG_LEVEL",
    "METEOBOT_TOOL_SEARCH",
    "METEOBOT_CODE_EXEC",
    "METEOBOT_EXECUTOR",
    "METEOBOT_SANDBOX_IMAGE",
    "METEOBOT_CODE_EXEC_TIMEOUT",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    "LANGSMITH_WORKSPACE_ID",
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in an empty cwd (no repo .env) with a scrubbed environment."""
    monkeypatch.chdir(tmp_path)
    for var in MANAGED_VARS:
        monkeypatch.delenv(var, raising=False)


def test_missing_api_key_raises_with_actionable_message() -> None:
    with pytest.raises(MissingAPIKeyError) as excinfo:
        load_settings()
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = load_settings()
    assert settings.openai_api_key == "sk-test"
    assert settings.model == DEFAULT_MODEL
    assert settings.http_timeout_seconds == DEFAULT_HTTP_TIMEOUT_SECONDS
    assert settings.request_limit == DEFAULT_REQUEST_LIMIT
    assert settings.total_tokens_limit == DEFAULT_TOTAL_TOKENS_LIMIT
    assert settings.log_level == DEFAULT_LOG_LEVEL
    # Tracing is off by default and fails open: no key required for a normal
    # local session.
    assert settings.tracing_enabled is False
    assert settings.langsmith_api_key is None
    assert settings.langsmith_endpoint == DEFAULT_LANGSMITH_ENDPOINT
    assert settings.langsmith_project == DEFAULT_LANGSMITH_PROJECT
    assert settings.langsmith_workspace_id is None
    # Advanced tool-use features default on, Docker executor.
    assert settings.tool_search_enabled is DEFAULT_TOOL_SEARCH_ENABLED
    assert settings.code_exec_enabled is DEFAULT_CODE_EXEC_ENABLED
    assert settings.executor == DEFAULT_EXECUTOR
    assert settings.sandbox_image == DEFAULT_SANDBOX_IMAGE
    assert settings.code_exec_timeout_seconds == DEFAULT_CODE_EXEC_TIMEOUT_SECONDS


def test_advanced_tool_use_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("METEOBOT_TOOL_SEARCH", "off")
    monkeypatch.setenv("METEOBOT_CODE_EXEC", "off")
    monkeypatch.setenv("METEOBOT_EXECUTOR", "inprocess")
    monkeypatch.setenv("METEOBOT_SANDBOX_IMAGE", "python:3.13-slim")
    monkeypatch.setenv("METEOBOT_CODE_EXEC_TIMEOUT", "12.5")
    settings = load_settings()
    assert settings.tool_search_enabled is False
    assert settings.code_exec_enabled is False
    assert settings.executor == "inprocess"
    assert settings.sandbox_image == "python:3.13-slim"
    assert settings.code_exec_timeout_seconds == 12.5


def test_invalid_executor_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("METEOBOT_EXECUTOR", "firecracker")
    with pytest.raises(ConfigError) as excinfo:
        load_settings()
    assert "METEOBOT_EXECUTOR" in str(excinfo.value)


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("METEOBOT_MODEL", "gpt-4.1")
    monkeypatch.setenv("METEOBOT_HTTP_TIMEOUT", "2.5")
    monkeypatch.setenv("METEOBOT_REQUEST_LIMIT", "3")
    monkeypatch.setenv("METEOBOT_TOTAL_TOKENS_LIMIT", "5000")
    monkeypatch.setenv("METEOBOT_LOG_LEVEL", "info")
    settings = load_settings()
    assert settings.model == "gpt-4.1"
    assert settings.http_timeout_seconds == 2.5
    assert settings.request_limit == 3
    assert settings.total_tokens_limit == 5000
    # Level names are normalized to upper case, so `logging.basicConfig` accepts
    # them and the value is case-insensitive to the user.
    assert settings.log_level == "INFO"


def test_langsmith_and_tracing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-secret")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://example.test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "my-project")
    monkeypatch.setenv("LANGSMITH_WORKSPACE_ID", "ws-123")
    settings = load_settings()
    assert settings.tracing_enabled is True
    assert settings.langsmith_api_key == "ls-secret"
    assert settings.langsmith_endpoint == "https://example.test"
    assert settings.langsmith_project == "my-project"
    assert settings.langsmith_workspace_id == "ws-123"


@pytest.mark.parametrize(
    "value, expected",
    [("1", True), ("true", True), ("YES", True), ("on", True),
     ("0", False), ("false", False), ("no", False), ("off", False)],
)
def test_tracing_toggle_parses_common_boolean_spellings(
    value: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGSMITH_TRACING", value)
    assert load_settings().tracing_enabled is expected


def test_empty_langsmith_key_is_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LANGSMITH_API_KEY", "   ")
    assert load_settings().langsmith_api_key is None


def test_dotenv_file_is_loaded(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-file\n")
    settings = load_settings()
    assert settings.openai_api_key == "sk-from-file"


def test_dotenv_does_not_override_exported_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    settings = load_settings()
    assert settings.openai_api_key == "sk-from-shell"


def test_missing_api_key_is_a_config_error() -> None:
    # MissingAPIKeyError is a ConfigError, so the entry point's single
    # ConfigError catch covers both a missing key and a bad tunable.
    assert issubclass(MissingAPIKeyError, ConfigError)


@pytest.mark.parametrize(
    "var, value",
    [
        ("METEOBOT_HTTP_TIMEOUT", "not-a-number"),
        ("METEOBOT_HTTP_TIMEOUT", "0"),
        ("METEOBOT_HTTP_TIMEOUT", "-1"),
        ("METEOBOT_REQUEST_LIMIT", "abc"),
        ("METEOBOT_REQUEST_LIMIT", "0"),
        ("METEOBOT_REQUEST_LIMIT", "-2"),
        ("METEOBOT_TOTAL_TOKENS_LIMIT", "lots"),
        ("METEOBOT_TOTAL_TOKENS_LIMIT", "0"),
        ("METEOBOT_LOG_LEVEL", "verbose"),
        ("METEOBOT_LOG_LEVEL", ""),
        ("LANGSMITH_TRACING", "maybe"),
    ],
)
def test_invalid_tunable_fails_fast_with_a_named_message(
    var: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A malformed or nonsensical tunable fails fast with the offending
    # variable named, not a raw parse traceback mid-startup.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv(var, value)
    with pytest.raises(ConfigError) as excinfo:
        load_settings()
    assert var in str(excinfo.value)


def test_settings_is_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = load_settings()
    with pytest.raises(Exception):
        settings.model = "other"  # type: ignore[misc]
