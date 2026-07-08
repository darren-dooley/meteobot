"""Settings loading: env precedence, defaults, and the fail-fast key check."""

from __future__ import annotations

from pathlib import Path

import pytest

from meteobot.config import (
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL,
    DEFAULT_TOOL_ROUND_CAP,
    ConfigError,
    MissingAPIKeyError,
    load_settings,
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in an empty cwd (no repo .env) with a scrubbed environment."""
    monkeypatch.chdir(tmp_path)
    for var in (
        "OPENAI_API_KEY",
        "METEOBOT_MODEL",
        "METEOBOT_HTTP_TIMEOUT",
        "METEOBOT_TOOL_ROUND_CAP",
        "METEOBOT_LOG_LEVEL",
    ):
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
    assert settings.tool_round_cap == DEFAULT_TOOL_ROUND_CAP
    assert settings.log_level == DEFAULT_LOG_LEVEL


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("METEOBOT_MODEL", "gpt-4.1")
    monkeypatch.setenv("METEOBOT_HTTP_TIMEOUT", "2.5")
    monkeypatch.setenv("METEOBOT_TOOL_ROUND_CAP", "3")
    monkeypatch.setenv("METEOBOT_LOG_LEVEL", "info")
    settings = load_settings()
    assert settings.model == "gpt-4.1"
    assert settings.http_timeout_seconds == 2.5
    assert settings.tool_round_cap == 3
    # Level names are normalized to upper case, so `logging.basicConfig` accepts
    # them and the value is case-insensitive to the user.
    assert settings.log_level == "INFO"


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
        ("METEOBOT_TOOL_ROUND_CAP", "abc"),
        ("METEOBOT_TOOL_ROUND_CAP", "0"),
        ("METEOBOT_TOOL_ROUND_CAP", "-2"),
        ("METEOBOT_LOG_LEVEL", "verbose"),
        ("METEOBOT_LOG_LEVEL", ""),
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
