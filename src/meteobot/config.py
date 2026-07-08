"""Configuration: a frozen Settings object read once at startup and injected everywhere.

Values come from the environment; a `.env` file is loaded via python-dotenv
without overriding variables already exported in the shell.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_TOOL_ROUND_CAP = 5
DEFAULT_LOG_LEVEL = "WARNING"


class ConfigError(Exception):
    """Raised at startup when configuration is missing or invalid.

    Carries a clear, actionable message so the app can fail fast with one line
    before any network call, rather than dying on a raw parse traceback.
    """


class MissingAPIKeyError(ConfigError):
    """Raised at startup when no OpenAI API key can be found."""


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from None
    if value <= 0:
        raise ConfigError(f"{name} must be greater than 0, got {value}.")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from None
    if value < 1:
        raise ConfigError(f"{name} must be at least 1, got {value}.")
    return value


def _log_level(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    level = raw.strip().upper()
    valid = logging.getLevelNamesMapping()
    if level not in valid:
        allowed = ", ".join(sorted(valid))
        raise ConfigError(f"{name} must be one of {allowed}, got {raw!r}.")
    return level


@dataclass(frozen=True)
class Settings:
    """All tunables, read once at startup and injected. Never a module-level global."""

    openai_api_key: str
    model: str
    http_timeout_seconds: float
    tool_round_cap: int
    log_level: str


def load_settings() -> Settings:
    """Read settings from the environment, loading `.env` first (non-overriding).

    Raises MissingAPIKeyError if no API key is present, so the app can fail fast
    with one clear message before any network call.
    """
    # Search for .env from the current working directory (where the user runs
    # the app), not from this source file. Never overrides exported variables.
    load_dotenv(find_dotenv(usecwd=True))

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set.\n"
            "Either copy .env.example to .env and paste your key in, or run:\n"
            "  OPENAI_API_KEY=sk-... uv run meteobot"
        )

    return Settings(
        openai_api_key=api_key,
        model=os.environ.get("METEOBOT_MODEL", DEFAULT_MODEL),
        http_timeout_seconds=_positive_float(
            "METEOBOT_HTTP_TIMEOUT", DEFAULT_HTTP_TIMEOUT_SECONDS
        ),
        tool_round_cap=_positive_int(
            "METEOBOT_TOOL_ROUND_CAP", DEFAULT_TOOL_ROUND_CAP
        ),
        log_level=_log_level("METEOBOT_LOG_LEVEL", DEFAULT_LOG_LEVEL),
    )
