"""Configuration: a frozen Settings object read once at startup and injected everywhere.

Values come from the environment; a `.env` file is loaded via python-dotenv
without overriding variables already exported in the shell. Every tunable —
the model, HTTP timeout, the usage limits that bound a Turn, the log level, and
the LangSmith tracing knobs — lives here so operational config is discoverable
in one place and a bad value names itself before any network call.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
# The request limit is PydanticAI's framework-enforced replacement for v1's
# hand-counted tool-round cap. It counts every model request in a Turn,
# including the final answer, so it sits a little above the tool-round budget:
# a normal multi-city question is one tool round (all cities concurrently) plus
# one answer request. Six leaves room for a few re-plans and still cuts off a
# runaway loop before it burns credits.
DEFAULT_REQUEST_LIMIT = 6
# A generous per-Turn token ceiling: a second guardrail so a confused model
# can't run up cost even within the request budget. Far above any real weather
# answer; present to bound the worst case, not to trim normal output.
DEFAULT_TOTAL_TOKENS_LIMIT = 100_000
DEFAULT_LOG_LEVEL = "WARNING"

DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com/otel"
DEFAULT_LANGSMITH_PROJECT = "meteobot"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off", ""})


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


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().casefold()
    if value in _TRUTHY:
        return True
    if value in _FALSEY:
        return False
    allowed = ", ".join(sorted(_TRUTHY | (_FALSEY - {""})))
    raise ConfigError(f"{name} must be a boolean ({allowed}), got {raw!r}.")


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
    request_limit: int
    total_tokens_limit: int
    log_level: str
    tracing_enabled: bool
    langsmith_api_key: str | None
    langsmith_endpoint: str
    langsmith_project: str


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

    langsmith_api_key = os.environ.get("LANGSMITH_API_KEY", "").strip() or None

    return Settings(
        openai_api_key=api_key,
        model=os.environ.get("METEOBOT_MODEL", DEFAULT_MODEL),
        http_timeout_seconds=_positive_float(
            "METEOBOT_HTTP_TIMEOUT", DEFAULT_HTTP_TIMEOUT_SECONDS
        ),
        request_limit=_positive_int("METEOBOT_REQUEST_LIMIT", DEFAULT_REQUEST_LIMIT),
        total_tokens_limit=_positive_int(
            "METEOBOT_TOTAL_TOKENS_LIMIT", DEFAULT_TOTAL_TOKENS_LIMIT
        ),
        log_level=_log_level("METEOBOT_LOG_LEVEL", DEFAULT_LOG_LEVEL),
        tracing_enabled=_bool("METEOBOT_TRACING", False),
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=os.environ.get(
            "LANGSMITH_OTEL_ENDPOINT", DEFAULT_LANGSMITH_ENDPOINT
        ),
        langsmith_project=os.environ.get(
            "LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT
        ),
    )
