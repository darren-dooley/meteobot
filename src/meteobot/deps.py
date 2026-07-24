"""The dependency-injection payload carried through a PydanticAI run.

A `Deps` value is assembled once at the composition root and passed on every
agent run; tools reach it through PydanticAI's `RunContext`. It preserves v1's
decision to share one pooled httpx client across the concurrent multi-city
fan-out, with its timeout and connect-retry policy configured in one place, and
to make the injected `Settings` reachable wherever a tool needs a tunable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from meteobot.config import Settings

if TYPE_CHECKING:
    from meteobot.codeexec import Sandbox


@dataclass
class Deps:
    """Shared runtime dependencies injected into every tool via `RunContext`."""

    http_client: httpx.AsyncClient
    settings: Settings
    # The code-execution sandbox, present only when code execution is enabled and
    # its MCP client is connected. `run_python` reads it here; every other tool
    # ignores it. Optional with a default so existing call sites and tests that
    # build a weather-only `Deps` keep working unchanged.
    sandbox: "Sandbox | None" = None
