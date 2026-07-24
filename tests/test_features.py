"""Assembly of the advanced tool-use features from settings.

`build_advanced_tooling` is the one place the two features are turned on or off
and their lifecycle is managed, so these tests pin what each setting produces and
that the Docker executor degrades to in-process when Docker is unusable.
"""

from __future__ import annotations

import pytest

from meteobot.codeexec.docker_executor import DockerExecutor
from meteobot.codeexec.executor import InProcessExecutor
from meteobot.config import Settings
from meteobot.features import _build_executor, build_advanced_tooling


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = dict(
        openai_api_key="test-key", model="gpt-4.1-mini", http_timeout_seconds=10.0,
        request_limit=6, total_tokens_limit=100_000, log_level="WARNING",
        tool_search_enabled=False, code_exec_enabled=False, executor="inprocess",
        sandbox_image="python:3.12-slim", code_exec_timeout_seconds=30.0,
        tracing_enabled=False, langsmith_api_key=None,
        langsmith_endpoint="https://example.test", langsmith_project="meteobot",
        langsmith_workspace_id=None,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


async def test_all_features_off_produces_nothing() -> None:
    async with build_advanced_tooling(settings()) as tooling:
        assert tooling.toolsets == []
        assert tooling.extra_tools == []
        assert tooling.capabilities == []
        assert tooling.sandbox is None
        assert tooling.enabled is False


async def test_tool_search_only_attaches_a_deferred_toolset_and_capability() -> None:
    async with build_advanced_tooling(settings(tool_search_enabled=True)) as tooling:
        assert len(tooling.toolsets) == 1
        assert len(tooling.capabilities) == 1
        assert tooling.sandbox is None
        assert tooling.extra_tools == []
        assert tooling.enabled is True


async def test_code_exec_only_attaches_run_python_and_a_sandbox() -> None:
    async with build_advanced_tooling(settings(code_exec_enabled=True)) as tooling:
        assert tooling.toolsets == []
        assert len(tooling.extra_tools) == 1
        assert tooling.sandbox is not None
        assert isinstance(tooling.sandbox.executor, InProcessExecutor)
        assert tooling.enabled is True


async def test_build_executor_honours_inprocess_setting() -> None:
    executor = await _build_executor(settings(executor="inprocess"))
    assert isinstance(executor, InProcessExecutor)


async def test_build_executor_falls_back_when_docker_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(*_args: object, **_kwargs: object) -> str:
        return "docker daemon not running"

    monkeypatch.setattr("meteobot.features.preflight", unavailable)
    executor = await _build_executor(settings(executor="docker"))
    assert isinstance(executor, InProcessExecutor)


async def test_build_executor_uses_docker_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def available(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr("meteobot.features.preflight", available)
    executor = await _build_executor(settings(executor="docker"))
    assert isinstance(executor, DockerExecutor)
