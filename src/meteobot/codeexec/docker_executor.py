"""The default executor: model code runs in a locked-down Docker container.

`DockerExecutor` launches `python:*-slim` with no network, all capabilities
dropped, and read-only access to just the runner script, then feeds it the
model's code over stdin. The container holds no MCP client; when the code calls
a tool, `runner.py` sends a `tool_call` frame and this class services it by
invoking `McpTools` in the trusted parent and sending the result back. Only the
code's captured stdout returns to the model.

Isolation comes from Docker flags, all kernel-enforced and free:

- `--network none`      no sockets: the snippet cannot reach the internet, and
                        MCP calls go through the parent, not from the container.
- `--cap-drop ALL` +    no privileged operations, no privilege escalation.
  `--security-opt no-new-privileges`
- `--read-only` + tmpfs a writable `/tmp` only; the rest of the filesystem is
                        immutable, and the runner is mounted `:ro`.
- `--memory` / `--cpus` / `--pids-limit`   bounded blast radius for a runaway.
- a wall-clock timeout in the parent kills a container that overstays.

The container is `--rm`, so nothing survives the run. `preflight()` checks Docker
is usable so the composition root can fall back to the in-process executor with a
clear log line rather than failing a Turn.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from meteobot.codeexec.executor import ExecResult
from meteobot.codeexec.mcp_tools import McpTools

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MEMORY = "256m"
DEFAULT_CPUS = "1.0"
DEFAULT_PIDS_LIMIT = 128

_RUNNER = Path(__file__).with_name("runner.py")


async def preflight(docker_bin: str = "docker", timeout: float = 10.0) -> str | None:
    """Return None if Docker looks usable, else a one-line reason it is not.

    Runs `docker info`, which fails fast when the daemon is unreachable — the
    composition root uses this to choose the container path or fall back.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            docker_bin,
            "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return f"{docker_bin!r} not found on PATH"
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return "`docker info` timed out"
    if proc.returncode != 0:
        first_line = stderr.decode().strip().splitlines()
        return first_line[0] if first_line else "`docker info` failed"
    return None


class DockerExecutor:
    """Runs code in an ephemeral, network-isolated container. See module docs."""

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        memory: str = DEFAULT_MEMORY,
        cpus: str = DEFAULT_CPUS,
        pids_limit: int = DEFAULT_PIDS_LIMIT,
        docker_bin: str = "docker",
    ) -> None:
        self._image = image
        self._timeout = timeout_seconds
        self._memory = memory
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._docker_bin = docker_bin

    def _command(self) -> list[str]:
        return [
            self._docker_bin,
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:size=16m",
            "--memory",
            self._memory,
            "--cpus",
            self._cpus,
            "--pids-limit",
            str(self._pids_limit),
            "-v",
            f"{_RUNNER}:/runner.py:ro",
            self._image,
            "python",
            "/runner.py",
        ]

    async def run(self, code: str, tools: McpTools) -> ExecResult:
        """Execute `code` in a container, servicing its MCP calls from here."""
        proc = await asyncio.create_subprocess_exec(
            *self._command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            return await asyncio.wait_for(
                self._drive(proc, code, tools), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(
                stdout="",
                error=f"Code execution exceeded the {self._timeout:g}s time limit.",
            )

    async def _drive(
        self, proc: asyncio.subprocess.Process, code: str, tools: McpTools
    ) -> ExecResult:
        assert proc.stdin is not None and proc.stdout is not None
        stdin, stdout = proc.stdin, proc.stdout
        write_lock = asyncio.Lock()

        async def send(obj: dict[str, Any]) -> None:
            async with write_lock:
                stdin.write((json.dumps(obj) + "\n").encode())
                await stdin.drain()

        await send({"type": "code", "source": code, "tools": tools.names})

        calls = 0
        in_flight: set[asyncio.Task[None]] = set()

        async def service(msg: dict[str, Any]) -> None:
            # A tool call from the sandbox: run it in the trusted parent and
            # return the result (or a readable error) so the snippet continues.
            try:
                result = await tools.call(msg["name"], msg.get("args", {}))
            except Exception as exc:  # noqa: BLE001 - surfaced into the snippet
                await send({"type": "tool_result", "id": msg["id"], "error": str(exc)})
            else:
                await send(
                    {"type": "tool_result", "id": msg["id"], "result": result}
                )

        result: ExecResult | None = None
        while True:
            line = await stdout.readline()
            if not line:
                break
            msg = json.loads(line)
            kind = msg.get("type")
            if kind == "tool_call":
                calls += 1
                task = asyncio.create_task(service(msg))
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)
            elif kind == "result":
                result = ExecResult(
                    stdout=msg.get("stdout", ""),
                    error=msg.get("error"),
                    tool_calls=calls,
                )
                break

        for task in in_flight:
            task.cancel()
        await proc.wait()
        if result is None:
            stderr = (await proc.stderr.read()).decode() if proc.stderr else ""
            detail = stderr.strip().splitlines()[-1] if stderr.strip() else "no output"
            return ExecResult(
                stdout="", error=f"sandbox exited early: {detail}", tool_calls=calls
            )
        return result
