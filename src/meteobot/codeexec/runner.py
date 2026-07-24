"""In-container runner for the Docker executor. Standard library only.

This file is mounted read-only into the sandbox container and run as its entry
point; it must import nothing outside the standard library, because the sandbox
image is a bare `python:*-slim` with no dependencies installed. It never runs in
the parent process.

Protocol, over stdin/stdout, one JSON object per line:

- parent -> runner, once:   {"type": "code", "source": <str>, "tools": [<str>]}
- runner -> parent, N times: {"type": "tool_call", "id": <int>, "name": <str>,
                              "args": <obj>}
- parent -> runner, per call: {"type": "tool_result", "id": <int>, "result": <obj>}
                              or {"type": "tool_result", "id": <int>,
                                 "error": <str>}
- runner -> parent, once:   {"type": "result", "stdout": <str>, "error": <str|null>}

The model's code runs inside `--network none`, so a tool proxy cannot reach the
MCP server itself. It sends a `tool_call` and awaits the parent, which holds the
only MCP client, to run the call and send back a `tool_result`. The runner binds
the real stdout pipe before redirecting the snippet's stdout into a buffer, so
protocol frames and the snippet's own `print()` output never collide.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
from typing import Any

# Bind the real stdout pipe before any redirect: RPC frames must reach the
# parent even while the snippet's own stdout is captured into a buffer.
_PIPE = sys.stdout


def _send(obj: dict[str, Any]) -> None:
    _PIPE.write(json.dumps(obj) + "\n")
    _PIPE.flush()


async def _main() -> None:
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    async def read_msg() -> dict[str, Any] | None:
        line = await reader.readline()
        return json.loads(line) if line else None

    first = await read_msg()
    if first is None or first.get("type") != "code":
        _send({"type": "result", "stdout": "", "error": "no code received"})
        return
    code: str = first["source"]
    tool_names: list[str] = first["tools"]

    pending: dict[int, asyncio.Future[Any]] = {}
    counter = 0

    def make_proxy(name: str):
        async def proxy(**kwargs: Any) -> Any:
            nonlocal counter
            counter += 1
            call_id = counter
            fut: asyncio.Future[Any] = loop.create_future()
            pending[call_id] = fut
            _send({"type": "tool_call", "id": call_id, "name": name, "args": kwargs})
            return await fut

        return proxy

    async def pump() -> None:
        while True:
            msg = await read_msg()
            if msg is None:
                break
            if msg.get("type") == "tool_result":
                fut = pending.pop(msg["id"], None)
                if fut is None or fut.done():
                    continue
                if "error" in msg and msg["error"] is not None:
                    fut.set_exception(RuntimeError(msg["error"]))
                else:
                    fut.set_result(msg.get("result"))

    namespace: dict[str, Any] = {"asyncio": asyncio, "json": json}
    for name in tool_names:
        namespace[name] = make_proxy(name)

    body = "".join("    " + line + "\n" for line in code.splitlines())
    wrapped = "async def __main__():\n" + (body or "    pass\n")

    buffer = io.StringIO()
    pump_task = asyncio.create_task(pump())
    error: str | None = None
    try:
        exec(wrapped, namespace)
        with contextlib.redirect_stdout(buffer):
            await namespace["__main__"]()
    except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
        error = f"{type(exc).__name__}: {exc}"
    finally:
        pump_task.cancel()
    _send({"type": "result", "stdout": buffer.getvalue(), "error": error})


if __name__ == "__main__":
    asyncio.run(_main())
