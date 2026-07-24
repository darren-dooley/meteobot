"""The `run_python` tool: the PydanticAI adapter over a `Sandbox`.

`build_run_python` turns a connected `Sandbox` into a registered tool. Two
things happen here that matter to how well the model uses it:

- The description is generated from the live MCP catalogue, so the model is told
  exactly which async functions its script may call and with what parameters —
  the tool-search corpus and the code-exec surface stay in sync by construction.
- The description carries worked examples (minimal, then a parallel `gather`).
  PydanticAI 2.14 has no `input_examples` field, so few-shot examples live in the
  description, which is the framework's documented way to demonstrate usage.

The tool returns a typed `ExecResult` and never raises across the boundary: a
disabled sandbox, a broken snippet, or a timeout all come back as data the model
can read and act on, keeping code execution in the same Tool Error tier as
`get_weather`.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext, Tool

from meteobot.codeexec.executor import ExecResult
from meteobot.codeexec.mcp_tools import McpTools
from meteobot.deps import Deps

_EXAMPLES = """\
Examples (the script body may use `await`, `asyncio.gather`, and `json`; both are
preloaded, no imports needed):

    # One conversion.
    f = await celsius_to_fahrenheit(celsius=21.8)
    print(f)

    # Several at once, combined into one printed result. Only what you print()
    # comes back — intermediate values stay in the sandbox.
    temps = [21.8, 18.0, 25.5]
    out = await asyncio.gather(*[celsius_to_fahrenheit(celsius=t) for t in temps])
    print(json.dumps(dict(zip(temps, out))))\
"""


def _params(schema: dict[str, Any]) -> str:
    """Render a tool's JSON-schema properties as a compact `name: type` list."""
    props = schema.get("properties", {})
    return ", ".join(f"{name}: {spec.get('type', 'any')}" for name, spec in props.items())


def describe_tools(tools: McpTools) -> str:
    """A bullet line per MCP tool: signature and one-line description."""
    lines = [
        f"  - {t.name}({_params(t.parameters)}): {t.description.strip()}"
        for t in tools.tools
    ]
    return "\n".join(lines)


def build_description(tools: McpTools) -> str:
    """The full `run_python` description: purpose, live catalogue, examples."""
    return (
        "Run a short Python 3 script in a sandbox to call conversion tools and "
        "return only what you print(). Prefer this over calling tools one at a "
        "time when a question needs several conversions or their results "
        "combined.\n\n"
        "Available async tools (call with await and keyword arguments):\n"
        f"{describe_tools(tools)}\n\n"
        f"{_EXAMPLES}"
    )


async def run_python(ctx: RunContext[Deps], code: str) -> ExecResult:
    """Execute `code` in the sandbox bound to this run's deps."""
    sandbox = ctx.deps.sandbox
    if sandbox is None:
        return ExecResult(
            stdout="",
            error="Code execution is not enabled in this session.",
        )
    return await sandbox.run(code)


def build_run_python(tools: McpTools) -> Tool[Deps]:
    """Build the `run_python` tool with a description drawn from `tools`."""
    return Tool(
        run_python,
        name="run_python",
        description=build_description(tools),
    )
