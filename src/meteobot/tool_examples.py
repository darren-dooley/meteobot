"""Tool-use examples: show the model concrete calls, not just a schema.

The blog's third feature attaches `input_examples` to a tool definition so the
model learns conventions a JSON schema can't express. PydanticAI 2.14 has no
such field, and its tool docs say to put few-shot examples in the description
instead — so that is what this does: `with_examples` renders a set of example
argument dicts as a call-shaped block and appends it to a base description.

`run_python` builds its own richer examples inline; this helper is for the plain
function tools (like `get_weather`) where one or two example calls remove any
ambiguity about how an argument should look.
"""

from __future__ import annotations

from typing import Any


def _render_call(tool_name: str, arguments: dict[str, Any]) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"    {tool_name}({args})"


def with_examples(
    description: str, tool_name: str, examples: list[dict[str, Any]]
) -> str:
    """Append an `Examples:` block of example calls to `description`.

    Each example is the keyword arguments of one call; they render as
    `tool_name(key=value, ...)` lines under an `Examples:` heading.
    """
    if not examples:
        return description
    rendered = "\n".join(_render_call(tool_name, ex) for ex in examples)
    return f"{description}\n\nExamples:\n{rendered}"
