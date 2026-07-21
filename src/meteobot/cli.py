"""The interactive REPL: read a question, stream the answer, repeat.

The loop's control flow and error routing live in `run_repl`, tested against
injected fakes. The terminal I/O edge — pushing the blocking read onto a
worker thread, readline line-editing, and flushed per-delta writes — lives in
`start_repl`, which is thin, dependency-only wiring and is left untested (see
design-decision #10): asserting that `print` prints or that a thread runs a
function tests the standard library, not meteobot.

Error routing follows the two-tier model: Tool Errors are already structured
data the model explains, so the only failures that escape `run_turn` are
Infrastructure Errors (LLM API down, bad credentials, transport failure).
Those are caught here, surfaced as one friendly line, and the current Turn is
abandoned while the REPL survives to accept the next question.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from meteobot.agent import History, RunTurn

PROMPT = "\n> "
WELCOME = (
    "meteobot — ask about the current weather anywhere. "
    "Type 'quit' or 'exit' to leave.\n"
)
GOODBYE = "Bye."
INFRA_ERROR_LINE = (
    "Sorry — I couldn't reach the model just now, so I wasn't able to answer "
    "that. Please try again."
)
QUIT_WORDS = frozenset({"quit", "exit"})


async def run_repl(
    *,
    run_turn: RunTurn,
    history: History,
    read_input: Callable[[], Awaitable[str | None]],
    render: Callable[[str], None],
    notify: Callable[[str], None],
) -> None:
    """Drive Turns until the user quits or input ends.

    `read_input` returns the next line, or None at end of input (Ctrl-D /
    closed stdin). `render` receives streamed text deltas; `notify` receives
    whole-line messages (sign-off, Infrastructure Errors).
    """
    while True:
        line = await read_input()
        if line is None:  # end of input
            notify(GOODBYE)
            return
        text = line.strip()
        if not text:
            continue
        if text.casefold() in QUIT_WORDS:
            notify(GOODBYE)
            return
        try:
            await run_turn(text, history, render)
        except Exception:
            # Anything raised past the agent is an Infrastructure Error: the
            # agent turns every model-actionable failure into a Tool Error data
            # result, so this is for the human, never the model. Abandon the
            # Turn, keep the REPL alive.
            notify(INFRA_ERROR_LINE)


async def start_repl(run_turn: RunTurn, history: History) -> None:
    """Wire real terminal I/O to `run_repl`. Untested edge (design-decision #10)."""
    import readline  # noqa: F401  (importing it enables line editing + history)

    def render(delta: str) -> None:
        print(delta, end="", flush=True)

    def notify(message: str) -> None:
        print(message, flush=True)

    async def read_input() -> str | None:
        # Keep the event loop free: the blocking read runs on a worker thread.
        try:
            return await asyncio.to_thread(input, PROMPT)
        except EOFError:
            return None

    print(WELCOME, end="", flush=True)
    await run_repl(
        run_turn=run_turn,
        history=history,
        read_input=read_input,
        render=render,
        notify=notify,
    )
