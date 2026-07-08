"""REPL loop contract tests.

The loop is tested at its injected seam: a fake `read_input` scripts the
terminal, a fake `run_turn` stands in for the agent, and `render`/`notify`
record output. This covers the behavioral acceptance criteria — quit/exit,
EOF, Infrastructure-Error survival, and History carried across Turns — while
the actual terminal I/O edge (asyncio.to_thread(input), readline, flushed
per-delta writes) is left untested per design-decision #10.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from meteobot.cli import INFRA_ERROR_LINE, run_repl
from meteobot.llm import History


@dataclass
class ScriptedInput:
    """Fake read_input: yields queued lines, then None (EOF) when exhausted."""

    lines: list[str]
    reads: int = 0

    async def __call__(self) -> str | None:
        self.reads += 1
        if not self.lines:
            return None
        return self.lines.pop(0)


@dataclass
class RecordingTurns:
    """Fake run_turn: records the History length it saw and appends to it,
    exactly as the real agent mutates client-owned History in place."""

    seen_history_lengths: list[int] = field(default_factory=list)

    async def run_turn(
        self, user_input: str, history: History, render: Callable[[str], None]
    ) -> None:
        self.seen_history_lengths.append(len(history))
        history.append({"role": "user", "content": user_input})
        render(f"answer to {user_input!r}")
        history.append({"role": "assistant", "content": f"answer to {user_input!r}"})


@pytest.mark.parametrize("word", ["quit", "exit", "QUIT", " exit "])
async def test_a_quit_word_ends_the_loop(word: str) -> None:
    # A quit word ends the loop before the sentinel line is ever read.
    read_input = ScriptedInput([word, "should never be read"])
    turns = RecordingTurns()
    notify: list[str] = []

    await run_repl(
        run_turn=turns.run_turn,
        history=[],
        read_input=read_input,
        render=lambda _: None,
        notify=notify.append,
    )

    assert turns.seen_history_lengths == []  # no Turn ran
    assert read_input.lines == ["should never be read"]  # loop stopped at the quit word


async def test_eof_ends_the_loop() -> None:
    # read_input returning None (Ctrl-D / closed stdin) ends the loop.
    read_input = ScriptedInput([])
    turns = RecordingTurns()

    await run_repl(
        run_turn=turns.run_turn,
        history=[],
        read_input=read_input,
        render=lambda _: None,
        notify=lambda _: None,
    )

    assert read_input.reads == 1
    assert turns.seen_history_lengths == []


async def test_infrastructure_error_is_caught_turn_abandoned_repl_survives() -> None:
    async def failing_turn(
        user_input: str, history: History, render: Callable[[str], None]
    ) -> None:
        raise RuntimeError("LLM API is down")

    read_input = ScriptedInput(["what's the weather?", "quit"])
    rendered: list[str] = []
    notify: list[str] = []

    await run_repl(
        run_turn=failing_turn,
        history=[],
        read_input=read_input,
        render=rendered.append,
        notify=notify.append,
    )

    # The failure surfaced as one friendly line, and the loop lived on to read
    # the next input (the quit word) rather than crashing.
    assert INFRA_ERROR_LINE in notify
    assert read_input.reads == 2
    assert rendered == []  # the abandoned Turn streamed nothing usable


async def test_history_is_carried_across_turns() -> None:
    # Two questions then quit: the second Turn must see the History the first
    # one grew, proving the same client-owned list is carried between Turns.
    read_input = ScriptedInput(["first", "second", "quit"])
    turns = RecordingTurns()
    history: History = []

    await run_repl(
        run_turn=turns.run_turn,
        history=history,
        read_input=read_input,
        render=lambda _: None,
        notify=lambda _: None,
    )

    # First Turn saw an empty History; the second saw the two items the first
    # appended (its user message and answer).
    assert turns.seen_history_lengths == [0, 2]
    assert [item["content"] for item in history] == [
        "first",
        "answer to 'first'",
        "second",
        "answer to 'second'",
    ]


async def test_blank_input_is_skipped_without_running_a_turn() -> None:
    read_input = ScriptedInput(["", "   ", "quit"])
    turns = RecordingTurns()

    await run_repl(
        run_turn=turns.run_turn,
        history=[],
        read_input=read_input,
        render=lambda _: None,
        notify=lambda _: None,
    )

    assert turns.seen_history_lengths == []  # neither blank line ran a Turn
    assert read_input.reads == 3
