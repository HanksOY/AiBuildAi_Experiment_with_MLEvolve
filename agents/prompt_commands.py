"""Shared input handling for every interactive prompt.

One parser, used by setup, the clarifying questions, the plan review and the
settings review, so a command means the same thing everywhere.

Two ways to say the same thing, deliberately:

  /ask what does 1800 mean?     explicit, parsed without a model, never misread
  what does 1800 mean?          inferred by the caller, as before

The slash form exists because inference is not reliable enough to be the only
route: it costs a model call, adds latency, and misjudges. It is an override,
not a requirement — being forced to prefix every message is friction nobody
keeps up.

The rule that matters most here is neither of those: input that cannot be
classified with confidence must send the prompt round again. It must never fall
through to a default action. A settings prompt that quietly kept every value
because it found the digits in "what does 1800 mean in 1?" is exactly the
failure this prevents.
"""

import logging
import re
import sys
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

logger = logging.getLogger("MLEvolve")

# Every spelling maps to one canonical kind.
ALIASES = {
    "ask": "ask", "question": "ask", "q": "ask", "?": "ask", "explain": "ask",
    "change": "change", "edit": "change", "revise": "change", "fix": "change",
    "back": "back", "b": "back", "rewind": "back", "undo": "back", "prev": "back",
    "skip": "skip", "s": "skip", "next": "skip",
    "edit": "edit", "reload": "edit", "reread": "edit",
    "go": "go", "start": "go", "continue": "go", "ok": "go", "done": "go",
    "y": "go", "yes": "go", "a": "go", "all": "go", "apply": "go",
    "quit": "quit", "exit": "quit", "cancel": "quit", "abort": "quit",
    "n": "quit", "no": "quit",
    "keep": "keep", "k": "keep",
    "help": "help", "h": "help",
}

HELP_ROWS = [
    ("/ask <text>", "answer my question, change nothing"),
    ("/change <text>", "request a change"),
    ("/back", "return to the previous question or step"),
    ("/edit", "reload a file after you changed it"),
    ("/skip", "leave this one unanswered"),
    ("/go", "accept and continue"),
    ("/quit", "stop without running"),
    ("/help", "this list"),
]


@dataclass
class Command:
    """A parsed prompt entry. `kind` is canonical; `text` is whatever followed."""

    kind: str
    text: str = ""

    @property
    def is_raw(self) -> bool:
        """True when nothing was recognised, so the caller should interpret it."""
        return self.kind == "raw"


def render_help(console: Console) -> None:
    lines = [f"[cyan]{cmd}[/cyan]".ljust(34) + desc for cmd, desc in HELP_ROWS]
    lines.append("")
    lines.append("[dim]The slash is optional — plain text works too, and anything")
    lines.append("unrecognised is read as a question or an instruction.[/dim]")
    console.print(Panel("\n".join(lines), title="[bold]Commands[/bold]",
                        border_style="cyan", expand=False))


def parse(line: str, bare: bool = True) -> Command:
    """Turn raw input into a Command.

    `bare` allows the unprefixed spellings ("go", "s", "back"). Slash forms are
    always accepted. Anything else comes back as kind "raw".
    """
    text = (line or "").strip()
    if not text:
        return Command("empty")

    if text.startswith("/"):
        word, _, rest = text[1:].partition(" ")
        kind = ALIASES.get(word.lower())
        if kind:
            return Command(kind, rest.strip())
        # A slash the user meant as a command but we do not know. Say so rather
        # than silently treating "/aks foo" as prose.
        return Command("unknown", text)

    if bare:
        word, _, rest = text.partition(" ")
        kind = ALIASES.get(word.lower())
        # Bare single words only. "go" is a command; "go deeper on the imbalance"
        # is an instruction, and treating it as "go" would start the run.
        if kind and not rest.strip():
            return Command(kind, "")

    return Command("raw", text)


def is_selection(text: str) -> bool:
    """True only when the whole entry is index tokens, e.g. "1 3" or "2,4".

    Scanning prose for stray digits is how "what does 1800 mean in 1?" turned
    into a selection, so the entire string has to look like one.
    """
    return bool(text) and bool(re.fullmatch(r"[\d,\s]+", text.strip()))


def selected_indices(text: str) -> set[int]:
    return {int(t) for t in re.split(r"[,\s]+", text.strip()) if t.isdigit()}


def discard_typeahead() -> None:
    """Drop keystrokes buffered while the previous step was still working.

    Terminals hold input typed during a model call and hand it to the next
    prompt, turning an idle thought into an answer to a question not yet asked.
    """
    try:
        import termios

        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass  # not a POSIX terminal, nothing buffered


def chat_prompt(console: Console, hint: str = "", default: str = "") -> str:
    """A bordered input area, so typing reads as a chat box, not a bare prompt.

    A terminal cannot pin a box to the bottom of the screen without a full TUI
    framework, so the frame is drawn around the cursor instead.
    """
    discard_typeahead()
    console.print()
    console.print(Rule(style="grey39"))
    if hint:
        console.print(f"[dim]{hint}  ·  /help[/dim]")
    try:
        answer = Prompt.ask("[bold cyan]>[/bold cyan]", default=default,
                            show_default=False, console=console)
    finally:
        console.print(Rule(style="grey39"))
    return answer.strip()


def ask_command(console: Console, hint: str = "", default: str = "") -> Command:
    """Prompt until something recognisable comes back.

    /help is handled here. "unknown" re-prompts rather than guessing, which is
    the whole point: no entry should ever fall through to a default action.
    """
    while True:
        raw = chat_prompt(console, hint=hint, default=default)
        command = parse(raw)
        if command.kind == "help":
            render_help(console)
            continue
        if command.kind == "unknown":
            console.print(f"  [yellow]Unknown command:[/yellow] {command.text.split()[0]}")
            render_help(console)
            continue
        return command
