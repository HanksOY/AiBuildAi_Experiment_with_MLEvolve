"""Human review of a draft after it is written and before it is run.

Placed between generation and execution, where the search already leaves a gap:
Phase 1 produces drafts with ``execute_immediately=False`` and only runs them
later via ``execute_deferred_node``. Nothing new has to be interrupted.

Scoped to the initial drafts on purpose. They set the direction for their whole
branch, they are generated one at a time, and there are only a handful. Gating
every step of a 200 step run would be manual operation, not review.

The code is written to a file and its path printed rather than dumped into the
terminal: several hundred lines do not fit in a panel, and an editor is the
right place to read them.
"""

import logging
import re
import sys
import threading
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.prompt_commands import Command, ask_command, render_help

logger = logging.getLogger("MLEvolve")

# Only one review is ever open, so parallel workers queue instead of fighting
# over stdin. Redundant in Phase 1, which is sequential, but keeps the gate safe
# if it is ever reached from the thread pool.
_gate = threading.Lock()

ACCEPT, REJECT, ABORT = "accept", "reject", "abort"

KNOWN_MODELS = [
    "LGBMClassifier", "XGBClassifier", "CatBoost", "HistGradientBoosting",
    "RandomForest", "ExtraTrees", "LogisticRegression", "LinearSVC", "SVC",
    "MLPClassifier", "Ridge", "Lasso", "ElasticNet", "KNeighbors",
    "GradientBoosting", "AdaBoost", "VotingClassifier", "StackingClassifier",
]

QA_SYSTEM = (
    "You answer questions about a machine learning solution that is about to run. "
    "You are concise and concrete, and you quote the code when it helps."
)

QA_TEMPLATE = """The owner is reviewing this solution before it runs, and asked a question.

Answer in two to five sentences. Point at specific lines or functions where useful.
Do not rewrite the solution; they will ask for that separately if they want it.

# Its plan
{plan}

# The code
{code}

# Their question
{question}
"""


def summarise(code: str) -> dict:
    """Facts worth seeing before deciding, extracted without a model call."""
    found = [m for m in KNOWN_MODELS if re.search(r"\b" + m, code)]
    return {
        "models": ", ".join(found) if found else "none detected",
        "lines": len(code.splitlines()),
        "writes_submission": "submission.csv" in code,
        "prints_score": "Final Validation Score" in code,
        "cv": bool(re.search(r"StratifiedKFold|KFold|cross_val", code)),
    }


def _write_draft_file(cfg, node, index: int):
    """Save the draft where the user can open it. Returns (path, windows path)."""
    from utils.run_paths import run_root, to_windows_path

    drafts_dir = run_root(cfg) / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    path = drafts_dir / f"draft_{index}_{node.id[:8]}.py"
    path.write_text(node.code, encoding="utf-8")
    return path, to_windows_path(path)


def _answer_question(agent, node, question: str, console: Console) -> None:
    from llm import generate
    from utils.response import trim_long_string

    console.print("[dim]thinking ...[/dim]")
    try:
        reply = generate(
            prompt={
                "system": QA_SYSTEM,
                "user": QA_TEMPLATE.format(
                    plan=node.plan or "(no plan recorded)",
                    code=trim_long_string(node.code, threshold=12000, k=6000),
                    question=question,
                ),
            },
            cfg=agent.cfg,
            stage="feedback",
        ).strip()
    except Exception as e:
        logger.warning(f"Could not answer the draft question: {e}")
        reply = "I could not answer that. /go runs it, /change regenerates, /skip drops it."
    console.print(Panel(reply, border_style="cyan", expand=False))


def _show(node, index: int, total: int, windows_path: str, console: Console) -> None:
    facts = summarise(node.code)
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("Models", facts["models"])
    table.add_row("Length", f"{facts['lines']} lines")
    table.add_row("Cross-validation", "yes" if facts["cv"] else "[yellow]not detected[/yellow]")
    table.add_row("Writes submission", "yes" if facts["writes_submission"] else "[red]NO[/red]")
    table.add_row("Prints score", "yes" if facts["prints_score"] else "[red]NO[/red]")
    table.add_row("Code", windows_path)
    console.print(
        Panel(table, title=f"[bold]Draft {index} of {total} — before it runs[/bold]",
              border_style="blue", expand=False)
    )
    if node.plan:
        console.print(Panel(node.plan.strip(), title="[bold]Its plan[/bold]",
                            border_style="grey50", expand=False))


def _rereview(agent, node, console: Console) -> None:
    """Run the code reviewer over an edited draft before it is executed."""
    from agents import code_review_agent

    console.print("[dim]Checking your edit ...[/dim]")
    try:
        reviewed = code_review_agent.run(agent, node)
    except Exception as e:
        logger.warning(f"Code review of the edited draft failed: {e}")
        console.print("[yellow]Could not review the edit; it will run as you wrote it.[/yellow]")
        return

    if reviewed.strip() != node.code.strip():
        console.print("[yellow]The reviewer adjusted your edit.[/yellow]")
        node.code = reviewed
    else:
        console.print("[green]Your edit passed review unchanged.[/green]")


def review_draft(agent, node, index: int, total: int, console: Console | None = None) -> str:
    """Block until the user accepts, rejects or aborts. Returns ACCEPT/REJECT/ABORT.

    Time spent here is added to ``agent.paused_seconds`` so reading a draft does
    not consume the search budget.
    """
    console = console or Console()
    cfg = agent.cfg
    if not getattr(getattr(cfg, "draft_review", None), "enabled", False):
        return ACCEPT

    # No terminal to ask on. Unlike the plan gate, this fires mid-run, so
    # aborting would throw away work already paid for. Accept and say so.
    if not sys.stdin.isatty():
        logger.warning(
            "draft_review.enabled is true but stdin is not a terminal; accepting draft "
            f"{index} unreviewed. Pass draft_review.enabled=False for unattended runs."
        )
        return ACCEPT

    with _gate:
        started = time.time()
        try:
            path, windows_path = _write_draft_file(cfg, node, index)
            logger.info(f"Draft {index} written for review: {path}")
            _show(node, index, total, windows_path, console)

            while True:
                command = ask_command(
                    console,
                    hint="/go run it · /edit reload my changes · /change <what to fix> · "
                         "/ask a question · /skip drop it · /quit",
                )

                if command.kind in {"go", "empty"}:
                    return ACCEPT
                if command.kind == "quit":
                    return ABORT
                if command.kind == "skip":
                    logger.info(f"Draft {index} dropped by the user.")
                    return REJECT
                if command.kind == "change":
                    node.review_feedback = command.text
                    logger.info(f"Draft {index} rejected with feedback: {command.text}")
                    return REJECT
                if command.kind == "ask":
                    _answer_question(agent, node, command.text, console)
                    continue
                if command.kind == "back":
                    console.print("[dim]Nothing precedes a draft. /skip drops this one.[/dim]")
                    continue

                if command.kind == "raw":
                    if command.text.lower() in {"edit", "reload", "reread"}:
                        command = Command("edit")
                    else:
                        _answer_question(agent, node, command.text, console)
                        continue

                if command.kind == "edit":
                    reloaded = path.read_text(encoding="utf-8")
                    if reloaded.strip() == node.code.strip():
                        console.print("[yellow]That file has not changed.[/yellow]")
                        continue
                    node.code = reloaded
                    console.print(f"[green]Reloaded {len(reloaded.splitlines())} lines.[/green]")
                    _rereview(agent, node, console)
                    path.write_text(node.code, encoding="utf-8")
                    _show(node, index, total, windows_path, console)
                    continue

                render_help(console)
        finally:
            waited = time.time() - started
            agent.paused_seconds += waited
            logger.info(f"Draft review took {waited:.0f}s, excluded from the search budget.")
