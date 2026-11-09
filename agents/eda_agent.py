"""Exploratory data analysis, run before anything is planned.

Until now the only thing any agent knew about the data was structural: shape,
column names, dtypes and a few rows, from ``utils.data_preview``. Feature
engineering was designed essentially blind.

This writes an analysis script, runs it against the real files, and turns the
output into ``EDA_report.md``. The report is reviewed, then used to draft a data
plan covering cleaning, feature engineering and the train/val/test split. Both
are confirmed by the user, then folded into the task description so every agent
downstream — the data_processing_and_feature_engineering step in particular —
works from measured facts rather than assumptions.

The analysis is deliberately read-only and short-lived: it inspects the data, it
does not train anything.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agents.prompt_commands import ask_command, render_help

logger = logging.getLogger("MLEvolve")

REPORT_NAME = "EDA_report.md"
DEFAULT_TIMEOUT = 900

EDA_CODE_SYSTEM = (
    "You write exploratory data analysis scripts. Your output is read by another agent "
    "that will design the feature engineering, so you report measured facts, not opinions."
)

EDA_CODE_TEMPLATE = """Write one Python script that explores the data in ./input/ and prints a
markdown report to stdout. The report is the only output that matters; print it, do not
save it.

Cover, as far as the data allows:

- Shape of every file, and how the files relate to each other.
- The target: its distribution, and class balance if it is a classification task.
- Missing values: which columns, what proportion.
- For numeric columns, the spread. For categorical, the cardinality.
- Columns that look like identifiers, constants, or duplicates of each other.
- LEAKAGE CHECKS. This matters most. Look for any column whose relationship with the
  target is suspiciously strong, columns present in one file but not another, and any
  structure that would let a model recover the label without learning anything.
- Anything that would break naive loading code: misleading file extensions, mixed
  dtypes, memory size large enough to need care.

Rules:
- KEEP IT SHORT: aim for 120 lines, absolutely no more than 200. A long script takes
  longer to write than to run and gives more places to fail. Prefer a loop over a list
  of columns to a block per column.
- Read-only. Do not write files, do not train models, do not fit estimators beyond
  cheap descriptive statistics.
- Use `nrows` or `usecols` where a file is large enough to be slow.
- Print ONLY markdown. Start with `# EDA Report`. Use tables for numbers.
- Report what you measure, including counts. Never state a conclusion you did not compute.
- It must finish in a few minutes on CPU.
- Use plain, boring pandas. Avoid anything clever. Never use a bare builtin name such as
  `object`, `type` or `id` as a dict key or variable — write the string `"object"` when
  you mean a dtype.
- Wrap each section in try/except and print the error into the report rather than
  crashing: a partial report is far more useful than none.

# Task
{task_desc}

# What is known about the files so far
{data_preview}
"""

DATA_PLAN_SYSTEM = (
    "You plan the data handling for an automated machine learning run, working strictly "
    "from a measured EDA report rather than from assumptions."
)

DATA_PLAN_TEMPLATE = """Using the EDA report below, write the data plan for this task.

Cover exactly these sections, as markdown headings:

## Loading
Which files, how to read them, and any trap the report revealed (extensions, dtypes,
memory). Name the columns involved.

## Cleaning
What to drop and why: identifiers, constants, duplicates. Cite the numbers from the report.

## Leakage controls
Every leakage risk the report found, and the concrete rule that prevents it. Be explicit
about which columns are excluded and why.

## Feature engineering
What to build from what exists, ordered by expected value. Say what each feature is meant
to capture. Keep it proportionate to the data size the report measured.

## Splitting
Train/validation/test strategy, including stratification or grouping if the report shows
it is needed, and where each preprocessing step must be fitted relative to the splits.

Keep the whole plan under 450 words. Do not write code. Every claim should trace back to
something the report measured.
{feedback}
# Task
{task_desc}

# EDA report
{report}
"""


def _stage_cfg_text(task_desc) -> str:
    from agents.plan_mode import _as_text

    return _as_text(task_desc)


def generate_eda_code(cfg, task_desc, data_preview: str) -> str:
    """Ask the code model for an analysis script."""
    from llm import generate
    from utils.response import extract_code, trim_long_string

    raw = generate(
        prompt={
            "system": EDA_CODE_SYSTEM,
            "user": EDA_CODE_TEMPLATE.format(
                task_desc=_stage_cfg_text(task_desc),
                data_preview=trim_long_string(data_preview or "", threshold=6000, k=3000),
            ),
        },
        cfg=cfg,
        stage="code",
    )
    code = extract_code(raw)
    if not code:
        raise ValueError("The EDA script could not be extracted from the model response.")
    return code


def run_eda_script(cfg, code: str, timeout: int = DEFAULT_TIMEOUT):
    """Execute the analysis in the workspace so ./input/ resolves. Returns (ok, output).

    Uses a plain subprocess rather than the search Interpreter: this runs before
    the interpreter exists, needs no CPU pinning or parallel slots, and is
    read-only and short.
    """
    workspace = Path(cfg.workspace_dir).resolve()
    script = workspace / "eda_script.py"
    script.write_text(code, encoding="utf-8")

    started = time.time()
    try:
        # Absolute path, because cwd is the workspace: a relative one would be
        # resolved against it a second time.
        proc = subprocess.run(
            [sys.executable, "-u", str(script)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"EDA script exceeded {timeout}s and was stopped.")
        return False, f"The analysis did not finish within {timeout} seconds."
    finally:
        elapsed = time.time() - started

    logger.info(f"EDA script finished in {elapsed:.0f}s with code {proc.returncode}")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-25:]
        return False, "\n".join(tail) or "The analysis failed with no error output."
    return True, proc.stdout


def save_report(cfg, report: str) -> Path:
    """Write EDA_report.md next to the run's other outputs."""
    from utils.run_paths import run_root

    path = run_root(cfg) / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    logger.info(f"EDA report written to {path}")
    return path


def generate_data_plan(cfg, task_desc, report: str, feedback=None) -> str:
    """Turn the measured report into a data plan."""
    from llm import generate
    from utils.response import trim_long_string

    notes = ""
    if feedback:
        notes = "\n# Revision requests from the task owner (apply all)\n" + "\n".join(
            f"- {f}" for f in feedback
        ) + "\n"

    return generate(
        prompt={
            "system": DATA_PLAN_SYSTEM,
            "user": DATA_PLAN_TEMPLATE.format(
                task_desc=_stage_cfg_text(task_desc),
                report=trim_long_string(report, threshold=14000, k=7000),
                feedback=notes,
            ),
        },
        cfg=cfg,
        stage="code",
    ).strip()


def _answer(cfg, question: str, context: str, console: Console) -> None:
    from llm import generate
    from utils.response import trim_long_string

    console.print("[dim]thinking ...[/dim]")
    try:
        reply = generate(
            prompt={
                "system": "You answer questions about a data analysis report, concisely and "
                          "only from what the report actually measured.",
                "user": f"# Report\n{trim_long_string(context, threshold=12000, k=6000)}\n\n"
                        f"# Question\n{question}\n\nAnswer in two to five sentences.",
            },
            cfg=cfg,
            stage="feedback",
        ).strip()
    except Exception as e:
        logger.warning(f"Could not answer the EDA question: {e}")
        reply = "I could not answer that."
    console.print(Panel(reply, border_style="cyan", expand=False))


def review_report(cfg, report: str, path: Path, console: Console) -> str:
    """Confirm the EDA report. Returns 'accept', 'redo' or 'quit'."""
    from utils.run_paths import to_windows_path

    while True:
        console.print(
            Panel(
                Markdown(report[:4000] + ("\n\n*(truncated — open the file for the rest)*"
                                          if len(report) > 4000 else "")),
                title="[bold]EDA report[/bold]",
                subtitle=f"[dim]{to_windows_path(path)}[/dim]",
                border_style="blue",
                expand=False,
            )
        )
        command = ask_command(
            console,
            hint="/go accept · /change rerun the analysis · /ask a question · /quit",
        )
        if command.kind in {"go", "empty"}:
            return "accept"
        if command.kind == "quit":
            return "quit"
        if command.kind in {"change", "raw"}:
            if command.kind == "raw":
                _answer(cfg, command.text, report, console)
                continue
            return "redo"
        if command.kind == "ask":
            _answer(cfg, command.text, report, console)
            continue
        render_help(console)


def review_data_plan(cfg, task_desc, report: str, console: Console):
    """Confirm the data plan, redrafting on request. Returns the plan or None."""
    feedback: list[str] = []
    console.print("\n[dim]Drafting the data plan from the report ...[/dim]")
    plan = generate_data_plan(cfg, task_desc, report)

    show = True
    while True:
        if show:
            console.print(
                Panel(Markdown(plan), title="[bold]Data plan[/bold]",
                      subtitle="[dim]loading · cleaning · leakage · features · splitting[/dim]",
                      border_style="cyan", expand=False)
            )
        show = True
        command = ask_command(
            console,
            hint="/go accept · /change <what to fix> · /ask a question · /quit",
        )
        if command.kind in {"go", "empty"}:
            return plan
        if command.kind == "quit":
            return None
        if command.kind == "ask":
            _answer(cfg, command.text, report + "\n\n" + plan, console)
            show = False
            continue
        if command.kind in {"change", "raw"}:
            from agents.plan_mode import interpret_review_input

            if command.kind == "raw":
                intent, answer = interpret_review_input(cfg, command.text, plan)
                if intent == "question":
                    console.print(Panel(answer, border_style="cyan", expand=False))
                    show = False
                    continue
                feedback.append(command.text)
            else:
                feedback.append(command.text)
            console.print("[dim]Redrafting ...[/dim]")
            plan = generate_data_plan(cfg, task_desc, report, feedback)
            continue
        render_help(console)


def build_context_block(report_path: Path, plan: str) -> str:
    """The block appended to the task description every agent receives."""
    return (
        "\n\n" + "=" * 60
        + "\n**CONFIRMED DATA PLAN**\n"
        + "=" * 60
        + "\nThis plan was derived from a measured analysis of the actual data and\n"
        f"approved by the task owner. The full report is at {report_path.name}.\n"
        "Follow it for loading, cleaning, leakage control, feature engineering and\n"
        "splitting. Say so explicitly if you deviate.\n\n"
        + plan
    )


def run_eda_gate(cfg, task_desc, data_preview: str, console: Console | None = None):
    """Run analysis, confirm the report, draft and confirm the data plan.

    Returns (report_path, data_plan) or None if the user stopped. Never raises:
    a failure here should degrade to the previous behaviour, not end the run.
    """
    console = console or Console()
    ecfg = getattr(cfg, "eda", None)
    if not getattr(ecfg, "enabled", False):
        return None

    if not sys.stdin.isatty():
        logger.warning(
            "eda.enabled is true but stdin is not a terminal; skipping the EDA gate. "
            "Pass eda.enabled=False for unattended runs."
        )
        return None

    attempts = 0
    max_attempts = max(1, getattr(ecfg, "max_attempts", 2))
    timeout = getattr(ecfg, "timeout", DEFAULT_TIMEOUT)

    while attempts < max_attempts:
        attempts += 1
        console.print(f"\n[dim]Writing an analysis script ({attempts}/{max_attempts}) ...[/dim]")
        try:
            code = generate_eda_code(cfg, task_desc, data_preview)
        except Exception as e:
            logger.warning(f"EDA script generation failed: {e}")
            console.print(f"[yellow]Could not write the analysis: {e}[/yellow]")
            continue

        console.print("[dim]Running it against your data ...[/dim]")
        ok, output = run_eda_script(cfg, code, timeout=timeout)
        if not ok:
            console.print(Panel(output, title="[red]The analysis failed[/red]",
                                border_style="red", expand=False))
            logger.warning("EDA execution failed; retrying with a fresh script.")
            continue

        report = output.strip()
        if not report:
            console.print("[yellow]The analysis produced no output.[/yellow]")
            continue

        path = save_report(cfg, report)
        verdict = review_report(cfg, report, path, console)
        if verdict == "quit":
            return None
        if verdict == "redo":
            console.print("[dim]Rerunning the analysis ...[/dim]")
            continue

        plan = review_data_plan(cfg, task_desc, report, console)
        if plan is None:
            return None

        (path.parent / REPORT_NAME).write_text(
            report + "\n\n---\n\n## Confirmed data plan\n\n" + plan, encoding="utf-8"
        )
        console.print("[green]Data plan approved.[/green]")
        logger.info("EDA gate complete; data plan folded into the task description.")
        return path, plan

    console.print("[yellow]Continuing without an EDA report.[/yellow]")
    logger.warning(f"EDA gate gave up after {attempts} attempt(s).")
    return None
