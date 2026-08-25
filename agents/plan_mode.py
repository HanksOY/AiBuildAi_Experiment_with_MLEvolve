"""Interactive run planning with a user confirmation gate.

Sits between workspace preparation and the start of the search, and does four
things before any compute is spent:

1. Surfaces the ``clean_task_desc`` rewrite as a before/after diff. That step
   sends the user's ``goal`` to an LLM and hands agents the rewritten version;
   without this the user never sees what changed.
2. Asks only the clarifying questions the task description does not already
   answer, each with numbered options plus free-text entry.
3. Drafts a run-level plan and lets the user revise it in free text until they
   are happy with it.
4. Folds the answers and the approved plan back into the task description, so
   what the user agreed to is what the coding agents actually receive.

Controlled by ``plan_mode.require_confirmation``. A headless session cannot
answer the prompts, so it aborts rather than starting an unapproved run.
"""

import difflib
import json
import logging
import re
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax

logger = logging.getLogger("MLEvolve")

from agents.prompt_commands import ask_command, chat_prompt, render_help
from agents.setup_agent import (
    OVERRIDABLE_KEYS,
    coerce_value,
    review_settings_for_plan,
    settings_summary,
)

# Returned by a step the user stepped back out of, so the caller reruns the
# previous one instead of treating it as a normal result.
BACK = "__back__"

MIN_QUESTIONS = 3
MAX_QUESTIONS = 5

QUESTION_SYSTEM_PROMPT = (
    "You are a senior ML engineer about to spend many hours of compute on an automated "
    "model-search run. Before starting, you get to ask the task owner a few questions."
)

QUESTION_TEMPLATE = """Read the task below and decide what you still need to know.

Ask between {min_q} and {max_q} questions. Two kinds are worth asking:

1. Genuine ambiguities the task description does not answer.
2. Consequential choices where more than one defensible option exists and the owner's
   preference should decide it — feature representation, which model families to
   prioritise, how to trade recall against precision, how much compute to spend on
   tuning versus breadth, whether to ship a single model or an ensemble.

Never ask about something already specified (metric, split, feature count, file formats).
Every question must change what the run actually does; skip anything cosmetic.

Reply with JSON only, in this exact shape:

{{
  "questions": [
    {{
      "question": "the question, one sentence",
      "why": "one short line on what changes depending on the answer",
      "options": ["option 1", "option 2", "option 3"],
      "recommended": 0
    }}
  ]
}}

Each question needs 3 to 5 concrete, mutually exclusive options. "recommended" is the
0-based index of the option you would choose and must be a valid index.

The run is already configured as follows. Do not ask about anything settled here
unless the task description contradicts it; if you do ask about one of these, state
the current value in the question so the owner knows what they are changing.

{settings}

# Task
{task_desc}
"""

CONFLICT_SYSTEM_PROMPT = (
    "You reconcile a task owner's stated intent against the configured settings of a run."
)

CONFLICT_TEMPLATE = """The task owner answered some questions. Check those answers against the
run's current settings and report only genuine contradictions — cases where following the
answer would require a setting to change.

Ignore anything that merely elaborates or is consistent with a setting. Most answers will
not conflict; an empty list is the normal result.

Reply with JSON only:

{{
  "conflicts": [
    {{
      "config_key": "one of the setting keys listed below, exactly as written",
      "answer": "the owner's answer, quoted",
      "suggested_value": <the concrete new value implied by the answer, correct JSON type>,
      "explanation": "one short line on the contradiction"
    }}
  ]
}}

If nothing conflicts, reply exactly: {{"conflicts": []}}

# Current settings
{settings}

# The owner's answers
{answers}
"""

PLAN_SYSTEM_PROMPT = (
    "You are a senior ML engineer writing the plan for an automated model-search run. "
    "Produce a short, concrete plan a reviewer can sanity-check before compute is spent. "
    "Be specific about validation, since a flawed protocol invalidates the whole run."
)

PLAN_TEMPLATE = """Write the plan for this machine learning task.

Cover exactly these sections, in order, using markdown headings:

## Understanding
Two or three sentences: what is being predicted, from what, and how success is measured.

## Data handling
Key preprocessing steps. Explicitly state any leakage risks you can see and how the plan
avoids them, including where each preprocessing step is fitted relative to the splits.

## Validation protocol
The split strategy and the metric used to rank candidate solutions.

## Modelling approach
Three to five candidate model families worth trying, ordered by expected value, with one
line of justification each.

## Risks
The two or three most likely ways this run produces a misleading or useless result.

Keep the whole plan under 400 words. Do not write code.
{clarifications}{feedback}
# Task
{task_desc}
"""


ASK_BACK_SYSTEM = (
    "You are interviewing the owner of a machine learning task before a long run starts. "
    "They may answer your question, or they may ask you something about it first."
)

ASK_BACK_TEMPLATE = """You asked the task owner this question:

  {question}
  Options: {options}

They replied:

  {message}

Decide whether that reply answers your question, or asks you something about it.

Reply with JSON only:

{{
  "intent": "answer" | "question",
  "reply": "your response, only when intent is question"
}}

Choose "answer" when the reply names one of the options, or states any preference,
decision or instruction — however informally, and even if it also adds commentary.
"Option 1 please", "keep the raw counts", "whatever you think best", and "do 2 but
watch the imbalance" are all answers.

Choose "question" only when they are clearly asking you something rather than telling
you something: they want clarification, an example, a definition, or the consequences
of an option, and have not indicated a choice. Then "reply" must actually help —
explain the choice in plain terms with a concrete example, in three or four sentences.
"""

CHAT_SYSTEM = (
    "You are the planning assistant for an automated machine learning run. The plan has "
    "been drafted and the operator is reviewing it. You either answer their question or "
    "recognise that they are asking for the plan to change."
)

CHAT_TEMPLATE = """The operator typed something while reviewing the plan below. Decide what it is.

Reply with JSON only:

{{
  "intent": "question" | "revision",
  "reply": "your answer, two or three sentences, only when intent is question"
}}

Use "question" when they are asking about the plan, the task, the run, or you — anything
that should be answered without altering the plan. Use "revision" only when they are
asking for the plan itself to change. When unsure, choose "question": answering is
harmless, silently rewriting the plan is not.

# The plan under review
{plan}

# What the operator typed
{message}
"""


def interpret_answer_input(cfg, message: str, question: dict,
                           force_question: bool = False) -> tuple[str, str]:
    """Decide whether the user answered the question or asked about it.

    Without this, "I don't get what you are asking" is stored as the answer and
    the run proceeds on it.
    """
    from llm import generate
    from utils.response import extract_review

    # "option 2", "2)", "#3" — unambiguous, so resolve without a model call.
    picked = re.match(r"^\s*(?:option|opt|#)?\s*([1-9])\b", message, re.IGNORECASE)
    if picked:
        index = int(picked.group(1))
        if 1 <= index <= len(question["options"]):
            return "answer", question["options"][index - 1]

    try:
        raw = generate(
            prompt={
                "system": ASK_BACK_SYSTEM,
                "user": ASK_BACK_TEMPLATE.format(
                    question=question["question"],
                    options="; ".join(question["options"]),
                    message=message,
                ),
            },
            cfg=cfg,
            stage="feedback",
        )
        data = extract_review(raw)
    except Exception as e:
        logger.warning(f"Could not interpret the reply: {e}")
        return "answer", ""

    if not isinstance(data, dict):
        return "answer", ""
    intent = str(data.get("intent", "answer")).lower()
    reply = str(data.get("reply", "")).strip()
    if intent == "question" and reply:
        return "question", reply
    return "answer", reply


def interpret_review_input(cfg, message: str, plan: str,
                           force_question: bool = False) -> tuple[str, str]:
    """Classify review input as a question or a revision request.

    Without this, anything typed is taken as a revision — so asking a question
    silently rewrites the plan around it.
    """
    from llm import generate
    from utils.response import extract_review

    try:
        raw = generate(
            prompt={"system": CHAT_SYSTEM,
                    "user": CHAT_TEMPLATE.format(plan=plan, message=message)},
            cfg=cfg,
            stage="feedback",
        )
        data = extract_review(raw)
    except Exception as e:
        logger.warning(f"Could not interpret review input: {e}")
        return "revision", ""

    if not isinstance(data, dict):
        return "revision", ""
    intent = str(data.get("intent", "revision")).lower()
    reply = str(data.get("reply", "")).strip()
    if force_question:
        return "question", reply or "I could not answer that."
    if intent == "question" and reply:
        return "question", reply
    return "revision", reply


# ── task description handling ────────────────────────────────────────────────

def _as_text(task_desc) -> str:
    """Task descriptions arrive as a str (desc_file) or a dict (goal/eval).

    load_task_desc returns file contents verbatim when ``desc_file`` is set, but
    ``{"Task goal": ..., "Task evaluation": ...}`` when built from the
    ``goal``/``eval`` config keys. Render both to the markdown agents see.
    """
    if isinstance(task_desc, str):
        return task_desc
    if task_desc is None:
        return ""
    from llm import compile_prompt_to_md

    return compile_prompt_to_md(task_desc)


def _task_desc_diff(original, cleaned) -> str | None:
    """Unified diff of the task description rewrite, or None if unchanged."""
    original, cleaned = _as_text(original), _as_text(cleaned)
    if original.strip() == cleaned.strip():
        return None
    diff = difflib.unified_diff(
        original.splitlines(),
        cleaned.splitlines(),
        fromfile="your goal (config.yaml)",
        tofile="what agents actually receive",
        lineterm="",
        n=2,
    )
    return "\n".join(diff)


def show_task_diff(original, cleaned, console: Console) -> bool:
    """Display the clean_task_desc rewrite. Returns True if anything changed."""
    diff = _task_desc_diff(original, cleaned)
    if diff is None:
        console.print(
            Panel(
                "Your task description was passed through unchanged.",
                title="[bold]Task description[/bold]",
                border_style="green",
                expand=False,
            )
        )
        return False

    console.print(
        Panel(
            Syntax(diff, "diff", theme="ansi_dark", word_wrap=True),
            title="[bold yellow]Your task description was rewritten before the agents saw it[/bold yellow]",
            subtitle="[dim]red = removed from your goal, green = added[/dim]",
            border_style="yellow",
            expand=False,
        )
    )
    logger.info("Task description was modified by clean_task_desc; diff shown to user.")
    return True


# ── clarifying questions ─────────────────────────────────────────────────────

def _parse_questions(raw: str) -> list[dict]:
    """Parse and sanity-check the question JSON. Returns [] on anything malformed."""
    from utils.response import extract_review

    try:
        data = extract_review(raw)
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(f"Could not parse clarifying questions: {e}")
        return []

    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list):
        return []

    clean: list[dict] = []
    for q in questions[:MAX_QUESTIONS]:
        if not isinstance(q, dict):
            continue
        text = str(q.get("question", "")).strip()
        options = [str(o).strip() for o in q.get("options", []) if str(o).strip()]
        if not text or len(options) < 2:
            continue
        rec = q.get("recommended", 0)
        rec = rec if isinstance(rec, int) and 0 <= rec < len(options) else 0
        clean.append(
            {"question": text, "why": str(q.get("why", "")).strip(), "options": options, "recommended": rec}
        )
    return clean


def generate_questions(cfg, task_desc) -> list[dict]:
    """Ask the model what it still needs to know. Empty list is a valid outcome."""
    from llm import generate

    try:
        raw = generate(
            prompt={
                "system": QUESTION_SYSTEM_PROMPT,
                "user": QUESTION_TEMPLATE.format(
                    min_q=MIN_QUESTIONS,
                    max_q=MAX_QUESTIONS,
                    task_desc=_as_text(task_desc),
                    settings=settings_summary(cfg),
                ),
            },
            cfg=cfg,
            stage="code",
        )
    except Exception as e:
        logger.warning(f"Clarifying-question generation failed: {e}")
        return []
    return _parse_questions(raw)


# ── conflicts between the owner's answers and the run's settings ─────────────

def detect_conflicts(cfg, answers: list[tuple[str, str]]) -> list[dict]:
    """Find answers that contradict a current setting. Empty list is the norm."""
    if not answers:
        return []

    from omegaconf import OmegaConf
    from llm import generate
    from utils.response import extract_review

    rendered = "\n".join(f"- Q: {q}\n  A: {a}" for q, a in answers)
    try:
        raw = generate(
            prompt={
                "system": CONFLICT_SYSTEM_PROMPT,
                "user": CONFLICT_TEMPLATE.format(settings=settings_summary(cfg), answers=rendered),
            },
            cfg=cfg,
            stage="code",
        )
        data = extract_review(raw)
    except Exception as e:
        logger.warning(f"Conflict detection failed: {e}")
        return []

    found = data.get("conflicts") if isinstance(data, dict) else None
    if not isinstance(found, list):
        return []

    conflicts = []
    for c in found:
        if not isinstance(c, dict):
            continue
        key = str(c.get("config_key", "")).strip()
        # Only ever touch keys we listed; a hallucinated key is dropped.
        if key not in OVERRIDABLE_KEYS:
            if key:
                logger.warning(f"Ignoring conflict on non-overridable key: {key}")
            continue
        current = OmegaConf.select(cfg, key)
        new_value = coerce_value(c.get("suggested_value"), current)
        if new_value is None or new_value == current:
            continue
        conflicts.append(
            {
                "key": key,
                "current": current,
                "new": new_value,
                "answer": str(c.get("answer", "")).strip(),
                "explanation": str(c.get("explanation", "")).strip(),
            }
        )
    return conflicts


def resolve_conflicts(cfg, conflicts: list[dict], console: Console) -> list[tuple[str, str]]:
    """Put each conflict to the user and apply their choice. Returns notes for the agents."""
    if not conflicts:
        return []

    from omegaconf import OmegaConf

    console.print(
        f"\n[bold yellow]{len(conflicts)} of your answers conflict with the current settings.[/bold yellow]"
    )

    notes: list[tuple[str, str]] = []
    for i, c in enumerate(conflicts, 1):
        body = [
            f"You answered: [bold]{c['answer']}[/bold]",
            f"Config says:  [bold]{c['key']} = {c['current']}[/bold]",
        ]
        if c["explanation"]:
            body.append(f"[dim]{c['explanation']}[/dim]")
        body += [
            "",
            f"  [cyan]1.[/cyan] Use my answer — set [bold]{c['key']} = {c['new']}[/bold] for this run"
            "  [green](recommended)[/green]",
            f"  [cyan]2.[/cyan] Keep the config value ([bold]{c['current']}[/bold]) and ignore my answer",
        ]
        console.print(
            Panel("\n".join(body), title=f"[bold]Conflict {i} of {len(conflicts)}[/bold]",
                  border_style="yellow", expand=False)
        )

        try:
            reply = Prompt.ask("  Which wins", choices=["1", "2"], default="1", console=console).strip()
        except (EOFError, KeyboardInterrupt):
            reply = "2"
            console.print("\n[yellow]Keeping the config value.[/yellow]")

        if reply == "1":
            OmegaConf.update(cfg, c["key"], c["new"], merge=False)
            console.print(f"  [green]->[/green] {c['key']} = {c['new']}\n")
            logger.info(f"Plan gate: user overrode {c['key']} {c['current']} -> {c['new']}")
            notes.append((f"Setting {c['key']}", f"changed from {c['current']} to {c['new']} on the owner's instruction"))
        else:
            console.print(f"  [green]->[/green] keeping {c['key']} = {c['current']}\n")
            logger.info(f"Plan gate: user kept {c['key']} = {c['current']} despite answer '{c['answer']}'")
            notes.append((f"Setting {c['key']}", f"stays at {c['current']}; the owner's earlier answer does not apply"))

    return notes


def ask_questions(cfg, questions: list[dict], console: Console) -> list[tuple[str, str]]:
    """Put each question to the user. Returns the (question, answer) pairs they answered."""
    if not questions:
        console.print(
            "[dim]No clarifying questions — the task description already covers what's needed.[/dim]"
        )
        return []

    console.print(
        f"\n[bold]{len(questions)} question(s) before planning[/bold] "
        "[dim](number to pick, or type your own answer; Enter = recommended, 's' = skip)[/dim]"
    )

    answers: list[tuple[str, str]] = []
    for idx, q in enumerate(questions, 1):
        lines = [f"[bold]{q['question']}[/bold]"]
        if q["why"]:
            lines.append(f"[dim]{q['why']}[/dim]")
        lines.append("")
        for i, opt in enumerate(q["options"]):
            tag = "  [green](recommended)[/green]" if i == q["recommended"] else ""
            lines.append(f"  [cyan]{i + 1}.[/cyan] {opt}{tag}")
        console.print(
            Panel("\n".join(lines), title=f"[bold]Question {idx} of {len(questions)}[/bold]",
                  border_style="cyan", expand=False)
        )

        # Stay on this question until it is actually answered or skipped: asking
        # about it must not be recorded as the answer to it.
        # Stay on this question until it is answered, skipped, or stepped back
        # from. Asking about it must never be recorded as the answer to it.
        answer = None
        while answer is None:
            try:
                command = ask_command(
                    console,
                    hint="a number · your own answer · /ask about it · /skip · /back · Enter = recommended",
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Skipping remaining questions.[/yellow]")
                return answers

            if command.kind == "skip":
                console.print("  [dim]skipped[/dim]")
                break
            if command.kind == "quit":
                return answers
            if command.kind == "back":
                return BACK
            if command.kind == "empty":
                answer = q["options"][q["recommended"]]
            elif command.kind == "ask":
                _, response = interpret_answer_input(cfg, command.text, q, force_question=True)
                console.print(Panel(response, border_style="cyan", expand=False))
            elif command.kind == "change":
                answer = command.text
            elif command.text.isdigit() and 1 <= int(command.text) <= len(q["options"]):
                answer = q["options"][int(command.text) - 1]
            else:
                intent, response = interpret_answer_input(cfg, command.text, q)
                if intent == "question":
                    console.print(Panel(response, border_style="cyan", expand=False))
                    continue  # same question again
                answer = response or command.text

        if answer is not None:
            console.print(f"  [green]->[/green] {answer}")
            answers.append((q["question"], answer))

    return answers


# ── plan drafting and review ─────────────────────────────────────────────────

def _format_clarifications(answers: list[tuple[str, str]]) -> str:
    if not answers:
        return ""
    lines = ["\n# Answers from the task owner (these take precedence)"]
    lines += [f"- {q}\n  -> {a}" for q, a in answers]
    return "\n".join(lines) + "\n"


def _format_feedback(feedback: list[str]) -> str:
    if not feedback:
        return ""
    lines = ["\n# Revision requests from the task owner (apply all of these)"]
    lines += [f"- {f}" for f in feedback]
    return "\n".join(lines) + "\n"


def generate_plan(cfg, task_desc, answers=None, feedback=None) -> str:
    """Draft the run-level plan. Returns the plan text, or an error note on failure."""
    from llm import generate

    try:
        return generate(
            prompt={
                "system": PLAN_SYSTEM_PROMPT,
                "user": PLAN_TEMPLATE.format(
                    task_desc=_as_text(task_desc),
                    clarifications=_format_clarifications(answers or []),
                    feedback=_format_feedback(feedback or []),
                ),
            },
            cfg=cfg,
            stage="code",
        ).strip()
    except Exception as e:
        logger.warning(f"Plan generation failed: {e}")
        return f"_Plan generation failed ({e}). Review the task description above instead._"


def review_plan(cfg, cleaned_task_desc, answers, console: Console) -> tuple[str, list[str]] | None:
    """Show the plan and take free-text revisions until approved.

    Returns (approved_plan, feedback_given), or None if the user quit.
    """
    feedback: list[str] = []
    console.print("\n[dim]Drafting the run plan ...[/dim]")
    plan = generate_plan(cfg, cleaned_task_desc, answers)

    show_plan = True
    while True:
        # Only re-print when it actually changed; answering a question should not
        # scroll the whole plan past again.
        if show_plan:
            console.print(
                Panel(
                    Markdown(plan),
                    title="[bold]Proposed plan[/bold]",
                    subtitle=f"[dim]{cfg.exp_name}[/dim]",
                    border_style="cyan",
                    expand=False,
                )
            )
        show_plan = True
        try:
            command = ask_command(
                console,
                hint="/go start the run · /change <what to change> · /ask a question · /back · /quit",
            )
        except (EOFError, KeyboardInterrupt):
            console.print("[red]Aborted at the plan gate.[/red]")
            return None

        if command.kind in {"go", "empty"}:
            return plan, feedback
        if command.kind == "quit":
            return None
        if command.kind == "back":
            return BACK
        if command.kind == "ask":
            _, response = interpret_review_input(cfg, command.text, plan, force_question=True)
            console.print(Panel(response, border_style="cyan", expand=False))
            show_plan = False
            continue

        request = command.text
        if command.kind == "raw":
            # Distinguish a question from a change request; treating everything
            # as a revision means asking "who are you" rewrites the plan.
            intent, response = interpret_review_input(cfg, request, plan)
            if intent == "question":
                console.print(Panel(response, border_style="cyan", expand=False))
                show_plan = False
                continue

        feedback.append(request)
        logger.info(f"Plan revision requested: {request}")
        console.print("[dim]Redrafting with your feedback ...[/dim]")
        plan = generate_plan(cfg, cleaned_task_desc, answers, feedback)


# ── output ───────────────────────────────────────────────────────────────────

def build_augmented_task_desc(cleaned_task_desc, answers, plan: str) -> str:
    """Fold the answers and approved plan into what the agents receive.

    Without this the plan is decoration: the user refines it and the coding
    agents never see any of it.
    """
    parts = [_as_text(cleaned_task_desc)]

    if answers:
        parts.append(
            "\n\n"
            + "=" * 60
            + "\n**CLARIFICATIONS FROM THE TASK OWNER**\n"
            + "=" * 60
            + "\nThese answers were given explicitly and override any conflicting\n"
            "assumption you might otherwise make.\n\n"
            + "\n".join(f"- {q}\n  {a}" for q, a in answers)
        )

    if plan:
        parts.append(
            "\n\n"
            + "=" * 60
            + "\n**APPROVED PLAN**\n"
            + "=" * 60
            + "\nThe task owner reviewed and approved this plan. Follow it unless you\n"
            "find a concrete reason not to, and say so if you deviate.\n\n"
            + plan
        )

    return "".join(parts)


def save_plan_transcript(cfg, diff: str | None, answers, plan: str) -> None:
    """Write the planning session to logs/plan.md so the run stays auditable."""
    try:
        log_dir = cfg.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"# Plan — {cfg.exp_name}\n"]
        if diff:
            lines.append("## Task description rewrite\n\n```diff\n" + diff + "\n```\n")
        if answers:
            lines.append("## Clarifications\n")
            lines += [f"**{q}**\n\n> {a}\n" for q, a in answers]
        lines.append("## Approved plan\n\n" + plan + "\n")
        (log_dir / "plan.md").write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Plan transcript written to {log_dir / 'plan.md'}")
    except Exception as e:
        logger.warning(f"Could not write plan transcript: {e}")


# ── entry point ──────────────────────────────────────────────────────────────

def run_plan_gate(cfg, original_task_desc, cleaned_task_desc):
    """Run the planning session.

    Returns the augmented task description to run with, or None to abort.
    Never raises: a failure here should not take down an otherwise fine run.
    """
    console = Console()
    pcfg = getattr(cfg, "plan_mode", None)

    if not getattr(pcfg, "require_confirmation", False):
        return _as_text(cleaned_task_desc)

    # Check before drafting anything: a headless session (cron, nohup, piped
    # stdin) cannot answer, and proceeding would recreate the very problem this
    # gate prevents — a run starting without consent.
    if not sys.stdin.isatty():
        console.print(
            "\n[red]Plan confirmation is enabled, but this session has no interactive terminal.[/red]\n"
            "[yellow]Either run from a terminal, or set "
            "plan_mode.require_confirmation=False for unattended runs.[/yellow]"
        )
        logger.warning(
            "plan_mode.require_confirmation is true but stdin is not a TTY; aborting "
            "rather than starting a run the user has not approved."
        )
        return None

    diff = _task_desc_diff(original_task_desc, cleaned_task_desc)
    if getattr(pcfg, "show_task_diff", True):
        show_task_diff(original_task_desc, cleaned_task_desc, console)

    # Phase machine, so /back steps to the previous phase instead of only
    # working inside one. Questions -> plan review -> settings review.
    questions = None
    answers: list[tuple[str, str]] = []
    plan = ""
    phase = "questions"

    while phase != "done":
        if phase == "questions":
            if questions is None:
                console.print("\n[dim]Checking whether anything needs clarifying ...[/dim]")
                questions = generate_questions(cfg, cleaned_task_desc)
            answers = ask_questions(cfg, questions, console)
            if answers is BACK:
                console.print("[dim]Nothing before this step — starting from the questions.[/dim]")
                answers = []
                continue
            # An answer may contradict a configured setting. Never silently pick
            # a winner: put the conflict to the user and apply their choice.
            if answers:
                console.print("[dim]Checking your answers against the run settings ...[/dim]")
                answers += resolve_conflicts(cfg, detect_conflicts(cfg, answers), console)
            phase = "plan"

        elif phase == "plan":
            reviewed = review_plan(cfg, cleaned_task_desc, answers, console)
            if reviewed is BACK:
                phase = "questions"
                continue
            if reviewed is None:
                console.print("[red]Stopped. Nothing was executed.[/red]")
                logger.info("User rejected the plan; exiting before the search started.")
                return None
            plan, feedback = reviewed
            phase = "settings"

        elif phase == "settings":
            # Only now does a plan exist to tune the run's parameters against.
            notes = review_settings_for_plan(cfg, plan, console)
            if notes is None:
                phase = "plan"
                continue
            answers += notes
            phase = "done"

    save_plan_transcript(cfg, diff, answers, plan)
    logger.info(f"Plan approved after {len(feedback)} revision(s); {len(answers)} clarification(s).")
    console.print("[green]Plan approved — starting the search.[/green]")

    return build_augmented_task_desc(cleaned_task_desc, answers, plan)


def _demo() -> None:
    """Exercise the real gate with canned content, so commands can be tested fast.

    Question drafting, plan drafting and setting recommendations are stubbed, so
    the whole flow is instant and free. Everything else is the real code path —
    the same parser, the same phase machine, the same /back transitions — so
    what you type here behaves exactly as it will in a run. /ask still calls the
    model, because answering you is the thing being tested.
    """
    import agents.setup_agent as setup_agent
    from config import _load_cfg

    console = Console()
    cfg = _load_cfg(use_cli_args=False)
    cfg.plan_mode.require_confirmation = True

    canned_questions = [
        {
            "question": "How should the ICD-9 count columns be represented as features?",
            "why": "Changes preprocessing and can swing PRAUC, especially for linear models.",
            "options": ["Use raw integer counts as-is", "Binarize to presence/absence",
                        "Apply log(1+count)", "Try several and pick by cross-validation"],
            "recommended": 3,
        },
        {
            "question": "Should the submission come from one model or an ensemble?",
            "why": "An ensemble needs blending logic and changes how the threshold is chosen.",
            "options": ["Single best by CV PRAUC", "Ensemble of the top 3",
                        "Try both, submit whichever scores higher"],
            "recommended": 2,
        },
    ]
    canned_plan = (
        "## Understanding\nPredict meningitis from ICD-9 counts; ranked by mean "
        "out-of-fold CV PRAUC.\n\n## Validation protocol\nStratified 20% hold-out, "
        "then stratified 5-fold CV on the remainder, seed 42.\n\n## Modelling approach\n"
        "1. Logistic regression  2. LightGBM  3. Extra-trees\n\n## Risks\n"
        "Residual leakage; hold-out contamination; high variance at 0.46% prevalence."
    )

    plan_mode = sys.modules[__name__]
    plan_mode.generate_questions = lambda *a, **k: canned_questions
    plan_mode.generate_plan = lambda *a, **k: canned_plan
    plan_mode.detect_conflicts = lambda *a, **k: []
    setup_agent.recommend_settings = lambda *a, **k: []

    console.print(
        Panel(
            "Canned questions and plan, so nothing waits on generation.\n\n"
            "Try: a number · [cyan]/ask[/cyan] something · [cyan]/back[/cyan] · "
            "[cyan]/skip[/cyan] · [cyan]/help[/cyan] · [cyan]/go[/cyan] · [cyan]/quit[/cyan]\n"
            "[dim]/back from the settings table returns to the plan, and from the plan "
            "returns to the questions.[/dim]",
            title="[bold]Plan gate demo[/bold]",
            border_style="green",
            expand=False,
        )
    )

    task = "Predict meningitis from per-patient ICD-9 diagnosis code counts."
    result = run_plan_gate(cfg, task, task)
    if result is None:
        console.print("\n[red]Gate returned None — the run would not start.[/red]")
    else:
        console.print(f"\n[green]Gate approved.[/green] Task description is now "
                      f"{len(result)} chars (was {len(task)}).")


if __name__ == "__main__":
    _demo()
