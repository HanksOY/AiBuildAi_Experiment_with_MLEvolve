"""First-run setup: get credentials working, then build the rest of the config.

Bootstrapping order matters. Nothing else in this program works without a
reachable model, so credentials come first and are verified with a live call.
Once that works the agent can inspect the dataset and draft the task description
itself, which is far more useful than asking the user to fill in a form.

Only three things genuinely need asking beyond credentials — ``data_dir``,
``goal`` and ``eval``. Everything else has a workable default or can be
detected, and can later be adjusted from the plan gate.

Run explicitly:
    python -m agents.setup_agent
It also triggers automatically from run.py when no API key is configured.
"""

import logging
import os
import shutil
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

logger = logging.getLogger("MLEvolve")

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

# Settings this agent may change. Anything outside this list is never written
# to, so a hallucinated key cannot reach credentials or paths.
OVERRIDABLE_KEYS = {
    "agent.time_limit": "wall-clock budget for the search, in seconds",
    "agent.steps": "number of search steps",
    "agent.initial_drafts": "how many independent drafts to start from",
    "agent.search.parallel_search_num": "solutions executed in parallel",
    "agent.search.num_gpus": "GPUs available",
    "agent.search.num_drafts": "draft branches allowed at the root",
    "exec.timeout": "per-solution execution timeout, in seconds",
    "agent.check_data_leakage": "run the data-leakage audit",
    "agent.use_global_memory": "reuse knowledge across search nodes",
    "agent.use_fusion": "allow cross-branch fusion once the search stagnates",
    "agent.search.fusion_min_time_hours": "hours that must elapse before fusion may trigger",
    "cpu_number": "CPU cores the run may use",
}

# One search step is an LLM generation plus a code review plus an execution.
# Observed on this repo: roughly 15 minutes. Anything under a few minutes per
# step is not reachable, so a step budget implying less is simply wrong.
MIN_SECONDS_PER_STEP = 180
TYPICAL_SECONDS_PER_STEP = 600

SETTINGS_SYSTEM_PROMPT = (
    "You tune the parameters of an automated ML model-search run so they match the plan "
    "that was just approved. You are conservative: you only propose a change when the "
    "plan clearly implies the current value is wrong."
)

SETTINGS_TEMPLATE = """The task owner approved the plan below. Check the run's settings against it
and propose changes only where the plan clearly implies the current value is wrong.

Weigh these in particular:
- Time and budget. Does agent.time_limit leave room for the work the plan describes? Does
  exec.timeout suit the heaviest single model the plan names? Are agent.steps and
  agent.initial_drafts enough to cover the model families listed, without being far more
  than the plan needs?
- Breadth. Does agent.search.num_drafts match how many distinct approaches the plan wants
  explored in parallel?
- Anything the plan relies on that a setting has disabled.

Do not propose changes for style or preference, and do not touch a setting the plan says
nothing about. Proposing nothing is a good answer when the settings already fit.

Reply with JSON only:

{{
  "recommendations": [
    {{
      "config_key": "one of the keys below, exactly as written",
      "suggested_value": <new value, correct JSON type>,
      "reason": "one short line tying this to something specific in the plan"
    }}
  ]
}}

If the settings already fit the plan, reply exactly: {{"recommendations": []}}

# Current settings
{settings}

# Approved plan
{plan}
"""

# base_url + a sensible default model pair per provider.
PROVIDERS = [
    ("GLM / BigModel", "https://open.bigmodel.cn/api/paas/v4", "glm-5.2", "glm-4.5-air"),
    ("OpenAI", "https://api.openai.com/v1", "gpt-5.2", "gpt-5.2-mini"),
    ("DeepSeek", "https://api.deepseek.com", "deepseek-chat", "deepseek-chat"),
    ("Gemini", "", "gemini-3-pro-preview", "gemini-3-flash"),
]

HELP = {
    "provider": "Which API serves your model. Pick 'Other' for a self-hosted or proxy endpoint.",
    "model": "The code model writes solutions; the feedback model reads results. "
             "The feedback model can be smaller and cheaper.",
    "api_key": "Written to .env, which is git-ignored. It never goes into config.yaml.",
    "data_dir": "Folder holding your input files. It is copied or symlinked into the "
                "run workspace as ./input/, which is what generated code reads.",
    "goal": "What to build, plus anything the model cannot infer from the files alone: "
            "traps, required preprocessing, constraints like CPU-only.",
    "eval": "How solutions are ranked, and what must be written out. The ranking metric "
            "matters most — it drives the entire search.",
    "cpu": "Cores the run may use. Also caps how many solutions run in parallel. "
           "Leaving one core free keeps the machine responsive.",
    "gpu": "0 means solutions must stay CPU-friendly — no deep learning. This shapes "
           "which model families are worth trying at all.",
}

DRAFT_SYSTEM = (
    "You write task specifications for an automated ML system. You are precise, and you "
    "call out data traps explicitly because the downstream agent only sees your text."
)

DRAFT_TEMPLATE = """Write a task specification from the user's description and a preview of their data.

Return two markdown sections and nothing else:

## GOAL
What to build and predict. State the target column, which columns to drop, and any
constraint the user gave. Call out anything in the preview that would break naive code:
misleading file extensions, mismatched schemas between files, columns that leak the
label, or sizes large enough to need dtype care. Be concrete about column names.

## EVAL
The metric solutions are ranked by, the validation split, and the required outputs.
Always require ./submission/submission.csv, and always require the final line to be
exactly: print(f"Final Validation Score: {{score}}")

Keep each section under 300 words. Do not write code.

# What the user said
{description}

# Data preview
{preview}
"""


# ── small helpers ────────────────────────────────────────────────────────────

def _ask(console: Console, prompt: str, default: str = "", help_text: str = "") -> str | None:
    """Prompt accepting free text. Returns None if the user asked to go back."""
    while True:
        answer = Prompt.ask(f"  {prompt}", default=default, console=console).strip()
        low = answer.lower()
        if low in {"?", "help"} and help_text:
            console.print(f"  [dim]{help_text}[/dim]")
            continue
        if low in {"b", "back"}:
            return None
        return answer


def _detect_cpus() -> int:
    """Usable cores, leaving one free. Beats the shipped default of 21."""
    try:
        total = len(os.sched_getaffinity(0))
    except AttributeError:
        total = os.cpu_count() or 4
    return max(1, total - 1)


def _detect_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def settings_summary(cfg) -> str:
    """The tunable settings, as `key = value  # meaning` lines."""
    from omegaconf import OmegaConf

    lines = []
    for key, meaning in OVERRIDABLE_KEYS.items():
        value = OmegaConf.select(cfg, key)
        if value is not None:
            lines.append(f"{key} = {value}  # {meaning}")
    return "\n".join(lines)


def coerce_value(value, current):
    """Cast a proposed value to the current setting's type, or None if impossible."""
    try:
        if isinstance(current, bool):
            if isinstance(value, str):
                low = value.strip().lower()
                if low in {"true", "yes", "on"}:
                    return True
                if low in {"false", "no", "off"}:
                    return False
                return None
            return bool(value)
        if isinstance(current, int):
            return int(float(value))
        if isinstance(current, float):
            return float(value)
        return type(current)(value)
    except (TypeError, ValueError):
        return None


def _collect_proposals(cfg, raw: str, json_key: str, value_key: str, note_key: str) -> list[dict]:
    """Parse proposed setting changes, dropping anything unsafe or already true."""
    from omegaconf import OmegaConf
    from utils.response import extract_review

    try:
        data = extract_review(raw)
    except Exception as e:
        logger.warning(f"Could not parse setting proposals: {e}")
        return []

    items = data.get(json_key) if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []

    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("config_key", "")).strip()
        if key not in OVERRIDABLE_KEYS:
            if key:
                logger.warning(f"Ignoring proposal for non-overridable key: {key}")
            continue
        current = OmegaConf.select(cfg, key)
        new_value = coerce_value(item.get(value_key), current)
        if new_value is None or new_value == current:
            continue
        out.append({"key": key, "current": current, "new": new_value,
                    "note": str(item.get(note_key, "")).strip()})
    return out


def check_settings_consistency(cfg) -> list[dict]:
    """Internal contradictions in the settings, found arithmetically.

    These are relationships between numbers and the machine, so they are checked
    directly rather than asked of a model — a language model reliably overlooks
    them (e.g. exec.timeout being nine times agent.time_limit).
    """
    from omegaconf import OmegaConf

    def get(key):
        return OmegaConf.select(cfg, key)

    issues = []
    time_limit, exec_timeout = get("agent.time_limit"), get("exec.timeout")
    if time_limit and exec_timeout and exec_timeout > time_limit:
        issues.append({
            "key": "exec.timeout",
            "current": exec_timeout,
            "new": max(60, int(time_limit * 0.5)),
            "note": (f"one solution may run {exec_timeout / 3600:.1f}h but the whole search only "
                     f"gets {time_limit / 3600:.1f}h; a single step could consume the entire budget"),
        })

    cpus, detected = get("cpu_number"), _detect_cpus() + 1
    if cpus and cpus > detected:
        issues.append({
            "key": "cpu_number",
            "current": cpus,
            "new": max(1, detected - 1),
            "note": f"only {detected} core(s) available on this machine",
        })

    parallel = get("agent.search.parallel_search_num")
    if parallel and cpus and parallel > cpus:
        issues.append({
            "key": "agent.search.parallel_search_num",
            "current": parallel,
            "new": max(1, cpus),
            "note": f"more parallel solutions than the {cpus} core(s) allotted",
        })

    gpus, have_gpus = get("agent.search.num_gpus"), _detect_gpus()
    if gpus and gpus > have_gpus:
        issues.append({
            "key": "agent.search.num_gpus",
            "current": gpus,
            "new": have_gpus,
            "note": f"only {have_gpus} GPU(s) detected",
        })

    # Can the step budget actually be spent in the time allowed?
    steps = get("agent.steps")
    if steps and time_limit:
        per_step = time_limit / steps
        if per_step < MIN_SECONDS_PER_STEP:
            fits = max(1, int(time_limit // TYPICAL_SECONDS_PER_STEP))
            issues.append({
                "key": "agent.steps",
                "current": steps,
                "new": fits,
                "note": (f"{steps} steps in {time_limit / 3600:.1f}h allows only {per_step:.0f}s per "
                         f"step; one step takes minutes. Either drop to ~{fits} steps or raise "
                         f"agent.time_limit to about {steps * TYPICAL_SECONDS_PER_STEP / 3600:.0f}h"),
            })

    drafts = get("agent.initial_drafts")
    if drafts and steps and drafts > steps:
        issues.append({
            "key": "agent.initial_drafts",
            "current": drafts,
            "new": max(1, steps),
            "note": f"more initial drafts than the {steps} step(s) available to run them",
        })

    max_drafts = get("agent.search.num_drafts")
    if drafts and max_drafts and drafts > max_drafts:
        issues.append({
            "key": "agent.search.num_drafts",
            "current": max_drafts,
            "new": drafts,
            "note": f"the root allows {max_drafts} draft branch(es) but {drafts} initial drafts are requested",
        })

    # Fusion only unlocks after a set number of hours; a short run never reaches it.
    if get("agent.use_fusion") and time_limit:
        fusion_after = get("agent.search.fusion_min_time_hours")
        if fusion_after and fusion_after * 3600 > time_limit:
            issues.append({
                "key": "agent.search.fusion_min_time_hours",
                "current": fusion_after,
                "new": round(max(0.25, (time_limit / 3600) / 2), 2),
                "note": (f"fusion unlocks after {fusion_after}h but the run ends at "
                         f"{time_limit / 3600:.1f}h, so it can never trigger"),
            })

    return issues


def recommend_settings(cfg, plan: str) -> list[dict]:
    """Propose settings that fit the approved plan. Empty list is a good outcome."""
    from llm import generate

    try:
        raw = generate(
            prompt={
                "system": SETTINGS_SYSTEM_PROMPT,
                "user": SETTINGS_TEMPLATE.format(settings=settings_summary(cfg), plan=plan),
            },
            cfg=cfg,
            stage="code",
        )
    except Exception as e:
        logger.warning(f"Setting recommendation failed: {e}")
        return []
    return _collect_proposals(cfg, raw, "recommendations", "suggested_value", "reason")


def review_settings_for_plan(cfg, plan: str, console: Console) -> list[tuple[str, str]]:
    """Show plan-derived setting recommendations and apply what the user accepts.

    Returns notes describing what was decided, so the agents see it too.
    """
    from omegaconf import OmegaConf

    console.print("\n[dim]Checking the run settings against the approved plan ...[/dim]")
    # Arithmetic contradictions first — they are certain, and a model misses them.
    proposals = check_settings_consistency(cfg)
    seen = {p["key"] for p in proposals}
    proposals += [p for p in recommend_settings(cfg, plan) if p["key"] not in seen]
    if not proposals:
        console.print("[green]Current settings already fit the plan — nothing to change.[/green]")
        return []

    table = Table(box=None, padding=(0, 2, 0, 0))
    table.add_column("#", style="cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Now")
    table.add_column("Suggested", style="green")
    table.add_column("Why", style="dim", overflow="fold")
    for i, p in enumerate(proposals, 1):
        table.add_row(str(i), p["key"], str(p["current"]), str(p["new"]), p["note"])
    console.print(
        Panel(table, title="[bold]Recommended settings for this plan[/bold]",
              border_style="yellow", expand=False)
    )

    console.print(
        "\n  [dim]'a' = apply all (recommended) · 'k' = keep everything as-is · "
        "or list the numbers to apply, e.g. '1 3'[/dim]"
    )
    try:
        reply = Prompt.ask("  >", default="a", show_default=False, console=console).strip().lower()
    except (EOFError, KeyboardInterrupt):
        reply = "k"

    if reply in {"k", "keep", "n", "no"}:
        chosen = []
    elif reply in {"a", "all", "y", "yes", ""}:
        chosen = proposals
    else:
        picked = {int(t) for t in reply.replace(",", " ").split() if t.isdigit()}
        chosen = [p for i, p in enumerate(proposals, 1) if i in picked]

    notes = []
    for p in proposals:
        if p in chosen:
            OmegaConf.update(cfg, p["key"], p["new"], merge=False)
            console.print(f"  [green]->[/green] {p['key']} = {p['new']}")
            logger.info(f"Plan settings: {p['key']} {p['current']} -> {p['new']} ({p['note']})")
            notes.append((f"Setting {p['key']}", f"set to {p['new']} to match the plan"))
        else:
            console.print(f"  [dim]->[/dim] {p['key']} stays at {p['current']}")
            logger.info(f"Plan settings: kept {p['key']} = {p['current']}")

    if not chosen:
        console.print("  [dim]No changes applied.[/dim]")
    return notes


def _yaml_block(text: str, indent: str = "  ") -> str:
    """Render text as a YAML literal block body."""
    lines = (text or "").strip().splitlines() or [""]
    return "\n".join(indent + ln if ln.strip() else "" for ln in lines)


# ── steps ────────────────────────────────────────────────────────────────────

def choose_provider(console: Console) -> tuple[str, str, str] | None:
    """Returns (base_url, code_model, feedback_model)."""
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    for i, (name, url, code, _fb) in enumerate(PROVIDERS, 1):
        table.add_row(f"[cyan]{i}.[/cyan]", name, f"[dim]{url or 'native SDK'}[/dim]")
    table.add_row(f"[cyan]{len(PROVIDERS) + 1}.[/cyan]", "Other", "[dim]self-hosted or proxy[/dim]")
    console.print(Panel(table, title="[bold]Which API provider?[/bold]", border_style="cyan", expand=False))

    choice = _ask(console, "Number", default="1", help_text=HELP["provider"])
    if choice is None:
        return None

    if choice.isdigit() and 1 <= int(choice) <= len(PROVIDERS):
        _name, base_url, code, fb = PROVIDERS[int(choice) - 1]
    else:
        base_url = choice if choice.startswith("http") else ""
        if not base_url:
            base_url = _ask(console, "Base URL", default="") or ""
        code, fb = "", ""

    code_model = _ask(console, "Code model (writes solutions)", default=code, help_text=HELP["model"])
    if code_model is None:
        return None
    fb_model = _ask(console, "Feedback model (reads results)", default=fb or code_model, help_text=HELP["model"])
    if fb_model is None:
        return None
    return base_url, code_model, fb_model


def verify_key(console: Console, base_url: str, model: str, api_key: str) -> bool:
    """One tiny live call, so a bad key fails here instead of ten minutes in."""
    console.print("  [dim]Checking the key with a test request ...[/dim]")
    try:
        if model.lower().startswith("gemini"):
            from google import genai

            client = genai.Client(api_key=api_key)
            client.models.generate_content(model=model, contents="ok")
        else:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=60.0)
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "ok"}], max_tokens=4
            )
    except Exception as e:
        console.print(f"  [red]Failed:[/red] {str(e)[:220]}")
        return False
    console.print("  [green]Key works.[/green]")
    return True


def collect_credentials(console: Console) -> dict | None:
    """Step 1: a verified, working model connection. Nothing else works without it."""
    console.print(
        Panel(
            "Everything else depends on a working model connection, so we start there.\n"
            "Your key is written to [bold].env[/bold], which is git-ignored.",
            title="[bold]Step 1 of 3 — model access and compute[/bold]",
            border_style="green",
            expand=False,
        )
    )
    while True:
        chosen = choose_provider(console)
        if chosen is None:
            return None
        base_url, code_model, fb_model = chosen

        api_key = _ask(console, "API key", default="", help_text=HELP["api_key"])
        if api_key is None:
            continue
        if not api_key:
            console.print("  [red]A key is required.[/red]")
            continue

        if verify_key(console, base_url, code_model, api_key):
            creds = {
                "base_url": base_url,
                "code_model": code_model,
                "feedback_model": fb_model,
                "api_key": api_key,
            }
            creds.update(collect_compute(console))
            return creds
        if not Confirm.ask("  Try again?", default=True, console=console):
            return None


def collect_data_dir(console: Console) -> tuple[str, str] | None:
    """Step 2: a real directory, plus a preview of what's in it."""
    console.print(
        Panel(
            "Point me at your data folder and I'll read it, so the task description can "
            "mention your actual columns and file quirks.",
            title="[bold]Step 2 of 3 — your data[/bold]",
            border_style="green",
            expand=False,
        )
    )
    while True:
        raw = _ask(console, "Path to your data folder", default="", help_text=HELP["data_dir"])
        if raw is None or not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_dir():
            console.print(f"  [red]Not a directory:[/red] {path}")
            continue

        files = sorted(p for p in path.iterdir() if p.is_file())
        if not files:
            console.print(f"  [yellow]No files in {path}.[/yellow]")
            if not Confirm.ask("  Use it anyway?", default=False, console=console):
                continue

        table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
        for f in files[:12]:
            table.add_row(f.name, f"[dim]{f.stat().st_size / 1e6:.1f} MB[/dim]")
        if len(files) > 12:
            table.add_row(f"[dim]... {len(files) - 12} more[/dim]", "")
        console.print(Panel(table, title=f"[bold]{len(files)} file(s)[/bold]",
                            border_style="cyan", expand=False))

        console.print("  [dim]Reading file structure ...[/dim]")
        try:
            from utils.data_preview import generate

            preview = generate(str(path))
        except Exception as e:
            logger.warning(f"Data preview failed: {e}")
            preview = "\n".join(f"- {f.name} ({f.stat().st_size / 1e6:.1f} MB)" for f in files)

        return str(path.resolve()), preview


def draft_task(console, cfg_stub, description: str, preview: str) -> tuple[str, str]:
    """Ask the model to turn a one-liner plus the data preview into goal/eval."""
    from llm import generate
    from utils.response import trim_long_string

    console.print("  [dim]Drafting the task description from your data ...[/dim]")
    try:
        text = generate(
            prompt={
                "system": DRAFT_SYSTEM,
                "user": DRAFT_TEMPLATE.format(
                    description=description, preview=trim_long_string(preview, threshold=6000, k=3000)
                ),
            },
            cfg=cfg_stub,
            stage="code",
        )
    except Exception as e:
        console.print(f"  [yellow]Drafting failed ({e}). You can write these yourself.[/yellow]")
        return description, ""

    goal, evaluation = text, ""
    upper = text.upper()
    if "## EVAL" in upper:
        idx = upper.index("## EVAL")
        goal, evaluation = text[:idx], text[idx:]
    for marker in ("## GOAL", "## Goal"):
        goal = goal.replace(marker, "")
    for marker in ("## EVAL", "## Eval"):
        evaluation = evaluation.replace(marker, "")
    return goal.strip(), evaluation.strip()


def collect_task(console: Console, cfg_stub, preview: str) -> tuple[str, str] | None:
    """Step 3: goal and eval, drafted from the data and then edited by the user."""
    console.print(
        Panel(
            "Describe in one or two sentences what you want built. I'll expand it into a "
            "full specification using what I just read from your files, and you can revise it.",
            title="[bold]Step 3 of 3 — the task[/bold]",
            border_style="green",
            expand=False,
        )
    )
    description = _ask(console, "What should this build?", default="", help_text=HELP["goal"])
    if description is None or not description:
        return None

    goal, evaluation = draft_task(console, cfg_stub, description, preview)

    while True:
        console.print(Panel(Markdown(goal or "_(empty)_"), title="[bold]goal[/bold]",
                            border_style="cyan", expand=False))
        console.print(Panel(Markdown(evaluation or "_(empty)_"), title="[bold]eval[/bold]",
                            border_style="cyan", expand=False))
        console.print(
            "\n  [dim]Type any correction and I'll redraft, or 'ok' to accept.[/dim]"
        )
        reply = Prompt.ask("  >", default="ok", show_default=False, console=console).strip()
        if reply.lower() in {"ok", "y", "yes", "go", ""}:
            return goal, evaluation
        description = f"{description}\n\nCorrection: {reply}"
        goal, evaluation = draft_task(console, cfg_stub, description, preview)


def collect_compute(console: Console) -> dict:
    """Hardware, detected and confirmed.

    Asked alongside credentials because it decides what kind of solution is even
    viable — CPU-only rules out deep learning, and core count caps parallelism.
    Search parameters (steps, budget) are deliberately NOT asked here: before a
    plan exists there is nothing to tune them against, so they stay at their
    defaults until review_settings_for_plan revisits them.
    """
    cpus, gpus = _detect_cpus(), _detect_gpus()
    console.print(
        f"\n  [dim]Detected {cpus} usable CPU core(s) and {gpus} GPU(s).[/dim]"
    )
    answer = _ask(console, "CPU cores to use", default=str(cpus), help_text=HELP["cpu"])
    if answer and answer.isdigit():
        cpus = max(1, int(answer))
    answer = _ask(console, "GPUs to use", default=str(gpus), help_text=HELP["gpu"])
    if answer and answer.isdigit():
        gpus = max(0, int(answer))

    if gpus == 0:
        console.print(
            "  [dim]No GPUs: solutions will be restricted to CPU-friendly models "
            "(gradient boosting, linear models, forests).[/dim]"
        )
    return {"cpu_number": cpus, "num_gpus": gpus}


# ── writing it out ───────────────────────────────────────────────────────────

def write_env(creds: dict) -> None:
    ENV_PATH.write_text(
        "MLEVOLVE_CODE_API_KEY={k}\nMLEVOLVE_FEEDBACK_API_KEY={k}\n".format(k=creds["api_key"]),
        encoding="utf-8",
    )


def write_config(creds: dict, data_dir: str, goal: str, evaluation: str, compute: dict) -> Path | None:
    """Patch config.yaml in place, keeping a .bak of whatever was there."""
    if not CONFIG_PATH.exists():
        return None
    backup = CONFIG_PATH.with_suffix(".yaml.bak")
    shutil.copy2(CONFIG_PATH, backup)

    text = CONFIG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    out, skipping = [], False
    # Which stage block we are inside, so model:/base_url: lines are rewritten
    # with the right value regardless of what models the file currently names.
    stage = None

    for line in lines:
        # Drop the old goal:/eval: literal blocks; they are rewritten below.
        if skipping:
            if line and not line[0].isspace():
                skipping = False
            else:
                continue
        if line.startswith("goal:"):
            out.append("goal: |")
            out.append(_yaml_block(goal))
            skipping = True
            continue
        if line.startswith("eval:"):
            out.append("eval: |")
            out.append(_yaml_block(evaluation))
            skipping = True
            continue
        if line.startswith("data_dir:"):
            out.append(f'data_dir: "{data_dir}"')
            continue
        if line.startswith("dataset_dir:"):
            out.append(f'dataset_dir: "{data_dir}"')
            continue
        if line.startswith("cpu_number:"):
            out.append(f"cpu_number: {compute['cpu_number']}")
            continue
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        if stripped in ("code:", "feedback:"):
            stage = stripped[:-1]
            out.append(line)
            continue
        if stripped and not line[0].isspace():
            stage = None  # left the agent: block entirely

        if stage and stripped.startswith("model:"):
            out.append(f"{indent}model: {creds[stage + '_model']}")
            continue
        if stage and stripped.startswith("base_url:"):
            out.append(f'{indent}base_url: "{creds["base_url"]}"')
            continue
        if stripped.startswith("num_gpus:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}num_gpus: {compute['num_gpus']}")
            continue
        out.append(line)

    CONFIG_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    return backup


def needs_setup() -> bool:
    """True when no API key is reachable, i.e. this looks like a first run."""
    if os.environ.get("MLEVOLVE_CODE_API_KEY"):
        return False
    if not ENV_PATH.exists():
        return True
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("MLEVOLVE_CODE_API_KEY=") and line.split("=", 1)[1].strip():
            return False
    return True


def run_setup() -> bool:
    """Walk the user through a working config. Returns True if one was written."""
    console = Console()
    console.print(
        Panel(
            "No API key found, so this looks like a first run.\n\n"
            "I'll get your model connection working, then read your data and draft the task "
            "description for you. Four steps.\n\n"
            "[dim]At any prompt: type freely, '?' for help, 'back' to go back, Ctrl+C to quit.[/dim]",
            title="[bold]MLEvolve setup[/bold]",
            border_style="green",
            expand=False,
        )
    )

    try:
        creds = collect_credentials(console)
        if not creds:
            return False

        # A stub config is enough for the drafting call, and lets us reuse llm.generate.
        from omegaconf import OmegaConf

        stage = {
            "model": creds["code_model"],
            "temp": 1,
            "base_url": creds["base_url"],
            "api_key": creds["api_key"],
            "max_tokens": 65536,
        }
        cfg_stub = OmegaConf.create({"agent": {"code": stage, "feedback": dict(stage, model=creds["feedback_model"])}})

        collected = collect_data_dir(console)
        if not collected:
            return False
        data_dir, preview = collected

        task = collect_task(console, cfg_stub, preview)
        if not task:
            return False
        goal, evaluation = task

    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup cancelled. Nothing was written.[/yellow]")
        return False

    write_env(creds)
    backup = write_config(creds, data_dir, goal, evaluation, creds)

    written = [f"[green]✓[/green] {ENV_PATH}  [dim](your API key, git-ignored)[/dim]",
               f"[green]✓[/green] {CONFIG_PATH}"]
    if backup:
        written.append(f"[dim]  previous config saved to {backup.name}[/dim]")
    console.print(
        Panel(
            "\n".join(written) + "\n\n[bold]Run it with:[/bold]\n  python run.py",
            title="[bold]Setup complete[/bold]",
            border_style="green",
            expand=False,
        )
    )
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run_setup() else 1)
