"""EDA and Plan steps for the dashboard, run inside WSL with the MLEvolve venv.

The dashboard server (Windows) cannot import MLEvolve, so it starts

    .venv/bin/python -m dashboard.gates

with one JSON request on stdin and reads one JSON answer from the last stdout
line that starts with RESULT_MARK. Each request is one step. The functions are
the ones the terminal gates use (agents/eda_agent.py, agents/plan_mode.py), so
the prompts and models are the same; only the questions and answers go through
the chat instead of the terminal.

Request: {"cmd": ..., "project": <project folder>, "data": <data folder>,
          "overrides": ["agent.code.model=...", ...], ...command fields}
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

RESULT_MARK = "@@GATES@@"


def make_cfg(req: dict):
    """The run config without prep_cfg's run folders: the workspace is <project>/workspace."""
    from omegaconf import OmegaConf
    from config import Config, _load_cfg

    project = Path(req["project"]).resolve()
    cfg = _load_cfg(use_cli_args=False)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([str(o) for o in req.get("overrides", [])]))
    cfg.data_dir = str(req.get("data") or project)
    cfg.desc_file = None
    cfg.goal = "dashboard"
    cfg.exp_name = project.name
    cfg.log_dir = str(project / "gate_logs")
    cfg.workspace_dir = str(project / "workspace")
    cfg = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    return cfg


def prepare_workspace(cfg) -> Path:
    """<project>/workspace/input links to the data, so ./input/ works as in a run."""
    from utils import copytree

    ws = Path(cfg.workspace_dir)
    inp = ws / "input"
    if not inp.exists() or not any(inp.iterdir()):
        inp.mkdir(parents=True, exist_ok=True)
        copytree(Path(cfg.data_dir), inp, use_symlinks=True)
    return ws


def eda_report(cfg, req: dict) -> dict:
    from agents import eda_agent
    from agents.prompts.framing import framing_of
    from utils.data_preview import generate as preview_data

    ws = prepare_workspace(cfg)
    try:
        preview = preview_data(str(ws), framing=framing_of(cfg))
    except Exception as e:  # the analysis can still run without a preview
        preview = f"(preview failed: {e})"
    attempts = max(1, int(getattr(cfg.eda, "max_attempts", 2)))
    timeout = int(getattr(cfg.eda, "timeout", eda_agent.DEFAULT_TIMEOUT))
    errors = []
    for attempt in range(1, attempts + 1):
        started = time.time()
        try:
            code = eda_agent.generate_eda_code(cfg, req["task"], preview)
        except Exception as e:
            errors.append(f"attempt {attempt}: could not write the script: {e}")
            continue
        ok, output = eda_agent.run_eda_script(cfg, code, timeout=timeout)
        if ok and output.strip():
            return {"ok": True, "report": output.strip(), "seconds": round(time.time() - started),
                    "attempts": attempt, "script": str(ws / "eda_script.py")}
        errors.append(f"attempt {attempt}: " + (output.strip()[-1500:] or "no output"))
    return {"ok": False, "errors": errors}


def eda_plan(cfg, req: dict) -> dict:
    from agents import eda_agent

    return {"ok": True, "plan": eda_agent.generate_data_plan(cfg, req["task"], req["report"],
                                                             req.get("feedback") or None)}


def interpret_review(cfg, req: dict) -> dict:
    from agents.plan_mode import interpret_review_input

    intent, reply = interpret_review_input(cfg, req["message"], req["plan"])
    return {"ok": True, "intent": intent, "reply": reply}


def plan_start(cfg, req: dict) -> dict:
    from agents.plan_mode import _task_desc_diff, generate_questions
    from utils.data_preview import clean_task_desc

    prepare_workspace(cfg)
    cleaned = clean_task_desc(req["task"], cfg)
    return {"ok": True, "cleaned": cleaned, "diff": _task_desc_diff(req["task"], cleaned) or "",
            "questions": generate_questions(cfg, cleaned)}


def plan_answer(cfg, req: dict) -> dict:
    from agents.plan_mode import interpret_answer_input

    intent, reply = interpret_answer_input(cfg, req["message"], req["question"])
    return {"ok": True, "intent": intent, "reply": reply}


def plan_draft(cfg, req: dict) -> dict:
    from agents.plan_mode import generate_plan

    answers = [tuple(a) for a in req.get("answers") or []]
    return {"ok": True, "plan": generate_plan(cfg, req["cleaned"], answers, req.get("feedback") or None)}


def compose(cfg, req: dict) -> dict:
    """The task description the run gets: cleaned task, answers, plan, then the data plan."""
    from agents.eda_agent import build_context_block
    from agents.plan_mode import build_augmented_task_desc

    text = build_augmented_task_desc(req["cleaned"], [tuple(a) for a in req.get("answers") or []], req["plan"])
    if req.get("data_plan"):
        text += build_context_block(Path("EDA_report.md"), req["data_plan"])
    return {"ok": True, "text": text}


COMMANDS = {"eda_report": eda_report, "eda_plan": eda_plan, "interpret_review": interpret_review,
            "plan_start": plan_start, "plan_answer": plan_answer, "plan_draft": plan_draft, "compose": compose}


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        req = json.loads(sys.stdin.read())
        fn = COMMANDS[req["cmd"]]
        result = fn(make_cfg(req), req)
    except Exception as e:
        result = {"ok": False, "errors": [f"{type(e).__name__}: {e}"]}
    sys.stdout.write("\n" + RESULT_MARK + json.dumps(result, default=str) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
