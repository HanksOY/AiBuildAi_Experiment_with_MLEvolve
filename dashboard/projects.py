"""Projects: one guided path from an empty setup to a trained model.

A project lives in projects/<slug>/:
    project.json   settings, data profile, task, stage facts, linked run names
    chat.jsonl     the conversation (written only by the dashboard server)
    task.md        the accepted task description (passed to run.py as desc_file)
    launch.sh      the command that starts the run in WSL
    launch.log     its console output

Once the run starts, its folder in runs/ is linked to the project, and the
run's own chat (the Main agent inside the run) is merged into the project chat.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import threading
import time
from pathlib import Path

from dashboard import wslexec

REPO = Path(__file__).resolve().parent.parent
STAGES = ["environment", "configuration", "data", "task", "eda", "plan", "training", "results"]
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
PENDING_S = 300
CHAT_LIMIT = 400
RUN_FIND_S = 240

# Settings the chat may change, with their type and limits.
CONFIG_SPEC = {
    "code_model": {"type": "model", "label": "Code model"},
    "feedback_model": {"type": "model", "label": "Feedback model"},
    "steps": {"type": int, "min": 1, "max": 1000, "label": "Steps"},
    "time_limit_h": {"type": float, "min": 0.1, "max": 72, "label": "Time limit (hours)"},
    "parallel": {"type": int, "min": 1, "max": 8, "label": "Parallel solutions"},
    "initial_drafts": {"type": int, "min": 1, "max": 10, "label": "First drafts"},
    "exec_timeout_min": {"type": int, "min": 1, "max": 1440, "label": "Script time limit (minutes)"},
    "draft_review": {"type": bool, "label": "Review first drafts before they run"},
    "inspector": {"type": bool, "label": "Inspector checks each agent's result"},
}
DRAFT_REVIEW_EXTRA_S = 7200  # wall-clock allowance for reading drafts (the run's own clock pauses)
MODEL_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,80}$")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_-")[:48]
    return slug or "project"


def default_config(config_yaml: Path = REPO / "config" / "config.yaml") -> dict:
    """Start from config/config.yaml (read with regular expressions: no YAML library needed)."""
    cfg = {"code_model": "claude-code:opus", "feedback_model": "claude-code:sonnet", "steps": 30,
           "time_limit_h": 4.0, "parallel": 3, "initial_drafts": 3, "exec_timeout_min": 60,
           "draft_review": True, "inspector": True}
    try:
        text = config_yaml.read_text(encoding="utf-8")
    except OSError:
        return cfg
    for stage in ("code", "feedback"):
        m = re.search(rf"^\s+{stage}:\s*\n\s+model:\s*(\S+)", text, re.M)
        if m:
            cfg[f"{stage}_model"] = m.group(1).strip("'\"")
    for key, name in (("parallel_search_num", "parallel"), ("initial_drafts", "initial_drafts")):
        m = re.search(rf"^\s+{key}:\s*(\d+)", text, re.M)
        if m:
            cfg[name] = int(m.group(1))
    return cfg


def normalize_model(value: str) -> str:
    v = str(value).strip().strip("`'\"")
    if v.lower() in ("opus", "sonnet", "haiku") or (v.startswith("claude-") and not v.startswith("claude-code")):
        return f"claude-code:{v.lower() if v.lower() in ('opus', 'sonnet', 'haiku') else v}"
    return v


def validate_change(key: str, value) -> tuple[bool, object, str]:
    """(ok, clean value, error) for one setting."""
    spec = CONFIG_SPEC.get(key)
    if spec is None:
        return False, None, f"`{key}` is not a setting I can change"
    if spec["type"] == "model":
        v = normalize_model(value)
        return (True, v, "") if MODEL_RE.match(v) else (False, None, f"`{value}` is not a model name")
    if spec["type"] is bool:
        word = str(value).strip().lower()
        if word in ("true", "on", "yes", "1"):
            return True, True, ""
        if word in ("false", "off", "no", "0"):
            return True, False, ""
        return False, None, f"{spec['label']} must be on or off"
    try:
        v = spec["type"](float(value)) if spec["type"] is int else spec["type"](value)
    except (TypeError, ValueError):
        return False, None, f"`{value}` is not a number"
    if not spec["min"] <= v <= spec["max"]:
        return False, None, f"{spec['label']} must be between {spec['min']} and {spec['max']}"
    return True, v, ""


def uses_claude_code(config: dict) -> bool:
    return any(str(config.get(k, "")).startswith("claude-code") for k in ("code_model", "feedback_model"))


class Project:
    def __init__(self, folder: Path):
        self.folder = folder
        self.slug = folder.name
        self.file = folder / "project.json"
        self.chat_file = folder / "chat.jsonl"
        self._lock = threading.RLock()

    # stored facts -----------------------------------------------------------

    def load(self) -> dict:
        with self._lock:
            try:
                return json.loads(self.file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}

    def save(self, data: dict) -> None:
        with self._lock:
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.file)

    def update(self, fn) -> dict:
        """Load, change with fn(data), save; atomic for this process."""
        with self._lock:
            data = self.load()
            fn(data)
            self.save(data)
            return data

    # conversation -----------------------------------------------------------

    def post(self, role: str, text: str, agent: str | None = None, kind: str = "reply", **extra) -> dict:
        msg = {"id": f"{'u' if role == 'user' else 'a'}-{time.time():.6f}-{secrets.token_hex(2)}",
               "role": role, "agent": agent, "text": text, "kind": kind, "t": time.time()}
        msg.update({k: v for k, v in extra.items() if v is not None})
        with self._lock, open(self.chat_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, default=str) + "\n")
        return msg

    def messages(self) -> list[dict]:
        if not self.chat_file.exists():
            return []
        out = []
        for line in self.chat_file.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def task_path(self) -> Path:
        return self.folder / "task.md"


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, Project] = {}
        self._lock = threading.Lock()

    def get(self, slug: str) -> Project | None:
        if not SLUG_RE.match(slug or ""):
            return None
        folder = self.root / slug
        if not (folder / "project.json").exists():
            return None
        with self._lock:
            return self._cache.setdefault(slug, Project(folder))

    def list(self) -> list[Project]:
        if not self.root.is_dir():
            return []
        return [p for p in (self.get(d.name) for d in self.root.iterdir() if d.is_dir()) if p]

    def create(self, name: str) -> Project:
        base = slugify(name)
        slug, i = base, 2
        while (self.root / slug).exists():
            slug, i = f"{base}_{i}", i + 1
        folder = self.root / slug
        folder.mkdir(parents=True)
        project = Project(folder)
        project.save({"name": name.strip() or slug, "slug": slug, "created": time.time(),
                      "config": default_config(), "config_history": [],
                      "env": {"checked": None, "ok": None, "items": [], "fixed": False},
                      "cfg": {"ok": None, "items": [], "changed": False},
                      "data": {}, "task": {}, "runs": [], "run": None, "launch": None, "announced": []})
        with self._lock:
            self._cache[slug] = project
        return project


# ── stages ──────────────────────────────────────────────────────────────────


def stages(data: dict, run_status: str | None) -> list[dict]:
    """The six stages with status todo / now / done / skipped and a short note."""
    env, cfg = data.get("env", {}), data.get("cfg", {})
    out = {}
    if env.get("ok"):
        out["environment"] = ("done", "fixed") if env.get("fixed") else ("skipped", "ready, skipped")
    else:
        bad = sum(1 for i in env.get("items", []) if i.get("blocking") and not i.get("ok"))
        out["environment"] = ("todo", f"{bad} to fix" if env.get("checked") else "checking")
    if cfg.get("ok"):
        out["configuration"] = ("done", "changed") if cfg.get("changed") else ("skipped", "ready, skipped")
    else:
        out["configuration"] = ("todo", "")
    prof = data.get("data", {}).get("profile")
    out["data"] = ("done", "") if prof and not prof.get("error") else ("todo", "")
    task = data.get("task", {})
    out["task"] = ("done", "") if task.get("accepted") else ("todo", "draft ready" if task.get("draft") else "")
    eda = data.get("eda", {})
    out["eda"] = {
        "approved": ("done", "data plan approved"), "skipped": ("skipped", "skipped"),
        "running": ("todo", "analysing"), "report": ("todo", "check the report"),
        "plan": ("todo", "check the data plan"), "failed": ("todo", "analysis failed"),
    }.get(eda.get("status"), ("todo", ""))
    plan = data.get("plan", {})
    out["plan"] = {
        "approved": ("done", "approved"), "starting": ("todo", "preparing"),
        "questions": ("todo", "question %d of %d" % (len(plan.get("answers", [])) + 1, len(plan.get("questions", []))
                                                     ) if plan.get("questions") else "questions"),
        "drafting": ("todo", "writing the plan"), "review": ("todo", "waiting for you"),
        "failed": ("todo", "failed"),
    }.get(plan.get("status"), ("todo", ""))
    if data.get("run"):
        out["training"] = ("now", "running") if run_status == "live" else (
            ("done", "") if run_status in ("finished", "stale") else ("now", "starting"))
        out["results"] = ("todo", "") if run_status == "live" else (("now", "") if run_status else ("todo", ""))
    elif data.get("launch"):
        out["training"], out["results"] = ("now", "starting"), ("todo", "")
    else:
        out["training"], out["results"] = ("todo", ""), ("todo", "")
    # The Inspector's open findings on a stage's result, for example "plan · 1 problem found".
    for stage, rec in data.get("inspect", {}).items():
        name = {"setup": "configuration"}.get(stage, stage)
        if rec.get("status") == "open" and rec.get("note") and name in out:
            status, note = out[name]
            out[name] = (status, f"{note} · {rec['note']}" if note else rec["note"])
    result, current_set = [], False
    for name in STAGES:
        status, note = out[name]
        if status == "todo" and not current_set:
            status, current_set = "now", True
        elif status == "now":
            if current_set:
                status = "todo"
            current_set = True
        result.append({"name": name, "status": status, "note": note})
    return result


def current_stage(data: dict, run_status: str | None) -> str:
    for s in stages(data, run_status):
        if s["status"] == "now":
            return s["name"]
    return "results"


# ── launching a run ─────────────────────────────────────────────────────────


def has_prediction_target(profile: dict) -> bool:
    """True when the data folder has a test or sample-submission file, so a submission.csv makes sense."""
    names = [str(f.get("name", "")).lower().rsplit("/", 1)[-1] for f in profile.get("files", [])]
    return any(("test" in n or "submission" in n) and not n.endswith((".md", ".txt", ".py")) for n in names)


def launch_script(project: Project, data: dict, repo_wsl: str) -> str:
    """The bash script that starts the run. Every value is quoted; settings were validated on entry."""
    c = data["config"]
    for key in ("code_model", "feedback_model"):
        if not MODEL_RE.match(str(c[key])):
            raise ValueError(f"bad model name in {key}")
    time_limit = int(float(c["time_limit_h"]) * 3600)
    args = [
        f"data_dir={data['data']['path_wsl']}", f"dataset_dir={data['data']['path_wsl']}",
        f"desc_file={wslexec.to_wsl_path(project.task_path())}", "goal=null", "eval=null",
        f"exp_name={project.slug}", "task.profile=generic",
        f"agent.code.model={c['code_model']}", f"agent.feedback.model={c['feedback_model']}",
        f"agent.steps={int(c['steps'])}", f"agent.time_limit={time_limit}",
        f"agent.initial_drafts={int(c['initial_drafts'])}", f"agent.search.parallel_search_num={int(c['parallel'])}",
        f"exec.timeout={int(c['exec_timeout_min']) * 60}",
        # The dashboard ran the EDA and plan steps before launch (the terminal gates cannot be answered here).
        "eda.enabled=False", "plan_mode.require_confirmation=False",
    ]
    if data.get("plan", {}).get("status") == "approved":
        args.append("plan_mode.preapproved=True")
    if not has_prediction_target(data.get("data", {}).get("profile") or {}):
        # No test or sample-submission file: there is nothing to predict, so a solution passes on its
        # validation score alone (a correct script that writes no submission.csv was marked failed).
        args.append("task.require_submission=False")
    review = bool(c.get("draft_review", True))
    args += ["draft_review.enabled=True", "draft_review.channel=file"] if review else ["draft_review.enabled=False"]
    # Upstream does not enforce time_limit: end the run 30 min after it, plus time to read drafts.
    hard_stop = time_limit + 1800 + (DRAFT_REVIEW_EXTRA_S if review else 0)
    return ("#!/bin/bash\n# Written by the MLEvolve dashboard (Run agent).\n"
            f"cd {shlex.quote(repo_wsl)} || exit 1\n"
            f"exec timeout --signal=INT --kill-after=300 {hard_stop} .venv/bin/python run.py \\\n  "
            + " \\\n  ".join(shlex.quote(a) for a in args) + " \\\n  < /dev/null\n")


def find_run(slug: str, runs_root: Path, since: float) -> str | None:
    """Newest runs/<timestamp>_<slug> folder created after `since`."""
    if not runs_root.is_dir():
        return None
    pattern = re.compile(rf"^\d{{8}}_\d{{6}}_{re.escape(slug)}$")
    found = [d for d in runs_root.iterdir()
             if d.is_dir() and pattern.match(d.name) and d.stat().st_mtime >= since - 5]
    return max(found, key=lambda d: d.stat().st_mtime).name if found else None
