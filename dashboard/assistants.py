"""The chat agents that guide a project from setup to results.

    Setup    environment and configuration: checks first, skips what is ready,
             guides what is not, and changes settings when asked
    Data     finds the data folder and profiles it
    Task     writes the task description with the user
    EDA      runs an analysis script on the data, then drafts the data plan
    Plan     shows the task rewrite, asks clarifying questions, writes the plan
    Run      launches training, then passes messages to the Main agent inside the run
             and handles the first-draft reviews
    Analyst  answers questions about the results
    Inspector checks each result above (settings, data, task, data plan, plan) with
             rule checks and one Claude call, and warns; it never blocks
             (dashboard/inspector.py, dashboard/inspect_agent.py)

"Main agent (auto)" in the page is the router below: it picks the agent from an
@mention, a /command, a live run, a few keywords, or the current stage. Routing
uses no LLM call. Free-text answers, the task draft and settings changes in
plain words make one `claude -p` call each. The EDA and Plan steps call the
terminal gates' own functions in WSL (dashboard/gates.py).
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from dashboard import dataprofile, envcheck, projects, wslexec
from dashboard.gate_agents import GateAgents
from dashboard.inspect_agent import InspectAgent
from dashboard.projects import CONFIG_SPEC, Project

AGENTS = {
    "setup": "Setup agent", "data": "Data agent", "task": "Task agent", "eda": "EDA agent",
    "plan": "Plan agent", "run": "Run agent", "analyst": "Analyst agent", "inspect": "Inspector agent",
    "auto": "Main agent",
}
ALIASES = {
    "setup": "setup", "config": "setup", "configuration": "setup", "env": "setup", "environment": "setup",
    "data": "data", "task": "task", "run": "run", "train": "run", "training": "run",
    "eda": "eda", "explore": "eda", "plan": "plan", "planning": "plan", "planner": "plan",
    "analyst": "analyst", "analysis": "analyst", "results": "analyst", "main": "auto", "auto": "auto",
    "inspect": "inspect", "inspector": "inspect", "check": "inspect",
}
STAGE_AGENT = {"environment": "setup", "configuration": "setup", "data": "data", "task": "task",
               "eda": "eda", "plan": "plan", "training": "run", "results": "analyst"}
LLM_TIMEOUT_S = 300
WORKER_IDLE_S = 300

CONFIG_WORDS = re.compile(r"\b(model|opus|sonnet|haiku|steps?|time ?limit|parallel|drafts?|gpu|cpu|install|"
                          r"wsl|venv|package|login|setting|configur\w*|timeout)\b", re.I)
LAUNCH_WORDS = re.compile(r"\b(launch|start (the )?(training|run)|begin training|train it|go ahead and train)\b", re.I)
RESULT_WORDS = re.compile(r"\b(result|score|best|plot|figure|recall|precision|threshold|auc|prauc|why|compare|"
                          r"summar\w*|report|confusion)\b", re.I)
PATH_RE = re.compile(r"((?:/mnt/[a-zA-Z]|/home|/data|/srv|/opt|/tmp|/root|/media|/workspace)(?:/[^\n`'\"]*)?"
                     r"|[A-Za-z]:\\[^\n`'\"]*|[A-Za-z]:/[^\n`'\"]*)")

HELP_TEXT = """**How to talk to the agents**

| Agent | Call it with | What it does |
|---|---|---|
| Setup | `@setup` | Checks WSL, Python, packages and the Claude login; changes settings (models, steps, time limit) |
| Data | `@data` | Profiles your data folder: send its path |
| Task | `@task` | Writes the task description with you: goal, metric, validation, deliverables |
| EDA | `@eda` | Runs an analysis script on your data, then writes the data plan with you |
| Plan | `@plan` | Asks a few questions, then writes the training plan with you; training starts only after you approve it |
| Run | `@run` | Launches training; while it runs: `/status`, `guide <text>`, `/stop`, `/abort`; shows each first draft for review (`accept`, `reject <reason>`, or a question) |
| Analyst | `@analyst` | Answers questions about the results |

Without an @name, the Main agent picks the agent for the current stage. `/check` runs the setup checks again."""

TASK_SYSTEM = """You write the task description for MLEvolve, a system that searches for a good machine learning
solution by writing and running Python scripts. Write Markdown with exactly these three sections:

# Goal
What to predict, the target column, the training file(s), the type of problem, and constraints
the user stated (hardware, model families, script time).

# Evaluation
The metric and whether higher or lower is better; the validation scheme (for example a hold-out
split and/or stratified k-fold with seed 42) on the training data only. End this section with:
The script must print its final line as `Final Validation Score: <score>`.

# Deliverables
Files to save in ./working/ (for example metrics.csv, metrics.md, plots as PNG).

Rules: use only facts from the data profile and the user's messages. Never use test labels. Name data
files by file name only (the run reads them from ./input/), never by a folder path. Keep it under 250
words. Plain English, short sentences."""

ANALYST_SYSTEM = (
    "You are the Analyst agent of the MLEvolve dashboard. Answer questions about a search run from the run "
    "facts given. Be concise and specific: cite solution ids and scores. Use Markdown. You cannot change a "
    "run; when asked to change something, say what to put in the task or in `guide` for the next run. "
    "Never invent numbers that are not in the facts.")


def ask_claude(prompt: str, model: str = "sonnet", timeout: float = LLM_TIMEOUT_S) -> str:
    """One `claude -p` call with no tools and no settings. Raises RuntimeError on failure."""
    claude = shutil.which("claude")
    if claude is None:
        raise RuntimeError("the `claude` CLI is not on PATH for the dashboard server")
    cmd = [claude, "-p", "--output-format", "json", "--tools", "", "--setting-sources", "",
           "--no-session-persistence", "--model", model]
    with tempfile.TemporaryDirectory(prefix="mlevolve-chat-") as cwd:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                              cwd=cwd, timeout=timeout)
    try:
        result = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        raise RuntimeError((proc.stderr or proc.stdout or "no answer")[:300])
    if result.get("is_error") or result.get("subtype") != "success":
        raise RuntimeError(str(result.get("result") or proc.stderr)[:300])
    return result.get("result") or ""


def json_from(text: str) -> dict:
    """The first JSON object in a model answer (tolerates code fences and prose around it)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the answer")
    return json.loads(text[start:end + 1])


def strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```$", t, re.S)
    return m.group(1).strip() if m else t


def chat_model(data: dict) -> str:
    model = str(data.get("config", {}).get("feedback_model") or "")
    if model.startswith("claude-code") and model.partition(":")[2]:
        return model.partition(":")[2]
    return "sonnet"


def route(text: str, data: dict, run_status: str | None, chosen: str | None = None) -> tuple[str, str]:
    """(agent key, message text without the @mention)."""
    text = text.strip()
    m = re.match(r"^@(\w+)[\s,:]*(.*)$", text, re.S)
    if m and m.group(1).lower() in ALIASES:
        chosen, text = ALIASES[m.group(1).lower()], (m.group(2).strip() or text)
    if chosen and chosen != "auto":
        return chosen, text
    if text.startswith("/"):
        cmd = text[1:].split(maxsplit=1)[0].lower() if len(text) > 1 else ""
        if cmd == "help":
            return "help", text
        if cmd in ("check", "setup"):
            return "setup", text
        return "run", text
    if run_status == "live":
        return "run", text
    if PATH_RE.search(text) and not data.get("data", {}).get("profile"):
        return "data", text
    stage = projects.current_stage(data, run_status)
    if stage in ("eda", "plan") and data.get(stage, {}).get("status"):
        return stage, text  # the step waits for the user: plain text is the answer
    if LAUNCH_WORDS.search(text):
        return "run", text
    if data.get("run") and run_status in ("finished", "stale") and RESULT_WORDS.search(text):
        return "analyst", text
    if CONFIG_WORDS.search(text):
        return "setup", text
    return STAGE_AGENT[projects.current_stage(data, run_status)], text


class Assistants(GateAgents, InspectAgent):
    """Runs agent work for each project on its own worker thread, one job at a time, in order.

    The EDA agent, the Plan agent and the draft review come from GateAgents (gate_agents.py)."""

    def __init__(self, store: projects.ProjectStore, runs_root: Path, reader_for, run_status_of):
        self.store = store
        self.runs_root = runs_root
        self.reader_for = reader_for          # run name -> RunReader | None
        self.run_status_of = run_status_of    # run name -> "live" | "finished" | "stale" | None
        self.busy: dict[str, dict] = {}
        self.inspecting: dict[tuple, float] = {}   # (project slug, stage) -> start time
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    # queue ------------------------------------------------------------------

    def submit(self, project: Project, job: tuple) -> None:
        with self._lock:
            q = self._queues.get(project.slug)
            if q is None:
                q = self._queues[project.slug] = queue.Queue()
                threading.Thread(target=self._worker, args=(project, q), daemon=True,
                                 name=f"agents-{project.slug}").start()
            q.put(job)

    def _worker(self, project: Project, q: queue.Queue) -> None:
        while True:
            try:
                job = q.get(timeout=WORKER_IDLE_S)
            except queue.Empty:
                with self._lock:
                    if q.empty():
                        self._queues.pop(project.slug, None)
                        return
                continue
            agent = {"message": lambda: job[1], "action": lambda: job[1].split(".")[0]}.get(job[0], lambda: "setup")()
            self.busy[project.slug] = {"agent": agent if agent in AGENTS else "auto", "since": time.time()}
            try:
                self.handle(project, job)
            except Exception as e:  # never let one job stop the worker
                project.post("agent", f"Something went wrong: {e}", agent=agent if agent in AGENTS else "auto",
                             kind="error")
            finally:
                self.busy.pop(project.slug, None)

    def run_sync(self, project: Project, job: tuple) -> None:
        """Run a job on the calling thread (tests)."""
        self.handle(project, job)

    # dispatch ---------------------------------------------------------------

    def handle(self, project: Project, job: tuple) -> None:
        kind = job[0]
        if kind == "start":
            return self.setup_check(project, first=True)
        if kind == "message":
            _, agent, text, msg_id = job
            handler = {"setup": self.setup_message, "data": self.data_message, "task": self.task_message,
                       "eda": self.eda_message, "plan": self.plan_message, "run": self.run_message,
                       "analyst": self.analyst_message, "inspect": self.inspect_message, "help": self.help}[agent]
            return handler(project, text, msg_id)
        if kind == "action":
            _, name, args = job
            actions = {
                "setup.check": lambda: self.setup_check(project, force_login=bool(args.get("login"))),
                "setup.fix": lambda: self.setup_fix(project, str(args.get("id", ""))),
                "setup.undo": lambda: self.setup_undo(project, int(args.get("index", -1))),
                "setup.keep": lambda: self.advance(project),
                "task.accept": lambda: self.task_accept(project),
                "task.save": lambda: self.task_save(project, str(args.get("text", ""))),
                "task.redraft": lambda: self.task_draft(project),
                "eda.accept": lambda: self.eda_accept(project),
                "eda.approve": lambda: self.eda_approve(project),
                "eda.redo": lambda: self.eda_start(project, echo="Run a new analysis"),
                "eda.skip": lambda: self.eda_skip(project),
                "plan.answer": lambda: self.plan_pick(project, args),
                "plan.approve": lambda: self.plan_approve(project),
                "plan.back": lambda: self.plan_back(project),
                "plan.restart": lambda: self.plan_start(project),
                "run.launch": lambda: self.run_launch(project),
                "run.draft": lambda: self.run_draft(project, args),
                "inspect.send": lambda: self.inspect_send(project, str(args.get("stage", ""))),
                "inspect.fix": lambda: self.inspect_fix(project, str(args.get("stage", ""))),
                "inspect.ignore": lambda: self.inspect_ignore(project, str(args.get("stage", ""))),
            }
            if name not in actions:
                return project.post("agent", f"Unknown action `{name}`.", agent="auto", kind="error")
            return actions[name]()

    def help(self, project: Project, text: str, msg_id: str | None) -> None:
        project.post("agent", HELP_TEXT, agent="auto", reply_to=msg_id)

    # flow -------------------------------------------------------------------

    def advance(self, project: Project) -> None:
        """Start the next stage's agent once the current stage is complete."""
        d = project.load()
        if not (d["env"].get("ok") and d["cfg"].get("ok")):
            return
        if not (d["data"].get("profile") and not d["data"]["profile"].get("error")):
            if not d["data"].get("asked"):
                project.post("agent", "Send me the path of your data folder. A Windows path such as "
                             "`E:\\data\\my_task` or a WSL path such as `/mnt/e/data/my_task` both work. "
                             "Tell me in the same message what you want to predict, if you know.", agent="data")
                project.update(lambda x: x["data"].update(asked=True))
            return
        if not d["task"].get("draft"):
            return self.task_draft(project)
        if not d["task"].get("accepted"):
            return
        if d.get("eda", {}).get("status") not in ("approved", "skipped"):
            if not d.get("eda", {}).get("status"):
                return self.eda_start(project)
            return
        if d.get("plan", {}).get("status") != "approved":
            if not d.get("plan", {}).get("status"):
                return self.plan_start(project)
            return
        if not d.get("run") and not d.get("launch"):
            return self.run_ready(project)

    # Setup agent ------------------------------------------------------------

    def setup_check(self, project: Project, first: bool = False, force_login: bool = False) -> None:
        d = project.load()
        cfg = d["config"]
        facts = envcheck.probe()
        needs_claude = projects.uses_claude_code(cfg)
        login = None
        if needs_claude and facts.get("shell") == "ok" and facts.get("claude", "missing") != "missing":
            login = envcheck.check_login(force_login)
        env_items = envcheck.environment_checks(facts, needs_claude, login)
        cfg_items = envcheck.config_checks(cfg)
        env_ok, cfg_ok = envcheck.all_ok(env_items), envcheck.all_ok(cfg_items)

        def save(x):
            was_fixing = x["env"].get("checked") is not None and x["env"].get("ok") is False
            x["env"].update(checked=time.time(), ok=env_ok, items=env_items,
                            fixed=bool(x["env"].get("fixed") or (env_ok and was_fixing)))
            x["cfg"].update(ok=cfg_ok, items=cfg_items)
        project.update(save)
        checks = env_items + cfg_items
        if env_ok and cfg_ok:
            passed = sum(1 for i in checks if i["ok"] or i["ok"] is None)
            text = ("I checked your environment and configuration. Everything is ready, so I skipped setup."
                    if first else "I checked again. Everything is ready.")
            optional = [i for i in checks if i["ok"] is False and not i["blocking"]]
            actions = [{"label": "Change configuration", "prefill": "@setup "},
                       {"label": "Run setup again", "action": "setup.check"}]
            for i in optional:
                text += f"\n\nOptional: {i['label']} — {i['detail']}."
                if i["fix"] in envcheck.FIXES:
                    actions.insert(0, {"label": envcheck.FIXES[i["fix"]]["label"], "action": "setup.fix",
                                       "args": {"id": i["fix"]}})
            project.post("agent", text, agent="setup", checks=checks, fold=True,
                         summary=f"{passed} of {len(checks)} checks passed", actions=actions)
            self.inspect(project, "setup")
            return self.advance(project)
        failing = [i for i in checks if i["blocking"] and not i["ok"]]
        step = failing[0]
        text = (f"I checked your environment and configuration. {len(failing)} item(s) are not set up. "
                f"I will guide you through them one at a time.\n\n**Step 1 of {len(failing)}: {step['label']}.** ")
        actions = []
        if step["fix"] in envcheck.FIXES:
            fix = envcheck.FIXES[step["fix"]]
            text += f"I can run this for you in the MLEvolve folder:\n\n`{fix['command']}`"
            actions = [{"label": "Run it for me", "action": "setup.fix", "args": {"id": step["fix"]}, "primary": True},
                       {"label": "I ran it, check again", "action": "setup.check"}]
        elif step["fix"] in envcheck.GUIDES:
            text += envcheck.GUIDES[step["fix"]]
            actions = [{"label": "Check again", "action": "setup.check", "args": {"login": step["id"] == "login"},
                        "primary": True}]
        else:
            text += f"Detail: {step['detail']}. Tell me the value to use, for example `@setup steps=20`."
            actions = [{"label": "Change configuration", "prefill": "@setup "}]
        project.post("agent", text, agent="setup", checks=checks, fold=False, actions=actions)

    def setup_fix(self, project: Project, fix_id: str) -> None:
        fix = envcheck.FIXES.get(fix_id)
        if fix is None:
            return project.post("agent", f"`{fix_id}` is not a fix I can run.", agent="setup", kind="error")
        project.post("agent", f"Running: **{fix['label']}**. This can take several minutes; I will tell you when "
                     f"it ends.", agent="setup", kind="progress")
        ok, tail = envcheck.run_fix(fix_id)
        project.post("agent", ("Done. " if ok else "It failed. ") + f"Last lines of output:\n\n```\n{tail or '(none)'}\n```",
                     agent="setup", kind="action" if ok else "error")
        self.setup_check(project)

    def setup_message(self, project: Project, text: str, msg_id: str | None) -> None:
        t = text.strip()
        if re.fullmatch(r"/?(check|setup)|.*\b(check|test|verify)\b.*\b(again|setup|login|environment)\b.*", t, re.I | re.S) and len(t) < 80:
            return self.setup_check(project, force_login="login" in t.lower())
        d = project.load()
        changes = dict(re.findall(r"\b([a-z_]+)\s*=\s*([^\s,;]+)", t))
        reply = ""
        if not changes or any(k not in CONFIG_SPEC for k in changes):
            spec = "\n".join(f"- {k}: {v['label']}" + (f" ({v['min']} to {v['max']})" if "min" in v else "")
                             for k, v in CONFIG_SPEC.items())
            prompt = (
                "You are the Setup agent of the MLEvolve dashboard. You manage the environment and the run settings.\n"
                f"Current settings: {json.dumps(d['config'])}\nSettings you can change:\n{spec}\n"
                "Model values: claude-code:opus, claude-code:sonnet, claude-code:haiku (Claude subscription, no API "
                "key), or another provider's model id (needs an API key in .env).\n"
                f"Last environment check: {json.dumps([(i['label'], i['ok'], i['detail']) for i in d['env'].get('items', [])])}\n"
                f"The user says: {t}\n\n"
                'Answer with JSON only: {"changes": {"<setting>": <value>}, "reply": "<short plain-English answer>"}. '
                "Put a setting in changes only when the user asks to change it. For a question, answer it in reply "
                "and recommend values when asked (for example a time limit that fits the steps).")
            answer = json_from(ask_claude(prompt, chat_model(d)))
            changes, reply = answer.get("changes") or {}, str(answer.get("reply") or "")
        if not changes:
            return project.post("agent", reply or "I did not find a setting to change.", agent="setup", reply_to=msg_id)
        self.apply_changes(project, changes, reply, msg_id)

    def apply_changes(self, project: Project, changes: dict, reply: str = "", msg_id: str | None = None) -> None:
        d = project.load()
        clean, errors = {}, []
        for key, value in changes.items():
            ok, v, err = projects.validate_change(key, value)
            if ok:
                if d["config"].get(key) != v:
                    clean[key] = v
            else:
                errors.append(err)
        if not clean:
            msg = "Nothing changed." + (" " + " ".join(errors) if errors else "")
            return project.post("agent", msg, agent="setup", kind="error" if errors else "reply", reply_to=msg_id)
        before = {k: d["config"].get(k) for k in clean}
        claude_before = projects.uses_claude_code(d["config"])

        def save(x):
            x["config"].update(clean)
            x["config_history"].append({"t": time.time(), "before": before, "after": clean})
            x["cfg"]["changed"] = True
            x["cfg"]["items"] = envcheck.config_checks(x["config"])
            x["cfg"]["ok"] = envcheck.all_ok(x["cfg"]["items"])
        d = project.update(save)
        rows = "\n".join(f"| {CONFIG_SPEC[k]['label']} | `{before[k]}` | `{clean[k]}` |" for k in clean)
        text = (reply + "\n\n" if reply else "") + f"I changed {len(clean)} setting(s) for this project:\n\n" \
            f"| Setting | Before | After |\n|---|---|---|\n{rows}"
        if errors:
            text += "\n\nNot changed: " + " ".join(errors)
        if d.get("run") and self.run_status_of(d["run"]) == "live":
            text += "\n\nThe run that is going on keeps its old settings. The new ones apply to the next launch."
        index = len(d["config_history"]) - 1
        project.post("agent", text, agent="setup", kind="action", reply_to=msg_id, diff=True,
                     actions=[{"label": "Keep and continue", "action": "setup.keep", "primary": True},
                              {"label": "Undo", "action": "setup.undo", "args": {"index": index}}])
        self.inspect(project, "setup")
        if d.get("plan", {}).get("status") == "review" and d["plan"].get("plan"):
            # The change closed the plan's buttons; show the plan again with the new settings (and its new check).
            self._post_plan(project, d["plan"]["plan"], "The settings changed. Here is the plan again with the new "
                            "settings.")
        if projects.uses_claude_code(d["config"]) != claude_before:
            self.setup_check(project)  # a provider change needs a new login or key check

    def setup_undo(self, project: Project, index: int) -> None:
        d = project.load()
        history = d.get("config_history", [])
        if not 0 <= index < len(history) or history[index].get("undone"):
            return project.post("agent", "There is nothing to undo for that change.", agent="setup", kind="error")
        before = history[index]["before"]

        def save(x):
            x["config"].update(before)
            x["config_history"][index]["undone"] = True
            x["cfg"]["items"] = envcheck.config_checks(x["config"])
            x["cfg"]["ok"] = envcheck.all_ok(x["cfg"]["items"])
        project.update(save)
        rows = ", ".join(f"{CONFIG_SPEC[k]['label']} = `{v}`" for k, v in before.items())
        project.post("agent", f"Undone. Back to: {rows}.", agent="setup", kind="action")

    # Data agent -------------------------------------------------------------

    def find_path(self, text: str) -> tuple[str | None, str | None]:
        """(existing path, last candidate tried). Paths may contain spaces, so shorter cuts are tried too."""
        quoted = re.findall(r"`([^`]+)`|\"([^\"]+)\"|'([^']+)'", text)
        candidates = [next(g for g in q if g) for q in quoted]
        candidates += [m.group(1) for m in PATH_RE.finditer(text)]
        tried, remote_checks = None, 0
        for cand in candidates:
            words = wslexec.clean_path(cand).split(" ")
            tried = tried or wslexec.clean_path(words[0].rstrip(".,;:!?)"))
            for cut in range(len(words), 0, -1):  # longest first: folder names may contain spaces
                c = wslexec.clean_path(" ".join(words[:cut]).rstrip(".,;:!?)"))
                if not c:
                    continue
                local = wslexec.to_local_path(c)
                if local is not None:
                    if local.exists():
                        return c, tried
                elif remote_checks < 12:  # a path only WSL can see costs one WSL call per check
                    remote_checks += 1
                    if self.path_exists(c):
                        return c, tried
        return None, tried

    @staticmethod
    def path_exists(path: str) -> bool:
        local = wslexec.to_local_path(path)
        if local is not None and local.exists():
            return True
        if path.startswith("/") and wslexec.ON_WINDOWS and wslexec.shell_available():
            try:
                return wslexec.run_shell(f"test -e {_q(path)}", timeout=30).returncode == 0
            except Exception:
                return False
        return False

    def data_message(self, project: Project, text: str, msg_id: str | None) -> None:
        path, tried = self.find_path(text)
        if path:
            return self.data_profile(project, path, msg_id)
        d = project.load()
        prof = d["data"].get("profile")
        if tried:
            return project.post("agent", f"I could not find `{tried}`. Check the spelling, and send the full path "
                                "of the folder.", agent="data", kind="error", reply_to=msg_id)
        if not prof:
            return project.post("agent", "Send me the full path of your data folder, for example "
                                "`/mnt/e/data/my_task` or `E:\\data\\my_task`.", agent="data", reply_to=msg_id)
        prompt = ("You are the Data agent of the MLEvolve dashboard. Answer from this data profile only. Be short and "
                  f"use Markdown.\n\nData profile: {json.dumps(prof)[:12000]}\n\nQuestion: {text}")
        project.post("agent", ask_claude(prompt, chat_model(d)), agent="data", reply_to=msg_id)

    def data_profile(self, project: Project, path: str, msg_id: str | None = None) -> None:
        local = wslexec.to_local_path(path)
        if local is not None and local.exists():
            prof = dataprofile.profile(local)
            prof["path"] = path
        else:
            proc = wslexec.run_shell(f"cd {_q(envcheck.REPO_WSL)} && python3 dashboard/dataprofile.py {_q(path)}",
                                     timeout=600)
            try:
                prof = json.loads(proc.stdout)
            except json.JSONDecodeError:
                prof = {"path": path, "error": (proc.stderr or "no output")[:300]}
        path_wsl = wslexec.to_wsl_path(path)
        project.update(lambda x: x["data"].update(path=path, path_wsl=path_wsl, profile=prof))
        if prof.get("error"):
            return project.post("agent", dataprofile.markdown(prof), agent="data", kind="error", reply_to=msg_id)
        project.post("agent", dataprofile.markdown(prof), agent="data", reply_to=msg_id)
        self.inspect(project, "data")
        d = project.load()
        if d["task"].get("draft") and not d["task"].get("accepted"):
            return self.task_draft(project, request="The data folder changed. Update the draft to the new data.")
        self.advance(project)

    # Task agent -------------------------------------------------------------

    def _task_prompt(self, project: Project, d: dict, request: str | None) -> str:
        said = [m["text"] for m in project.messages() if m.get("role") == "user"][-12:]
        prof = json.dumps(d["data"].get("profile", {}))[:12000]
        prompt = f"{TASK_SYSTEM}\n\n# Data profile\n{prof}\n\n# What the user said so far\n" + \
            ("\n".join(f"- {s[:600]}" for s in said) or "(nothing)")
        if d["task"].get("draft") and request:
            prompt += f"\n\n# Current draft\n{d['task']['draft']}\n\n# Change requested\n{request}\n\n" \
                "Write the full new draft."
        return prompt + "\n\nAnswer with the Markdown task description only."

    def task_draft(self, project: Project, request: str | None = None, msg_id: str | None = None) -> None:
        d = project.load()
        if not d["data"].get("profile"):
            return project.post("agent", "I need the data first. Send the path of your data folder.",
                                agent="task", reply_to=msg_id)
        draft = strip_fences(ask_claude(self._task_prompt(project, d, request), chat_model(d)))
        project.update(lambda x: x["task"].update(draft=draft, accepted=False))
        lead = "I updated the task draft." if request else "I wrote a task draft."
        project.post("agent", f"{lead} Read it, then accept it or tell me what to change.", agent="task",
                     card=draft, reply_to=msg_id, actions=self._task_actions())
        self.inspect(project, "task")

    @staticmethod
    def _task_actions() -> list[dict]:
        return [{"label": "Accept task", "action": "task.accept", "primary": True},
                {"label": "Edit text", "ui": "edit_task"},
                {"label": "Ask for a change", "prefill": "@task "}]

    def task_message(self, project: Project, text: str, msg_id: str | None) -> None:
        d = project.load()
        if not d["task"].get("draft"):
            return self.task_draft(project, request=text, msg_id=msg_id)
        prompt = (self._task_prompt(project, d, None) +
                  f"\n\n# Current draft\n{d['task']['draft']}\n\n# The user says\n{text}\n\n"
                  'Answer with JSON only: {"action": "revise" or "answer", "task": "<full new Markdown draft when '
                  'revising, else empty>", "reply": "<one or two short sentences>"}')
        answer = json_from(ask_claude(prompt, chat_model(d)))
        if answer.get("action") == "revise" and str(answer.get("task", "")).strip():
            draft = strip_fences(str(answer["task"]))
            project.update(lambda x: x["task"].update(draft=draft, accepted=False))
            note = ""
            if d.get("run") and self.run_status_of(d["run"]) == "live":
                note = " The run that is going on keeps the old task; use `guide <text>` to steer it."
            project.post("agent", (answer.get("reply") or "I updated the draft.") + note, agent="task",
                         card=draft, reply_to=msg_id, actions=self._task_actions())
            return self.inspect(project, "task")
        project.post("agent", answer.get("reply") or "(no answer)", agent="task", reply_to=msg_id)

    def task_accept(self, project: Project) -> None:
        d = project.load()
        draft = d["task"].get("draft")
        if not draft:
            return project.post("agent", "There is no draft to accept yet.", agent="task", kind="error")
        project.task_path().write_text(draft.strip() + "\n", encoding="utf-8", newline="\n")
        # A new task needs a new analysis and plan.
        project.update(lambda x: (x["task"].update(accepted=True, accepted_t=time.time()),
                                  x.update(eda={}, plan={})))
        project.post("agent", f"Task accepted and saved to `projects/{project.slug}/task.md`. Next, the EDA agent "
                     "analyses the data.", agent="task", kind="action")
        self.advance(project)

    def task_save(self, project: Project, text: str) -> None:
        text = text.strip()
        if not text:
            return project.post("agent", "The edited task was empty, so I kept the old draft.", agent="task",
                                kind="error")
        project.update(lambda x: x["task"].update(draft=text[:20000], accepted=False))
        project.post("agent", "I saved your edited task.", agent="task", card=text[:20000],
                     actions=self._task_actions())
        self.inspect(project, "task")

    # Run agent --------------------------------------------------------------

    def run_ready(self, project: Project) -> None:
        d = project.load()
        c = d["config"]
        rows = "\n".join(f"| {CONFIG_SPEC[k]['label']} | `{c.get(k, projects.default_config().get(k))}` |"
                         for k in CONFIG_SPEC)
        project.post("agent", "The plan is approved and everything is ready to start training.\n\n| Setting | Value |\n|---|---|\n" + rows +
                     f"\n| Data | `{d['data'].get('path_wsl')}` |", agent="run",
                     actions=[{"label": "Launch training", "action": "run.launch", "primary": True},
                              {"label": "Change configuration", "prefill": "@setup "},
                              {"label": "Change the task", "prefill": "@task "}])

    def run_message(self, project: Project, text: str, msg_id: str | None) -> None:
        d = project.load()
        status = self.run_status_of(d["run"]) if d.get("run") else None
        low = text.lower().strip().lstrip("/")
        if LAUNCH_WORDS.search(text) or low in ("go", "launch", "start"):
            if d.get("plan", {}).get("status") == "review":
                return self.plan_approve(project)  # "start training" while the plan waits means approve it
            return self.run_launch(project)
        if low.startswith("status"):
            return project.post("agent", self.status_text(project), agent="run", reply_to=msg_id)
        if status == "live":  # normally forwarded before it reaches here
            return project.post("agent", "The run is live; I passed your message to its Main agent.", agent="run",
                                reply_to=msg_id)
        if low.split(" ")[0] in ("stop", "abort", "guide", "seed"):
            return project.post("agent", "No run is live, so there is nothing to stop or guide.", agent="run",
                                reply_to=msg_id)
        project.post("agent", "I start the training run and pass your messages to it while it runs. Say "
                     "**launch** when you are ready, or `/status` to see where the project is.", agent="run",
                     reply_to=msg_id)

    def status_text(self, project: Project) -> str:
        d = project.load()
        status = self.run_status_of(d["run"]) if d.get("run") else None
        lines = ["| Stage | Status |", "|---|---|"]
        for s in projects.stages(d, status):
            lines.append(f"| {s['name'].capitalize()} | {s['status']}{' · ' + s['note'] if s['note'] else ''} |")
        if d.get("run"):
            lines.append(f"\nRun: `{d['run']}` ({status})")
        return "\n".join(lines)

    def run_launch(self, project: Project) -> None:
        d = project.load()
        missing = []
        if not d["env"].get("ok"):
            missing.append("the environment checks")
        if not d["cfg"].get("ok"):
            missing.append("the configuration")
        if not d["data"].get("profile"):
            missing.append("the data folder")
        if not d["task"].get("accepted"):
            missing.append("an accepted task")
        elif d.get("eda", {}).get("status") not in ("approved", "skipped"):
            missing.append("an approved data plan (EDA stage)")
        elif d.get("plan", {}).get("status") != "approved":
            missing.append("an approved training plan")
        if missing:
            project.post("agent", "I cannot launch yet. Still needed: " + ", ".join(missing) + ".",
                         agent="run", kind="error")
            # A project that a pre-EDA server took past the task (or an old "Launch" button) starts the
            # missing EDA or Plan step here, so the user is not left without a way forward.
            if d["task"].get("accepted") and len(missing) == 1:
                self.advance(project)
            return
        if d.get("run") and self.run_status_of(d["run"]) == "live":
            return project.post("agent", f"Run `{d['run']}` is already live.", agent="run", kind="error")
        if d.get("launch") and not d.get("run") and time.time() - d["launch"]["t"] < projects.RUN_FIND_S:
            return project.post("agent", "A launch is already starting.", agent="run", kind="error")
        script = project.folder / "launch.sh"
        script.write_text(projects.launch_script(project, d, envcheck.REPO_WSL), encoding="utf-8", newline="\n")
        # The run gets the approved text: cleaned task, answers, plan and data plan.
        project.task_path().write_text((d["plan"].get("final") or d["task"]["draft"]).strip() + "\n",
                                       encoding="utf-8", newline="\n")
        t0 = time.time()
        pid = wslexec.spawn_detached(wslexec.to_wsl_path(script), project.folder / "launch.log")
        project.update(lambda x: x.update(launch={"t": t0, "pid": pid}, run=None))
        review = d["config"].get("draft_review", True)
        project.post("agent", "Training started. I will link the run here as soon as its folder appears, usually "
                     "within a minute. Open the monitor with the arrow at the top to watch it." +
                     (" Before each first draft runs, I show it here for your review." if review else ""),
                     agent="run",
                     kind="action", actions=[{"label": "Show monitor", "ui": "monitor"}])
        threading.Thread(target=self._link_run, args=(project, t0), daemon=True, name="link-run").start()

    def _link_run(self, project: Project, t0: float) -> None:
        while time.time() - t0 < projects.RUN_FIND_S:
            name = projects.find_run(project.slug, self.runs_root, t0)
            if name:
                project.update(lambda x: (x.update(run=name), x["runs"].append(name)))
                self._copy_plan_files(project, name)
                return project.post("agent", f"Linked run `{name}`. Messages to the Run agent now go to the Main "
                                    "agent inside the run: try `/status` or `guide <text>`.", agent="run",
                                    kind="action")
            time.sleep(3)
        log = project.folder / "launch.log"
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]) if log.exists() else ""
        project.update(lambda x: x.update(launch=None))
        project.post("agent", "The run folder did not appear within 4 minutes, so the launch probably failed. "
                     f"Last lines of `launch.log`:\n\n```\n{tail or '(empty)'}\n```", agent="run", kind="error",
                     actions=[{"label": "Try again", "action": "run.launch"}])

    # Analyst agent ----------------------------------------------------------

    def analyst_message(self, project: Project, text: str, msg_id: str | None) -> None:
        from dashboard import server
        d = project.load()
        reader = self.reader_for(d["run"]) if d.get("run") else None
        if reader is None:
            return project.post("agent", "There are no results yet. Launch training first; I can answer questions "
                                "once solutions have scores.", agent="analyst", reply_to=msg_id)
        convo = "\n".join(f"{m['role']}: {m['text'][:600]}" for m in project.messages()[-10:])
        prompt = (f"{ANALYST_SYSTEM}\n\n# Task\n{d['task'].get('draft', '')[:3000]}\n\n# Run facts\n"
                  f"{server.review_context(reader)}\n\n# Conversation so far\n{convo}\n\n# Question\n{text}")
        project.post("agent", ask_claude(prompt, chat_model(d)), agent="analyst", reply_to=msg_id)


def _q(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"
