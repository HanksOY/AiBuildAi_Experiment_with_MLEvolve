"""Local dashboard for MLEvolve runs.

    python -m dashboard.server [--runs ./runs] [--port 8765]

then open http://127.0.0.1:8765. Standard library only, so it runs from the
Windows Python or the WSL venv alike. It reads the files a run writes
(logs/journal.json, logs/MLEvolve.log, logs/monitor_*.json[l],
workspace/node_outputs/) and never imports MLEvolve, so it can watch a run
that is going on in another process.

It also hosts projects (projects/<slug>/, see dashboard/projects.py): a chat
with the Setup, Data, Task, Run and Analyst agents (dashboard/assistants.py)
that takes a user from an empty setup to a trained model. Writes to a run are
Stop/Abort requests (logs/control_requests.jsonl) and chat messages
(logs/chat_requests.jsonl), which utils/monitor.py forwards inside the run.
Every POST needs the per-server token embedded in the page, so another
website open in the same browser cannot send them.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

EVENTS_FILE = "monitor_events.jsonl"
RUN_FILE = "monitor_run.json"
CONTROL_FILE = "control_requests.jsonl"
# Chat: operator messages go to CHAT_REQUESTS (written only by this server);
# replies go to CHAT (written by the run while it is live, by this server after).
CHAT_REQUESTS_FILE = "chat_requests.jsonl"
CHAT_FILE = "chat.jsonl"
CHAT_LIMIT = 300
CHAT_MAX_CHARS = 4000
# A message with no reply after this long is no longer shown as "thinking".
CHAT_PENDING_S = 300
REVIEW_TIMEOUT_S = 300

# A run counts as live when its files changed within this window and no
# run_end event was written.
LIVE_WINDOW_S = 180
TIMELINE_LIMIT = 600
LOG_TAIL_LINES = 120
TERM_OUT_LIMIT = 60_000
CSV_ROW_LIMIT = 200

FILE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
    ".csv": "text/csv; charset=utf-8", ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".html": "text/plain; charset=utf-8",  # shown as text, never rendered
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg"}

# Log lines worth showing in the activity feed.
LOG_PATTERN = re.compile(
    r"\[(parse|best|draft|improve|debug|evolution|fusion|aggregation|select|eval)\]"
    r"|Executing deferred node|Submitted|Code review for node|ERROR|WARNING|Exception|"
    r"Operator requested|Dashboard requested|KeyboardInterrupt|Phase \d|Notebook exported"
)
LOG_LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\] (\w+): (.*)$")


# ── reading a run ───────────────────────────────────────────────────────────


class RunReader:
    """Cached, read-only view of one run folder. Re-reads a file only when it changes."""

    def __init__(self, root: Path):
        self.root = root
        self.logs = root / "logs"
        self.workspace = root / "workspace"
        self._lock = threading.Lock()
        self._journal_key = None
        self._journal: dict = {"nodes": [], "by_id": {}}
        self._events: list[dict] = []
        self._events_offset = 0
        self.pid = None  # launch PID (wsl.exe on Windows) when a project started this run

    # journal.json ---------------------------------------------------------

    def journal(self) -> dict:
        path = self.logs / "journal.json"
        try:
            st = path.stat()
        except FileNotFoundError:
            return self._journal
        key = (st.st_mtime, st.st_size)
        with self._lock:
            if key == self._journal_key:
                return self._journal
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return self._journal  # mid-write; keep the previous copy
            parents = raw.get("node2parent", {})
            nodes, by_id = [], {}
            for n in raw.get("nodes", []):
                if n.get("stage") == "root":
                    continue
                metric = (n.get("metric") or {}).get("value")
                summary = {
                    "id": n["id"], "step": n.get("step"), "stage": n.get("stage"),
                    "parent": parents.get(n["id"]), "branch": n.get("branch_id"),
                    "metric": metric, "buggy": bool(n.get("is_buggy")),
                    "exec_time": n.get("exec_time"), "created": n.get("created_time"),
                    "finished": n.get("finish_time"), "ctime": n.get("ctime"),
                    "exc_type": n.get("exc_type"),
                    "plan_head": _first_line(plan_markdown(n.get("plan"))),
                }
                nodes.append(summary)
                by_id[n["id"]] = n
            nodes.sort(key=lambda x: (x["ctime"] or 0))
            self._journal = {"nodes": nodes, "by_id": by_id,
                             "maximize": _maximize(raw.get("nodes", []))}
            self._journal_key = key
            return self._journal

    # monitor_events.jsonl --------------------------------------------------

    def events(self) -> list[dict]:
        path = self.logs / EVENTS_FILE
        with self._lock:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                return self._events
            if size < self._events_offset:  # file was replaced
                self._events, self._events_offset = [], 0
            if size > self._events_offset:
                with open(path, "rb") as f:
                    f.seek(self._events_offset)
                    chunk = f.read()
                end = chunk.rfind(b"\n") + 1  # leave a half-written last line for later
                for line in chunk[:end].splitlines():
                    try:
                        self._events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                self._events_offset += end
            return self._events

    def chat(self) -> list[dict]:
        """Operator messages and replies, oldest first."""
        messages = []
        for name, role in ((CHAT_REQUESTS_FILE, "operator"), (CHAT_FILE, None)):
            path = self.logs / name
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if role:
                    m = {"id": m.get("id"), "role": role, "text": m.get("text", ""), "kind": "message", "t": m.get("t")}
                messages.append(m)
        messages.sort(key=lambda m: m.get("t") or 0)
        answered = {m.get("reply_to") for m in messages if m.get("role") == "agent"}
        now = time.time()
        for m in messages:
            if m["role"] == "operator" and m.get("kind") == "message":
                m["pending"] = m.get("id") not in answered and now - (m.get("t") or 0) < CHAT_PENDING_S
        return messages[-CHAT_LIMIT:]

    def meta(self) -> dict:
        path = self.logs / RUN_FILE
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        meta = {"exp_name": self.root.name, **_config_meta(self.logs / "config.yaml")}
        first = _first_log_time(self.logs / "MLEvolve.log")
        if first:
            meta["started"] = first
        return meta

    def last_activity(self) -> float:
        # draft_review.jsonl gets a heartbeat while the run waits for the user's draft review
        times = [p.stat().st_mtime for p in (self.logs / EVENTS_FILE, self.logs / "MLEvolve.log",
                                            self.logs / "journal.json", self.logs / "draft_review.jsonl")
                 if p.exists()]
        return max(times) if times else 0.0

    def status(self, events: list[dict]) -> str:
        if any(e.get("type") == "run_end" for e in events[-20:]):
            return "finished"
        idle = time.time() - self.last_activity()
        if idle <= LIVE_WINDOW_S:
            return "live"
        # Quiet, but its process still runs: a long LLM call, a long script, or a wait for the user.
        from dashboard import wslexec
        if self.pid and wslexec.pid_alive(self.pid):
            return "live"
        # A monitored run with no run_end that went quiet probably crashed or was killed.
        return "stale" if (self.logs / RUN_FILE).exists() else "finished"

    def log_tail(self) -> list[dict]:
        path = self.logs / "MLEvolve.log"
        if not path.exists():
            return []
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - 400_000))
            text = f.read().decode("utf-8", errors="replace")
        out = []
        for line in text.splitlines():
            m = LOG_LINE.match(line)
            if m and LOG_PATTERN.search(m.group(3)):
                out.append({"time": m.group(1), "level": m.group(2), "msg": m.group(3)[:400]})
        return out[-LOG_TAIL_LINES:]

    def stop_facts(self, tail_lines: int = 12, max_errors: int = 5) -> dict:
        """Where the run's log ends, and the error lines in it (with their tracebacks). Read-only."""
        path = next((p for p in (self.logs / "MLEvolve.verbose.log", self.logs / "MLEvolve.log") if p.exists()), None)
        if path is None:
            return {"log": None, "last_lines": [], "errors": []}
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - 300_000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        errors, i = [], 0
        while i < len(lines):
            line = lines[i]
            if ERROR_LINE.search(line):
                block = [line]
                j = i + 1
                while j < len(lines) and not LOG_LINE.match(lines[j]) and len(block) < 25:
                    block.append(lines[j])  # traceback lines have no log prefix
                    j += 1
                errors.append("\n".join(block)[:3000])
                i = j
            else:
                i += 1
        return {"log": path.name, "last_lines": [l[:400] for l in lines[-tail_lines:]], "errors": errors[-max_errors:]}


ERROR_LINE = re.compile(r"\] (ERROR|CRITICAL):|^Traceback \(most recent call last\)|\b(MemoryError|Killed)\b")


def plan_markdown(plan) -> str:
    """Improve/debug plans are stored as JSON ({reason, module, plan, ...}); render them as Markdown."""
    text = plan or ""
    if not text.lstrip().startswith("{"):
        return text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(data, dict):
        return text
    parts = []
    if data.get("reason"):
        parts.append(f"**Why:** {data['reason']}")
    body = data.get("plan")
    if isinstance(body, str) and body.lstrip().startswith("{"):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            pass
    if isinstance(body, dict):
        parts.append("**Changes:**")
        parts += [f"#### {k.replace('_', ' ')}\n{v}" for k, v in body.items()]
    elif body:
        parts.append(f"**Changes:** {body}")
    return "\n\n".join(parts) or text


def _first_line(text) -> str:
    for line in (text or "").splitlines():
        line = line.replace("**Why:**", "").strip(" #*-")
        if line:
            return line[:200]
    return ""


def _maximize(nodes) -> bool | None:
    for n in nodes:
        m = (n.get("metric") or {}).get("maximize")
        if m is not None:
            return bool(m)
    return None


def _config_meta(path: Path) -> dict:
    """A few settings from the saved config.yaml, for runs without monitor_run.json (no YAML library needed)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    meta = {}
    for key in ("time_limit", "steps"):
        m = re.search(rf"^\s+{key}:\s*(\d+)\s*$", text, re.M)
        if m:
            meta[key] = int(m.group(1))
    for stage in ("code", "feedback"):
        m = re.search(rf"^\s+{stage}:\s*\n\s+model:\s*(\S+)", text, re.M)
        if m:
            meta[f"{stage}_model"] = m.group(1).strip("'\"")
    return meta


def _first_log_time(path: Path) -> float | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            m = LOG_LINE.match(f.readline())
        if m:
            return time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    except OSError:
        pass
    return None


# ── summarising ─────────────────────────────────────────────────────────────


def aggregate(events: list[dict]) -> dict:
    """Token, time and activity totals from the monitor events."""
    agents: dict[str, dict] = {}
    per_node: dict[str, dict] = {}
    open_llm: dict[int, dict] = {}
    open_exec: dict[str, dict] = {}
    timeline: list[dict] = []
    totals = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "llm_seconds": 0.0,
              "exec_seconds": 0.0, "execs": 0, "errors": 0, "estimated": False,
              "cost_usd_equivalent": 0.0}
    controls = []
    for e in events:
        kind = e.get("type")
        if kind == "llm_start":
            open_llm[e.get("call")] = e
        elif kind == "llm_end":
            open_llm.pop(e.get("call"), None)
            name = e.get("agent") or "unknown"
            a = agents.setdefault(name, {"agent": name, "calls": 0, "in_tokens": 0, "out_tokens": 0,
                                         "seconds": 0.0, "errors": 0, "estimated": False})
            a["calls"] += 1
            a["in_tokens"] += e.get("in_tokens", 0)
            a["out_tokens"] += e.get("out_tokens", 0)
            a["seconds"] += e.get("seconds", 0.0)
            a["errors"] += 0 if e.get("ok", True) else 1
            a["estimated"] |= bool(e.get("estimated"))
            totals["calls"] += 1
            totals["in_tokens"] += e.get("in_tokens", 0)
            totals["out_tokens"] += e.get("out_tokens", 0)
            totals["llm_seconds"] += e.get("seconds", 0.0)
            totals["errors"] += 0 if e.get("ok", True) else 1
            totals["estimated"] |= bool(e.get("estimated"))
            totals["cost_usd_equivalent"] += e.get("cost_usd") or 0.0
            if e.get("node"):
                n = per_node.setdefault(e["node"], {"calls": 0, "in_tokens": 0, "out_tokens": 0, "llm_seconds": 0.0})
                n["calls"] += 1
                n["in_tokens"] += e.get("in_tokens", 0)
                n["out_tokens"] += e.get("out_tokens", 0)
                n["llm_seconds"] += e.get("seconds", 0.0)
            timeline.append({"kind": "llm", "agent": name, "node": e.get("node"), "thread": e.get("thread"),
                             "start": e["t"] - e.get("seconds", 0.0), "end": e["t"], "ok": e.get("ok", True),
                             "tokens": e.get("in_tokens", 0) + e.get("out_tokens", 0)})
        elif kind == "exec_start":
            open_exec[e.get("node")] = e
        elif kind == "exec_end":
            open_exec.pop(e.get("node"), None)
            totals["execs"] += 1
            totals["exec_seconds"] += e.get("seconds", 0.0)
            timeline.append({"kind": "exec", "agent": "script", "node": e.get("node"), "thread": e.get("thread"),
                             "start": e["t"] - e.get("seconds", 0.0), "end": e["t"], "ok": e.get("ok", True)})
        elif kind == "control":
            controls.append({"t": e.get("t"), "action": e.get("action")})
    now = time.time()
    active = [{"kind": "llm", "agent": e.get("agent") or "unknown", "func": e.get("func"), "node": e.get("node"),
               "model": e.get("model"), "thread": e.get("thread"), "since": now - e["t"]} for e in open_llm.values()]
    active += [{"kind": "exec", "agent": "script", "node": e.get("node"), "thread": e.get("thread"),
                "since": now - e["t"]} for e in open_exec.values()]
    for a in agents.values():
        a["seconds"] = round(a["seconds"], 1)
    return {
        "totals": totals,
        "agents": sorted(agents.values(), key=lambda a: -(a["in_tokens"] + a["out_tokens"])),
        "per_node": per_node,
        "active": sorted(active, key=lambda a: -a["since"]),
        "timeline": timeline[-TIMELINE_LIMIT:],
        "controls": controls,
        "recorded": bool(events),
    }


def run_state(reader: RunReader) -> dict:
    journal = reader.journal()
    events = reader.events()
    agg = aggregate(events)
    status = reader.status(events)
    if status != "live":
        agg["active"] = []  # nothing can still be running
    nodes = journal["nodes"]
    maximize = journal.get("maximize")
    scored = [n for n in nodes if n["metric"] is not None and not n["buggy"]]
    best = None
    if scored:
        best = (max if maximize is not False else min)(scored, key=lambda n: n["metric"])
    for n in nodes:
        n["tokens"] = agg["per_node"].get(n["id"])
    # A draft's LLM calls happen before its node exists, so they name the hidden
    # search root; show those as "new draft" rather than as an unknown solution.
    known = {n["id"] for n in nodes}
    for item in agg["active"] + agg["timeline"]:
        if item.get("node") and item["node"] not in known:
            item["node"], item["new_draft"] = None, True
    meta = reader.meta()
    return {
        "name": reader.root.name, "status": status, "meta": meta, "now": time.time(),
        "last_activity": reader.last_activity(), "maximize": maximize,
        "nodes": nodes, "best": best["id"] if best else None,
        "counts": {"total": len(nodes), "passed": len(scored), "failed": sum(1 for n in nodes if n["buggy"])},
        "monitor": {k: v for k, v in agg.items() if k != "per_node"},
        "log": reader.log_tail(),
        "control_available": status == "live" and (reader.logs / RUN_FILE).exists(),
        "chat": reader.chat(),
        # live: the run's main agent answers and can act; review: Claude answers
        # questions about the saved results and cannot change anything.
        "chat_mode": ("live" if (reader.logs / RUN_FILE).exists() else "none") if status == "live"
                     else ("review" if shutil.which("claude") else "none"),
    }


# ── review chat for runs that are not live ──────────────────────────────────

REVIEW_SYSTEM = (
    "You help a machine learning engineer review a finished MLEvolve search run. Answer from the run "
    "facts given. Be concise and specific: cite solution ids and scores. Use Markdown. The run has "
    "ended, so you cannot change it; when asked to change something, say what to put in the task "
    "description or in `guide` for the next run. Never invent numbers that are not in the facts."
)


def review_context(reader: RunReader) -> str:
    """Compact, factual summary of a run for the review chat."""
    state = run_state(reader)
    nodes = state["nodes"]
    scored = sorted([n for n in nodes if n["metric"] is not None and not n["buggy"]],
                    key=lambda n: n["metric"], reverse=state["maximize"] is not False)
    lines = [f"Run: {state['name']} ({state['status']})",
             f"Settings: {json.dumps({k: v for k, v in state['meta'].items() if k != 'pid'}, default=str)}",
             f"Solutions: {state['counts']}; metric direction: "
             f"{'higher' if state['maximize'] is not False else 'lower'} is better",
             "Top solutions (id, kind, score, built from, plan):"]
    lines += [f"- {n['id'][:8]} {n['stage']} {n['metric']:.4f} from {(n['parent'] or '')[:8]}: {n['plan_head'][:200]}"
              for n in scored[:8]]
    failed = [n for n in nodes if n["buggy"]]
    if failed:
        lines.append("Failed solutions: " + ", ".join(f"{n['id'][:8]} ({n['exc_type'] or 'no score'})" for n in failed[:10]))
    totals = state["monitor"]["totals"]
    if state["monitor"]["recorded"]:
        lines.append(f"LLM use: {totals['calls']} calls, {totals['in_tokens']} input / {totals['out_tokens']} output tokens")
    if state["best"]:
        best = node_detail(reader, state["best"])
        lines.append(f"\nBest solution {best['id'][:8]} plan:\n{best['plan'][:3000]}")
        lines.append(f"\nBest solution analysis:\n{best['analysis'][:2000]}")
        for t in best["outputs"]["texts"]:
            if t["name"].lower().endswith(".md"):
                lines.append(f"\n{t['name']} of the best solution:\n{t['text'][:4000]}")
        lines.append(f"\nEnd of the best solution's output:\n{best['term_out'][-3000:]}")
    if state["status"] == "stale":
        facts = reader.stop_facts()
        lines.append("\nThe run stopped without its normal end record (crash, kill, or machine shutdown).")
        lines.append("Error lines in the log:\n" + ("\n".join(facts["errors"]) if facts["errors"] else "(none)"))
        lines.append(f"Last lines of {facts['log']}:\n" + "\n".join(facts["last_lines"]))
    return "\n".join(lines)


def _review_model(reader: RunReader) -> str:
    """The run's feedback model when it was a Claude Code model, else Sonnet."""
    model = str(reader.meta().get("feedback_model") or "")
    if model.startswith("claude-code") and model.partition(":")[2]:
        return model.partition(":")[2]
    return "sonnet"


def answer_review(reader: RunReader, msg_id: str, text: str) -> None:
    """Answer one review question with the local Claude Code CLI; runs on its own thread."""
    history = [m for m in reader.chat() if m.get("id") != msg_id][-10:]
    convo = "\n".join(f"{m['role']}: {m['text'][:800]}" for m in history) or "(none)"
    prompt = (f"{REVIEW_SYSTEM}\n\n# Run facts\n{review_context(reader)}\n\n"
              f"# Conversation so far\n{convo}\n\n# Question\n{text}")
    claude = shutil.which("claude")
    reply, kind = "", "reply"
    try:
        if claude is None:
            raise RuntimeError("the `claude` CLI is not on PATH for the dashboard server")
        cmd = [claude, "-p", "--output-format", "json", "--tools", "", "--setting-sources", "",
               "--no-session-persistence", "--model", _review_model(reader)]
        with tempfile.TemporaryDirectory(prefix="mlevolve-review-") as cwd:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                  cwd=cwd, timeout=REVIEW_TIMEOUT_S)
        result = json.loads(proc.stdout or "{}")
        if result.get("is_error") or result.get("subtype") != "success":
            raise RuntimeError(str(result.get("result") or proc.stderr)[:300])
        reply = result.get("result") or "(empty answer)"
    except Exception as e:
        reply, kind = f"Could not get an answer: {e}", "error"
    _append(reader.logs / CHAT_FILE, {"id": f"agent-{time.time():.6f}", "role": "agent", "text": reply,
                                      "kind": kind, "reply_to": msg_id, "t": time.time(), "source": "review"})


def _append(path: Path, obj: dict) -> None:
    with _append_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


_append_lock = threading.Lock()


def draft_code(reader: RunReader, draft_id: str) -> tuple[int, dict]:
    """The code of a first draft that the run wrote for review (runs/<run>/drafts/)."""
    from dashboard.gate_agents import DRAFT_REVIEW_FILE, read_jsonl
    for e in read_jsonl(reader.logs / DRAFT_REVIEW_FILE):
        if e.get("type") == "draft" and e.get("id") == draft_id:
            name = f"draft_{int(e.get('index') or 0)}_{str(e.get('node') or '')[:8]}.py"
            path = reader.root / "drafts" / name
            if path.is_file():
                return 200, {"code": path.read_text(encoding="utf-8", errors="replace"), "path": e.get("path"),
                             "index": e.get("index")}
    return 404, {"error": "unknown draft"}


def node_detail(reader: RunReader, node_id: str) -> dict | None:
    journal = reader.journal()
    n = journal["by_id"].get(node_id)
    if n is None:
        return None
    term = "".join(n.get("_term_out") or [])
    if len(term) > TERM_OUT_LIMIT:
        term = term[:TERM_OUT_LIMIT // 2] + "\n\n... (output shortened) ...\n\n" + term[-TERM_OUT_LIMIT // 2:]
    outputs = _node_outputs(reader, node_id, journal)
    return {
        "id": node_id, "stage": n.get("stage"), "step": n.get("step"),
        "metric": (n.get("metric") or {}).get("value"), "buggy": bool(n.get("is_buggy")),
        "plan": plan_markdown(n.get("plan")), "analysis": n.get("analysis") or "", "code": n.get("code") or "",
        "code_summary": n.get("code_summary") or "", "term_out": term,
        "exc_type": n.get("exc_type"), "exec_time": n.get("exec_time"),
        "created": n.get("created_time"), "finished": n.get("finish_time"),
        "outputs": outputs,
    }


def _node_outputs(reader: RunReader, node_id: str, journal: dict) -> dict:
    """Tables and figures for one solution, with an honest note on where they came from."""
    folder = reader.workspace / "node_outputs" / node_id
    note, overlap = None, []
    if folder.is_dir():
        manifest = folder / "manifest.json"
        if manifest.exists():
            try:
                overlap = json.loads(manifest.read_text(encoding="utf-8")).get("overlapping_runs", [])
            except (json.JSONDecodeError, OSError):
                pass
        if overlap:
            note = (f"{len(overlap)} other solution(s) ran at the same time and write to the same folder, "
                    "so a file here may come from one of them.")
    else:
        # Runs recorded before per-solution snapshots: only the shared folder exists.
        folder = reader.workspace / "working"
        note = ("This run has no per-solution copies. These are the latest files in the shared "
                "working/ folder, written by whichever solution ran last.")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in FILE_TYPES
                   and p.name != "manifest.json") if folder.is_dir() else []
    figures, tables, texts = [], [], []
    for p in files:
        rel = p.relative_to(reader.root).as_posix()
        if p.suffix.lower() in IMAGE_SUFFIXES:
            figures.append({"name": p.name, "path": rel})
        elif p.suffix.lower() == ".csv":
            tables.append({"name": p.name, "path": rel, **_read_csv(p)})
        elif p.suffix.lower() in {".md", ".txt"} and p.stat().st_size < 200_000:
            texts.append({"name": p.name, "path": rel, "text": p.read_text(encoding="utf-8", errors="replace")})
    return {"figures": figures, "tables": tables, "texts": texts, "note": note, "overlap": overlap}


def _read_csv(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            rows = []
            for i, row in enumerate(csv.reader(f)):
                if i > CSV_ROW_LIMIT:
                    break
                rows.append(row[:40])
        header, body = (rows[0], rows[1:]) if rows else ([], [])
        total = sum(1 for _ in open(path, "rb")) - 1
        return {"header": header, "rows": body, "total_rows": max(total, 0)}
    except OSError:
        return {"header": [], "rows": [], "total_rows": 0}


def list_runs(runs_root: Path, readers: dict) -> list[dict]:
    out = []
    for d in runs_root.iterdir() if runs_root.is_dir() else []:
        if not (d.is_dir() and (d / "logs").is_dir()):
            continue
        reader = readers.setdefault(d.name, RunReader(d))
        events = reader.events()
        out.append({"name": d.name, "status": reader.status(events), "last_activity": reader.last_activity(),
                    "recorded": bool(events)})
    return sorted(out, key=lambda r: -r["last_activity"])


# ── HTTP ────────────────────────────────────────────────────────────────────


def _best(reader: RunReader) -> float | None:
    journal = reader.journal()
    scored = [n["metric"] for n in journal["nodes"] if n["metric"] is not None and not n["buggy"]]
    if not scored:
        return None
    return (max if journal.get("maximize") is not False else min)(scored)


class Dashboard:
    def __init__(self, runs_root: Path, token: str | None = None, projects_root: Path | None = None):
        from dashboard import assistants, projects
        self.runs_root = runs_root.resolve()
        self.token = token or secrets.token_urlsafe(24)
        self.readers: dict[str, RunReader] = {}
        self._pids: dict = {}
        self._pids_t = 0.0
        self.projects = projects.ProjectStore((projects_root or REPO / "projects").resolve())
        self.agents = assistants.Assistants(self.projects, self.runs_root, self.reader, self.run_status)

    def run_status(self, name: str | None) -> str | None:
        reader = self.reader(name or "")
        return reader.status(reader.events()) if reader else None

    # projects and the sidebar --------------------------------------------------

    def items(self) -> list[dict]:
        """Projects plus runs that belong to no project, newest activity first."""
        out, linked = [], set()
        for p in self.projects.list():
            d = p.load()
            linked.update(d.get("runs", []))
            status = self.run_status(d.get("run")) if d.get("run") else None
            reader = self.reader(d["run"]) if d.get("run") else None
            from dashboard import projects as pj
            stage = pj.current_stage(d, status)
            best = _best(reader) if reader else None
            times = [d.get("created", 0)] + ([p.chat_file.stat().st_mtime] if p.chat_file.exists() else []) + \
                ([reader.last_activity()] if reader else [])
            out.append({"id": p.slug, "kind": "project", "name": d.get("name", p.slug),
                        "status": status or ("starting" if d.get("launch") else "setup"), "stage": stage,
                        "best": best, "last_activity": max(times)})
        for r in list_runs(self.runs_root, self.readers):
            if r["name"] in linked:
                continue
            reader = self.reader(r["name"])
            out.append({"id": "run:" + r["name"], "kind": "run", "name": r["name"], "status": r["status"],
                        "stage": "training" if r["status"] == "live" else "results",
                        "best": _best(reader) if reader else None, "last_activity": r["last_activity"]})
        return sorted(out, key=lambda x: -x["last_activity"])

    def create_project(self, name: str) -> tuple[int, dict]:
        name = (name or "").strip()
        if not name or len(name) > 80:
            return 400, {"error": "give the project a name of 1 to 80 characters"}
        project = self.projects.create(name)
        self.agents.submit(project, ("start",))
        return 200, {"ok": True, "id": project.slug}

    def project_view(self, item_id: str) -> tuple[int, dict]:
        from dashboard import projects as pj
        if item_id.startswith("run:"):
            reader = self.reader(item_id[4:])
            if reader is None:
                return 404, {"error": "unknown run"}
            state = run_state(reader)
            msgs = []
            for m in state["chat"]:
                if m["role"] == "operator":
                    msgs.append({**m, "role": "user", "agent": "terminal" if m.get("kind") == "terminal" else None})
                else:
                    msgs.append({**m, "agent": "analyst" if m.get("source") == "review" else "run"})
            done = [{"name": s, "status": "done", "note": ""} for s in pj.STAGES[:-2]]
            live = state["status"] == "live"
            done += [{"name": "training", "status": "now" if live else "done", "note": "running" if live else ""},
                     {"name": "results", "status": "todo" if live else "now", "note": ""}]
            return 200, {"id": item_id, "kind": "run", "name": reader.root.name, "run": reader.root.name,
                         "run_status": state["status"], "stages": done, "messages": msgs,
                         "busy": None, "chat_mode": state["chat_mode"]}
        project = self.projects.get(item_id)
        if project is None:
            return 404, {"error": "unknown project"}
        d = project.load()
        status = self.run_status(d.get("run")) if d.get("run") else None
        if d.get("run") and status in ("finished", "stale") and d["run"] not in d.get("announced", []):
            self._announce(project, d["run"], status)
            d = project.load()
        stages = pj.stages(d, status)
        if status == "live" and self.agents.pending_draft(d.get("run")) is not None:
            for s in stages:
                if s["name"] == "training":
                    s["note"] = "waiting for your draft review"
        return 200, {"id": project.slug, "kind": "project", "name": d.get("name"), "run": d.get("run"),
                     "run_status": status, "stages": stages, "messages": self._merged(project, d),
                     "busy": self.agents.busy.get(project.slug), "chat_mode": "project",
                     "config": d.get("config"), "task": d.get("task", {}).get("draft", ""),
                     "launching": bool(d.get("launch") and not d.get("run")),
                     "inspecting": sorted(st for (slug, st) in list(self.agents.inspecting) if slug == project.slug)}

    @staticmethod
    def _stop_note(reader: RunReader) -> str:
        facts = reader.stop_facts(tail_lines=1)
        last = facts["last_lines"][-1] if facts["last_lines"] else "(empty log)"
        err = facts["errors"][-1].splitlines()[0] if facts["errors"] else None
        return (f"\n\nThe log ends with:\n```\n{last}\n```\n"
                + (f"Last error line:\n```\n{err[:400]}\n```\n" if err else "The log has no error line. "))

    def error_report(self, reader: RunReader) -> dict:
        """The Errors tab for one run. The project that launched it gives launch.log and launch.sh. Cached per change."""
        from dashboard import errors
        key = (reader.last_activity(), reader.status(reader.events()))
        cached = getattr(reader, "_errors_cache", None)
        if cached and cached[0] == key:
            return cached[1]
        folder = next((p.folder for p in self.projects.list() if reader.root.name in p.load().get("runs", [])), None)
        out = errors.report(reader, folder / "launch.log" if folder else None,
                            folder / "launch.sh" if folder else None)
        reader._errors_cache = (key, out)
        return out

    def _announce(self, project, run: str, status: str) -> None:
        reader = self.reader(run)
        best = _best(reader) if reader else None
        state = run_state(reader) if reader else None
        rep = self.error_report(reader) if reader else None
        normal = rep is None or rep["kind"] == "ok"
        text = ((f"Training finished. " if normal else f"Training stopped: **{rep['headline']}**. ")
                + (f"Best score **{best:.4f}** (solution `{state['best'][:8]}`), "
                   f"{state['counts']['passed']} of {state['counts']['total']} solutions passed. "
                   if best is not None and state else "No solution got a score. ")
                + (self._stop_note(reader) if status == "stale" and reader else "")
                + ("" if normal else f"\n\n**What to do:** {rep['next']} ")
                + ("Ask me about the results, or open the monitor for tables and figures." if normal else
                   "The **Errors** tab in the monitor has the details."))
        actions = [{"label": "Summarize the results", "prefill": "@analyst Summarize the results: what was "
                    "tried, what worked, and the final scores."},
                   {"label": "Show monitor", "ui": "monitor"}]
        if not normal:
            actions.insert(0, {"label": "Show errors", "ui": "errors", "primary": True})
        project.post("agent", text, agent="analyst", actions=actions)
        project.update(lambda x: x.setdefault("announced", []).append(run))

    def _merged(self, project, d: dict) -> list[dict]:
        from dashboard.gate_agents import draft_messages
        msgs = project.messages()
        reader = self.reader(d["run"]) if d.get("run") else None
        if reader is not None:
            for m in reader.chat():
                if m["role"] == "operator":
                    if m.get("kind") == "terminal":
                        msgs.append({**m, "role": "user", "agent": "terminal"})
                else:
                    msgs.append({**m, "agent": "analyst" if m.get("source") == "review" else "run"})
            msgs += draft_messages(reader.root)
        msgs.sort(key=lambda m: m.get("t") or 0)
        # One-time choices (EDA, plan questions) keep their buttons only while they are open: a newer choice or
        # a step change ("Data plan approved", "Training started") closes them. A question back does not.
        # Inspector findings keep their buttons while they are the stage's newest, unhandled check and no step
        # change came after them; they never close other messages' buttons.
        from dashboard.inspect_agent import open_inspections
        still_open = open_inspections(d)
        closed = False
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if m.get("kind") == "inspect" and m.get("actions") and (closed or m.get("id") not in still_open):
                msgs[i] = {**m, "actions": None}
                continue
            if m.get("once") and m.get("actions") and closed:
                msgs[i] = {**m, "actions": None}
            if m.get("role") == "agent" and (m.get("once") or m.get("kind") == "action"):
                closed = True
        answered = {m.get("reply_to") for m in msgs if m.get("role") == "agent"}
        now = time.time()
        for m in msgs:
            if m.get("forward_id"):
                m["pending"] = m["forward_id"] not in answered and now - m.get("t", 0) < CHAT_PENDING_S
        return msgs[-CHAT_LIMIT:]

    def request_project_chat(self, item_id: str, text: str, agent: str | None) -> tuple[int, dict]:
        from dashboard import assistants
        text = (text or "").strip()
        if not text:
            return 400, {"error": "empty message"}
        if len(text) > CHAT_MAX_CHARS:
            return 400, {"error": f"message longer than {CHAT_MAX_CHARS} characters"}
        if item_id.startswith("run:"):
            return self.request_chat(item_id[4:], text)
        project = self.projects.get(item_id)
        if project is None:
            return 404, {"error": "unknown project"}
        if agent and agent not in assistants.AGENTS:
            return 400, {"error": "unknown agent"}
        d = project.load()
        status = self.run_status(d.get("run")) if d.get("run") else None
        target, body = assistants.route(text, d, status, agent)
        if target == "run" and status == "live" and not re.match(r"^(/|status\b|guide\b|stop\b|abort\b|seed\b)",
                                                                  body.strip(), re.I):
            pending = self.agents.pending_draft(d["run"])
            if pending is not None:  # the run waits on a draft review: the message answers it
                verdict, rest = self.agents.draft_command(body)
                msg = project.post("user", text, agent=target, **({"forward_id": f"dq-{time.time():.6f}"}
                                                                  if verdict == "ask" else {}))
                self.agents.draft_answer(d["run"], verdict, rest, msg_id=msg.get("forward_id") or msg["id"])
                return 200, {"ok": True, "id": msg["id"], "agent": target, "draft": verdict}
        if target == "run" and status == "live":
            reader = self.reader(d["run"])
            if not (reader.logs / RUN_FILE).exists():
                msg = project.post("user", text, agent=target)
                project.post("agent", "This run was started without dashboard recording, so it cannot receive "
                             "messages.", agent="run", kind="error", reply_to=msg["id"])
                return 200, {"ok": True, "id": msg["id"], "agent": target}
            fwd = f"op-{time.time():.6f}-{secrets.token_hex(3)}"
            msg = project.post("user", text, agent=target, forward_id=fwd)
            _append(reader.logs / CHAT_REQUESTS_FILE, {"id": fwd, "text": body, "t": time.time(), "mode": "live",
                                                       "via": "project"})
            return 200, {"ok": True, "id": msg["id"], "agent": target, "forwarded": True}
        msg = project.post("user", text, agent=target)
        self.agents.submit(project, ("message", target, body, msg["id"]))
        return 200, {"ok": True, "id": msg["id"], "agent": target}

    def request_project_action(self, item_id: str, action: str, args) -> tuple[int, dict]:
        project = self.projects.get(item_id)
        if project is None:
            return 404, {"error": "unknown project"}
        if not re.fullmatch(r"[a-z]+\.[a-z]+", action or ""):
            return 400, {"error": "bad action"}
        self.agents.submit(project, ("action", action, args if isinstance(args, dict) else {}))
        return 200, {"ok": True}

    def reader(self, name: str) -> RunReader | None:
        if not name or "/" in name or "\\" in name or name.startswith("."):
            return None
        d = self.runs_root / name
        if not (d / "logs").is_dir():
            return None
        reader = self.readers.setdefault(name, RunReader(d))
        reader.pid = self._run_pids().get(name)
        return reader

    def _run_pids(self) -> dict:
        """run name -> launch PID, from the projects. Cached for 10 s."""
        now = time.time()
        if now - self._pids_t > 10:
            pids = {}
            for p in self.projects.list():
                d = p.load()
                if d.get("run") and (d.get("launch") or {}).get("pid"):
                    pids[d["run"]] = d["launch"]["pid"]
            self._pids, self._pids_t = pids, now
        return self._pids

    def page(self) -> bytes:
        html = (HERE / "index.html").read_text(encoding="utf-8")
        return html.replace("__CONTROL_TOKEN__", self.token).encode("utf-8")

    def request_control(self, name: str, action: str) -> tuple[int, dict]:
        if action not in ("stop", "abort"):
            return 400, {"error": "action must be stop or abort"}
        reader = self.reader(name)
        if reader is None:
            return 404, {"error": "unknown run"}
        if not run_state(reader)["control_available"]:
            return 409, {"error": "this run is not live, or was started without dashboard recording"}
        _append(reader.logs / CONTROL_FILE, {"action": action, "t": time.time(), "source": "dashboard"})
        return 200, {"ok": True, "action": action}

    def request_chat(self, name: str, text: str) -> tuple[int, dict]:
        text = (text or "").strip()
        if not text:
            return 400, {"error": "empty message"}
        if len(text) > CHAT_MAX_CHARS:
            return 400, {"error": f"message longer than {CHAT_MAX_CHARS} characters"}
        reader = self.reader(name)
        if reader is None:
            return 404, {"error": "unknown run"}
        mode = run_state(reader)["chat_mode"]
        if mode == "none":
            return 409, {"error": "chat is not available for this run"}
        msg_id = f"op-{time.time():.6f}-{secrets.token_hex(3)}"
        _append(reader.logs / CHAT_REQUESTS_FILE, {"id": msg_id, "text": text, "t": time.time(), "mode": mode})
        if mode == "review":
            threading.Thread(target=answer_review, args=(reader, msg_id, text), daemon=True,
                             name="review-chat").start()
        return 200, {"ok": True, "id": msg_id, "mode": mode}


def make_handler(app: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MLEvolveDashboard/1"

        def log_message(self, fmt, *args):  # quiet: the page polls every few seconds
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=60" if cache else "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, default=str).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, app.page(), "text/html; charset=utf-8")
                if url.path == "/api/runs":
                    return self._json(200, list_runs(app.runs_root, app.readers))
                if url.path == "/api/items":
                    return self._json(200, app.items())
                if url.path == "/api/project":
                    return self._json(*app.project_view(q.get("id", "")))
                reader = app.reader(q.get("run", ""))
                if url.path.startswith("/api/") and reader is None:
                    return self._json(404, {"error": "unknown run"})
                if url.path == "/api/state":
                    return self._json(200, run_state(reader))
                if url.path == "/api/node":
                    detail = node_detail(reader, q.get("id", ""))
                    return self._json(200, detail) if detail else self._json(404, {"error": "unknown node"})
                if url.path == "/api/file":
                    return self._file(reader, q.get("path", ""))
                if url.path == "/api/draft":
                    return self._json(*draft_code(reader, q.get("id", "")))
                if url.path == "/api/errors":
                    return self._json(200, app.error_report(reader))
                return self._json(404, {"error": "not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _file(self, reader: RunReader, rel: str):
            target = (reader.root / rel).resolve()
            if not target.is_relative_to(reader.root.resolve()) or not target.is_file():
                return self._json(404, {"error": "not found"})
            ctype = FILE_TYPES.get(target.suffix.lower())
            if ctype is None:
                return self._json(403, {"error": "file type not served"})
            return self._send(200, target.read_bytes(), ctype, cache=target.suffix.lower() in IMAGE_SUFFIXES)

        def do_POST(self):
            url = urlparse(self.path)
            if url.path not in ("/api/control", "/api/chat", "/api/projects", "/api/project/chat",
                                "/api/project/action"):
                return self._json(404, {"error": "not found"})
            if not secrets.compare_digest(self.headers.get("X-Control-Token", ""), app.token):
                return self._json(403, {"error": "missing or wrong control token"})
            origin = self.headers.get("Origin")
            if origin and not re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", origin):
                return self._json(403, {"error": "cross-origin request refused"})
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 64_000)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json(400, {"error": "bad JSON"})
            if url.path == "/api/chat":
                code, obj = app.request_chat(str(body.get("run", "")), str(body.get("text", "")))
            elif url.path == "/api/projects":
                code, obj = app.create_project(str(body.get("name", "")))
            elif url.path == "/api/project/chat":
                code, obj = app.request_project_chat(str(body.get("id", "")), str(body.get("text", "")),
                                                     body.get("agent") or None)
            elif url.path == "/api/project/action":
                code, obj = app.request_project_action(str(body.get("id", "")), str(body.get("action", "")),
                                                       body.get("args") or {})
            else:
                code, obj = app.request_control(str(body.get("run", "")), str(body.get("action", "")))
            return self._json(code, obj)

    return Handler


class _Server(ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second server bind a port that is already
    # in use, and requests then go to either one. Refuse instead.
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def serve(runs_root: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    app = Dashboard(runs_root)
    try:
        httpd = _Server((host, port), make_handler(app))
    except OSError as e:
        raise SystemExit(f"Port {port} is already in use ({e}). Another dashboard may be running: "
                         f"stop it, or start this one with --port {port + 1}.")
    print(f"MLEvolve dashboard: http://{host}:{port}  (runs folder: {app.runs_root})")
    print("Press Ctrl+C to stop the dashboard. This does not stop any run.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Local dashboard for MLEvolve runs.")
    parser.add_argument("--runs", default=str(REPO / "runs"), help="folder that holds the run folders")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind; keep 127.0.0.1 unless you know you need more")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(Path(args.runs), args.host, args.port)


if __name__ == "__main__":
    main()
