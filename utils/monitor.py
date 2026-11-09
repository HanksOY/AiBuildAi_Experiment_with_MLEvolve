"""Run recording for the dashboard (dashboard/server.py).

Writes, under the run's log_dir:
  monitor_run.json          run metadata, written once at start
  monitor_events.jsonl      one JSON object per line:
      llm_start / llm_end   each LLM call: agent, node, model, tokens, seconds
      exec_start / exec_end each solution script run
      control               a stop/abort request received from the dashboard
      run_end               the run ended: reason finished | stopped | aborted | interrupted |
                            crashed (with error and traceback) | killed (with signal) | exited
  control_requests.jsonl    written by the dashboard, read here
  chat_requests.jsonl       operator messages from the dashboard chat, read here
  chat.jsonl                the conversation: operator messages and agent replies

and copies the tables and figures each solution writes into
workspace/node_outputs/<node_id>/, because every solution writes into the
same shared ./working/ folder and later solutions overwrite earlier ones.

Nothing here changes search behaviour, and no function raises: a recording
failure is logged once and otherwise ignored. Until run_started() is called
(tests, setup agent, plan gate before the run) every hook is a no-op.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

logger = logging.getLogger("MLEvolve")

EVENTS_FILE = "monitor_events.jsonl"
RUN_FILE = "monitor_run.json"
CONTROL_FILE = "control_requests.jsonl"
CHAT_REQUESTS_FILE = "chat_requests.jsonl"
CHAT_FILE = "chat.jsonl"
OUTPUTS_DIR = "node_outputs"

# Output files worth keeping per solution; models and data dumps are skipped.
SNAPSHOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".csv", ".md", ".json", ".txt", ".html"}
SNAPSHOT_MAX_BYTES = 25 * 1024 * 1024

# Local variable names that hold the node an agent is working on, innermost first.
_NODE_LOCALS = ("node", "result_node", "child_node", "cur_node", "new_node", "parent_node")

_lock = threading.Lock()
_state: dict = {"log_dir": None, "workspace": None, "llm": True, "snapshots": True}
_call_seq = 0
_active_execs: dict[str, float] = {}   # node_id -> start time
_open_llm: dict[int, dict] = {}        # call id -> handle, for current_activity()
_chat_queue = None                     # created by the first chat request
_recent_execs: list[tuple[str, float, float]] = []  # (node_id, start, end), last 50
_warned = False


def _warn_once(e: Exception) -> None:
    global _warned
    if not _warned:
        _warned = True
        logger.warning(f"Dashboard recording failed (further failures are silent): {e}")


def active() -> bool:
    return _state["log_dir"] is not None


def run_started(cfg) -> None:
    """Turn recording on for this run and write monitor_run.json."""
    mcfg = getattr(cfg, "monitor", None)
    if mcfg is not None and not getattr(mcfg, "enabled", True):
        return
    try:
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _state.update(
            log_dir=log_dir,
            started=time.time(),
            workspace=Path(cfg.workspace_dir),
            llm=bool(getattr(mcfg, "record_llm_calls", True)),
            snapshots=bool(getattr(mcfg, "snapshot_outputs", True)),
            ended=None,
            signal=None,
        )
        agent = cfg.agent
        meta = {
            "exp_name": str(cfg.exp_name),
            "started": time.time(),
            "pid": os.getpid(),
            "task_profile": str(getattr(getattr(cfg, "task", None), "profile", "kaggle")),
            "code_model": str(agent.code.model),
            "feedback_model": str(agent.feedback.model),
            "steps": int(agent.steps),
            "time_limit": int(agent.time_limit),
            "parallel": int(agent.search.parallel_search_num),
            "data_dir": str(cfg.data_dir),
        }
        (log_dir / RUN_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:  # never fatal
        _state["log_dir"] = None
        _warn_once(e)
        return
    if Path(sys.argv[0]).name == "run.py":  # only a real run, never tests or other tools
        _install_exit_hooks()


def _install_exit_hooks() -> None:
    """Record why the run ends when it is not a normal end: an uncaught error (with its traceback), Ctrl+C,
    SIGTERM or SIGHUP, or an exit without an end record. A hard kill (SIGKILL, WSL shutdown) cannot be caught."""
    try:
        previous_hook = sys.excepthook

        def on_uncaught(exc_type, exc, tb):
            if issubclass(exc_type, KeyboardInterrupt):
                run_ended("interrupted", signal=_state.get("signal") or "SIGINT")
            else:
                text = "".join(traceback.format_exception(exc_type, exc, tb))
                run_ended("crashed", error=f"{exc_type.__name__}: {exc}"[:500], traceback=text[-8000:])
            previous_hook(exc_type, exc, tb)
        sys.excepthook = on_uncaught

        if threading.current_thread() is threading.main_thread():
            if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
                def on_int(signum, frame):
                    _state["signal"] = "SIGINT"
                    signal.default_int_handler(signum, frame)
                signal.signal(signal.SIGINT, on_int)
            for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
                if sig is not None and signal.getsignal(sig) in (signal.SIG_DFL, None):
                    def on_term(signum, frame):
                        run_ended("killed", signal=signal.Signals(signum).name)
                        signal.signal(signum, signal.SIG_DFL)
                        os.kill(os.getpid(), signum)
                    signal.signal(sig, on_term)
        atexit.register(lambda: run_ended("exited"))  # no-op after any other end record
    except Exception as e:  # never fatal
        _warn_once(e)


def _emit(event: dict) -> None:
    log_dir = _state["log_dir"]
    if log_dir is None:
        return
    event.setdefault("t", time.time())
    event.setdefault("thread", threading.current_thread().name)
    line = json.dumps(event, default=str)
    try:
        with _lock, open(log_dir / EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        _warn_once(e)


def run_ended(reason: str = "finished", **extra) -> None:
    """Write the run_end record once; later calls (an exit hook after a normal end) do nothing."""
    if _state["log_dir"] is None or _state.get("ended") == _state["log_dir"]:
        return
    _state["ended"] = _state["log_dir"]  # once per run folder
    _emit({"type": "run_end", "reason": reason, **extra})


# ── LLM calls ───────────────────────────────────────────────────────────────


def _caller() -> tuple[str | None, str | None, str | None]:
    """(agent module, function, node id) of the code that made the LLM call.

    Agents do not say who they are when they call the LLM, so this reads the
    call stack: the innermost frame under agents/ or engine/ names the agent,
    and the innermost local called node/result_node/... names the solution.
    """
    agent = func = node_id = None
    frame = sys._getframe(2)
    while frame is not None and (agent is None or node_id is None):
        path = frame.f_code.co_filename.replace("\\", "/")
        if agent is None and ("/agents/" in path or "/engine/" in path):
            agent, func = Path(path).stem, frame.f_code.co_name
        if node_id is None:
            for name in _NODE_LOCALS:
                value = frame.f_locals.get(name)
                nid = getattr(value, "id", None)
                if isinstance(nid, str) and len(nid) == 32:
                    node_id = nid
                    break
        frame = frame.f_back
    return agent, func, node_id


def llm_start(model: str, kind: str) -> dict | None:
    """Record the start of an LLM call; returns a handle for llm_end()."""
    global _call_seq
    if not (active() and _state["llm"]):
        return None
    try:
        agent, func, node_id = _caller()
    except Exception:
        agent = func = node_id = None
    with _lock:
        _call_seq += 1
        call_id = _call_seq
    handle = {"call": call_id, "t0": time.time(), "agent": agent, "func": func, "node": node_id,
              "model": model, "kind": kind}
    with _lock:
        _open_llm[call_id] = handle
    _emit({"type": "llm_start", **{k: v for k, v in handle.items() if k != "t0"}})
    return handle


def llm_end(handle: dict | None, in_tokens: int | None = None, out_tokens: int | None = None,
            ok: bool = True, error: str | None = None, estimated: bool = False,
            cost_usd: float | None = None) -> None:
    if handle is None:
        return
    with _lock:
        _open_llm.pop(handle["call"], None)
    _emit({
        "type": "llm_end",
        "call": handle["call"], "agent": handle["agent"], "func": handle["func"],
        "node": handle["node"], "model": handle["model"], "kind": handle["kind"],
        "seconds": round(time.time() - handle["t0"], 3),
        "in_tokens": int(in_tokens or 0), "out_tokens": int(out_tokens or 0),
        "estimated": estimated, "ok": ok, "error": (error or "")[:300] or None,
        "cost_usd": cost_usd,
    })


def current_activity() -> list[dict]:
    """LLM calls and solution scripts running right now in this process (for the main agent)."""
    now = time.time()
    with _lock:
        calls = [{"what": "llm_call", "agent": h["agent"], "step": h["func"],
                  "solution": (h["node"] or "")[:8] or None, "seconds": round(now - h["t0"])}
                 for h in _open_llm.values() if h["agent"] != "main_agent"]
        scripts = [{"what": "solution_script", "solution": n[:8], "seconds": round(now - t0)}
                   for n, t0 in _active_execs.items()]
    return calls + scripts


def estimate_tokens(text) -> int:
    """About 4 characters per token; used only where the backend reports no usage."""
    try:
        return max(0, len(text if isinstance(text, str) else json.dumps(text, default=str)) // 4)
    except Exception:
        return 0


# ── solution script runs ────────────────────────────────────────────────────


def exec_started(node_id) -> float:
    t0 = time.time()
    if active() and node_id:
        with _lock:
            _active_execs[str(node_id)] = t0
        _emit({"type": "exec_start", "node": str(node_id)})
    return t0


def exec_finished(node_id, t0: float, ok: bool = True) -> None:
    if not (active() and node_id):
        return
    node_id, t1 = str(node_id), time.time()
    with _lock:
        _active_execs.pop(node_id, None)
        # Runs still going, or finished after this one started, shared working/.
        overlap = sorted(({n for n, s in _active_execs.items() if s < t1}
                          | {n for n, s, e in _recent_execs if s < t1 and e > t0}) - {node_id})
        _recent_execs.append((node_id, t0, t1))
        del _recent_execs[:-50]
    files = _snapshot(node_id, t0, t1, overlap) if _state["snapshots"] else []
    _emit({"type": "exec_end", "node": node_id, "seconds": round(t1 - t0, 3), "ok": ok,
           "files": files, "overlap": overlap})


def _snapshot(node_id: str, t0: float, t1: float, overlap: list[str]) -> list[str]:
    """Copy files this run wrote to working/ into node_outputs/<node_id>/."""
    try:
        working = _state["workspace"] / "working"
        if not working.is_dir():
            return []
        dest = _state["workspace"] / OUTPUTS_DIR / node_id
        copied = []
        for p in sorted(working.iterdir()):
            if not p.is_file() or p.suffix.lower() not in SNAPSHOT_SUFFIXES:
                continue
            st = p.stat()
            if st.st_mtime < t0 - 1 or st.st_size > SNAPSHOT_MAX_BYTES:
                continue
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / p.name)
            copied.append(p.name)
        if copied:
            manifest = {"node": node_id, "start": t0, "end": t1, "files": copied,
                        # Solutions that ran at the same time write to the same
                        # folder, so a file here may come from one of them.
                        "overlapping_runs": overlap}
            (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return copied
    except Exception as e:
        _warn_once(e)
        return []


# ── chat ────────────────────────────────────────────────────────────────────


def chat_message(role: str, text: str, kind: str = "reply", reply_to=None, msg_id=None) -> None:
    """Append one message to logs/chat.jsonl (the dashboard shows this file)."""
    log_dir = _state["log_dir"]
    if log_dir is None:
        return
    line = json.dumps({"id": msg_id or f"{role}-{time.time():.6f}", "role": role, "text": text,
                       "kind": kind, "reply_to": reply_to, "t": time.time()}, default=str)
    try:
        with _lock, open(log_dir / CHAT_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        _warn_once(e)


# ── dashboard control requests ──────────────────────────────────────────────


class _Tail:
    """New complete lines of a file that another process appends to.

    Lines stamped before `since` (the run's start) are skipped, so requests
    left over from an earlier run are ignored but requests sent while this run
    was still setting up are kept.
    """

    def __init__(self, path: Path, since: float):
        self.path = path
        self.since = since
        self.offset = 0

    def lines(self) -> list[str]:
        if not self.path.exists() or self.path.stat().st_size <= self.offset:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1
        self.offset += end
        fresh = []
        for line in chunk[:end].decode("utf-8", errors="replace").splitlines():
            try:
                stamp = json.loads(line).get("t")
            except (json.JSONDecodeError, AttributeError):
                stamp = None
            if stamp is None or stamp >= self.since:
                fresh.append(line)
        return fresh


def start_control_watcher(cfg, bus, interval: float = 1.0, chat_handler=None) -> threading.Thread | None:
    """Poll the dashboard's request files and act on them.

    control_requests.jsonl: stop/abort, forwarded to the ControlBus.
    chat_requests.jsonl:    chat messages, passed to chat_handler(text, msg_id)
                            (the Main Agent) on a short-lived thread, because
                            answering may take an LLM call.
    Works without a terminal, so an unattended run can still be steered from
    the dashboard. Requests already in the files when the run starts are ignored.
    """
    mcfg = getattr(cfg, "monitor", None)
    if not active() or (mcfg is not None and not getattr(mcfg, "control", True)):
        return None
    # 60 s margin: the dashboard stamps requests with the Windows clock, the run uses WSL's.
    since = (_state.get("started") or time.time()) - 60
    control = _Tail(_state["log_dir"] / CONTROL_FILE, since)
    chat = _Tail(_state["log_dir"] / CHAT_REQUESTS_FILE, since)

    def loop():
        while True:
            time.sleep(interval)
            try:
                for line in control.lines():
                    handle_control_request(line, bus)
                for line in chat.lines():
                    _dispatch_chat(line, chat_handler)
            except Exception as e:
                _warn_once(e)

    thread = threading.Thread(target=loop, name="dashboard-control", daemon=True)
    thread.start()
    return thread


def _dispatch_chat(line: str, handler) -> None:
    try:
        req = json.loads(line)
        text, msg_id = str(req.get("text", "")).strip(), req.get("id")
    except (json.JSONDecodeError, AttributeError):
        return
    if not text:
        return
    if handler is None:
        chat_message("agent", "This run has no main agent to answer chat messages.", kind="error", reply_to=msg_id)
        return
    # One worker answers messages in the order they were sent, so an answer can
    # rely on guidance given in an earlier message.
    global _chat_queue
    if _chat_queue is None:
        import queue

        _chat_queue = queue.Queue()

        def work():
            while True:
                fn, args = _chat_queue.get()
                try:
                    fn(*args)
                except Exception as e:
                    _warn_once(e)

        threading.Thread(target=work, name="dashboard-chat", daemon=True).start()
    _chat_queue.put((handler, (text, msg_id)))


def handle_control_request(line: str, bus) -> str | None:
    """Apply one request line; returns the action applied, or None."""
    try:
        action = json.loads(line).get("action")
    except (json.JSONDecodeError, AttributeError):
        return None
    if action == "stop":
        bus.request_stop()
        bus.push("stop")
    elif action == "abort":
        bus.request_abort()
        bus.push("abort")
    else:
        return None
    logger.info(f"Dashboard requested {action}.")
    _emit({"type": "control", "action": action})
    return action
