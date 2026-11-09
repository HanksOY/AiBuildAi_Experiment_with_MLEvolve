"""The monitor's Errors tab: why a run stopped, its errors, and where its log ends. Read-only.

Sources, all inside the run folder or its project folder:
  logs/monitor_events.jsonl   run_end {reason, error?, traceback?, signal?} (utils/monitor.py)
  logs/MLEvolve*.log          ERROR lines and tracebacks, the last lines
  logs/journal.json           failed solutions (via server.node_detail)
  logs/draft_review.jsonl     a draft that waits for the user
  <project>/launch.log        output before Python starts; <project>/launch.sh for the hard stop
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

SOLUTION_ERRORS_MAX = 10
HARD_STOP_RE = re.compile(r"timeout\s+--signal=\w+\s+(?:--kill-after=\d+\s+)?(\d+)\s")
SAVE_RE = re.compile(r"in (save_run|dump_json|_replace_with_retry)\b")
PROGRESS_RE = re.compile(r"Progress: (\d+)/(\d+) steps completed")


def steps_done(lines: list[str]) -> int | None:
    """N when the newest progress line says N/N steps completed, else None."""
    for line in reversed(lines or []):
        if m := PROGRESS_RE.search(line):
            return int(m.group(1)) if m.group(1) == m.group(2) else None
    return None


def _fmt_time(t: float | None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) if t else "?"


def _pending_draft(logs: Path) -> dict | None:
    path = logs / "draft_review.jsonl"
    if not path.exists():
        return None
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    done = {e.get("id") for e in events if e.get("type") == "result"}
    waiting = [e for e in events if e.get("type") == "draft" and e.get("id") not in done]
    return waiting[-1] if waiting else None


def hard_stop_s(launch_sh: Path | None) -> int | None:
    if launch_sh is None or not launch_sh.exists():
        return None
    m = HARD_STOP_RE.search(launch_sh.read_text(encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None


def advice(error: str) -> str:
    e = error or ""
    if "MemoryError" in e or "Killed" in e or "out of memory" in e.lower():
        return ("The data does not fit in memory. Ask the Plan agent to read the data in chunks, or lower the "
                "parallel solutions with `@setup parallel=1`.")
    if "ModuleNotFoundError" in e or "ImportError" in e:
        return "A Python package is missing in `.venv`. Ask `@setup` to install it, then launch again."
    if "FileNotFoundError" in e:
        return "A file is missing. Check the data folder with `@data`, then launch again."
    if "claude" in e.lower() and ("login" in e.lower() or "auth" in e.lower()):
        return "The Claude login failed. Run `@setup check login`, then launch again."
    return "Ask the Analyst about it, or send the traceback to `@run` before you launch again."


def report(reader, launch_log: Path | None = None, launch_sh: Path | None = None) -> dict:
    """{kind: ok|warn|bad, headline, detail[], next, run_errors[], solution_errors[], last_lines[], launch[]}."""
    from dashboard import server
    events = reader.events()
    status = reader.status(events)
    end = next((e for e in reversed(events) if e.get("type") == "run_end"), None)
    facts = reader.stop_facts(tail_lines=15, max_errors=8)
    meta = reader.meta()
    started, last = meta.get("started"), reader.last_activity()
    run_errors = [{"time": (m.group(1)[11:19] if (m := re.match(r"\[(\S+ \S+?),\d+\]", e)) else ""),
                   "text": e} for e in facts["errors"]]
    kind, headline, detail, nxt = "ok", "", [], ""
    reason = ((end or {}).get("reason") or "finished") if end else None  # an end record without reason: finished
    hard = hard_stop_s(launch_sh)
    hit_hard = bool(hard and started and last and last - started >= hard - 120)

    if status == "live":
        pending = _pending_draft(reader.logs)
        if pending:
            kind, headline = "ok", f"Running: waiting for your review of draft {pending.get('index')}"
            detail = [f"Draft {pending.get('index')} has waited since {_fmt_time(pending.get('t'))}. The run's clock "
                      "is paused while it waits."]
            nxt = "Answer the draft card in the chat: accept, reject with a reason, or ask a question."
        else:
            headline = "Running"
            detail = [f"Last activity at {_fmt_time(last)}."]
            nxt = "Nothing to do."
    elif reason == "finished":
        headline, detail, nxt = "Finished normally", [f"Ended at {_fmt_time(end.get('t'))}."], "Nothing to do."
    elif reason in ("stopped", "aborted"):
        kind = "ok" if reason == "stopped" else "warn"
        headline = f"{'Stopped' if reason == 'stopped' else 'Aborted'} by you at {_fmt_time(end.get('t'))}"
        nxt = "Nothing is broken. Launch again when you want to train."
    elif reason == "interrupted" and hit_hard:
        kind = "warn"
        headline = f"Stopped by the hard time limit at {_fmt_time(end.get('t'))}"
        detail = [f"`launch.sh` stops a run after {hard / 3600:.1f} h of real time, whatever the run's own clock says."]
        nxt = "Raise the time limit with `@setup time_limit_h=<hours>`, or answer draft reviews sooner."
    elif reason == "interrupted":
        kind = "warn"
        headline = f"Stopped: interrupted ({end.get('signal') or 'SIGINT'}) at {_fmt_time(end.get('t'))}"
        detail = ["The run got an interrupt signal (Ctrl+C or `kill -INT`). Python recorded no error."]
        nxt = "Nothing is broken. Launch again when you want to train."
    elif reason == "crashed":
        kind = "bad"
        err = str(end.get("error") or "an error")
        tb = str(end.get("traceback") or "")
        done = steps_done(facts["last_lines"])
        if end.get("traceback"):
            run_errors.append({"time": _fmt_time(end.get("t")), "text": tb})
        if SAVE_RE.search(tb) and done:
            kind = "warn"
            headline = f"Crashed while saving, after all {done} steps finished, at {_fmt_time(end.get('t'))}"
            saved = [f for f in ("journal.json", "best_solution.py") if (reader.logs / f).exists()]
            detail = [f"The training is complete. The last save failed: {err[:160]}"]
            if saved:
                detail.append("Your results are saved: " + ", ".join(f"`logs/{f}`" for f in saved) + ".")
            if "PermissionError" in err:
                detail.append("On the Windows drive, a rename fails while another program (the dashboard or a virus "
                              "scanner) has the file open. The save now tries again for about 2 s.")
            nxt = "Nothing to redo. Look at the results in the Solutions tab."
        else:
            headline = f"Crashed: {err[:120]} at {_fmt_time(end.get('t'))}"
            detail = ["`run.py` stopped on an error it could not handle. The traceback is under Run errors."]
            nxt = advice(err + " " + tb)
    elif reason in ("killed", "exited"):
        kind = "bad" if reason == "killed" else "warn"
        headline = (f"Killed by {end.get('signal') or 'a signal'} at {_fmt_time(end.get('t'))}" if reason == "killed"
                    else f"Exited early at {_fmt_time(end.get('t'))}, without an error")
        detail = ["Something outside MLEvolve stopped the run, for example `wsl --shutdown`, a Windows restart, "
                  "or closing the WSL window."] if reason == "killed" else []
        nxt = "Keep WSL and the computer running during training, then launch again."
    else:  # no end record and no live process
        if run_errors:
            kind, headline = "bad", "Crashed (no end record)"
            detail = ["The log has an error, and the run wrote no end record. The newest error is under Run errors."]
            nxt = advice(run_errors[-1]["text"])
        elif hit_hard:
            kind, headline = "warn", "Stopped, most likely by the hard time limit (no end record)"
            detail = [f"The run was {(last - started) / 3600:.1f} h old, past the {hard / 3600:.1f} h hard limit of "
                      "`launch.sh`, so the `timeout` command most likely stopped it. If the computer slept, the "
                      "stop comes late."]
            nxt = ("Raise the time limit with `@setup time_limit_h=<hours>`, answer draft reviews sooner, or keep "
                   "the computer awake during training.")
        else:
            kind, headline = "bad", "Killed: no error recorded"
            detail = ["The run stopped without an error and without an end record. Only a hard kill does that: WSL "
                      "shut down, the computer slept or restarted, or the system ended the process (often for lack "
                      "of memory)."]
            nxt = "Keep the computer awake during training, then launch again. If it happens again, lower the " \
                  "parallel solutions."
        detail.append(f"The log ends at {_fmt_time(last)}.")

    state = server.run_state(reader)
    failed = [n for n in state["nodes"] if n["buggy"]][-SOLUTION_ERRORS_MAX:]
    solution_errors = []
    for n in reversed(failed):
        det = server.node_detail(reader, n["id"]) or {}
        solution_errors.append({"id": n["id"][:8], "step": n.get("step"), "exc": n.get("exc_type") or "no score",
                                "tail": (det.get("term_out") or det.get("analysis") or "")[-1500:]})
    launch = []
    if launch_log is not None and launch_log.exists():
        launch = launch_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
    return {"run": reader.root.name, "status": status, "kind": kind, "headline": headline, "detail": detail,
            "next": nxt, "run_errors": run_errors, "solution_errors": solution_errors,
            "last_lines": facts["last_lines"], "log": facts["log"], "launch": launch,
            "count": len(run_errors) + len(solution_errors)}
