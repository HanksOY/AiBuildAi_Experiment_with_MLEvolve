"""A run that waits quietly (draft review, long LLM call) is live, not "stale"; a stopped run shows why."""

import json
import os
import time
from pathlib import Path

from agents import draft_review
from dashboard import server, wslexec

from test_gate_agents import FakeGate, answer_later, app, draft_agent, last, task_accepted, write_draft_request  # noqa: F401


def quiet_run(root: Path, age_s: float = 600) -> Path:
    """A monitored run folder whose files were last written `age_s` seconds ago."""
    logs = root / "logs"
    logs.mkdir(parents=True)
    (logs / server.RUN_FILE).write_text(json.dumps({"exp_name": root.name}))
    (logs / "MLEvolve.log").write_text("[2026-11-08 23:33:03,187] INFO: Draft 1 written for review\n")
    past = time.time() - age_s
    for p in logs.iterdir():
        os.utime(p, (past, past))
    return root


def test_quiet_run_is_stale_without_a_live_process(tmp_path):
    reader = server.RunReader(quiet_run(tmp_path / "r"))
    assert reader.status([]) == "stale"


def test_quiet_run_with_a_live_process_is_live(tmp_path, monkeypatch):
    reader = server.RunReader(quiet_run(tmp_path / "r"))
    reader.pid = 1234
    monkeypatch.setattr(wslexec, "pid_alive", lambda pid: pid == 1234)
    assert reader.status([]) == "live"
    monkeypatch.setattr(wslexec, "pid_alive", lambda pid: False)
    assert reader.status([]) == "stale"


def test_draft_review_heartbeat_keeps_the_run_live(tmp_path):
    root = quiet_run(tmp_path / "r")
    (root / "logs" / "draft_review.jsonl").write_text("{}\n")  # fresh: the run touched it while waiting
    assert server.RunReader(root).status([]) == "live"


def test_waiting_run_touches_the_review_file(tmp_path, monkeypatch):
    monkeypatch.setattr(draft_review, "POLL_S", 0.05)
    monkeypatch.setattr(draft_review, "HEARTBEAT_S", 0.1)
    touched = []
    real = os.utime
    monkeypatch.setattr(draft_review.os, "utime", lambda p, *a, **k: (touched.append(Path(p).name), real(p, *a, **k)))
    agent, node, run = draft_agent(tmp_path)
    answer_later(run, [{"verdict": "accept"}], delay=0.6)
    assert draft_review.review_draft(agent, node, 1, 1) == draft_review.ACCEPT
    assert touched and set(touched) == {"draft_review.jsonl"}


def test_dashboard_gives_the_reader_the_project_launch_pid(app, tmp_path, monkeypatch):
    run = "20261108_235715_flow"
    quiet_run(tmp_path / "runs" / run)
    p = app.projects.create("flow")
    p.update(lambda x: x.update(run=run, launch={"t": time.time(), "pid": 4321}))
    monkeypatch.setattr(wslexec, "pid_alive", lambda pid: pid == 4321)
    assert app.run_status(run) == "live"


def test_live_waiting_run_blocks_a_second_launch_and_shows_the_wait(app, tmp_path, monkeypatch):
    gate = FakeGate()
    p = task_accepted(app, tmp_path, gate)
    for action in ("eda.accept", "eda.approve"):
        app.agents.run_sync(p, ("action", action, {}))
    p.update(lambda x: x["plan"].update(status="approved", final="TASK"))
    run = "20261108_235715_flow"
    write_draft_request(tmp_path / "runs" / run)
    quiet_run_dir = tmp_path / "runs" / run / "logs"
    past = time.time() - 600
    for f in quiet_run_dir.iterdir():
        os.utime(f, (past, past))
    (quiet_run_dir / server.RUN_FILE).write_text("{}")
    os.utime(quiet_run_dir / server.RUN_FILE, (past, past))
    p.update(lambda x: x.update(run=run, runs=[run], launch={"t": past, "pid": 4321}))
    monkeypatch.setattr(wslexec, "pid_alive", lambda pid: pid == 4321)
    spawned = []
    monkeypatch.setattr(wslexec, "spawn_detached", lambda *a: spawned.append(a) or 1)

    view = app.project_view(p.slug)[1]
    assert view["run_status"] == "live"
    assert next(s for s in view["stages"] if s["name"] == "training")["note"] == "waiting for your draft review"
    assert not any(m.get("agent") == "analyst" for m in p.messages())  # no "stopped responding"
    app.agents.run_sync(p, ("action", "run.launch", {}))
    assert not spawned and "already live" in last(p)["text"]


def test_stop_facts_and_stale_announcement_show_the_log_end(app, tmp_path):
    run = "20261108_120000_flow"
    root = quiet_run(tmp_path / "runs" / run)
    (root / "logs" / "MLEvolve.verbose.log").write_text(
        "[2026-11-08 12:00:00,000] INFO: Step 1/3\n"
        "[2026-11-08 12:00:01,000] ERROR: execution failed\n"
        "Traceback (most recent call last):\n"
        '  File "runfile.py", line 3, in <module>\n'
        "MemoryError\n"
        "[2026-11-08 12:00:02,000] INFO: Merging all steps...\n")
    facts = server.RunReader(root).stop_facts()
    assert len(facts["errors"]) == 1 and facts["errors"][0].endswith("MemoryError")
    assert facts["last_lines"][-1].endswith("Merging all steps...")
    assert "Error lines in the log" in server.review_context(server.RunReader(root))

    p = app.projects.create("flow")
    p.update(lambda x: x.update(run=run, runs=[run]))
    app.project_view(p.slug)
    msg = last(p)
    assert msg["agent"] == "analyst" and "Training stopped: **Crashed" in msg["text"]
    assert msg["actions"][0]["ui"] == "errors"
    assert "Merging all steps" in msg["text"] and "execution failed" in msg["text"]
