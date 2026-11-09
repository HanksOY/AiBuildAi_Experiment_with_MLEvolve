"""Inspector agent (dashboard/inspector.py, dashboard/inspect_agent.py), the Errors tab (dashboard/errors.py) and the
run's exit records (utils/monitor.py)."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dashboard import assistants, errors, inspect_agent, inspector, server

from test_gate_agents import FakeGate, TASK_MD, app, last, task_accepted  # noqa: F401

REPO = Path(__file__).resolve().parent.parent


def inspector_answer(checks):
    return json.dumps({"checks": checks})


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(inspect_agent.InspectAgent, "inspect_mode", "sync")
    calls = []

    def fake(prompt, model="sonnet", timeout=0):
        calls.append(prompt)
        if prompt.startswith("You are the Inspector agent"):
            return inspector_answer([{"label": "Agrees with the data", "ok": True, "detail": ""},
                                     {"label": "Class balance handled", "ok": False, "severity": "warning",
                                      "detail": "Say how to handle the imbalance."}])
        if 'Answer with JSON only: {"action"' in prompt:
            return json.dumps({"action": "revise", "task": TASK_MD + "\n\nUse class weights.", "reply": "Done."})
        return TASK_MD
    monkeypatch.setattr(assistants, "ask_claude", fake)
    return calls


# ── rule checks ─────────────────────────────────────────────────────────────


def meningitis_like(tmp_path):
    data = tmp_path / "public"
    data.mkdir()
    (data / "pos.csv").write_text("SUBJECT_ID,CLASSIFIER\n1,1\n2,1\n")
    (data / "neg.xls").write_text("SUBJECT_ID,CLASSIFIER\n3,0\n")   # a CSV with an .xls name
    (data / "~$prompt.doc").write_bytes(b"\x00lock")
    (data / "prompt.doc").write_bytes(b"\xd0\xcf\x11\xe0word")
    prof = {"files": [{"name": n, "size": (data / n).stat().st_size} for n in sorted(os.listdir(data))],
            "tables": [{"name": "pos.csv", "role": "train", "rows": 214, "columns": 983, "missing_cells": 0,
                        "target_guess": "CLASSIFIER", "target_counts": {"1": 214}}]}
    return {"data": {"path": str(data), "profile": prof}}, data


def test_data_rules_find_the_testing_3_problems(tmp_path):
    d, folder = meningitis_like(tmp_path)
    facts = inspector.file_facts(d["data"]["profile"], folder)
    checks = {c["label"]: c for c in inspector.data_rules(d, facts)}
    assert not checks["File type matches its name"]["ok"]
    assert "`neg.xls` is really a CSV file" in checks["File type matches its name"]["detail"]
    assert not checks["Only data files in the folder"]["ok"] and "~$prompt.doc" in checks["Only data files in the folder"]["detail"]
    assert not checks["Target has 2 or more values in pos.csv"]["ok"]
    assert checks["No answer or label file"]["ok"] and checks["Target column found"]["ok"]


def test_answer_files_are_flagged():
    d = {"data": {"profile": {"files": [{"name": "train.csv"}, {"name": "test_labels.csv"}], "tables": []}}}
    facts = inspector.file_facts(d["data"]["profile"], None)
    c = next(c for c in inspector.data_rules(d, facts) if c["label"] == "No answer or label file")
    assert not c["ok"] and "test_labels.csv" in c["detail"]


def test_task_rules_require_the_score_line():
    checks = {c["label"]: c for c in inspector.task_rules({"task": {"draft": "# Goal\nPredict y with ROC AUC, 5-fold."}})}
    assert checks["Metric named"]["ok"] and checks["Validation method named"]["ok"]
    assert not checks["Score line required"]["ok"] and checks["Score line required"]["blocking"]


def test_plan_budget_for_the_testing_3_settings():
    d = {"config": {"steps": 10, "time_limit_h": 0.5, "parallel": 3, "initial_drafts": 3},
         "plan": {"plan": "1. logistic regression\n2. random forest, extra trees\n3. LightGBM, XGBoost"}}
    facts = [{"name": "neg.xls", "size": 645_000_000, "ext": ".xls", "kind": "csv"}]
    checks, fix = inspector.plan_rules(d, facts)
    budget = checks[0]
    assert not budget["ok"] and budget["blocking"] and "Estimate 1.0 h, limit 0.5 h" in budget["detail"]
    assert fix == {"label": "Set the time limit to 1 h", "changes": {"time_limit_h": 1.0}}
    assert not next(c for c in checks if c["label"] == "Memory for parallel scripts")["ok"]
    d["config"]["time_limit_h"] = 2.0
    checks, fix = inspector.plan_rules(d, facts)
    assert checks[0]["ok"] and fix is None


def test_setup_rules():
    d = {"config": {"steps": 100, "time_limit_h": 1, "parallel": 16, "exec_timeout_min": 90}}
    checks = {c["label"]: c for c in inspector.setup_rules(d, cpus=12)}
    assert not checks["Time per step"]["ok"] and not checks["Parallel solutions fit the CPU"]["ok"]
    assert not checks["Script time limit fits the budget"]["ok"]


def test_llm_answer_parsing_keeps_at_most_5_checks():
    many = [{"label": f"c{i}", "ok": i % 2 == 0, "severity": "problem", "detail": "x"} for i in range(8)]
    out = inspector.parse_llm({"checks": many + ["junk", {"label": ""}]})
    assert len(out) == 5 and all(c["src"] == "Claude" for c in out)
    assert out[1]["blocking"] and not out[0]["blocking"]


# ── the agent in the chat ───────────────────────────────────────────────────


def test_inspector_checks_data_and_task_and_sends_issues(app, tmp_path, on):
    p = task_accepted(app, tmp_path, FakeGate())
    msgs = [m for m in p.messages() if m.get("agent") == "inspect"]
    assert [m["inspect_stage"] for m in msgs] == ["data", "task"]
    task_msg = msgs[-1]
    assert "Score line required" in json.dumps(task_msg["checks"])          # the rule layer
    assert any(c["src"] == "Claude" and c["label"] == "Class balance handled" for c in task_msg["checks"])
    assert any("You are the Inspector agent" in c and "# Result to check: the task draft" in c for c in on)
    d = p.load()
    assert d["inspect"]["task"]["status"] == "open" and d["inspect"]["task"]["msg"] == task_msg["id"]
    stages = {s["name"]: s for s in app.project_view(p.slug)[1]["stages"]}
    assert "problem" in stages["task"]["note"]

    # The task was accepted after the check, so its buttons are closed (a step change came after them).
    view_msg = next(m for m in app.project_view(p.slug)[1]["messages"] if m.get("id") == task_msg["id"])
    assert view_msg["actions"] is None

    # "Send" goes to the Task agent as a user message, and the new draft gets a new check.
    p.update(lambda x: x["task"].update(accepted=False))
    app.agents.run_sync(p, ("action", "inspect.send", {"stage": "task"}))
    sent = [m for m in p.messages() if m["role"] == "user"][-1]
    assert sent["agent"] == "task" and "Score line required" in sent["text"]
    assert "Use class weights." in p.load()["task"]["draft"]
    assert p.load()["inspect"]["task"]["status"] == "open"  # the new draft's check
    app.agents.run_sync(p, ("action", "inspect.send", {"stage": "data"}))
    assert "Add these notes about the data to the task" in [m for m in p.messages() if m["role"] == "user"][-1]["text"]


def test_ignore_closes_the_buttons_and_the_stage_note(app, tmp_path, on):
    p = task_accepted(app, tmp_path, FakeGate())
    p.update(lambda x: x["task"].update(accepted=False))
    app.agents.run_sync(p, ("action", "task.redraft", {}))
    rec = p.load()["inspect"]["task"]
    view = app.project_view(p.slug)[1]
    assert next(m for m in view["messages"] if m.get("id") == rec["msg"])["actions"]
    app.agents.run_sync(p, ("action", "inspect.ignore", {"stage": "task"}))
    view = app.project_view(p.slug)[1]
    assert next(m for m in view["messages"] if m.get("id") == rec["msg"])["actions"] is None
    assert "problem" not in next(s for s in view["stages"] if s["name"] == "task")["note"]
    app.agents.run_sync(p, ("action", "inspect.ignore", {"stage": "task"}))
    assert "already handled" in last(p)["text"]


def test_a_check_of_an_old_result_is_dropped(app, tmp_path, on):
    p = task_accepted(app, tmp_path, FakeGate())
    old = inspector.digest("task", p.load())
    p.update(lambda x: x["task"].update(draft="# Goal\nsomething else"))
    n = len(p.messages())
    app.agents._inspect_run(p, "task", old)
    assert len(p.messages()) == n


def test_claude_failure_leaves_the_rule_checks(app, tmp_path, on, monkeypatch):
    def broken(prompt, model="sonnet", timeout=0):
        if prompt.startswith("You are the Inspector agent"):
            raise RuntimeError("rate limited")
        return TASK_MD
    monkeypatch.setattr(assistants, "ask_claude", broken)
    p = task_accepted(app, tmp_path, FakeGate())
    msg = [m for m in p.messages() if m.get("agent") == "inspect"][-1]
    assert "Only the rule checks ran" in msg["text"] and "rate limited" in msg["text"]
    assert all(c["src"] == "Rule" for c in msg["checks"])


def test_inspector_off_and_on_demand(app, tmp_path, on):
    p = app.projects.create("quiet")
    p.update(lambda x: x["config"].update(inspector=False))
    app.agents.inspect(p, "setup")
    assert not [m for m in p.messages() if m.get("agent") == "inspect"]
    p.update(lambda x: x["config"].update(inspector=True, steps=500, time_limit_h=1))
    app.agents.run_sync(p, ("message", "inspect", "check again", "m1"))
    msg = last(p)
    assert msg["agent"] == "inspect" and msg["inspect_stage"] == "setup" and "Time per step" in json.dumps(msg["checks"])
    assert assistants.route("@inspector check the plan", p.load(), None)[0] == "inspect"


def test_fix_button_changes_the_time_limit_and_shows_the_plan_again(app, tmp_path, on):
    gate = FakeGate()
    p = task_accepted(app, tmp_path, gate)
    p.update(lambda x: x["config"].update(steps=40, time_limit_h=0.2, parallel=1))
    for action in ("eda.accept", "eda.approve"):
        app.agents.run_sync(p, ("action", action, {}))
    for o in (0, 0):
        app.agents.run_sync(p, ("action", "plan.answer", {"q": len(p.load()["plan"]["answers"]), "o": o}))
    rec = p.load()["inspect"]["plan"]
    assert rec["status"] == "open" and rec["fix"]["changes"]["time_limit_h"] > 0.2
    plan_msg = next(m for m in app.project_view(p.slug)[1]["messages"] if m.get("id") == rec["msg"])
    assert plan_msg["actions"][0]["action"] == "inspect.fix"
    app.agents.run_sync(p, ("action", "inspect.fix", {"stage": "plan"}))
    d = p.load()
    assert d["config"]["time_limit_h"] == rec["fix"]["changes"]["time_limit_h"]
    plans = [m for m in p.messages() if m.get("agent") == "plan" and m.get("card")]
    assert plans[-1]["text"].startswith("The settings changed") and f"{d['config']['time_limit_h']} h" in plans[-1]["card"]
    view = app.project_view(p.slug)[1]
    assert next(m for m in reversed(view["messages"]) if m.get("agent") == "plan")["actions"][0]["action"] == "plan.approve"


# ── Errors tab ──────────────────────────────────────────────────────────────


def make_run(root: Path, events: list[dict], log: str = "", age_s: float = 600) -> server.RunReader:
    logs = root / "logs"
    logs.mkdir(parents=True)
    (logs / server.RUN_FILE).write_text(json.dumps({"exp_name": root.name, "started": time.time() - 3600}))
    (logs / server.EVENTS_FILE).write_text("".join(json.dumps(e) + "\n" for e in events))
    (logs / "MLEvolve.log").write_text(log or "[2026-11-09 10:00:00,000] INFO: Step 1/3\n")
    past = time.time() - age_s
    for f in logs.iterdir():
        os.utime(f, (past, past))
    return server.RunReader(root)


def test_error_report_for_a_crash(tmp_path):
    tb = 'Traceback (most recent call last):\n  File "run.py", line 9, in <module>\nMemoryError'
    r = make_run(tmp_path / "r", [{"type": "run_end", "reason": "crashed", "error": "MemoryError: ", "traceback": tb,
                                   "t": time.time() - 600}])
    rep = errors.report(r)
    assert rep["kind"] == "bad" and rep["headline"].startswith("Crashed: MemoryError")
    assert rep["run_errors"][-1]["text"] == tb and "in chunks" in rep["next"]


def test_error_report_for_a_crash_in_the_final_save(tmp_path):
    tb = ('Traceback (most recent call last):\n  File "config/__init__.py", line 365, in save_run\n'
          "PermissionError: [Errno 13] Permission denied: 'journal.json.tmp' -> 'journal.json'")
    log = ("[2026-11-09 17:15:04,940] INFO: Task completed: step=10\n"
           "[2026-11-09 17:15:04,986] INFO: 📊 Progress: 10/10 steps completed, 0 tasks running\n")
    r = make_run(tmp_path / "r", [{"type": "run_end", "reason": "crashed", "error": "PermissionError: [Errno 13]",
                                   "traceback": tb, "t": time.time() - 600}], log=log)
    (r.logs / "journal.json").write_text("{}")
    rep = errors.report(r)
    assert rep["kind"] == "warn" and rep["headline"].startswith("Crashed while saving, after all 10 steps finished")
    assert any("`logs/journal.json`" in x for x in rep["detail"]) and rep["next"].startswith("Nothing to redo")


def test_a_save_crash_before_the_last_step_is_a_normal_crash(tmp_path):
    tb = 'Traceback (most recent call last):\n  File "config/__init__.py", line 365, in save_run\nOSError: disk full'
    log = "[2026-11-09 17:00:00,000] INFO: Progress: 4/10 steps completed, 2 tasks running\n"
    r = make_run(tmp_path / "r", [{"type": "run_end", "reason": "crashed", "error": "OSError: disk full",
                                   "traceback": tb, "t": time.time() - 600}], log=log)
    rep = errors.report(r)
    assert rep["kind"] == "bad" and rep["headline"].startswith("Crashed: OSError")


def test_dump_json_tries_the_rename_again(tmp_path, monkeypatch):
    from utils import serialize
    monkeypatch.setattr(serialize, "REPLACE_WAIT_S", 0)
    real, calls = Path.replace, []

    def flaky(self, target):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "Permission denied")
        return real(self, target)
    monkeypatch.setattr(Path, "replace", flaky)
    monkeypatch.setattr(serialize, "dumps_json", lambda obj: '{"a": 1}')
    serialize.dump_json(object(), tmp_path / "journal.json")
    assert len(calls) == 3 and (tmp_path / "journal.json").read_text() == '{"a": 1}'
    calls.clear()
    monkeypatch.setattr(Path, "replace", lambda self, t: (calls.append(1), (_ for _ in ()).throw(PermissionError(13, "x"))))
    try:
        serialize.dump_json(object(), tmp_path / "journal.json")
        assert False, "should raise"
    except PermissionError:
        assert len(calls) == serialize.REPLACE_TRIES


def test_error_report_for_a_kill_without_a_record(tmp_path):
    rep = errors.report(make_run(tmp_path / "r", []))
    assert rep["status"] == "stale" and rep["kind"] == "bad" and rep["headline"] == "Killed: no error recorded"


def test_error_report_for_the_hard_stop(tmp_path):
    r = make_run(tmp_path / "r", [{"type": "run_end", "reason": "interrupted", "signal": "SIGINT", "t": time.time()}],
                 age_s=0)
    sh = tmp_path / "launch.sh"
    sh.write_text("exec timeout --signal=INT --kill-after=300 3600 .venv/bin/python run.py \\\n")
    rep = errors.report(r, None, sh)
    assert rep["headline"].startswith("Stopped by the hard time limit") and rep["kind"] == "warn"
    assert errors.report(r)["headline"].startswith("Stopped: interrupted (SIGINT)")


def test_errors_api_uses_the_project_launch_log(app, tmp_path):
    run = "20261109_100000_errs"
    make_run(app.runs_root / run, [])
    p = app.projects.create("errs")
    (p.folder / "launch.log").write_text("bash: .venv/bin/python: No such file or directory\n")
    p.update(lambda x: x.update(run=run, runs=[run]))
    rep = app.error_report(app.reader(run))
    assert rep["launch"] == ["bash: .venv/bin/python: No such file or directory"]


# ── exit records written by the run ─────────────────────────────────────────

RUN_SCRIPT = """
import os, signal, sys, time
from types import SimpleNamespace as N
sys.path.insert(0, {repo!r})
from utils import monitor
cfg = N(log_dir={logs!r}, workspace_dir={logs!r}, exp_name="t", data_dir=".", task=N(profile="generic"),
        agent=N(code=N(model="m"), feedback=N(model="m"), steps=1, time_limit=1, search=N(parallel_search_num=1)))
monitor.run_started(cfg)
mode = sys.argv[1]
if mode == "crash":
    raise ValueError("bad column")
if mode == "int":
    raise KeyboardInterrupt
if mode == "term":
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(5)
if mode == "normal":
    monitor.run_ended("finished")
"""


@pytest.mark.parametrize("mode,reason", [("crash", "crashed"), ("int", "interrupted"), ("normal", "finished"),
                                         ("exit", "exited")] + ([("term", "killed")] if os.name != "nt" else []))
def test_the_run_records_why_it_ended(tmp_path, mode, reason):
    logs = tmp_path / "logs"
    script = tmp_path / "run.py"   # the hooks are installed only for a script named run.py
    script.write_text(RUN_SCRIPT.format(repo=str(REPO), logs=str(logs)))
    subprocess.run([sys.executable, str(script), mode], capture_output=True, timeout=60)
    ends = [json.loads(x) for x in (logs / "monitor_events.jsonl").read_text().splitlines()
            if json.loads(x)["type"] == "run_end"]
    assert [e["reason"] for e in ends] == [reason]
    if mode == "crash":
        assert ends[0]["error"] == "ValueError: bad column" and "Traceback" in ends[0]["traceback"]
    if mode == "term":
        assert ends[0]["signal"] == "SIGTERM"


def test_the_checking_flag_is_cleared_after_every_check(app, tmp_path, on):
    p = task_accepted(app, tmp_path, FakeGate())
    app.agents.run_sync(p, ("message", "inspect", "check again", "m1"))
    assert app.project_view(p.slug)[1]["inspecting"] == [] and not app.agents.inspecting


def test_error_report_old_run_past_the_hard_limit(tmp_path):
    """A run from before the exit hooks: no end record, but it was older than the launch.sh hard limit."""
    r = make_run(tmp_path / "r", [], age_s=0)
    sh = tmp_path / "launch.sh"
    sh.write_text("exec timeout --signal=INT --kill-after=300 1800 .venv/bin/python run.py \\n")
    r.pid = None
    import os as _os, time as _t
    past = _t.time() - 600
    for f in (tmp_path / "r" / "logs").iterdir():
        _os.utime(f, (past, past))
    rep = errors.report(r, None, sh)
    assert rep["kind"] == "warn" and rep["headline"].startswith("Stopped, most likely by the hard time limit")
