"""EDA agent, Plan agent and first-draft review in the dashboard (dashboard/gate_agents.py,
dashboard/gates.py, agents/draft_review.py file channel). The WSL gate calls are stubbed."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents import draft_review
from dashboard import assistants, envcheck, gate_agents, gates, server, wslexec

READY_FACTS = {"shell": "ok", "repo": "ok", "venv": "ok", "pyver": "3.12", "core_missing": "", "ml_missing": "",
               "import": "ok", "claude": "2.1.0 (Claude Code)", "gpus": "0", "cpus": "8", "env_file": "missing"}
TASK_MD = "# Goal\nPredict `target`.\n\n# Evaluation\nROC-AUC, 5-fold.\n\n# Deliverables\nmetrics.csv"
QUESTIONS = [{"question": "Tune the threshold for recall?", "why": "Missed cases cost more.",
              "options": ["Yes, recall >= 0.9", "No, use 0.5"], "recommended": 0},
             {"question": "Explainable models?", "why": "", "options": ["Yes", "No"], "recommended": 1}]


class FakeGate:
    """Answers like dashboard/gates.py; records every call."""

    def __init__(self, eda_ok=True):
        self.calls = []
        self.eda_ok = eda_ok

    def __call__(self, project, cmd, **f):
        self.calls.append((cmd, f))
        if cmd == "eda_report":
            return {"ok": True, "report": "# EDA Report\n- 3 rows", "seconds": 4} if self.eda_ok else \
                {"ok": False, "errors": ["attempt 1: KeyError: 'x'"]}
        if cmd == "eda_plan":
            return {"ok": True, "plan": "## Loading\nread train.csv" + ("\n- " + "\n- ".join(f["feedback"])
                                                                       if f.get("feedback") else "")}
        if cmd == "interpret_review":
            q = f["message"].endswith("?")
            return {"ok": True, "intent": "question" if q else "revision", "reply": "Because AUC." if q else ""}
        if cmd == "plan_start":
            return {"ok": True, "cleaned": "CLEAN TASK", "diff": "- old\n+ new", "questions": QUESTIONS}
        if cmd == "plan_answer":
            return {"ok": True, "intent": "answer", "reply": ""}
        if cmd == "plan_draft":
            return {"ok": True, "plan": "1. Logistic regression\n2. LightGBM" +
                    ("\n3. " + "; ".join(f["feedback"]) if f.get("feedback") else "")}
        if cmd == "compose":
            return gates.compose(None, f)
        raise AssertionError(cmd)


@pytest.fixture
def app(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(envcheck, "probe", lambda: dict(READY_FACTS))
    monkeypatch.setattr(envcheck, "check_login", lambda force=False: (True, "logged in"))
    monkeypatch.setattr(assistants, "ask_claude", lambda prompt, model="sonnet", timeout=0: TASK_MD)
    return server.Dashboard(tmp_path / "runs", token="tok", projects_root=tmp_path / "projects")


def last(p, role="agent"):
    return [m for m in p.messages() if m["role"] == role][-1]


def task_accepted(app, tmp_path, gate):
    app.agents.gate = gate
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text("a,target\n1,0\n2,1\n3,1\n")
    p = app.projects.create("flow")
    app.agents.run_sync(p, ("start",))
    app.agents.run_sync(p, ("message", "data", f"data is in {data}", None))
    app.agents.run_sync(p, ("action", "task.accept", {}))
    return p


def test_full_flow_eda_then_plan_then_launch(app, tmp_path, monkeypatch):
    gate = FakeGate()
    p = task_accepted(app, tmp_path, gate)
    # EDA ran right after the task was accepted, and nothing was launched.
    d = p.load()
    assert d["eda"]["status"] == "report" and not d.get("launch")
    report_msg = last(p)
    assert report_msg["agent"] == "eda" and report_msg["once"] and report_msg["card"].startswith("# EDA Report")
    assert (p.folder / "EDA_report.md").exists()
    assert gate.calls[0][1]["task"] == TASK_MD

    # Launch is refused while the EDA and plan wait for approval.
    app.agents.run_sync(p, ("action", "run.launch", {}))
    assert "cannot launch" in last(p)["text"] and "data plan" in last(p)["text"]

    app.agents.run_sync(p, ("action", "eda.accept", {}))
    assert p.load()["eda"]["status"] == "plan" and last(p)["card"].startswith("## Loading")
    # A question is answered; a change request redrafts the data plan.
    app.agents.run_sync(p, ("message", "eda", "why roc auc?", "m1"))
    assert last(p)["text"] == "Because AUC." and p.load()["eda"].get("feedback") == []
    app.agents.run_sync(p, ("message", "eda", "impute with the median", "m2"))
    assert "impute with the median" in last(p)["card"]
    app.agents.run_sync(p, ("message", "eda", "looks good", "m3"))
    d = p.load()
    assert d["eda"]["status"] == "approved" and "Confirmed data plan" in (p.folder / "EDA_report.md").read_text()

    # Plan: task rewrite, then the questions one at a time.
    assert d["plan"]["status"] == "questions" and d["plan"]["cleaned"] == "CLEAN TASK"
    q1 = last(p)
    assert q1["agent"] == "plan" and q1["text"].startswith("**Question 1 of 2.**")
    assert [a.get("args") for a in q1["actions"][:2]] == [{"q": 0, "o": 0}, {"q": 0, "o": 1}]
    assert "(recommended)" in q1["actions"][0]["label"]
    assert any("```" in (m.get("card") or "") and "+ new" in m["card"] for m in p.messages())
    app.agents.run_sync(p, ("action", "plan.answer", {"q": 0, "o": 0}))
    app.agents.run_sync(p, ("action", "plan.answer", {"q": 0, "o": 1}))  # stale button: refused
    assert "already answered" in last(p)["text"]
    n_gate = len(gate.calls)
    app.agents.run_sync(p, ("message", "plan", "2", "m4"))  # option number: no model call
    assert len(gate.calls) == n_gate + 1 and gate.calls[-1][0] == "plan_draft"
    d = p.load()
    assert d["plan"]["answers"] == [[QUESTIONS[0]["question"], "Yes, recall >= 0.9"], [QUESTIONS[1]["question"], "No"]]
    assert d["plan"]["status"] == "review" and last(p)["actions"][0]["action"] == "plan.approve"

    # Free text while the plan waits goes to the Plan agent, even with setting words in it.
    assert assistants.route("add more steps of feature engineering", d, None)[0] == "plan"
    app.agents.run_sync(p, ("message", "plan", "add a random forest", "m5"))
    assert "add a random forest" in last(p)["card"]

    started = {}
    monkeypatch.setattr(wslexec, "spawn_detached", lambda script, log: started.update(script=script) or 7)
    monkeypatch.setattr(app.agents, "_link_run", lambda project, t0: None)
    app.agents.run_sync(p, ("action", "plan.approve", {}))
    d = p.load()
    assert d["plan"]["status"] == "approved" and d["launch"]["pid"] == 7
    task = p.task_path().read_text()
    assert task.startswith("CLEAN TASK") and "APPROVED PLAN" in task and "add a random forest" in task
    assert "CLARIFICATIONS FROM THE TASK OWNER" in task and "CONFIRMED DATA PLAN" in task
    script = (p.folder / "launch.sh").read_text()
    assert "plan_mode.preapproved=True" in script and "draft_review.channel=file" in script
    assert "eda.enabled=False" in script and "plan_mode.require_confirmation=False" in script
    assert "task.require_submission=False" in script  # train.csv only: nothing to predict
    assert (p.folder / "plan.md").read_text().count("Approved plan") == 1


def test_eda_failure_offers_retry_or_skip(app, tmp_path):
    p = task_accepted(app, tmp_path, FakeGate(eda_ok=False))
    msg = last(p)
    assert msg["kind"] == "error" and "KeyError" in msg["text"]
    assert [a["action"] for a in msg["actions"]] == ["eda.redo", "eda.skip"]
    app.agents.gate.eda_ok = True
    app.agents.run_sync(p, ("action", "eda.skip", {}))
    d = p.load()
    assert d["eda"]["status"] == "skipped" and d["plan"]["status"] == "questions"


def test_old_launch_button_starts_the_missing_eda_step(app, tmp_path):
    """A project that a pre-EDA server took straight to "Launch training" has no eda state."""
    p = task_accepted(app, tmp_path, FakeGate())
    p.update(lambda x: (x.pop("eda", None), x.pop("plan", None)))  # state as the old server left it
    app.agents.run_sync(p, ("action", "run.launch", {}))
    d = p.load()
    assert d.get("run") is None and d.get("launch") is None
    assert d["eda"]["status"] == "report" and last(p)["agent"] == "eda"
    assert any("cannot launch yet" in m["text"] for m in p.messages())


def test_plan_back_and_new_task_reset_later_steps(app, tmp_path):
    gate = FakeGate()
    p = task_accepted(app, tmp_path, gate)
    app.agents.run_sync(p, ("action", "eda.accept", {}))
    app.agents.run_sync(p, ("action", "eda.approve", {}))
    app.agents.run_sync(p, ("action", "plan.back", {}))
    d = p.load()
    assert d["eda"]["status"] == "plan" and d["plan"] == {} and last(p)["agent"] == "eda"
    app.agents.run_sync(p, ("action", "task.accept", {}))  # task accepted again: analysis starts over
    assert p.load()["eda"]["status"] == "report"


def test_once_buttons_disappear_after_the_user_answers(app, tmp_path):
    p = task_accepted(app, tmp_path, FakeGate())
    view = app.project_view(p.slug)[1]
    assert view["messages"][-1]["actions"]
    app.agents.run_sync(p, ("action", "eda.accept", {}))
    msgs = app.project_view(p.slug)[1]["messages"]
    report = next(m for m in msgs if (m.get("card") or "").startswith("# EDA Report"))
    assert not report.get("actions") and msgs[-1]["actions"]
    # A question back keeps the open choice's buttons.
    p.post("user", "why roc auc?", agent="eda")  # the server posts the user's message before the agent runs
    app.agents.run_sync(p, ("message", "eda", "why roc auc?", "m1"))
    msgs = app.project_view(p.slug)[1]["messages"]
    assert msgs[-1]["text"] == "Because AUC." and msgs[-3]["card"].startswith("## Loading") and msgs[-3]["actions"]


# ── draft review ────────────────────────────────────────────────────────────


def write_draft_request(run_dir: Path, rid="abcd1234-1-1", index=1):
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "drafts").mkdir(exist_ok=True)
    (run_dir / "drafts" / f"draft_{index}_abcd1234.py").write_text("print('Final Validation Score: 0.9')\n")
    with open(run_dir / "logs" / "draft_review.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "type": "draft", "id": rid, "index": index, "total": 2,
                            "node": "abcd1234ffff", "path": "E:\\runs\\x\\drafts\\d.py",
                            "facts": {"models": "LogisticRegression", "lines": 1, "cv": True, "prints_score": True},
                            "plan": "Logistic regression baseline"}) + "\n")


def test_draft_command_words():
    cmd = gate_agents.GateAgents.draft_command
    assert cmd("accept") == ("accept", "")
    assert cmd("Reject: use class weights") == ("reject", "use class weights")
    assert cmd("skip") == ("skip", "")
    assert cmd("why no scaling?") == ("ask", "why no scaling?")


def test_project_chat_answers_the_waiting_draft(app, tmp_path, monkeypatch):
    run = "20261108_101500_flow"
    run_dir = tmp_path / "runs" / run
    write_draft_request(run_dir)
    p = app.projects.create("flow")
    p.update(lambda x: x.update(run=run))
    monkeypatch.setattr(app, "run_status", lambda name: "live")

    msgs = app.project_view(p.slug)[1]["messages"]
    draft = next(m for m in msgs if m.get("draft"))
    assert draft["actions"][0]["args"] == {"id": "abcd1234-1-1", "verdict": "accept"}
    assert app.agents.pending_draft(run)["id"] == "abcd1234-1-1"
    assert server.draft_code(server.RunReader(run_dir), "abcd1234-1-1")[1]["code"].startswith("print")

    # A question goes to the draft, not to the Main agent; /status still goes to the Main agent.
    code, body = app.request_project_chat(p.slug, "why no scaling?", None)
    assert code == 200 and body["draft"] == "ask"
    code, body = app.request_project_chat(p.slug, "reject use class weights", None)
    answers = [json.loads(x) for x in (run_dir / "logs" / "draft_review_answers.jsonl").read_text().splitlines()]
    assert [(a["verdict"], a["text"]) for a in answers] == [("ask", "why no scaling?"), ("reject", "use class weights")]
    code, body = app.request_project_chat(p.slug, "/status", None)
    assert body.get("forwarded") or "cannot receive" in last(p)["text"]

    # Once the run records the result, the buttons go away and no draft is pending. The run's clock may be
    # behind (WSL): the result still sorts after the answer that caused it.
    with open(run_dir / "logs" / "draft_review.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time() - 30, "type": "result", "id": "abcd1234-1-1", "index": 1,
                            "verdict": "reject"}) + "\n")
    msgs = app.project_view(p.slug)[1]["messages"]
    assert not next(m for m in msgs if m.get("draft")).get("actions")
    order = [m["text"] for m in msgs if m["text"].startswith(("reject use", "Draft 1 rejected"))]
    assert order == ["reject use class weights", "Draft 1 rejected. The run writes a new one with your reason."]
    assert any("rejected" in m["text"] for m in msgs) and app.agents.pending_draft(run) is None
    app.agents.run_sync(p, ("action", "run.draft", {"id": "abcd1234-1-1", "verdict": "accept"}))
    assert "no longer waiting" in last(p)["text"]


class FakeBus:
    stop_requested = False


def draft_agent(tmp_path):
    run = tmp_path / "run"
    cfg = SimpleNamespace(log_dir=run / "logs", workspace_dir=run / "workspace",
                          draft_review=SimpleNamespace(enabled=True, channel="file"))
    node = SimpleNamespace(id="feedbeef0000", code="print(1)\n", plan="baseline", review_feedback=None)
    return SimpleNamespace(cfg=cfg, control=FakeBus(), paused_seconds=0.0), node, run


def answer_later(run: Path, records, delay=0.3):
    def go():
        time.sleep(delay)
        req = None
        while req is None:
            lines = (run / "logs" / "draft_review.jsonl").read_text().splitlines() \
                if (run / "logs" / "draft_review.jsonl").exists() else []
            req = next((json.loads(x) for x in lines if json.loads(x)["type"] == "draft"), None)
            time.sleep(0.05)
        for r in records:
            with open(run / "logs" / "draft_review_answers.jsonl", "a") as f:
                f.write(json.dumps({"id": req["id"], **r}) + "\n")
            time.sleep(0.2)
    threading.Thread(target=go, daemon=True).start()


def test_run_side_file_channel_accepts_answers_and_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(draft_review, "POLL_S", 0.05)
    monkeypatch.setattr(draft_review, "_answer_text", lambda agent, node, q: f"answer to {q}")
    agent, node, run = draft_agent(tmp_path)
    answer_later(run, [{"verdict": "ask", "text": "why?", "msg": "dq-1"}, {"verdict": "accept"}])
    assert draft_review.review_draft(agent, node, 1, 2) == draft_review.ACCEPT
    events = [json.loads(x) for x in (run / "logs" / "draft_review.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events] == ["draft", "reply", "result"]
    assert events[1]["text"] == "answer to why?" and events[1]["reply_to"] == "dq-1"
    assert events[0]["facts"]["lines"] == 1 and (run / "drafts").is_dir()
    assert agent.paused_seconds > 0  # the run's clock does not count the wait


def test_run_side_file_channel_reject_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(draft_review, "POLL_S", 0.05)
    agent, node, run = draft_agent(tmp_path)
    answer_later(run, [{"verdict": "reject", "text": "use class weights"}])
    assert draft_review.review_draft(agent, node, 1, 2) == draft_review.REJECT
    assert node.review_feedback == "use class weights"
    agent.control.stop_requested = True  # Stop from the dashboard ends the wait
    assert draft_review.review_draft(agent, node, 2, 2) == draft_review.ABORT


def test_terminal_channel_unchanged_without_a_terminal(tmp_path, monkeypatch):
    agent, node, run = draft_agent(tmp_path)
    agent.cfg.draft_review.channel = "terminal"
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert draft_review.review_draft(agent, node, 1, 1) == draft_review.ACCEPT
    assert not (run / "logs" / "draft_review.jsonl").exists()


def test_gate_overrides_carry_the_project_budget():
    c = {"code_model": "claude-code:sonnet", "feedback_model": "claude-code:sonnet", "steps": 4,
         "time_limit_h": 0.42, "parallel": 2, "initial_drafts": 2, "exec_timeout_min": 10}
    o = gate_agents.gate_overrides(c)
    assert "agent.time_limit=1512" in o and "exec.timeout=600" in o and "agent.steps=4" in o


def test_submission_required_only_with_a_test_file():
    from dashboard import projects
    assert not projects.has_prediction_target({"files": [{"name": "train.csv"}, {"name": "README.md"}]})
    assert projects.has_prediction_target({"files": [{"name": "train.csv"}, {"name": "public/test.csv"}]})
    assert projects.has_prediction_target({"files": [{"name": "sample_submission.csv"}]})


def test_compose_puts_answers_plan_and_data_plan_in_order():
    text = gates.compose(None, {"cleaned": "TASK", "answers": [["Q?", "A"]], "plan": "PLAN", "data_plan": "DATA"})["text"]
    assert text.index("TASK") < text.index("CLARIFICATIONS") < text.index("APPROVED PLAN") < text.index("DATA")
