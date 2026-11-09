"""Dashboard chat: the main agent's replies, guidance, the request watcher and the server side."""

import json
import subprocess
import time
from types import SimpleNamespace

import pytest

from agents.main_agent import ControlBus, MainAgent
from dashboard import server
from utils import monitor


@pytest.fixture
def recording(tmp_path):
    """Recording on, into a fresh run folder; restored afterwards."""
    root = tmp_path / "runs" / "20261106_000000_chat"
    (root / "logs").mkdir(parents=True)
    (root / "workspace").mkdir()
    saved = dict(monitor._state)
    monitor._state.update(log_dir=root / "logs", workspace=root / "workspace", llm=True, snapshots=True)
    yield root
    monitor._state.update(saved)


def _chat(root):
    path = root / "logs" / monitor.CHAT_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _agent(context=None):
    bus = ControlBus()
    bus.publish(progress="3/10 steps", best_metric=0.81, running=2)
    return MainAgent(cfg=SimpleNamespace(), bus=bus, context=context), bus


def test_status_reply_goes_to_chat(recording):
    agent, _ = _agent()
    agent.handle_message("status", msg_id="op-1")
    reply = _chat(recording)[-1]
    assert reply["role"] == "agent" and reply["reply_to"] == "op-1"
    assert "3/10 steps" in reply["text"] and "0.81" in reply["text"]


def test_terminal_input_is_copied_to_chat(recording):
    agent, _ = _agent()
    agent.handle_message("help")
    first, second = _chat(recording)
    assert (first["role"], first["kind"], first["text"]) == ("operator", "terminal", "help")
    assert "guide <text>" in second["text"]


@pytest.mark.parametrize("line, action", [("stop", "stop"), ("/abort", "abort")])
def test_stop_and_abort_commands(recording, line, action):
    agent, bus = _agent()
    agent.handle_message(line, msg_id="op-2")
    assert bus.stop_requested and bus.drain() == [(action, {})]
    assert bus.abort_requested is (action == "abort")
    assert _chat(recording)[-1]["kind"] == "action"


def test_guide_command_queues_guidance(recording):
    agent, bus = _agent()
    agent.handle_message("guide try XGBoost with class weights", msg_id="op-3")
    assert bus.drain() == [("guide", {"text": "try XGBoost with class weights"})]
    assert not bus.stop_requested
    assert "try XGBoost" in _chat(recording)[-1]["text"]


def test_guidance_applied_at_once_with_callback(recording):
    applied = []
    bus = ControlBus()
    agent = MainAgent(cfg=SimpleNamespace(), bus=bus, on_guidance=applied.append,
                      context=lambda: {"operator_guidance_in_effect": applied, "solutions_so_far": 0,
                                       "in_progress": [{"what": "llm_call", "agent": "stepwise_coder",
                                                        "step": "generate", "solution": None, "seconds": 12}]})
    agent.handle_message("guide use logistic regression", msg_id="op-g")
    assert applied == ["use logistic regression"] and bus.drain() == []
    agent.handle_message("status", msg_id="op-s")
    text = _chat(recording)[-1]["text"]
    assert "setting up" in text and "stepwise_coder" in text and "use logistic regression" in text


def test_empty_guide_is_an_error(recording):
    agent, bus = _agent()
    agent.handle_message("guide", msg_id="op-4")
    assert bus.drain() == [] and _chat(recording)[-1]["kind"] == "error"


def test_free_text_mapped_to_guidance(recording, monkeypatch):
    import llm

    seen = {}

    def fake_generate(prompt, cfg, stage):
        seen["prompt"] = prompt["user"]
        return json.dumps({"action": "guide", "guidance": "avoid SMOTE", "reply": "Will do."})

    monkeypatch.setattr(llm, "generate", fake_generate)
    agent, bus = _agent(context=lambda: {"top_solutions": [{"id": "abc12345", "score": 0.9}]})
    agent.handle_message("please stop using SMOTE", msg_id="op-5")
    assert bus.drain() == [("guide", {"text": "avoid SMOTE"})]
    assert "abc12345" in seen["prompt"]          # the run context reaches the model
    assert "please stop using SMOTE" in seen["prompt"]
    assert not bus.stop_requested                # "stop using" is not a stop request


def test_free_text_reply(recording, monkeypatch):
    import llm

    monkeypatch.setattr(llm, "generate", lambda prompt, cfg, stage: '{"action": "reply", "reply": "The best is 0.81."}')
    agent, bus = _agent()
    agent.handle_message("how good is it?", msg_id="op-6")
    assert _chat(recording)[-1]["text"] == "The best is 0.81." and bus.drain() == []


def test_model_failure_is_reported_not_raised(recording, monkeypatch):
    import llm

    def boom(prompt, cfg, stage):
        raise RuntimeError("usage limit")

    monkeypatch.setattr(llm, "generate", boom)
    agent, _ = _agent()
    agent.handle_message("anything", msg_id="op-7")
    assert _chat(recording)[-1]["kind"] == "error"


def test_watcher_passes_chat_requests_to_handler(recording):
    got = []
    monitor._state["started"] = time.time()
    # Sent while the run was still setting up, before the watcher started: still answered.
    with open(recording / "logs" / monitor.CHAT_REQUESTS_FILE, "a") as f:
        f.write(json.dumps({"id": "op-8", "text": "early", "t": time.time() - 5}) + "\n")
    cfg = SimpleNamespace(monitor=SimpleNamespace(control=True))
    monitor.start_control_watcher(cfg, ControlBus(), interval=0.05, chat_handler=lambda text, mid: got.append((text, mid)))
    with open(recording / "logs" / monitor.CHAT_REQUESTS_FILE, "a") as f:
        f.write(json.dumps({"id": "op-9", "text": "status", "t": time.time()}) + "\n")
    deadline = time.time() + 3
    while len(got) < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert sorted(got) == [("early", "op-8"), ("status", "op-9")]


# ── server ──────────────────────────────────────────────────────────────────


def _live_run(root):
    (root / "logs" / server.RUN_FILE).write_text(json.dumps({"exp_name": "x", "feedback_model": "claude-code:haiku"}))
    (root / "logs" / "journal.json").write_text(json.dumps({"nodes": [], "node2parent": {}}))


def test_chat_merges_requests_and_replies_and_marks_pending(recording):
    _live_run(recording)
    logs = recording / "logs"
    t = time.time()
    (logs / server.CHAT_REQUESTS_FILE).write_text(
        json.dumps({"id": "op-a", "text": "q1", "t": t - 5}) + "\n" + json.dumps({"id": "op-b", "text": "q2", "t": t - 1}) + "\n")
    (logs / server.CHAT_FILE).write_text(json.dumps({"id": "ag-1", "role": "agent", "text": "a1", "reply_to": "op-a", "t": t - 3}) + "\n")
    msgs = server.RunReader(recording).chat()
    assert [m["text"] for m in msgs] == ["q1", "a1", "q2"]
    assert [m.get("pending") for m in msgs] == [False, None, True]


def test_request_chat_live_writes_request_only(recording):
    _live_run(recording)
    app = server.Dashboard(recording.parent, token="t")
    code, reply = app.request_chat(recording.name, "status")
    assert code == 200 and reply["mode"] == "live"
    req = json.loads((recording / "logs" / server.CHAT_REQUESTS_FILE).read_text().splitlines()[-1])
    assert req["text"] == "status" and req["id"] == reply["id"]
    assert not (recording / "logs" / server.CHAT_FILE).exists()  # the run answers, not the server


@pytest.mark.parametrize("text, code", [("", 400), ("x" * 5000, 400)])
def test_request_chat_rejects_bad_text(recording, text, code):
    _live_run(recording)
    assert server.Dashboard(recording.parent).request_chat(recording.name, text)[0] == code


def test_review_answer_uses_claude_cli(recording, monkeypatch):
    _live_run(recording)
    monitor.run_ended()  # the run is finished, so the server answers
    calls = {}

    def fake_run(cmd, input, **kw):
        calls["cmd"], calls["input"] = cmd, input
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"subtype": "success", "result": "It went well."}), "")

    monkeypatch.setattr(server.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(server.subprocess, "run", fake_run)
    reader = server.RunReader(recording)
    assert server.run_state(reader)["chat_mode"] == "review"
    server.answer_review(reader, "op-r", "How did it go?")
    reply = server.RunReader(recording).chat()[-1]
    assert reply["text"] == "It went well." and reply["reply_to"] == "op-r" and reply["source"] == "review"
    assert calls["cmd"][calls["cmd"].index("--model") + 1] == "haiku"
    assert calls["cmd"][calls["cmd"].index("--tools") + 1] == ""
    assert "How did it go?" in calls["input"]


def test_review_without_cli_reports_error(recording, monkeypatch):
    _live_run(recording)
    monitor.run_ended()
    monkeypatch.setattr(server.shutil, "which", lambda name: None)
    reader = server.RunReader(recording)
    assert server.run_state(reader)["chat_mode"] == "none"
    server.answer_review(reader, "op-x", "hi")
    assert server.RunReader(recording).chat()[-1]["kind"] == "error"
