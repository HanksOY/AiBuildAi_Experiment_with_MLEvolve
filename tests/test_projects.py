"""Projects and the guiding chat agents (dashboard/projects.py, dashboard/assistants.py)."""

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from dashboard import assistants, dataprofile, envcheck, projects, server, wslexec

READY_FACTS = {"shell": "ok", "repo": "ok", "venv": "ok", "pyver": "3.12", "core_missing": "", "ml_missing": "",
               "import": "ok", "claude": "2.1.0 (Claude Code)", "gpus": "0", "cpus": "8", "env_file": "missing"}
TASK_MD = "# Goal\nPredict `target`.\n\n# Evaluation\nROC-AUC, 5-fold.\n\n# Deliverables\nmetrics.csv"


@pytest.fixture
def app(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(envcheck, "probe", lambda: dict(READY_FACTS))
    monkeypatch.setattr(envcheck, "check_login", lambda force=False: (True, "logged in"))
    monkeypatch.setattr(assistants, "ask_claude", lambda prompt, model="sonnet", timeout=0: TASK_MD)
    return server.Dashboard(tmp_path / "runs", token="tok", projects_root=tmp_path / "projects")


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "train.csv").write_text("a,b,target\n1,2,0\n3,,1\n5,6,1\n")
    (d / "test.csv").write_text("a,b,target\n9,9,1\n")
    return d


def texts(p):
    return [m["text"] for m in p.messages()]


def last(p, role="agent"):
    return [m for m in p.messages() if m["role"] == role][-1]


# ── routing and stages ──────────────────────────────────────────────────────


def test_route_rules():
    fresh = {"env": {}, "cfg": {}, "data": {}, "task": {}}
    assert assistants.route("@task use recall too", fresh, None) == ("task", "use recall too")
    assert assistants.route("@config steps=5", fresh, None)[0] == "setup"
    assert assistants.route("/help", fresh, None)[0] == "help"
    assert assistants.route("/check", fresh, None)[0] == "setup"
    assert assistants.route("/status", fresh, None)[0] == "run"
    assert assistants.route("anything", fresh, "live")[0] == "run"
    assert assistants.route("my data is in /mnt/e/data/x", fresh, None)[0] == "data"
    assert assistants.route("use sonnet for coding", fresh, None)[0] == "setup"
    assert assistants.route("hello", fresh, None)[0] == "setup"  # current stage: environment
    assert assistants.route("hello", fresh, None, chosen="analyst")[0] == "analyst"


def test_stages_progress():
    d = {"env": {"ok": True}, "cfg": {"ok": True}, "data": {}, "task": {}}
    st = {s["name"]: s for s in projects.stages(d, None)}
    assert st["environment"]["status"] == "skipped" and st["configuration"]["status"] == "skipped"
    assert st["data"]["status"] == "now"
    d["data"]["profile"] = {"tables": []}
    d["task"] = {"draft": "x", "accepted": True}
    st = {s["name"]: s for s in projects.stages(d, None)}
    assert st["task"]["status"] == "done" and st["eda"]["status"] == "now" and st["training"]["status"] == "todo"
    d["eda"] = {"status": "approved"}
    d["plan"] = {"status": "review"}
    st = {s["name"]: s for s in projects.stages(d, None)}
    assert st["eda"]["status"] == "done" and st["plan"]["status"] == "now" and st["plan"]["note"] == "waiting for you"
    d["plan"]["status"] = "approved"
    d["run"] = "20261107_000000_p"
    st = {s["name"]: s for s in projects.stages(d, "live")}
    assert st["task"]["status"] == "done" and st["training"]["status"] == "now" and st["results"]["status"] == "todo"
    st = {s["name"]: s for s in projects.stages(d, "finished")}
    assert st["training"]["status"] == "done" and st["results"]["status"] == "now"
    fresh = projects.stages({"env": {"checked": 1, "ok": False, "items": [{"blocking": True, "ok": False}]},
                             "cfg": {}, "data": {}, "task": {}}, None)
    assert fresh[0]["status"] == "now" and fresh[0]["note"] == "1 to fix"
    assert sum(s["status"] == "now" for s in fresh) == 1


def test_validate_change():
    assert projects.validate_change("code_model", "sonnet") == (True, "claude-code:sonnet", "")
    assert projects.validate_change("code_model", "claude-opus-5-5")[1] == "claude-code:claude-opus-5-5"
    assert projects.validate_change("steps", "20")[1] == 20
    assert not projects.validate_change("steps", "0")[0]
    assert not projects.validate_change("code_model", "a b; rm -rf")[0]
    assert not projects.validate_change("data_dir", "/x")[0]
    assert projects.validate_change("draft_review", "off") == (True, False, "")
    assert not projects.validate_change("draft_review", "maybe")[0]


# ── the guided flow ─────────────────────────────────────────────────────────


def test_ready_environment_is_skipped_then_data_is_asked(app):
    p = app.projects.create("My Test")
    app.agents.run_sync(p, ("start",))
    setup, data = [m for m in p.messages() if m["role"] == "agent"]
    assert "skipped setup" in setup["text"] and setup["fold"] and setup["agent"] == "setup"
    assert {i["id"] for i in setup["checks"]} >= {"venv", "login", "code_model"}
    assert data["agent"] == "data" and "path" in data["text"]
    stages = {s["name"]: s["status"] for s in projects.stages(p.load(), None)}
    assert stages["environment"] == "skipped" and stages["data"] == "now"


def test_missing_items_are_guided_one_at_a_time(app, monkeypatch):
    monkeypatch.setattr(envcheck, "probe", lambda: {**READY_FACTS, "venv": "missing"})
    p = app.projects.create("p")
    app.agents.run_sync(p, ("start",))
    msg = last(p)
    assert "Step 1 of" in msg["text"] and "venv" in msg["text"]
    assert msg["actions"][0] == {"label": "Run it for me", "action": "setup.fix", "args": {"id": "create_venv"},
                                 "primary": True}
    assert not p.load()["env"]["ok"]
    # Once fixed, the next check reports the stage as fixed (done), not skipped.
    monkeypatch.setattr(envcheck, "probe", lambda: dict(READY_FACTS))
    app.agents.run_sync(p, ("action", "setup.check", {}))
    assert {s["name"]: s["status"] for s in projects.stages(p.load(), None)}["environment"] == "done"


def test_unknown_fix_is_refused(app):
    p = app.projects.create("p")
    app.agents.run_sync(p, ("action", "setup.fix", {"id": "rm_everything"}))
    assert last(p)["kind"] == "error"
    assert envcheck.run_fix("rm_everything")[0] is False


def test_settings_change_and_undo(app):
    p = app.projects.create("p")
    app.agents.run_sync(p, ("message", "setup", "steps=12 code_model=haiku", None))
    cfg = p.load()["config"]
    assert cfg["steps"] == 12 and cfg["code_model"] == "claude-code:haiku"
    msg = last(p)
    assert msg["diff"] and "| Steps |" in msg["text"]
    undo = next(a for a in msg["actions"] if a["action"] == "setup.undo")
    app.agents.run_sync(p, ("action", "setup.undo", undo["args"]))
    assert p.load()["config"]["steps"] != 12 and "Undone" in last(p)["text"]


def test_settings_change_in_plain_words_uses_the_model(app, monkeypatch):
    seen = {}

    def fake(prompt, model="sonnet", timeout=0):
        seen["prompt"] = prompt
        return '{"changes": {"time_limit_h": 2}, "reply": "Two hours fits 20 steps."}'

    monkeypatch.setattr(assistants, "ask_claude", fake)
    p = app.projects.create("p")
    app.agents.run_sync(p, ("message", "setup", "give it two hours", None))
    assert p.load()["config"]["time_limit_h"] == 2.0
    assert "give it two hours" in seen["prompt"] and "Two hours" in last(p)["text"]


def test_data_profile_task_draft_accept_and_launch(app, data_dir, monkeypatch):
    p = app.projects.create("flow")
    app.agents.run_sync(p, ("start",))
    app.agents.run_sync(p, ("message", "data", f"My data is in {data_dir} and I want to predict target", None))
    d = p.load()
    train = next(t for t in d["data"]["profile"]["tables"] if t["name"] == "train.csv")
    test = next(t for t in d["data"]["profile"]["tables"] if t["name"] == "test.csv")
    assert train["rows"] == 3 and train["target_guess"] == "target" and train["target_counts"] == {"1": 2, "0": 1}
    assert "target_counts" not in test and test["rows"] == 1  # no label statistics from test files
    task_msg = last(p)
    assert task_msg["agent"] == "task" and task_msg["card"] == TASK_MD
    app.agents.run_sync(p, ("action", "task.accept", {}))
    assert p.task_path().read_text().startswith("# Goal")
    assert last(p)["agent"] == "eda"  # the EDA step starts after the task, not training
    # (the EDA and Plan steps are tested in test_gate_agents.py; mark them approved here)
    p.update(lambda x: x.update(eda={"status": "approved", "data_plan": "plan"},
                                plan={"status": "approved", "final": TASK_MD + "\n\nAPPROVED PLAN"}))

    started = {}
    monkeypatch.setattr(wslexec, "spawn_detached", lambda script, log: started.update(script=script) or 4242)
    monkeypatch.setattr(app.agents, "_link_run", lambda project, t0: None)  # no run folder in this test
    app.agents.run_sync(p, ("action", "run.launch", {}))
    script = (p.folder / "launch.sh").read_text()
    assert started["script"].endswith("launch.sh") and p.load()["launch"]["pid"] == 4242
    assert f"data_dir={wslexec.to_wsl_path(data_dir)}" in script and "exp_name=flow" in script
    assert "agent.steps=" in script and "< /dev/null" in script
    assert "plan_mode.preapproved=True" in script and "draft_review.channel=file" in script
    assert p.task_path().read_text().rstrip().endswith("APPROVED PLAN")  # the run gets the approved text


def test_launch_refused_until_ready(app):
    p = app.projects.create("p")
    app.agents.run_sync(p, ("action", "run.launch", {}))
    assert "cannot launch" in last(p)["text"] and not (p.folder / "launch.sh").exists()


def test_launch_script_quotes_paths_with_spaces(tmp_path):
    p = projects.ProjectStore(tmp_path).create("q")
    d = p.load()
    d["data"] = {"path_wsl": "/mnt/e/Test v2.0.1/public"}
    script = projects.launch_script(p, d, "/mnt/e/My Repo")
    assert "cd '/mnt/e/My Repo'" in script and "'data_dir=/mnt/e/Test v2.0.1/public'" in script
    d["config"]["code_model"] = "x; rm -rf /"
    with pytest.raises(ValueError):
        projects.launch_script(p, d, "/r")


def test_find_path_with_spaces(app, tmp_path):
    folder = tmp_path / "Test v2.0.1" / "public"
    folder.mkdir(parents=True)
    found, _ = app.agents.find_path(f"the data is at {folder} thanks")
    assert found == str(folder)
    long_tail = f"My data is in {folder} . Predict target with ROC-AUC, CPU only, classical models, scripts under 2 minutes."
    assert app.agents.find_path(long_tail)[0] == str(folder)
    assert app.agents.find_path("no path here") == (None, None)


def test_find_run_matches_exact_slug(tmp_path):
    runs = tmp_path / "runs"
    (runs / "20261107_101010_dashboard_smoke").mkdir(parents=True)
    assert projects.find_run("smoke", runs, 0) is None
    (runs / "20261107_101011_smoke").mkdir()
    assert projects.find_run("smoke", runs, 0) == "20261107_101011_smoke"


# ── server side ─────────────────────────────────────────────────────────────


def _live_run(runs_root: Path, name: str):
    logs = runs_root / name / "logs"
    logs.mkdir(parents=True)
    (logs / server.RUN_FILE).write_text(json.dumps({"exp_name": name}))
    (logs / "journal.json").write_text(json.dumps({"nodes": [], "node2parent": {}}))
    (logs / server.EVENTS_FILE).write_text(json.dumps({"type": "run_start", "t": time.time()}) + "\n")
    return logs


def test_items_list_projects_and_unlinked_runs(app):
    p = app.projects.create("alpha")
    _live_run(app.runs_root, "20261107_000001_alpha")
    _live_run(app.runs_root, "20261107_000002_other")
    p.update(lambda x: (x.update(run="20261107_000001_alpha"), x["runs"].append("20261107_000001_alpha")))
    ids = {i["id"]: i for i in app.items()}
    assert ids["alpha"]["status"] == "live" and "run:20261107_000002_other" in ids
    assert "run:20261107_000001_alpha" not in ids  # shown inside its project


def test_messages_to_a_live_run_are_forwarded(app):
    p = app.projects.create("fwd")
    logs = _live_run(app.runs_root, "20261107_000003_fwd")
    p.update(lambda x: x.update(run="20261107_000003_fwd"))
    code, reply = app.request_project_chat("fwd", "/status", None)
    assert code == 200 and reply["forwarded"]
    req = json.loads((logs / server.CHAT_REQUESTS_FILE).read_text().splitlines()[-1])
    assert req["text"] == "/status" and req["via"] == "project"
    # The run's Main agent answers in its own chat file; the project view shows the reply under the run agent.
    (logs / server.CHAT_FILE).write_text(json.dumps({"id": "ag", "role": "agent", "text": "3/10 steps",
                                                     "reply_to": req["id"], "t": time.time()}) + "\n")
    view = app.project_view("fwd")[1]
    user = next(m for m in view["messages"] if m["role"] == "user")
    assert user["pending"] is False and view["messages"][-1]["agent"] == "run"


def test_finished_run_is_announced_once(app):
    p = app.projects.create("done")
    logs = _live_run(app.runs_root, "20261107_000004_done")
    with open(logs / server.EVENTS_FILE, "a") as f:
        f.write(json.dumps({"type": "run_end", "t": time.time()}) + "\n")
    p.update(lambda x: x.update(run="20261107_000004_done"))
    app.project_view("done")
    app.project_view("done")
    notes = [m for m in p.messages() if m.get("agent") == "analyst"]
    assert len(notes) == 1 and "Training finished" in notes[0]["text"]


def test_run_only_item_view(app):
    _live_run(app.runs_root, "20261107_000005_x")
    code, view = app.project_view("run:20261107_000005_x")
    assert code == 200 and view["kind"] == "run" and view["chat_mode"] == "live"
    assert app.project_view("run:nope")[0] == 404 and app.project_view("../etc")[0] == 404


def test_http_endpoints_need_the_token(app):
    httpd = server._Server(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body, token):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "X-Control-Token": token})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        assert call("/api/projects", {"name": "web"}, "wrong")[0] == 403
        code, made = call("/api/projects", {"name": "web"}, "tok")
        assert code == 200 and made["id"] == "web"
        assert call("/api/project/action", {"id": "web", "action": "bad action!"}, "tok")[0] == 400
        with urllib.request.urlopen(base + "/api/items") as r:
            assert any(i["id"] == "web" for i in json.loads(r.read()))
        deadline = time.time() + 5  # the start job runs on the project's worker thread
        while time.time() < deadline and not any(m["role"] == "agent" for m in app.projects.get("web").messages()):
            time.sleep(0.05)
        with urllib.request.urlopen(base + "/api/project?id=web") as r:
            view = json.loads(r.read())
        assert view["stages"][0]["name"] == "environment" and view["messages"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dataprofile_markdown_and_missing_folder(tmp_path):
    assert "could not read" in dataprofile.markdown(dataprofile.profile(tmp_path / "nope"))


def test_path_mapping():
    assert wslexec.to_wsl_path("E:\\Internship\\UCSD") == "/mnt/e/Internship/UCSD"
    assert wslexec.to_wsl_path("/mnt/e/x") == "/mnt/e/x"
    assert wslexec.clean_path("`/mnt/e/data/`") == "/mnt/e/data"
