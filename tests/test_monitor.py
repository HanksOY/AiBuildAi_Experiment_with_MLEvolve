"""Dashboard recording (utils/monitor.py) and the dashboard server, without a real run."""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from utils import monitor
from dashboard import server


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A run folder with recording switched on, reset afterwards."""
    root = tmp_path / "runs" / "20261106_000000_demo"
    (root / "logs").mkdir(parents=True)
    (root / "workspace" / "working").mkdir(parents=True)
    saved = dict(monitor._state)
    cfg = SimpleNamespace(
        log_dir=root / "logs", workspace_dir=root / "workspace", exp_name="demo", data_dir="/data",
        task=SimpleNamespace(profile="generic"),
        agent=SimpleNamespace(code=SimpleNamespace(model="claude-code:opus"),
                              feedback=SimpleNamespace(model="claude-code:sonnet"),
                              steps=10, time_limit=3600, search=SimpleNamespace(parallel_search_num=2)),
        monitor=SimpleNamespace(enabled=True, record_llm_calls=True, snapshot_outputs=True, control=True),
    )
    monitor.run_started(cfg)
    yield root, cfg
    monitor._state.update(saved)
    monitor._active_execs.clear()
    monitor._recent_execs.clear()


def _events(root):
    path = root / "logs" / monitor.EVENTS_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _Node:
    def __init__(self, nid):
        self.id = nid


def _fake_agent_call(node):
    """Stands in for an agent module: llm_start reads the caller from the stack."""
    return monitor.llm_start("claude-code:opus", "query")


def test_inactive_hooks_are_noops(tmp_path):
    saved = dict(monitor._state)
    monitor._state["log_dir"] = None
    try:
        assert monitor.llm_start("m", "query") is None
        monitor.llm_end(None, 1, 2)
        monitor.exec_finished("x" * 32, time.time())
    finally:
        monitor._state.update(saved)


def test_run_meta_written(run_dir):
    root, _ = run_dir
    meta = json.loads((root / "logs" / monitor.RUN_FILE).read_text())
    assert meta["code_model"] == "claude-code:opus" and meta["time_limit"] == 3600


def test_llm_call_records_node_and_tokens(run_dir):
    root, _ = run_dir
    node = _Node("a" * 32)
    handle = _fake_agent_call(node)
    monitor.llm_end(handle, 120, 30, cost_usd=0.01)
    start, end = _events(root)
    assert start["type"] == "llm_start" and end["type"] == "llm_end"
    assert end["node"] == "a" * 32
    assert (end["in_tokens"], end["out_tokens"], end["ok"]) == (120, 30, True)
    assert end["call"] == start["call"]


def test_llm_query_is_recorded_through_llm_module(run_dir, monkeypatch):
    import llm

    root, cfg = run_dir
    monkeypatch.setattr(llm._claude_code, "query", lambda **kw: ("hi", 0.1, 11, 4, {"cost_usd_equivalent": 0.002}))
    assert llm.query("S", "U", model="claude-code:opus", cfg=cfg) == "hi"
    end = [e for e in _events(root) if e["type"] == "llm_end"][-1]
    assert (end["in_tokens"], end["out_tokens"], end["estimated"]) == (11, 4, False)


def test_failed_llm_call_is_recorded_and_reraised(run_dir, monkeypatch):
    import llm

    root, cfg = run_dir

    def boom(**kw):
        raise RuntimeError("rate limit")

    monkeypatch.setattr(llm._claude_code, "query", boom)
    with pytest.raises(RuntimeError):
        llm.query("S", "U", model="claude-code:opus", cfg=cfg)
    end = [e for e in _events(root) if e["type"] == "llm_end"][-1]
    assert end["ok"] is False and "rate limit" in end["error"]


def test_exec_snapshot_copies_new_outputs_only(run_dir):
    root, _ = run_dir
    working = root / "workspace" / "working"
    old = working / "old.csv"
    old.write_text("a\n1\n")
    past = time.time() - 600
    import os
    os.utime(old, (past, past))
    nid = "b" * 32
    t0 = monitor.exec_started(nid)
    (working / "metrics.csv").write_text("metric,value\nauc,0.9\n")
    (working / "roc.png").write_bytes(b"\x89PNG fake")
    (working / "model.joblib").write_bytes(b"x")
    monitor.exec_finished(nid, t0, ok=True)
    dest = root / "workspace" / monitor.OUTPUTS_DIR / nid
    assert sorted(p.name for p in dest.iterdir()) == ["manifest.json", "metrics.csv", "roc.png"]
    end = [e for e in _events(root) if e["type"] == "exec_end"][-1]
    assert end["files"] == ["metrics.csv", "roc.png"] and end["overlap"] == []


def test_exec_snapshot_flags_overlapping_runs(run_dir):
    root, _ = run_dir
    a, b = "c" * 32, "d" * 32
    ta = monitor.exec_started(a)
    tb = monitor.exec_started(b)
    (root / "workspace" / "working" / "x.csv").write_text("v\n1\n")
    monitor.exec_finished(a, ta)
    manifest = json.loads((root / "workspace" / "node_outputs" / a / "manifest.json").read_text())
    assert manifest["overlapping_runs"] == [b]
    monitor.exec_finished(b, tb)


class _Bus:
    def __init__(self):
        self.stop = self.abort = False
        self.pushed = []

    def request_stop(self):
        self.stop = True

    def request_abort(self):
        self.abort = self.stop = True

    def push(self, action, **kw):
        self.pushed.append(action)


@pytest.mark.parametrize("line, action", [('{"action": "stop"}', "stop"), ('{"action": "abort"}', "abort"),
                                          ('{"action": "rm -rf"}', None), ("not json", None)])
def test_control_requests(run_dir, line, action):
    bus = _Bus()
    assert monitor.handle_control_request(line, bus) == action
    assert bus.pushed == ([action] if action else [])


def test_control_watcher_forwards_new_requests_only(run_dir):
    root, cfg = run_dir
    path = root / "logs" / monitor.CONTROL_FILE
    path.write_text(json.dumps({"action": "abort", "t": time.time() - 3600}) + "\n")  # before the run: ignored
    bus = _Bus()
    monitor.start_control_watcher(cfg, bus, interval=0.05)
    with open(path, "a") as f:
        f.write('{"action": "stop"}\n')
    deadline = time.time() + 3
    while not bus.pushed and time.time() < deadline:
        time.sleep(0.05)
    assert bus.pushed == ["stop"] and not bus.abort


# ── server ──────────────────────────────────────────────────────────────────


def _journal(root):
    nodes = [
        {"id": "r" * 32, "stage": "root", "metric": {"value": None, "maximize": None}},
        {"id": "a" * 32, "stage": "draft", "step": 1, "metric": {"value": 0.5, "maximize": True}, "is_buggy": False,
         "ctime": 1.0, "plan": "# Plan\nuse trees", "code": "print(1)", "_term_out": ["ok\n"], "analysis": "fine"},
        {"id": "b" * 32, "stage": "improve", "step": 2, "metric": {"value": 0.7, "maximize": True}, "is_buggy": False,
         "ctime": 2.0, "plan": "tune", "code": "print(2)", "_term_out": [], "analysis": ""},
        {"id": "c" * 32, "stage": "improve", "step": 3, "metric": {"value": None, "maximize": True}, "is_buggy": True,
         "ctime": 3.0, "plan": "bad", "code": "", "_term_out": [], "analysis": ""},
    ]
    parents = {"a" * 32: "r" * 32, "b" * 32: "a" * 32, "c" * 32: "a" * 32}
    (root / "logs" / "journal.json").write_text(json.dumps({"nodes": nodes, "node2parent": parents}))


def test_state_summarises_journal_and_events(run_dir):
    root, _ = run_dir
    _journal(root)
    h = _fake_agent_call(_Node("b" * 32))
    monitor.llm_end(h, 100, 20)
    monitor.llm_start("claude-code:opus", "generate")  # still running
    state = server.run_state(server.RunReader(root))
    assert state["best"] == "b" * 32
    assert state["counts"] == {"total": 3, "passed": 2, "failed": 1}
    assert [n["parent"] for n in state["nodes"]] == ["r" * 32, "a" * 32, "a" * 32]
    assert state["monitor"]["totals"]["in_tokens"] == 100
    assert state["status"] == "live" and state["control_available"]
    assert len(state["monitor"]["active"]) == 1
    assert next(n for n in state["nodes"] if n["id"] == "b" * 32)["tokens"]["calls"] == 1


def test_finished_run_has_no_active_work_or_control(run_dir):
    root, _ = run_dir
    _journal(root)
    monitor.llm_start("claude-code:opus", "query")
    monitor.run_ended()
    state = server.run_state(server.RunReader(root))
    assert state["status"] == "finished" and state["monitor"]["active"] == [] and not state["control_available"]


def test_node_detail_reads_outputs(run_dir):
    root, _ = run_dir
    _journal(root)
    out = root / "workspace" / "node_outputs" / ("a" * 32)
    out.mkdir(parents=True)
    (out / "metrics.csv").write_text("metric,value\nauc,0.9\n")
    (out / "metrics.md").write_text("| a | b |\n|---|---|\n| 1 | 2 |\n")
    (out / "roc.png").write_bytes(b"png")
    d = server.node_detail(server.RunReader(root), "a" * 32)
    assert d["outputs"]["tables"][0]["header"] == ["metric", "value"]
    assert d["outputs"]["figures"][0]["path"].endswith("node_outputs/" + "a" * 32 + "/roc.png")
    assert d["outputs"]["note"] is None and d["code"] == "print(1)"


@pytest.fixture
def http(run_dir):
    root, _ = run_dir
    _journal(root)
    app = server.Dashboard(root.parent, token="secret-token")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, root
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def _post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_page_runs_and_state(http):
    base, root = http
    status, page = _get(base + "/")
    assert status == 200 and b"secret-token" in page and b"__CONTROL_TOKEN__" not in page
    runs = json.loads(_get(base + "/api/runs")[1])
    assert runs[0]["name"] == root.name
    state = json.loads(_get(f"{base}/api/state?run={root.name}")[1])
    assert state["best"] == "b" * 32


@pytest.mark.parametrize("path", ["../../etc/passwd", "logs/../../x", "workspace/model.joblib"])
def test_http_file_stays_inside_run_and_types(http, path):
    base, root = http
    (root / "workspace" / "model.joblib").write_bytes(b"x")
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"{base}/api/file?run={root.name}&path={path}")
    assert e.value.code in (403, 404)


def test_http_unknown_run_is_404(http):
    base, _ = http
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/state?run=../logs")
    assert e.value.code == 404


def test_http_control_needs_token_and_same_origin(http):
    base, root = http
    url = base + "/api/control"
    body = {"run": root.name, "action": "stop"}
    assert _post(url, body, {})[0] == 403
    assert _post(url, body, {"X-Control-Token": "secret-token", "Origin": "https://evil.example"})[0] == 403
    assert _post(url, {**body, "action": "delete"}, {"X-Control-Token": "secret-token"})[0] == 400
    status, reply = _post(url, body, {"X-Control-Token": "secret-token", "Origin": base})
    assert status == 200 and reply["ok"]
    lines = (root / "logs" / monitor.CONTROL_FILE).read_text().splitlines()
    assert json.loads(lines[-1])["action"] == "stop"


def test_http_control_refused_for_finished_run(http):
    base, root = http
    monitor.run_ended()
    status, _ = _post(base + "/api/control", {"run": root.name, "action": "abort"}, {"X-Control-Token": "secret-token"})
    assert status == 409


def test_draft_calls_on_the_hidden_root_show_as_new_draft(run_dir):
    root, _ = run_dir
    _journal(root)
    handle = _fake_agent_call(_Node("r" * 32))  # the root is not listed as a solution
    state = server.run_state(server.RunReader(root))
    active = state["monitor"]["active"][0]
    assert active["node"] is None and active["new_draft"] is True
    monitor.llm_end(handle, 1, 1)


@pytest.mark.parametrize("plan, expected", [
    ("Use gradient boosting.", "Use gradient boosting."),
    ('{"reason": "AUC is flat", "plan": "{\\"model_design\\": \\"add LightGBM\\"}", "raw_response": "x"}',
     "**Why:** AUC is flat\n\n**Changes:**\n\n#### model design\nadd LightGBM"),
    ("{not json", "{not json"),
])
def test_plan_markdown(plan, expected):
    assert server.plan_markdown(plan) == expected
