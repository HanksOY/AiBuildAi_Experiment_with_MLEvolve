"""Claude Code CLI backend, with the CLI stubbed out (no model calls)."""

import json
import subprocess
from types import SimpleNamespace

import pytest

import llm
from llm import claude_code
from llm.gemini import FunctionSpec

SCHEMA = {"type": "object", "properties": {"lower_is_better": {"type": "boolean"}}, "required": ["lower_is_better"]}


def _cli_result(**overrides):
    result = {
        "type": "result", "subtype": "success", "is_error": False,
        "result": "hello", "structured_output": None,
        "usage": {"input_tokens": 3, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 7},
        "total_cost_usd": 0.01,
    }
    result.update(overrides)
    return json.dumps(result)


@pytest.fixture
def fake_cli(monkeypatch):
    """Record each CLI invocation and reply with a canned result."""
    calls = []
    reply = {"stdout": _cli_result(), "returncode": 0}

    def run(cmd, input, **kwargs):
        calls.append({"cmd": cmd, "input": input, "cwd": kwargs.get("cwd")})
        return subprocess.CompletedProcess(cmd, reply["returncode"], reply["stdout"], "")

    monkeypatch.setattr(claude_code.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(claude_code.subprocess, "run", run)
    return SimpleNamespace(calls=calls, reply=reply)


def _cfg(model="claude-code:opus", effort=None):
    stage = SimpleNamespace(model=model, effort=effort)
    return SimpleNamespace(agent=SimpleNamespace(code=stage, feedback=stage))


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


@pytest.mark.parametrize("model, expected", [
    ("claude-code:opus", "claude_code"),
    ("claude-code", "claude_code"),
    ("claude-opus-5-5", "openai"),   # API model names keep the OpenAI-compatible path
    ("gemini-3-pro", "gemini"),
    ("glm-5.2", "openai"),
])
def test_provider_routing(model, expected):
    assert llm._provider(model) == expected


@pytest.mark.parametrize("model, expected", [
    ("claude-code:opus", "opus"), ("claude-code:claude-sonnet-5-5", "claude-sonnet-5-5"), ("claude-code", None),
])
def test_cli_model(model, expected):
    assert claude_code.cli_model(model) == expected


def test_command_isolates_the_call(fake_cli):
    cmd = claude_code.build_command("claude-code:opus", SCHEMA, effort="high")
    assert cmd[:2] == ["/usr/bin/claude", "-p"]
    assert _flag(cmd, "--tools") == ""
    assert _flag(cmd, "--setting-sources") == ""     # no hooks, settings or CLAUDE.md
    assert _flag(cmd, "--output-format") == "json"
    assert "--no-session-persistence" in cmd
    assert "--bare" not in cmd                       # --bare ignores the OAuth login
    assert _flag(cmd, "--model") == "opus"
    assert _flag(cmd, "--effort") == "high"
    assert json.loads(_flag(cmd, "--json-schema")) == SCHEMA


def test_command_without_model_or_effort(fake_cli):
    cmd = claude_code.build_command("claude-code")
    assert "--model" not in cmd and "--effort" not in cmd and "--json-schema" not in cmd


def test_missing_cli_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(claude_code.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="claude"):
        claude_code.build_command("claude-code:opus")


def test_text_query_sends_prompt_on_stdin(fake_cli):
    output, _, in_tok, out_tok, info = llm._claude_code.query(
        system_message="SYSTEM", user_message="USER", cfg=_cfg(), model="claude-code:opus",
        temperature=0.5, max_tokens=100,
    )
    assert output == "hello"
    assert (in_tok, out_tok) == (18, 7)
    call = fake_cli.calls[0]
    assert call["input"] == "SYSTEM\n\nUSER"
    assert "SYSTEM" not in " ".join(call["cmd"])   # prompts never go on the command line
    assert info["model"] == "claude-code:opus"


def test_function_query_returns_structured_output(fake_cli):
    fake_cli.reply["stdout"] = _cli_result(structured_output={"lower_is_better": True})
    spec = FunctionSpec(name="direction", json_schema=SCHEMA, description="d")
    output, *_ = claude_code.query("SYSTEM", None, func_spec=spec, cfg=_cfg(effort="low"), model="claude-code:opus")
    assert output == {"lower_is_better": True}
    cmd = fake_cli.calls[0]["cmd"]
    assert json.loads(_flag(cmd, "--json-schema")) == SCHEMA
    assert _flag(cmd, "--effort") == "low"


def test_structured_output_falls_back_to_result_text(fake_cli):
    fake_cli.reply["stdout"] = _cli_result(result='{"lower_is_better": false}')
    spec = FunctionSpec(name="direction", json_schema=SCHEMA, description="d")
    output, *_ = claude_code.query("SYSTEM", None, func_spec=spec, cfg=_cfg(), model="claude-code:opus")
    assert output == {"lower_is_better": False}


@pytest.mark.parametrize("stdout", [
    _cli_result(is_error=True, subtype="error_during_execution", api_error_status=429),
    "not json",
])
def test_cli_errors_raise(fake_cli, stdout):
    fake_cli.reply["stdout"] = stdout
    with pytest.raises(claude_code.ClaudeCodeError):
        claude_code.query("SYSTEM", None, cfg=_cfg(), model="claude-code:opus")


def test_generate_flattens_chat_prompt(fake_cli):
    out = llm.generate({"system": "S", "user": "U", "assistant": "A"}, _cfg())
    assert out == "hello"
    assert fake_cli.calls[0]["input"] == "S\n\nU\n\nA"


def test_generate_with_schema_returns_json_text(fake_cli):
    fake_cli.reply["stdout"] = _cli_result(structured_output={"lower_is_better": True})
    out = llm.generate("P", _cfg(), json_schema=SCHEMA)
    assert json.loads(out) == {"lower_is_better": True}


def test_generate_retries_then_succeeds(fake_cli, monkeypatch):
    monkeypatch.setattr(claude_code.time, "sleep", lambda s: None)
    replies = iter([_cli_result(is_error=True, subtype="error_during_execution"), _cli_result()])

    def run(cmd, input, **kwargs):
        fake_cli.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, next(replies), "")

    monkeypatch.setattr(claude_code.subprocess, "run", run)
    assert llm.generate("P", _cfg(), max_retries=3) == "hello"
    assert len(fake_cli.calls) == 2


def test_calls_run_in_an_empty_directory(fake_cli, tmp_path):
    claude_code.query("SYSTEM", None, cfg=_cfg(), model="claude-code:opus")
    cwd = fake_cli.calls[0]["cwd"]
    assert cwd and not any(__import__("os").scandir(cwd))
