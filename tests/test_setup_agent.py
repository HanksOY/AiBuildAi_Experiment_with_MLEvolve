"""Setup agent with the Claude Code provider: no API key asked, nothing written to .env."""

from types import SimpleNamespace

import pytest

from agents import setup_agent


@pytest.fixture
def answers(monkeypatch):
    """Feed scripted answers to _ask and Confirm.ask; record the prompts shown."""
    script = SimpleNamespace(replies=[], prompts=[])

    def ask(console, prompt, default="", help_text=""):
        script.prompts.append(prompt)
        reply = script.replies.pop(0)
        return default if reply == "" else reply

    monkeypatch.setattr(setup_agent, "_ask", ask)
    monkeypatch.setattr(setup_agent.Confirm, "ask", lambda *a, **k: False)
    monkeypatch.setattr(setup_agent, "collect_compute", lambda console: {"cpu_number": 4, "num_gpus": 0})
    return script


@pytest.fixture
def fake_claude(monkeypatch):
    calls = []

    def query(system_message, user_message, **kwargs):
        calls.append(kwargs["model"])
        return "ok", 0.1, 1, 1, {}

    from llm import claude_code

    monkeypatch.setattr(claude_code, "query", query)
    return calls


def _claude_choice():
    return str(next(i for i, p in enumerate(setup_agent.PROVIDERS, 1) if p[0] == setup_agent.CLAUDE_CODE_PROVIDER))


def test_claude_code_defaults_need_no_key(answers, fake_claude):
    answers.replies = [_claude_choice(), "", ""]   # accept the default models
    creds = setup_agent.collect_credentials(setup_agent.Console())
    assert creds["code_model"] == "claude-code:opus"
    assert creds["feedback_model"] == "claude-code:sonnet"
    assert creds["api_key"] == ""
    assert not any("API key" in p for p in answers.prompts)
    assert fake_claude == ["claude-code:opus"]


@pytest.mark.parametrize("typed, expected", [
    ("haiku", "claude-code:haiku"),
    ("claude-opus-5-5", "claude-code:claude-opus-5-5"),
    ("claude-code:sonnet", "claude-code:sonnet"),   # prefix is not doubled
])
def test_claude_model_names(answers, fake_claude, typed, expected):
    answers.replies = [_claude_choice(), typed, typed]
    creds = setup_agent.collect_credentials(setup_agent.Console())
    assert creds["code_model"] == creds["feedback_model"] == expected


def test_failed_claude_login_returns_none(answers, monkeypatch):
    from llm import claude_code

    def broken(*a, **k):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(claude_code, "query", broken)
    answers.replies = [_claude_choice(), "", ""]
    assert setup_agent.collect_credentials(setup_agent.Console()) is None


@pytest.mark.parametrize("models, expected", [
    (("claude-code:opus", "claude-code:sonnet"), True),
    (("claude-code:opus", "glm-4.5-air"), False),
    (("glm-5.2", "glm-4.5-air"), False),
])
def test_uses_claude_code(models, expected):
    assert setup_agent.uses_claude_code(*models) is expected


def test_needs_setup_skipped_for_claude_code(monkeypatch, tmp_path):
    monkeypatch.delenv("MLEVOLVE_CODE_API_KEY", raising=False)
    monkeypatch.setattr(setup_agent, "ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setattr(setup_agent, "_configured_models", lambda: ("claude-code:opus", "claude-code:sonnet"))
    assert setup_agent.needs_setup() is False
    monkeypatch.setattr(setup_agent, "_configured_models", lambda: ("glm-5.2", "glm-4.5-air"))
    assert setup_agent.needs_setup() is True


def test_configured_models_read_cli_overrides(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run.py", "agent.code.model=claude-code:opus", "agent.feedback.model=claude-code:haiku"])
    assert setup_agent._configured_models() == ("claude-code:opus", "claude-code:haiku")


def test_write_config_sets_claude_models(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        'data_dir: "/old"\ndataset_dir: "/old"\ngoal: |\n  old goal\neval: |\n  old eval\n'
        "cpu_number: 11\nagent:\n  code:\n    model: glm-5.2\n    base_url: \"https://x\"\n"
        "  feedback:\n    model: glm-4.5-air\n    base_url: \"https://x\"\n  search:\n    num_gpus: 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(setup_agent, "CONFIG_PATH", config)
    creds = {"base_url": "", "code_model": "claude-code:opus", "feedback_model": "claude-code:sonnet", "api_key": ""}
    setup_agent.write_config(creds, "/mnt/e/data", "new goal", "new eval", {"cpu_number": 4, "num_gpus": 0})

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config)
    assert cfg.agent.code.model == "claude-code:opus"
    assert cfg.agent.feedback.model == "claude-code:sonnet"
    assert cfg.data_dir == "/mnt/e/data" and cfg.goal.strip() == "new goal"
