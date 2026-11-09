"""Task framing: upstream prompts unchanged, generic prompts free of Kaggle wording."""

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from agents.prompts.framing import (
    GENERIC,
    GENERIC_REWRITES,
    KAGGLE,
    OPTIONAL_SUBMISSION_REWRITES,
    UPSTREAM,
    Framing,
    framing_of,
)
from prompt_render import make_dataset, render_builders

REPO = Path(__file__).resolve().parent.parent
SNAPSHOT = Path(__file__).parent / "snapshots" / "prompts_kaggle.json"
KAGGLE_WORDS = re.compile(r"kaggle|competition|grandmaster|leaderboard|mle-?bench", re.IGNORECASE)

# Files whose string constants end up in prompts.
PROMPT_SOURCES = sorted((REPO / "agents").rglob("*.py")) + [
    REPO / "utils" / "data_preview.py",
    REPO / "engine" / "agent_search.py",
]
# The framing module holds the rewrite table itself.
PROMPT_SOURCES = [p for p in PROMPT_SOURCES if p.name != "framing.py"]

GENERIC_NO_SUBMISSION = Framing(profile=GENERIC, require_submission=False)


@pytest.fixture
def workspace(tmp_path):
    return make_dataset(tmp_path)


def _cfg(tmp_path, profile=KAGGLE, require_submission=True):
    return SimpleNamespace(
        workspace_dir=tmp_path,
        exp_id="task",
        task=SimpleNamespace(profile=profile, require_submission=require_submission),
    )


# ── Upstream (kaggle) output is byte-identical ──────────────────────────────


@pytest.mark.parametrize("framing", [None, UPSTREAM])
def test_upstream_prompts_match_snapshot(workspace, monkeypatch, framing):
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert render_builders(workspace, monkeypatch, framing) == expected


def test_upstream_adapt_returns_input_unchanged():
    text = "You are a Kaggle grandmaster attending a competition."
    lines = [text, "**2. Generate submission.csv**"]
    assert UPSTREAM.adapt(text) is text
    assert UPSTREAM.adapt_lines(lines) is lines


def test_default_config_is_upstream():
    from config import TaskConfig

    assert framing_of(SimpleNamespace()) == UPSTREAM
    assert framing_of(SimpleNamespace(task=TaskConfig())) == UPSTREAM
    yaml_cfg = OmegaConf.load(REPO / "config" / "config.yaml")
    assert framing_of(yaml_cfg) == UPSTREAM


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        Framing(profile="kagle")


def test_unknown_profile_fails_at_config_load(tmp_path):
    """A typo must stop the run before the interactive gates, not after them."""
    from config import _load_cfg, prep_cfg

    cfg = _load_cfg(use_cli_args=False)
    cfg.log_dir = cfg.workspace_dir = str(tmp_path)
    cfg.task.profile = "generic."
    with pytest.raises(ValueError, match="generic\\."):
        prep_cfg(cfg)


# ── Generic output carries no Kaggle wording ────────────────────────────────


@pytest.mark.parametrize("framing", [Framing(profile=GENERIC), GENERIC_NO_SUBMISSION])
def test_generic_prompts_have_no_kaggle_wording(workspace, monkeypatch, framing):
    rendered = json.dumps(render_builders(workspace, monkeypatch, framing), ensure_ascii=False)
    assert KAGGLE_WORDS.findall(rendered) == []


def _string_constants(tree):
    """(node, text) for every prompt string literal: plain strings and f-string parts.

    Docstrings and logger/exception messages are not prompt text.
    """
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                skip.add(id(body[0].value))
        if isinstance(node, ast.Call):
            func = node.func
            is_log = isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "logger"
            if is_log or (isinstance(func, ast.Name) and func.id.endswith(("Error", "Exception"))):
                for sub in ast.walk(node):
                    skip.add(id(sub))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            yield node, node.value


def _adapted_spans(tree):
    """Constants covered by an adapt()/adapt_lines() call, directly or through a variable."""
    covered, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("adapt", "adapt_lines"):
            for sub in ast.walk(node):
                covered.add(id(sub))
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            for sub in ast.walk(node.value):
                covered.add(id(sub))
        # name.append(...) / name.extend(...) on a list that is later adapted
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("append", "extend")
                and isinstance(node.func.value, ast.Name) and node.func.value.id in names):
            for sub in ast.walk(node):
                covered.add(id(sub))
    return covered


@pytest.mark.parametrize("path", PROMPT_SOURCES, ids=lambda p: str(p.relative_to(REPO)))
def test_every_kaggle_phrase_in_source_is_adapted(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    covered = _adapted_spans(tree)
    problems = []
    for node, text in _string_constants(tree):
        if not KAGGLE_WORDS.search(text):
            continue
        if id(node) not in covered:
            problems.append(f"line {node.lineno}: not passed through Framing.adapt: {text[:70]!r}")
        elif KAGGLE_WORDS.search(Framing(profile=GENERIC).adapt(text)):
            problems.append(f"line {node.lineno}: no generic rewrite covers: {text[:70]!r}")
    assert problems == []


def test_every_rewrite_matches_some_prompt_text():
    corpus = "\n".join(
        text for path in PROMPT_SOURCES for _, text in _string_constants(ast.parse(path.read_text(encoding="utf-8")))
    )
    dead = [old for old, _ in GENERIC_REWRITES + OPTIONAL_SUBMISSION_REWRITES if old not in corpus]
    assert dead == []


# ── Optional submission ─────────────────────────────────────────────────────


def test_optional_submission_prompts(workspace, monkeypatch):
    out = render_builders(workspace, monkeypatch, GENERIC_NO_SUBMISSION)
    impl = "\n".join(out["impl_guideline"]["Implementation guideline"])
    assert "**2. Generate submission.csv**" not in impl
    assert "**2. Prediction File (only if the task asks for one)**" in impl
    review = json.dumps(out["code_review_prompt"], ensure_ascii=False)
    assert "Must save the submission" not in review
    training = next(a for a in out["step_agents"] if a["name"] == "training_evaluation")
    assert "Only if the task asks for a prediction file" in training["guidelines"][0]
    # The score line stays mandatory: search cannot rank nodes without it.
    assert "Final Validation Score" in impl


def _parse_node():
    return SimpleNamespace(id="n1", exc_type=None, analysis="")


@pytest.mark.parametrize(
    "require, has_csv, metric, buggy",
    [
        (True, False, 0.9, True),    # upstream: no file means failure
        (True, True, 0.9, False),
        (False, False, 0.9, False),  # score alone is the result
        (False, False, None, True),  # still needs a score
        (False, True, 0.9, False),
    ],
)
def test_determine_buggy(require, has_csv, metric, buggy):
    from agents.result_parse_agent import _determine_buggy

    node = _parse_node()
    _determine_buggy(node, {"is_bug": False, "metric": metric}, has_csv, require)
    assert node.is_buggy is buggy
    assert node.completed_with_errors is False


def test_determine_buggy_default_is_upstream():
    from agents.result_parse_agent import _determine_buggy

    node = _parse_node()
    _determine_buggy(node, {"is_bug": False, "metric": 0.9}, False)
    assert node.is_buggy is True


@pytest.mark.parametrize("require, buggy", [(True, True), (False, False)])
def test_validate_executed_node_without_submission(tmp_path, require, buggy):
    from engine.execution import validate_executed_node
    from utils.metric import MetricValue

    agent = SimpleNamespace(cfg=_cfg(tmp_path, GENERIC, require), branch_successful_nodes={})
    node = SimpleNamespace(id="n1", is_buggy=False, metric=MetricValue(0.8, maximize=True), branch_id=1)
    validate_executed_node(agent, node)
    assert node.is_buggy is buggy
    assert (agent.branch_successful_nodes == {1: [node]}) is not buggy


def _no_server(*args, **kwargs):
    raise AssertionError("the MLE-Bench format server must not be called")


def test_generic_without_file_skips_validation(tmp_path, monkeypatch):
    from agents import result_parse_agent

    monkeypatch.setattr(result_parse_agent, "_validate_submission_with_retry", _no_server)
    agent = SimpleNamespace(cfg=_cfg(tmp_path, GENERIC, False))
    node = SimpleNamespace(id="n1", is_valid=None)
    result_parse_agent._validate_format_with_retry(agent, node)
    assert node.is_valid is True


def test_generic_with_file_checks_content_not_server(tmp_path, monkeypatch):
    from agents import result_parse_agent

    monkeypatch.setattr(result_parse_agent, "_validate_submission_with_retry", _no_server)
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "submission_n1.csv").write_text("id,p\n1,0.1\n2,0.7\n3,0.4\n")
    agent = SimpleNamespace(cfg=_cfg(tmp_path, GENERIC, True))
    node = SimpleNamespace(id="n1", is_valid=None, is_buggy=False)
    result_parse_agent._validate_format_with_retry(agent, node)
    assert node.is_valid is True


def test_save_best_solution_without_submission(tmp_path):
    import threading
    from engine.solution_manager import save_best_solution
    from utils.metric import MetricValue

    agent = SimpleNamespace(cfg=_cfg(tmp_path, GENERIC, False), save_node_lock=threading.Lock(), metric_maximize=True)
    node = SimpleNamespace(id="n1", code="print(1)", metric=MetricValue(0.8, maximize=True), exec_time=1.0)
    save_best_solution(agent, node, tmp_path / "submission" / "submission_n1.csv")
    assert (tmp_path / "best_solution" / "solution.py").read_text() == "print(1)"
    assert not (tmp_path / "best_submission" / "submission.csv").exists()
