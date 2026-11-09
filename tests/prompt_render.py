"""Render every prompt builder that can run without an LLM, for snapshot tests.

`render_builders()` with no framing reproduces upstream output, which is what
`snapshots/prompts_kaggle.json` records. Passing a Framing renders the same
builders under that framing.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def make_dataset(root: Path) -> Path:
    """A tiny workspace with train, labelled validation, and sample submission files."""
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": [1, 2, 3], "x": [0.1, 0.2, 0.3], "y": [0, 1, 0]}).to_csv(input_dir / "train.csv", index=False)
    pd.DataFrame({"id": [4, 5], "x": [0.4, 0.5], "y": [1, 0]}).to_csv(input_dir / "val.csv", index=False)
    pd.DataFrame({"id": [6, 7], "y": [0.5, 0.5]}).to_csv(input_dir / "sample_submission.csv", index=False)
    return root


def _kw(framing):
    return {} if framing is None else {"framing": framing}


def render_builders(workspace: Path, monkeypatch, framing=None) -> dict:
    import llm
    from agents.prompts.impl_guideline import get_impl_guideline
    from agents.prompts.shared import get_internet_clarification, prompt_resp_fmt
    from agents.prompts.validation_template_prompts import get_code_review_prompt
    from agents.coder.stepwise_coder import create_default_step_agents
    from agents.result_parse_agent import _build_introduction
    from agents import improve_agent, evolution_agent, fusion_agent
    from utils import data_preview

    out = {}
    out["impl_guideline"] = get_impl_guideline(
        tot_time_remaining=5 * 3600 + 125, steps_remaining=40, exec_timeout=7200, **_kw(framing)
    )
    out["impl_guideline_options"] = get_impl_guideline(
        tot_time_remaining=600, steps_remaining=3, exec_timeout=300,
        expose_prediction=True, k_fold_validation=5, pretrain_model_dir="/models", **_kw(framing),
    )
    out["code_review_prompt"] = get_code_review_prompt("TASK DESC", "print(1)", **_kw(framing))
    out["internet_clarification"] = get_internet_clarification("", **_kw(framing))
    out["internet_clarification_models"] = get_internet_clarification("/models", **_kw(framing))
    out["resp_fmt"] = prompt_resp_fmt()
    out["data_preview"] = data_preview.generate(workspace, **_kw(framing)).replace(str(workspace), "<WS>")
    out["step_agents"] = [
        {"name": a.name, "introduction": a.introduction, "description": a.description, "guidelines": a.guidelines}
        for a in create_default_step_agents(**_kw(framing))
    ]

    # clean_task_desc makes one LLM call; capture the prompt and echo a fixed reply.
    captured = {}

    def fake_query(system_message, user_message=None, **kwargs):
        captured["system_message"] = system_message
        return "CLEANED TASK"

    monkeypatch.setattr(llm, "query", fake_query)
    cfg = SimpleNamespace(
        agent=SimpleNamespace(code=SimpleNamespace(model="stub")),
        workspace_dir=workspace,
    )
    if framing is not None:
        cfg.task = SimpleNamespace(profile=framing.profile, require_submission=framing.require_submission)
    out["clean_task_desc_output"] = data_preview.clean_task_desc("TASK DESC", cfg)
    out["clean_task_desc_prompt"] = captured["system_message"]

    for use_memory in (False, True):
        agent = SimpleNamespace(acfg=SimpleNamespace(use_global_memory=use_memory), cfg=cfg)
        out[f"parse_intro_memory_{use_memory}"] = _build_introduction(agent)

    # Module-level intros are adapted where diff_generate_and_apply receives them.
    adapt = (lambda text: text) if framing is None else framing.adapt
    out["improve_diff_intro"] = adapt(improve_agent._IMPROVE_DIFF_INTRODUCTION)
    out["evolution_diff_intro"] = adapt(evolution_agent._EVOLUTION_DIFF_INTRODUCTION)
    out["fusion_diff_intro"] = adapt(fusion_agent._FUSION_DIFF_INTRODUCTION)
    out["multi_fusion_diff_intro"] = adapt(fusion_agent._MULTI_FUSION_DIFF_INTRODUCTION)

    # Round-trip through JSON so tuples and lists compare the same as the snapshot.
    return json.loads(json.dumps(out, ensure_ascii=False))
