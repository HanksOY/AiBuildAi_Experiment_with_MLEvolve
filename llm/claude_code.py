"""Claude through the local Claude Code CLI (`claude -p`), on the user's own login.

Select it with a model name of the form `claude-code:<model>` in
cfg.agent.code / cfg.agent.feedback, e.g. `claude-code:opus`. The part after the
colon is passed to `claude --model` (an alias such as opus/sonnet/haiku, or a
full model id); a bare `claude-code` uses the CLI's default model. base_url and
api_key are ignored: authentication is whatever `claude` is logged in with.

Each call runs one non-interactive `claude -p` in an empty directory with:
  --tools ""             no tools: the model answers, MLEvolve runs the code
  --setting-sources ""   no user/project settings, hooks or CLAUDE.md, so the
                         prompt is exactly what MLEvolve sends
  --no-session-persistence, --output-format json
Prompts go through stdin, which avoids command-line length limits.

The CLI has no max_tokens, temperature or stop-sequence flags; those
arguments are accepted and ignored. An optional `effort` key on the stage
config is passed to `claude --effort`.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time

from .gemini import FunctionSpec, compile_prompt_to_md

logger = logging.getLogger("MLEvolve")

PREFIX = "claude-code"
OutputType = str | dict

# Neutral role for Claude Code's own system prompt slot; MLEvolve's system
# content travels in the message body.
_SYSTEM_PROMPT = (
    "You are the language model behind an automated machine learning engineering system. "
    "Answer each request directly and in exactly the format it asks for. "
    "You have no tools; do not try to run, read or write anything."
)

_CALL_TIMEOUT_S = 1200
_workdir: str | None = None
# Usage of the last successful call on this thread, for the dashboard recorder.
_last = threading.local()


class ClaudeCodeError(RuntimeError):
    """The CLI ran but reported an error (rate limit, auth, bad request...)."""


def is_claude_code(model: str) -> bool:
    return (model or "").lower().startswith(PREFIX)


def cli_model(model: str) -> str | None:
    """`claude-code:opus` -> `opus`; `claude-code` -> None (CLI default)."""
    _, _, name = (model or "").partition(":")
    return name.strip() or None


def _empty_workdir() -> str:
    global _workdir
    if _workdir is None:
        _workdir = tempfile.mkdtemp(prefix="mlevolve-claude-")
    return _workdir


def build_command(model: str, json_schema: dict | None = None, effort: str | None = None) -> list[str]:
    claude = shutil.which("claude")
    if claude is None:
        raise FileNotFoundError(
            "The `claude` CLI is not on PATH. Install Claude Code and log in, "
            "or choose a different agent.<stage>.model."
        )
    cmd = [
        claude, "-p",
        "--output-format", "json",
        "--tools", "",
        "--setting-sources", "",
        "--no-session-persistence",
        "--system-prompt", _SYSTEM_PROMPT,
    ]
    name = cli_model(model)
    if name:
        cmd += ["--model", name]
    if effort:
        cmd += ["--effort", effort]
    if json_schema is not None:
        cmd += ["--json-schema", json.dumps(json_schema)]
    return cmd


def combine(system_message: str | None, user_message: str | None) -> str:
    """One message body: the CLI takes a single prompt."""
    if system_message and user_message:
        return f"{system_message}\n\n{user_message}"
    return system_message or user_message or ""


def parse_result(stdout: str) -> dict:
    """The CLI's final JSON object; raises ClaudeCodeError on a reported failure."""
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ClaudeCodeError(f"claude returned non-JSON output: {stdout[:500]!r}") from e
    if result.get("is_error") or result.get("subtype") != "success":
        status = result.get("api_error_status")
        raise ClaudeCodeError(
            f"claude reported an error (status={status}, subtype={result.get('subtype')}): "
            f"{str(result.get('result'))[:500]}"
        )
    return result


def _run(prompt: str, model: str, json_schema: dict | None, effort: str | None) -> tuple[dict, float]:
    cmd = build_command(model, json_schema, effort)
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=_empty_workdir(),
        timeout=_CALL_TIMEOUT_S,
    )
    req_time = time.time() - t0
    if proc.returncode != 0 and not proc.stdout.strip():
        raise ClaudeCodeError(f"claude exited with {proc.returncode}: {proc.stderr[-500:]}")
    result = parse_result(proc.stdout)
    in_tok, out_tok = _usage(result)
    _last.usage = {"in_tokens": in_tok, "out_tokens": out_tok, "cost_usd": result.get("total_cost_usd")}
    return result, req_time


def last_usage() -> dict | None:
    """Tokens of this thread's last successful CLI call, then cleared."""
    usage = getattr(_last, "usage", None)
    _last.usage = None
    return usage


def _usage(result: dict) -> tuple[int, int]:
    usage = result.get("usage") or {}
    in_tok = sum(usage.get(k, 0) or 0 for k in (
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return in_tok, usage.get("output_tokens", 0) or 0


def _structured(result: dict) -> dict:
    output = result.get("structured_output")
    if isinstance(output, dict):
        return output
    # Fall back to the text answer if the CLI did not fill structured_output.
    return json.loads(result.get("result") or "")


def _effort_for(cfg, model: str) -> str | None:
    agent = getattr(cfg, "agent", None)
    for stage in (getattr(agent, "code", None), getattr(agent, "feedback", None)):
        if stage is not None and getattr(stage, "model", None) == model:
            return getattr(stage, "effort", None)
    return None


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg=None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Same return shape as gemini.query / openai.query."""
    model = model_kwargs.get("model") or PREFIX
    schema = func_spec.json_schema if func_spec is not None else None
    result, req_time = _run(combine(system_message, user_message), model, schema, _effort_for(cfg, model))
    output: OutputType = _structured(result) if func_spec is not None else (result.get("result") or "")
    logger.info(f"Claude Code response: {output}", extra={"verbose": True})
    in_tok, out_tok = _usage(result)
    info = {"model": model, "created": int(time.time()), "cost_usd_equivalent": result.get("total_cost_usd")}
    return output, req_time, in_tok, out_tok, info


def prompt_text(prompt) -> str:
    """Flatten MLEvolve's {system, user, assistant} chat dict into one message.

    Claude Code cannot prefill an assistant turn, so the assistant text is
    appended to the message, as the OpenAI backend does for GPT models.
    """
    if isinstance(prompt, dict) and ("system" in prompt or "user" in prompt or "assistant" in prompt):
        parts = [str(prompt[k]) for k in ("system", "user", "assistant") if prompt.get(k)]
        if not parts:
            raise ValueError("Chat dict must have at least one of: system, user, assistant")
        return "\n\n".join(parts)
    return prompt if isinstance(prompt, str) else compile_prompt_to_md(prompt)


def generate(
    prompt,
    cfg,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 20,
    retry_delay: float = 3,
    stage: str = "code",
) -> str:
    """Text generation; with json_schema, returns the structured answer as JSON text."""
    stage_cfg = getattr(cfg.agent, stage, None) or cfg.agent.code
    model = stage_cfg.model
    text = prompt_text(prompt)
    effort = getattr(stage_cfg, "effort", None)
    for attempt in range(max_retries):
        try:
            result, _ = _run(text, model, json_schema, effort)
            output = json.dumps(_structured(result)) if json_schema is not None else (result.get("result") or "")
            logger.info(f"generate response: {output}", extra={"verbose": True})
            return output
        except Exception as e:
            logger.warning(f"generate failed, retrying {attempt + 1}/{max_retries}: {e}")
            if attempt >= max_retries - 1:
                logger.error("generate retry limit reached")
                raise
            time.sleep(retry_delay)
    return ""
