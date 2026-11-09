import logging
from . import gemini as _gemini
from . import openai as _openai
from . import claude_code as _claude_code
from .gemini import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from config import Config
from utils import monitor
logger = logging.getLogger("MLEvolve")


def _provider(model: str) -> str:
    """gemini* -> Gemini; claude-code[:model] -> local Claude Code CLI; else OpenAI-compatible (e.g. Qwen)."""
    if _claude_code.is_claude_code(model):
        return "claude_code"
    return "gemini" if (model or "").lower().startswith("gemini") else "openai"


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    cfg:Config=None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gemini-3-pro-preview")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("---Querying model---", extra={"verbose": True})
    system_message = compile_prompt_to_md(system_message) if system_message else None
    if system_message:
        if len(system_message) > 1000:
            logger.info(f"system: {system_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"system: {system_message}", extra={"verbose": True})
    user_message = compile_prompt_to_md(user_message) if user_message else None
    if user_message:
        if len(user_message) > 1000:
            logger.info(f"user: {user_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"user: {user_message}", extra={"verbose": True})
    if func_spec:
        logger.info(f"function spec: {func_spec.to_dict()}", extra={"verbose": True})

    provider = _provider(model)
    call = monitor.llm_start(model, "query")
    try:
        output, in_tok_count, out_tok_count, info = _query_backend(
            provider, system_message, user_message, func_spec, cfg, model_kwargs
        )
    except Exception as e:
        monitor.llm_end(call, ok=False, error=str(e))
        raise
    monitor.llm_end(call, in_tok_count, out_tok_count, cost_usd=(info or {}).get("cost_usd_equivalent"))
    logger.info("---Query complete---", extra={"verbose": True})

    return output


def _query_backend(provider, system_message, user_message, func_spec, cfg, model_kwargs):
    if provider == "claude_code":
        output, req_time, in_tok_count, out_tok_count, info = _claude_code.query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            cfg=cfg,
            **model_kwargs,
        )
    elif provider == "openai":
        output, req_time, in_tok_count, out_tok_count, info = _openai.query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            cfg=cfg,
            **model_kwargs,
        )
    else:
        output, req_time, in_tok_count, out_tok_count, info = _gemini.query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            cfg=cfg,
            **model_kwargs,
        )
    return output, in_tok_count, out_tok_count, info


def stage_config(cfg: Config, stage: str = "code"):
    """Return the cfg.agent.<stage> block, falling back to `code` if absent."""
    return getattr(cfg.agent, stage, None) or cfg.agent.code


def generate(
    prompt,
    cfg,
    temperature=None,
    max_tokens=None,
    stop_tokens=None,
    json_schema=None,
    max_retries=20,
    retry_delay=3,
    stage="code",
):
    """Streaming text generation.

    `stage` selects which cfg.agent.<stage> block supplies the model, base_url
    and api_key. It defaults to "code" because most callers are code-generation
    agents; result/validation agents should pass stage="feedback" so they are
    not silently billed to the code model's credentials.
    """
    model = getattr(stage_config(cfg, stage), "model", "") or ""
    call = monitor.llm_start(model, "generate")
    try:
        output = _generate_backend(model, prompt, cfg, temperature, max_tokens, stop_tokens,
                                   json_schema, max_retries, retry_delay, stage)
    except Exception as e:
        monitor.llm_end(call, ok=False, error=str(e))
        raise
    usage = _claude_code.last_usage() if _provider(model) == "claude_code" else None
    if usage:
        monitor.llm_end(call, usage["in_tokens"], usage["out_tokens"], cost_usd=usage.get("cost_usd"))
    else:
        # The streaming OpenAI/Gemini paths report no usage; estimate from text length.
        monitor.llm_end(call, monitor.estimate_tokens(prompt), monitor.estimate_tokens(output), estimated=True)
    return output


def _generate_backend(model, prompt, cfg, temperature, max_tokens, stop_tokens,
                      json_schema, max_retries, retry_delay, stage):
    if _provider(model) == "claude_code":
        return _claude_code.generate(
            prompt=prompt,
            cfg=cfg,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            json_schema=json_schema,
            max_retries=max_retries,
            retry_delay=retry_delay,
            stage=stage,
        )
    if _provider(model) == "openai":
        return _openai.generate(
            prompt=prompt,
            cfg=cfg,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            json_schema=json_schema,
            max_retries=max_retries,
            retry_delay=retry_delay,
            stage=stage,
        )
    return _gemini.generate(
        prompt=prompt,
        cfg=cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_tokens=stop_tokens,
        json_schema=json_schema,
        max_retries=max_retries,
        retry_delay=retry_delay,
        stage=stage,
    )
