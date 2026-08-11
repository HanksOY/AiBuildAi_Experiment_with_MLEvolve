"""Gemini API backend: function calling (query), streaming generation (generate),
   prompt compilation, retry logic, and function-calling specs."""

import json
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import backoff
import jsonschema
from dataclasses_json import DataClassJsonMixin
from funcy import notnone, once, select_values
from google import genai
from google.genai import types
from config import Config

logger = logging.getLogger("MLEvolve")

# ---------------------------------------------------------------------------
#  Type aliases
# ---------------------------------------------------------------------------
PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType


def _build_thinking_config(model_name: str, level: str = "high"):
    """Return a ThinkingConfig appropriate for the model, or None if unsupported.

    - Gemini 3.x: uses thinking_level ("minimal"/"low"/"medium"/"high")
    - Gemini 2.5 Pro: uses thinking_budget (128-32768; cannot disable)
    - Gemini 2.5 Flash/Flash-Lite: uses thinking_budget (0-24576; 0 disables)
    - Other Gemini: no thinking config
    """
    name = (model_name or "").lower()
    if name.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level=level)
    if name.startswith("gemini-2.5"):
        # Map level → budget. "high" forces max thinking.
        budget_map = {"minimal": 128, "low": 4096, "medium": 16384, "high": 32768}
        budget = budget_map.get(level, 32768)
        # Flash variants cap at 24576
        if "flash" in name:
            budget = min(budget, 24576)
        return types.ThinkingConfig(thinking_budget=budget)
    return None


# ---------------------------------------------------------------------------
#  Prompt & message helpers
# ---------------------------------------------------------------------------

@backoff.on_predicate(
    wait_gen=backoff.constant,
    interval=5,
    max_time=300,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    """Call *create_fn* with automatic retry on transient errors."""
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        logger.warning(f"Retryable error: {e}\n{traceback.format_exc()}")
        return False


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
            "strict": True,
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }

# ---------------------------------------------------------------------------
#  Gemini client
# ---------------------------------------------------------------------------

_client: genai.Client = None  # type: ignore


GEMINI_TIMEOUT_EXCEPTIONS = (
    Exception,  # Gemini SDK may throw various exceptions
)


@once
def _setup_gemini_client(cfg: Config):
    global _client
    _client = genai.Client(
        api_key=cfg.agent.code.api_key,
        http_options={'base_url': cfg.agent.code.base_url, 'timeout': 1200000}
    )


def _convert_func_spec_to_gemini_tool(func_spec: FunctionSpec) -> types.Tool:
    """Convert FunctionSpec to Gemini Tool format."""
    function_declaration = types.FunctionDeclaration(
        name=func_spec.name,
        description=func_spec.description,
        parameters=func_spec.json_schema
    )
    return types.Tool(function_declarations=[function_declaration])


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_gemini_client(cfg)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    # Construct contents for Gemini
    contents = []
    if system_message:
        if user_message:
            contents = f"{system_message}\n\n{user_message}"
        else:
            contents = system_message
    elif user_message:
        contents = user_message
    else:
        raise ValueError("Either system_message or user_message must be provided")

    # Build generation config with tools if func_spec is provided
    config_params = {
        "temperature": filtered_kwargs.get("temperature", 1.0),
        "max_output_tokens": filtered_kwargs.get("max_tokens", 16384),
    }
    thinking_cfg = _build_thinking_config(filtered_kwargs.get("model", ""), level="low")
    if thinking_cfg is not None:
        config_params["thinking_config"] = thinking_cfg

    if func_spec is not None:
        config_params["response_mime_type"] = "application/json"
        config_params["response_json_schema"] = func_spec.json_schema

    generation_config = types.GenerateContentConfig(**config_params)

    t0 = time.time()
    logger.info(f"Querying Gemini with model: {filtered_kwargs.get('model')}")

    try:
        response = _client.models.generate_content(
            model=filtered_kwargs.get("model", "gemini-3-pro-preview"),
            contents=contents,
            config=generation_config,
        )
        req_time = time.time() - t0

        # Parse response
        if func_spec is None:
            output = response.text
            logger.info(f"Gemini response: {output}", extra={"verbose": True})
        else:
            text = response.text
            if not text:
                raise ValueError("No response text from Gemini for structured output")
            output = json.loads(text)
            if isinstance(output, list):
                if len(output) > 0:
                    output = output[0]
                else:
                    raise ValueError("Gemini returned empty array for structured output")
            logger.info(f"Gemini structured output response: {output}", extra={"verbose": True})

        in_tokens = 0
        out_tokens = 0
        if hasattr(response, 'usage_metadata'):
            in_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            out_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

        info = {
            "model": filtered_kwargs.get("model", "gemini-3-pro-preview"),
            "created": int(time.time()),
        }

        return output, req_time, in_tokens, out_tokens, info

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        raise e


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 20,
    retry_delay: float = 3,
    stage: str = "code",
) -> str:
    """Streaming text generation via Gemini API.

    Args:
        prompt: The text prompt to complete.
        cfg: Config instance (provides model name and initializes client).
        temperature: Sampling temperature (default 1.0).
        max_tokens: Max output tokens (default 16384).
        stop_tokens: Optional stop sequences.
        json_schema: Optional JSON schema for structured output.
        max_retries: Max retry attempts on failure.
        retry_delay: Seconds to wait between retries.
        stage: which cfg.agent.<stage> block supplies the model name.

    Returns:
        The generated text (with <think> blocks stripped).
    """
    _setup_gemini_client(cfg)
    # NOTE: _setup_gemini_client is @once, so the client's api_key/base_url are
    # always the code stage's. Only the model name is stage-aware here; a
    # feedback stage on a different Gemini endpoint would need that lifted.
    stage_cfg = getattr(cfg.agent, stage, None) or cfg.agent.code

    # Convert dict/list prompts to markdown string
    if prompt is not None and not isinstance(prompt, str):
        prompt = compile_prompt_to_md(prompt)

    logger.info(f"generate prompt: {prompt}", extra={"verbose": True})

    if max_tokens is None:
        max_tokens = getattr(stage_cfg, "max_tokens", None) or 16384

    config_params = {
        "temperature": temperature if temperature is not None else 1.0,
        "max_output_tokens": max_tokens,
        "stop_sequences": stop_tokens,
    }
    thinking_cfg = _build_thinking_config(stage_cfg.model, level="high")
    if thinking_cfg is not None:
        config_params["thinking_config"] = thinking_cfg

    if json_schema is not None:
        config_params["response_mime_type"] = "application/json"
        config_params["response_json_schema"] = json_schema
        logger.info("Enforcing JSON output with schema", extra={"verbose": True})

    generation_config = types.GenerateContentConfig(**config_params)
    model_name = stage_cfg.model

    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
                config=generation_config,
            )
            full_text = ""
            for chunk in response:
                if chunk.text:
                    full_text += chunk.text

            # Strip thinking tags
            if "</think>" in full_text:
                full_text = full_text[full_text.find("</think>") + 8:]

            logger.info(f"generate response: {full_text}", extra={"verbose": True})
            return full_text

        except Exception as e:
            logger.warning(f"generate failed, retrying {attempt + 1}/{max_retries}: {e}")
            if attempt >= max_retries - 1:
                logger.error("generate retry limit reached")
                raise
            time.sleep(retry_delay)
