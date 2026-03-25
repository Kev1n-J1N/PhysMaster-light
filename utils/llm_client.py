import json
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml
from openai import OpenAI


def _load_llm_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    default_path = Path(__file__).resolve().parents[1] / "config.yaml"
    path = Path(config_path) if config_path else default_path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    llm_config = config.get("llm", {})
    if not isinstance(llm_config, dict):
        raise ValueError("`llm` section in config.yaml must be a mapping.")

    missing = [key for key in ("base_url", "api_key", "model") if not llm_config.get(key)]
    if missing:
        raise ValueError(f"Missing llm config fields: {', '.join(missing)}")

    return llm_config


class LLMClient:
    def __init__(self, config_path: str | Path | None = None):
        llm_config = _load_llm_config(config_path)
        self.model = str(llm_config["model"])
        self.client = OpenAI(
            api_key=str(llm_config["api_key"]),
            base_url=str(llm_config["base_url"]),
        )

    def call_without_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: Optional[str] = None,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=model_name or self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (completion.choices[0].message.content or "").strip()

    def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list | None = None,
        tool_functions: Optional[Dict[str, Callable]] = None,
        model_name: Optional[str] = None,
        max_tool_calls: int = 20,
    ) -> str:
        tools = tools or []
        tool_functions = tool_functions or {}
        messages: list[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for _ in range(max_tool_calls):
            completion = self.client.chat.completions.create(
                model=model_name or self.model,
                messages=messages,
                tools=tools if tools else None,
            )

            msg = completion.choices[0].message
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in (msg.tool_calls or [])
                    ] or None,
                }
            )

            tool_call_list = msg.tool_calls or []
            if not tool_call_list:
                if completion.choices[0].finish_reason == "stop":
                    return (msg.content or "").strip()
                continue

            for tool_call in tool_call_list:
                raw_args = tool_call.function.arguments or "{}"
                try:
                    call_args = json.loads(raw_args)
                except Exception:
                    call_args = {"_raw": raw_args}

                try:
                    fn = tool_functions.get(tool_call.function.name)
                    if fn is None:
                        result = f"[tool:{tool_call.function.name}] not implemented"
                    else:
                        result = fn(**call_args) if isinstance(call_args, dict) else fn(call_args)
                except Exception:
                    result = traceback.format_exc()

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                return str(message["content"]).strip()
        return ""


def call_model_without_tools(
    system_prompt: str,
    user_prompt: str,
    model_name: Optional[str] = None,
    config_path: str | Path | None = None,
) -> str:
    client = LLMClient(config_path=config_path)
    return client.call_without_tools(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
    )


def call_model(
    system_prompt: str,
    user_prompt: str,
    tools: list | None = None,
    tool_functions: Optional[Dict[str, Callable]] = None,
    model_name: Optional[str] = None,
    max_tool_calls: int = 20,
    config_path: str | Path | None = None,
) -> str:
    client = LLMClient(config_path=config_path)
    return client.call_with_tools(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools,
        tool_functions=tool_functions,
        model_name=model_name,
        max_tool_calls=max_tool_calls,
    )
