"""Thin OpenRouter wrapper using the OpenAI SDK.

OpenRouter is OpenAI-API-compatible, so we point the OpenAI client at
`https://openrouter.ai/api/v1`. Tool calls round-trip in the standard
OpenAI `tools` / `tool_calls` shape.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"


@dataclass
class LLMConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = OPENROUTER_BASE_URL
    temperature: float = 0.2

    @classmethod
    def from_env(cls, model_override: str | None = None) -> "LLMConfig":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        model = model_override or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL
        return cls(api_key=key, model=model)


def build_client(config: LLMConfig) -> OpenAI:
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        default_headers={
            # OpenRouter recommends these for attribution; they're optional.
            "HTTP-Referer": "https://github.com/eag4-downloads-organizer",
            "X-Title": "Downloads Folder Organizer",
        },
    )


def mcp_tools_to_openai(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool descriptions into OpenAI function-calling schemas."""
    openai_tools: list[dict[str, Any]] = []
    for t in mcp_tools:
        schema = t.inputSchema or {"type": "object", "properties": {}}
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": schema,
                },
            }
        )
    return openai_tools
