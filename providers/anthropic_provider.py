# providers/anthropic_provider.py
"""Anthropic (Claude) provider using structured-output via tool calling."""

import os
from pydantic import BaseModel
from typing import Type, List
from anthropic import Anthropic
from providers import AIProvider


DEFAULT_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


def _api_key() -> str:
    return os.environ.get("CLAUDE_API") or os.environ.get("ANTHROPIC_API_KEY") or ""


def get_anthropic_models() -> List[str]:
    """Anthropic doesn't expose a list-models API for all accounts; return a
    curated list of current Claude model IDs."""
    return list(DEFAULT_MODELS)


class AnthropicProvider(AIProvider):
    """Claude provider — uses the messages API with a tool-use schema to
    force structured JSON output that matches the given Pydantic model."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = Anthropic(api_key=_api_key())

    def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        json_schema = schema.model_json_schema()

        tool = {
            "name": "emit_structured_output",
            "description": "Emit the structured output matching the required schema.",
            "input_schema": json_schema,
        }

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit_structured_output"},
            messages=[{"role": "user", "content": prompt}],
        )

        payload = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                payload = block.input
                break

        if payload is None:
            raise RuntimeError(f"Claude did not return a tool_use block. Raw: {response}")

        validated = schema.model_validate(payload)
        return validated.model_dump()
