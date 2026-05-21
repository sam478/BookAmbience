"""llama.cpp provider — uses an OpenAI-compatible chat/completions endpoint.

This module keeps backward compatibility with the old LM Studio naming
because the rest of the app may still import LMStudioProvider.
"""

from __future__ import annotations

import json
import re
from typing import Type, Any, Dict

import requests
from pydantic import BaseModel

from . import AIProvider


def _normalize_base_url(base_url: str) -> str:
    """
    Accept either:
      - http://localhost:8080/v1/chat/completions
      - http://localhost:8080/v1
      - http://localhost:8080

    and normalize to a chat/completions endpoint.
    """
    base_url = base_url.rstrip("/")

    if base_url.endswith("/chat/completions"):
        return base_url

    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"

    # If user passed bare host/port, assume llama.cpp OpenAI server style.
    return f"{base_url}/v1/chat/completions"


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Remove leading/trailing fenced block markers if present.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class LlamaCppProvider(AIProvider):
    """Local provider using llama.cpp's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:8080/v1/chat/completions",
        temperature: float = 0.7,
        top_p: float = 0.95,
        timeout: int = 360,
    ):
        self.model = model
        self.base_url = _normalize_base_url(base_url)
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def _extract_text_from_response(self, resp_json: Dict[str, Any]) -> str:
        """Extract assistant text from common OpenAI-compatible response shapes."""
        choices = resp_json.get("choices") or []
        if not isinstance(choices, list) or not choices:
            return resp_json.get("content", "") or ""

        first = choices[0]
        if not isinstance(first, dict):
            return ""

        # OpenAI chat format
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                return content

        # Legacy/completion-like format
        text = first.get("text", "")
        if isinstance(text, str):
            return text

        # Some servers may return direct content
        content = first.get("content", "")
        if isinstance(content, str):
            return content

        return ""

    def _validate_structured_output(self, data: Any, schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        Validate and normalize the model output against the expected schema.
        Supports both pydantic v2 and v1.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}: {data!r}")

        # Reject obvious schema echoes.
        if {"properties", "required", "title", "type"}.issubset(data.keys()) and not any(
            key in data for key in ("prompt", "negative_prompt", "suggested_bpm")
        ):
            raise ValueError(f"Model returned a schema instead of data: {data!r}")

        try:
            if hasattr(schema, "model_validate"):  # pydantic v2
                validated = schema.model_validate(data)
                return validated.model_dump()
            else:  # pydantic v1
                validated = schema.parse_obj(data)
                return validated.dict()
        except Exception as e:
            raise ValueError(f"Structured output validation failed: {e}; raw={data!r}") from e

    def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        schema_name = getattr(schema, "__name__", "Schema")
        schema_json = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema()

        system_prompt = f"""
You are a JSON-only response engine.

Return ONLY valid JSON that conforms to the target schema.
Do not return the schema itself.
Do not include markdown, code fences, explanations, or extra keys.
If you must omit something, use null.

Target schema name: {schema_name}
Target JSON schema:
{json.dumps(schema_json, indent=2)}
""".strip()

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        resp = requests.post(self.base_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()

        resp_json = resp.json()
        content = self._extract_text_from_response(resp_json)
        if not content or not content.strip():
            raise RuntimeError(f"llama.cpp response missing content: {resp_json}")

        content = _strip_code_fences(content)
        # First try direct JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Model did not return valid JSON: {e}; raw={content!r}") from e

        # Validate against the expected Pydantic schema
        return self._validate_structured_output(data, schema)


# Backward compatibility with older imports
LMStudioProvider = LlamaCppProvider
