# providers/groq_provider.py
"""Groq AI provider using OpenAI-compatible API with JSON mode."""

import os
import json
from pydantic import BaseModel
from typing import Type, List
from groq import Groq
from providers import AIProvider


def get_groq_models() -> List[str]:
    """Fetch available models from Groq API dynamically."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return []

    try:
        client = Groq(api_key=api_key)
        models = client.models.list()
        model_ids = sorted([m.id for m in models.data if m.id])
        return model_ids
    except Exception as e:
        print(f"Failed to fetch Groq models: {e}")
        return [
            "moonshotai/kimi-k2-instruct-0905",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]


class GroqProvider(AIProvider):
    """Groq provider using JSON mode + schema-in-prompt strategy."""

    def __init__(self, model: str = "moonshotai/kimi-k2-instruct-0905"):
        self.model = model
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        json_schema = json.dumps(schema.model_json_schema(), indent=2)

        system_message = (
            "You are a structured data assistant. You MUST respond with ONLY valid JSON "
            "matching the following JSON Schema. No markdown, no explanation, just JSON.\n\n"
            f"JSON Schema:\n{json_schema}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        validated = schema.model_validate(parsed)
        return validated.model_dump()
