# providers/gemini_provider.py
"""Gemini AI provider using google-genai SDK."""

import os
import json
from pydantic import BaseModel
from typing import Type, List
from google import genai
from providers import AIProvider


def get_gemini_models() -> List[str]:
    """Fetch available Gemini models from the API dynamically.
    Filters to text-generation-capable models only."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return []

    try:
        client = genai.Client(api_key=api_key)
        ids = []
        for m in client.models.list():
            methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", [])
            if methods and "generateContent" not in methods:
                continue
            name = getattr(m, "name", "") or ""
            mid = name.split("/")[-1] if name.startswith("models/") else name
            if not mid or not mid.startswith("gemini-"):
                continue
            # Skip non-chat variants
            if any(tag in mid for tag in ("embedding", "aqa", "tts", "image-generation", "native-audio")):
                continue
            ids.append(mid)
        return sorted(set(ids))
    except Exception as e:
        print(f"Failed to fetch Gemini models: {e}")
        return [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ]


class GeminiProvider(AIProvider):
    """Google Gemini provider with native structured output."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )
        return json.loads(response.text)
