# providers/__init__.py
"""
AI provider abstraction layer.
Supports Gemini, Groq, Anthropic, and local (LM Studio / llama.cpp) models.
"""

import json
import pathlib
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type, Dict

SETTINGS_PATH = pathlib.Path(__file__).parent.parent / "settings.json"

DEFAULT_SETTINGS = {
    "provider": "gemini",
    "gemini_model": "gemini-2.5-flash",
    "groq_model": "moonshotai/kimi-k2-instruct-0905",
    "anthropic_model": "claude-sonnet-4-6",
    "lmstudio_model": "local-model",
    "lmstudio_url": "http://localhost:1234/v1/chat/completions",
}


def load_settings() -> dict:
    """Load settings from disk, returning defaults if missing."""
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            return {**DEFAULT_SETTINGS, **saved}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict):
    """Persist settings to disk."""
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


class AIProvider(ABC):
    """Base class for AI providers."""

    @abstractmethod
    def generate_structured(self, prompt: str, schema: Type[BaseModel]) -> dict:
        """Send a prompt and return a dict matching the given Pydantic schema."""
        ...


def get_available_providers() -> Dict[str, bool]:
    """Return a mapping of provider name -> importable (True/False)."""
    available = {}
    try:
        from .gemini_provider import GeminiProvider
        available["gemini"] = True
    except Exception:
        available["gemini"] = False

    try:
        from .groq_provider import GroqProvider
        available["groq"] = True
    except Exception:
        available["groq"] = False

    try:
        from .anthropic_provider import AnthropicProvider
        available["anthropic"] = True
    except Exception:
        available["anthropic"] = False

    try:
        from .lmstudio_provider import LMStudioProvider
        available["lmstudio"] = True
    except Exception:
        available["lmstudio"] = False

    return available


def _create_provider(provider_name: str, model: str) -> AIProvider:
    """Internal factory to create a provider instance."""
    if provider_name == "groq":
        from .groq_provider import GroqProvider
        return GroqProvider(model=model)

    elif provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model)

    elif provider_name == "lmstudio":
        from .lmstudio_provider import LMStudioProvider
        settings = load_settings()
        return LMStudioProvider(
            model=model,
            base_url=settings.get("lmstudio_url", DEFAULT_SETTINGS["lmstudio_url"]),
        )

    else:
        from .gemini_provider import GeminiProvider
        return GeminiProvider(model=model)


def get_provider(provider_name: str = None, model: str = None) -> AIProvider:
    """Factory: return an AI provider instance.

    If provider_name/model are not given, uses current settings.
    """
    settings = load_settings()
    name = provider_name or settings.get("provider", "gemini")

    if not model:
        if name == "groq":
            model = settings.get("groq_model", DEFAULT_SETTINGS["groq_model"])
        elif name == "anthropic":
            model = settings.get("anthropic_model", DEFAULT_SETTINGS["anthropic_model"])
        elif name == "lmstudio":
            model = settings.get("lmstudio_model", DEFAULT_SETTINGS["lmstudio_model"])
        else:
            model = settings.get("gemini_model", DEFAULT_SETTINGS["gemini_model"])

    return _create_provider(name, model)
