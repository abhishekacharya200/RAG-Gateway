"""Providers: the CLIs that answer requests with your subscription logins."""

from typing import Dict, Tuple

from ..config import Settings
from .antigravity import Antigravity
from .base import Provider, ProviderStatus
from .claude_code import ClaudeCode
from .codex import Codex

# Names accepted in "provider/model", e.g. "codex/gpt-6-luna" or "agy/gemini-3.8-flash-low".
ALIASES = {"claude": "claude", "claude-code": "claude", "anthropic": "claude",
           "codex": "codex", "openai": "codex",
           "antigravity": "antigravity", "agy": "antigravity", "google": "antigravity", "gemini": "antigravity"}


def build_providers(settings: Settings) -> Dict[str, Provider]:
    providers = (ClaudeCode(settings, settings.claude_bin), Codex(settings, settings.codex_bin),
                 Antigravity(settings, settings.agy_bin))
    return {p.id: p for p in providers}


def route(model: str, default_provider: str) -> Tuple[str, str]:
    """Which provider answers. Each API format has its own provider; a "provider/" prefix picks another one."""
    prefix, slash, rest = model.partition("/")
    if slash and rest and prefix.lower() in ALIASES:
        return ALIASES[prefix.lower()], rest
    return default_provider, model


__all__ = ["ALIASES", "Provider", "ProviderStatus", "build_providers", "route"]
