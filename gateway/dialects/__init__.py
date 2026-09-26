"""API formats: each one reads requests and writes responses exactly like a real provider API."""

from .anthropic import AnthropicMessages
from .base import Dialect
from .gemini import Gemini
from .openai_chat import OpenAIChat
from .openai_responses import OpenAIResponses

ANTHROPIC = AnthropicMessages()
OPENAI_CHAT = OpenAIChat()
OPENAI_RESPONSES = OpenAIResponses()
GEMINI = Gemini()
DIALECTS = {d.id: d for d in (ANTHROPIC, OPENAI_CHAT, OPENAI_RESPONSES, GEMINI)}

__all__ = ["ANTHROPIC", "DIALECTS", "Dialect", "GEMINI", "OPENAI_CHAT", "OPENAI_RESPONSES"]
