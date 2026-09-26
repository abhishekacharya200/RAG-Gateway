"""Shared building blocks: the provider-neutral request, the events providers emit, the result, and errors.

API formats (gateway/dialects) turn a client's request into a ChatRequest and turn events back into
that format's response. Providers (gateway/providers) turn a ChatRequest into events by running a CLI.
Neither side knows about the other.
"""

import hashlib
import json
import secrets
import string
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Union

_BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
ALNUM = string.ascii_letters + string.digits


def random_id(length: int, alphabet: str = _BASE58) -> str:
    return "".join(secrets.choice(alphabet) for _ in range(length))


def request_id() -> str:
    return "req_01" + random_id(22)


# ---------- request ----------

@dataclass
class Attachment:
    """An image or document sent with a message, inline or as an uploaded file."""

    kind: str          # "image" or "document"
    media_type: str    # e.g. image/png, application/pdf, text/plain
    data: bytes = field(repr=False)
    filename: Optional[str] = None

    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    call_id: str
    content: str
    name: Optional[str] = None
    is_error: bool = False


Part = Union[str, Attachment, ToolCall, ToolResult]


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    parts: List[Part] = field(default_factory=list)  # in the order the client sent them

    @property
    def text(self) -> str:
        return "\n\n".join(p for p in self.parts if isinstance(p, str))

    @property
    def attachments(self) -> List[Attachment]:
        return [p for p in self.parts if isinstance(p, Attachment)]


@dataclass
class ToolDef:
    """A function the client offers the model. parameters is a JSON Schema."""

    name: str
    description: str
    parameters: dict


@dataclass
class ChatRequest:
    """One request in provider-neutral form."""

    api: str                     # API format the client used, e.g. "openai-chat"
    provider: str                # provider that answers, e.g. "codex"
    model: str                   # model name for that provider (any "provider/" prefix removed)
    system: str
    turns: List[Turn]
    max_tokens: Optional[int]
    stream: bool
    body: dict                   # the request body exactly as the client sent it (files as sent too)
    question: str                # last text block of the last message, for the history view
    context: Optional[str] = None  # earlier text blocks of the last message (usually RAG passages)
    thinking_budget: Optional[int] = 0  # Claude: None = its default (adaptive), 0 = off, N = budget
    reasoning_effort: Optional[str] = None  # Codex and Antigravity: low, medium, high...
    tools: List[ToolDef] = field(default_factory=list)
    tool_choice: Optional[dict] = None  # {"type": "auto" | "any" | "none" | "tool", "name": ...}
    # Structured output: {"type": "json_schema", "schema": {...}, "name": ..., "strict": bool} or {"type": "json_object"}.
    # strict means the API would enforce the schema (Anthropic always does; OpenAI with strict: true).
    response_format: Optional[dict] = None
    ignored: List[str] = field(default_factory=list)  # parameters the CLIs cannot honour
    sampling: dict = field(default_factory=dict)  # ignored parameters that still belong in the cache key
    options: dict = field(default_factory=dict)   # format-specific switches, e.g. OpenAI's include_usage

    @property
    def attachments(self) -> List[Attachment]:
        return [a for turn in self.turns for a in turn.attachments]

    def cache_material(self) -> dict:
        """Everything that can change the answer. The API format is left out, so the same question asked
        through two formats of the same provider shares one cache entry."""
        def part(p):
            if isinstance(p, Attachment):
                return {"attachment": p.media_type, "sha256": p.digest()}
            return p if isinstance(p, str) else asdict(p)
        return {"provider": self.provider, "model": self.model, "system": self.system,
                "turns": [{"role": t.role, "parts": [part(p) for p in t.parts]} for t in self.turns],
                "max_tokens": self.max_tokens, "thinking_budget": self.thinking_budget,
                "reasoning_effort": self.reasoning_effort, "tools": [asdict(t) for t in self.tools],
                "tool_choice": self.tool_choice, "response_format": self.response_format, "sampling": self.sampling}


def render_prompt(turns: List[Turn], attachment: Callable[[int, Attachment], str]) -> str:
    """The conversation as one prompt. A single text message is sent as-is; earlier turns go into a
    history block before the last message. attachment(n, a) says how attachment number n appears."""
    counter = iter(range(1, 10_000))

    def render(turn: Turn) -> str:
        out = []
        for p in turn.parts:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, Attachment):
                out.append(attachment(next(counter), p))
            elif isinstance(p, ToolCall):
                out.append(f'<tool_call id="{p.id}" name="{p.name}">{json.dumps(p.arguments, ensure_ascii=False)}'
                           f'</tool_call>')
            elif isinstance(p, ToolResult):
                error = ' is_error="true"' if p.is_error else ""
                out.append(f'<tool_result call_id="{p.call_id}" name="{p.name or ""}"{error}>\n{p.content}\n'
                           f'</tool_result>')
        return "\n\n".join(x for x in out if x)

    rendered = [(t.role, render(t)) for t in turns]
    if len(rendered) == 1:
        return rendered[0][1]
    history = "\n\n".join(f"<{role}>\n{text}\n</{role}>" for role, text in rendered[:-1])
    return f"<conversation_history>\n{history}\n</conversation_history>\n\n{rendered[-1][1]}"


# ---------- events that providers emit ----------

@dataclass
class Usage:
    input_tokens: int = 0        # all input, including the cached part
    output_tokens: int = 0       # all output, including reasoning
    cached_tokens: int = 0       # part of the input read from a cache
    cache_write_tokens: int = 0  # part of the input written to a cache
    reasoning_tokens: int = 0    # part of the output spent on thinking


@dataclass
class Start:
    model: str


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolUse:
    """The model calls one of the client's tools. The client runs it and sends the result back."""

    call: ToolCall


@dataclass
class Finish:
    stop_reason: str  # "end_turn", "max_tokens", "stop_sequence", "tool_use" or "refusal"
    usage: Usage


@dataclass
class Native:
    """A raw Anthropic API event from Claude Code, passed through to Anthropic-format clients."""

    event: dict


Event = Union[Start, TextDelta, ToolUse, Finish, Native]


@dataclass
class Result:
    """The finished answer, kept in the cache and used to render non-streaming responses."""

    model: str
    text: str
    stop_reason: str
    usage: Usage
    tool_calls: List[ToolCall] = field(default_factory=list)
    native: Optional[dict] = None  # the exact Anthropic message, when Claude Code answered

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Result":
        return cls(**{**data, "usage": Usage(**data["usage"]),
                      "tool_calls": [ToolCall(**c) for c in data.get("tool_calls") or []]})


class Collector:
    """Builds the Result while events stream past."""

    def __init__(self):
        self.model = None
        self.parts = []
        self.tool_calls = []
        self.finish = None
        self.natives = []

    def add(self, event: Event) -> None:
        if isinstance(event, Start):
            self.model = event.model
        elif isinstance(event, TextDelta):
            self.parts.append(event.text)
        elif isinstance(event, ToolUse):
            self.tool_calls.append(event.call)
        elif isinstance(event, Finish):
            self.finish = event
        elif isinstance(event, Native):
            self.natives.append(event.event)

    def result(self, fallback_model: str) -> "Result":
        finish = self.finish or Finish("end_turn", Usage())
        return Result(model=self.model or fallback_model, text="".join(self.parts),
                      stop_reason=finish.stop_reason, usage=finish.usage, tool_calls=list(self.tool_calls))


# ---------- errors ----------

class GatewayError(Exception):
    """A failure, described once and rendered by each API format in its own error shape.

    kind is one of: invalid_request, authentication, permission, not_found, too_large,
    rate_limit, overloaded, timeout, api_error.
    """

    def __init__(self, kind: str, message: str, retry: bool = True, retry_after: Optional[int] = None,
                 code: Optional[str] = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retry = retry
        self.retry_after = retry_after
        self.code = code  # "missing_key" or "bad_key" for gateway-key errors, which each format words its own way

    def headers(self) -> dict:
        headers = {}
        if not self.retry:
            headers["x-should-retry"] = "false"
        if self.retry_after:
            headers["retry-after"] = str(self.retry_after)
        return headers
