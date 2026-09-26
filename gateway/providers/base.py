"""What every provider (a CLI that answers with a subscription login) looks like to the gateway."""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Sequence, Union

from ..config import Settings
from ..core import ChatRequest, Event, ToolDef, random_id, ALNUM
from ..process import WarmPool, resolve

# The MCP server that offers the client's tools to Claude Code and Codex.
BRIDGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_bridge.py")


@dataclass
class ProviderStatus:
    installed: bool = False
    ready: bool = False
    version: Optional[str] = None
    account: Optional[str] = None   # how it is logged in, e.g. "claude.ai, pro plan"
    problem: Optional[str] = None   # why it is not ready, in words a person can act on
    models: List[dict] = field(default_factory=list)  # [{"id": ..., "label": ...}]


class Provider:
    id = ""            # used in URLs, history and "provider/model" names
    name = ""          # the CLI, e.g. "Claude Code"
    vendor = ""        # whose models, e.g. "Anthropic"
    apis: List[str] = []   # the API formats this provider speaks natively
    default_model = ""
    # What the CLI can actually do with a request. Anything else is ignored and logged.
    honours = frozenset()
    tool_calling = ""  # how tools work: "native" (through MCP) or "prompt" (a reply format in the instructions)

    def __init__(self, settings: Settings, binary: Union[str, Sequence[str]]):
        self.settings = settings
        # The CLI, optionally with a prefix (tests run the fake CLIs as: python fake_claude.py).
        self.command = [binary] if isinstance(binary, str) else list(binary)
        self.status = ProviderStatus()
        self.rate_limit: Optional[dict] = None  # plan usage, for providers whose CLI reports it
        self.pool = WarmPool(enabled=False)     # replaced by the gateway's shared pool

    @property
    def binary(self) -> str:
        return self.command[0]

    def found(self) -> bool:
        """Resolves the CLI to its full path (shutil.which also finds claude.cmd / codex.cmd on Windows)."""
        path = resolve(self.command[0])
        if path:
            self.command[0] = path
        return bool(path)

    def check(self) -> ProviderStatus:
        """Runs quick CLI commands (version, login, models). Called at startup."""
        raise NotImplementedError

    def run(self, req: ChatRequest, request_id: str) -> AsyncIterator[Event]:
        """Answers one request. Yields Start, TextDelta and ToolUse events, then Finish, or raises GatewayError."""
        raise NotImplementedError

    def describe(self) -> dict:
        s = self.status
        return {"id": self.id, "name": self.name, "vendor": self.vendor, "apis": self.apis,
                "default_model": self.default_model, "installed": s.installed, "ready": s.ready,
                "version": s.version, "account": s.account, "problem": s.problem, "models": s.models,
                "rate_limit": self.rate_limit, "tool_calling": self.tool_calling}


def offered_tools(req: ChatRequest) -> List[ToolDef]:
    """The tools the model may call in this request (none when tool_choice is "none")."""
    return [] if (req.tool_choice or {}).get("type") == "none" else list(req.tools)


def mcp_tools_env(tools: List[ToolDef]) -> dict:
    """Environment for the MCP bridge: the tools as MCP tool definitions."""
    return {"GATEWAY_TOOLS": json.dumps([{"name": t.name, "description": t.description, "inputSchema": t.parameters}
                                         for t in tools])}


def bridge_command() -> List[str]:
    return [sys.executable, BRIDGE]


def tool_choice_note(req: ChatRequest) -> str:
    """Extra instruction for tool_choice "any" or a named tool (CLIs have no switch for it)."""
    choice = req.tool_choice or {}
    if choice.get("type") == "any":
        return "\n\nIn this reply you must call at least one of the available tools."
    if choice.get("type") == "tool" and choice.get("name"):
        return f"\n\nIn this reply you must call the tool `{choice['name']}`."
    return ""


def json_note(req: ChatRequest) -> str:
    """Instruction for JSON output, where the CLI cannot enforce it (the API's non-strict mode works the same way)."""
    fmt = req.response_format
    if not fmt:
        return ""
    if fmt["type"] == "json_object":
        return "\n\nReply with a single JSON object and nothing else: no other text, no code fences."
    return ("\n\nReply with a single JSON value that matches this JSON Schema and nothing else: no other text, "
            "no code fences.\n" + json.dumps(fmt["schema"], indent=1))


def clean_json(text: str) -> str:
    """The reply without the code fences models sometimes put around JSON."""
    body = text.strip()
    if body.startswith("```") and body.endswith("```"):
        body = body[3:-3]
        body = body[body.index("\n") + 1:] if "\n" in body else body  # drops "json" after the opening fence
    return body.strip()


def call_id() -> str:
    return "call_" + random_id(24, ALNUM)
