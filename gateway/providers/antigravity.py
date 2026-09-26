"""Gemini (and the other Antigravity) models through the Antigravity CLI (`agy`), using your Google login."""

import json
import logging
from typing import AsyncIterator, List, Optional

from .. import attachments
from ..core import (Attachment, ChatRequest, Event, Finish, GatewayError, Start, TextDelta, ToolCall, ToolUse,
                    Usage, render_prompt)
from ..process import CliRun, Launch, run_quick
from .base import Provider, ProviderStatus, call_id, clean_json, json_note, offered_tools, tool_choice_note

log = logging.getLogger("gateway.antigravity")

# A custom agent without Antigravity's default prompt sections and built-in tools (agy 1.2.1+), so a
# request is prompt in, text out. Its body is the request's system prompt.
AGENT_FILE = """---
name: gateway
description: Plain model call from the local gateway, with no tools.
excludeDefaultComponents: true
---
"""

# That agent cannot use MCP tools, so the client's tools are described in its instructions and the model
# answers with this block when it wants to call them.
TOOL_OPEN, TOOL_CLOSE = "<tool_calls>", "</tool_calls>"
TOOL_INSTRUCTIONS = """

# Functions
You can call the functions below. They are not part of your environment: calling one means writing a request
that the user's application runs; its result comes back in the next message.

To call functions, reply with only this block and nothing else:
<tool_calls>
[{"name": "<function name>", "arguments": {<arguments as JSON>}}]
</tool_calls>
Only the functions listed below exist for this; never put any other tool in the block. You may list several
calls. If no function is needed, answer normally.

<functions>
%s
</functions>"""


def _usage(u: dict) -> Usage:
    return Usage(input_tokens=u.get("input_tokens") or 0, output_tokens=u.get("output_tokens") or 0,
                 cached_tokens=u.get("cache_read_tokens") or 0, reasoning_tokens=u.get("thinking_tokens") or 0)


def _error_from(text: str, model: str) -> GatewayError:
    low = text.lower()
    first_line = text.strip().splitlines()[0][:300] if text.strip() else "Antigravity failed to produce a response."
    if "not recognized as a known model" in low or "invalid model" in low:
        return GatewayError("not_found", f"Model `{model}` is not available in Antigravity.", retry=False)
    if "quota" in low or "resource_exhausted" in low or "rate limit" in low or " 429" in low:
        return GatewayError("rate_limit", first_line, retry=False)
    if "sign in" in low or "signed in" in low or "unauthenticated" in low or "login" in low:
        return GatewayError("authentication", "Antigravity on the gateway host is not signed in. Run `agy` and sign in.",
                            retry=False)
    if "permission" in low:
        return GatewayError("permission", first_line, retry=False)
    return GatewayError("api_error", first_line)


def parse_tool_calls(text: str, names: set) -> Optional[List[ToolCall]]:
    """The calls in a <tool_calls> reply, or None when the reply is a normal answer."""
    body = text.strip()
    if not (body.startswith(TOOL_OPEN) and body.endswith(TOOL_CLOSE)):
        return None
    try:
        raw = json.loads(body[len(TOOL_OPEN):-len(TOOL_CLOSE)])
    except ValueError:
        return None
    raw = raw if isinstance(raw, list) else [raw]
    calls = []
    for entry in raw:
        # The model sometimes adds one of Antigravity's own tools; only the client's tools are kept.
        if isinstance(entry, dict) and entry.get("name") in names and isinstance(entry.get("arguments", {}), dict):
            calls.append(ToolCall(call_id(), entry["name"], entry.get("arguments") or {}))
        else:
            log.info("dropped a call to %r, which the client did not offer", entry.get("name") if isinstance(entry, dict) else entry)
    return calls or None


class Antigravity(Provider):
    id = "antigravity"
    name = "Antigravity"
    vendor = "Google"
    apis = ["gemini"]
    default_model = "gemini-3.8-flash-low"
    honours = frozenset()
    tool_calling = "prompt"

    def check(self) -> ProviderStatus:
        status = ProviderStatus()
        if not self.found():
            status.problem = f"`{self.binary}` is not installed."
            return status
        status.installed = True
        status.version = run_quick([*self.command, "--version"]).stdout.strip()
        out = run_quick([*self.command, "models"], timeout=60)
        for line in out.stdout.splitlines():
            model_id, tab, label = line.partition("\t")
            if tab and model_id.strip():
                status.models.append({"id": model_id.strip(), "label": label.strip()})
        if status.models:
            status.ready = True
            status.account = "Google account"
            if not any(m["id"] == self.default_model for m in status.models):
                self.default_model = status.models[0]["id"]
        else:
            status.problem = ((out.stderr or out.stdout).strip().splitlines() or ["Could not list models."])[-1][:300] \
                + " Run `agy` and sign in."
        return status

    def launch(self, req: ChatRequest) -> Launch:
        tools = offered_tools(req)
        agent = AGENT_FILE + (req.system or "") + json_note(req)
        if tools:
            described = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]
            agent += TOOL_INSTRUCTIONS % json.dumps(described, indent=1) + tool_choice_note(req)
        cmd = [*self.command, "-p=", "--agent", "gateway", "--input-format", "stream-json",
               "--output-format", "stream-json", "--model", req.model, "--disable-slash-commands", "--sandbox"]
        return Launch(self.id, cmd, {}, {".agents/agents/gateway.md": agent}, warm=True)

    async def run(self, req: ChatRequest, request_id: str) -> AsyncIterator[Event]:
        tools = offered_tools(req)
        started = await self.pool.take(self.launch(req), request_id)
        images = []

        def show(n: int, a: Attachment) -> str:
            if a.kind == "image":
                # Antigravity attaches an image when the prompt mentions its path with @.
                path = started.write(f"attachments/{n}{attachments.extension(a.media_type)}", a.data)
                images.append(path)
                return f"[Image {n}" + (f": {a.filename}" if a.filename else "") + f"] @{path} "
            name = f' name="{a.filename}"' if a.filename else ""
            return f"<document{name}>\n{attachments.document_text(a)}\n</document>"

        prompt = render_prompt(req.turns, show)
        line = json.dumps({"event": "user", "message": {"role": "user", "content": prompt}})
        async with CliRun(started, stdin=(line + "\n").encode("utf-8"), timeout=self.settings.request_timeout,
                          request_id=request_id) as run:
            began = False
            held = ""       # trailing whitespace, held back so the answer does not end with agy's closing newline
            buffered = ""   # with tools: the start of the reply, until it is clear whether it is a tool call
            deciding = bool(tools)
            whole = bool(req.response_format)  # JSON output: the reply is cleaned up, so it is sent in one piece
            async for obj in run.json_lines():
                kind = obj.get("event")
                if kind == "init" and not began:
                    began = True
                    yield Start((obj.get("init") or {}).get("model") or req.model)
                elif kind == "step_update":
                    step = obj.get("step_update") or {}
                    if step.get("step_type") != "agent_response" or not step.get("text_delta"):
                        continue
                    if whole:
                        buffered += step["text_delta"]
                        continue
                    if deciding:
                        buffered += step["text_delta"]
                        stripped = buffered.lstrip()
                        if len(stripped) < len(TOOL_OPEN) and TOOL_OPEN.startswith(stripped):
                            continue  # could still become a tool call
                        if stripped.startswith(TOOL_OPEN):
                            continue  # a tool call: keep collecting until the reply ends
                        deciding = False
                        delta = buffered
                    else:
                        delta = step["text_delta"]
                    text = held + delta
                    body = text.rstrip()
                    held = text[len(body):]
                    if body:
                        yield TextDelta(body)
                elif kind == "result":
                    result = obj.get("result") or {}
                    if result.get("status") != "SUCCESS":
                        raise _error_from(str(result.get("error") or ""), req.model)
                    run.completed = True
                    usage = _usage(result.get("usage") or {})
                    if deciding or whole:
                        calls = parse_tool_calls(buffered, {t.name for t in tools}) if tools else None
                        if calls:
                            for call in calls:
                                yield ToolUse(call)
                            yield Finish("tool_use", usage)
                            return
                        if buffered.strip():
                            yield TextDelta(clean_json(buffered) if whole else buffered.strip())
                    yield Finish("end_turn", usage)
                    return
            await run.finish()
            raise _error_from(run.stderr, req.model)
