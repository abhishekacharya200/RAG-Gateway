"""Claude models through Claude Code (`claude -p`), using your Claude Pro/Max login."""

import base64
import json
import logging
import time
from typing import AsyncIterator, List

from .. import attachments
from ..core import (Attachment, ChatRequest, Event, Finish, GatewayError, Native, Start, TextDelta, ToolCall,
                    ToolUse, Usage, render_prompt)
from ..process import CliRun, Launch, run_quick
from .base import (Provider, ProviderStatus, bridge_command, mcp_tools_env, offered_tools, tool_choice_note)

EFFORTS = ("low", "medium", "high", "xhigh", "max")  # `claude --effort`
STRUCTURED_TOOL = "StructuredOutput"  # the tool Claude Code uses to deliver --json-schema output

log = logging.getLogger("gateway.claude")

TOOL_PREFIX = "mcp__gateway__"  # Claude Code's name for the tools the gateway's MCP server offers

# Flags checked against `claude --help` (Claude Code 2.1.283).
COMMON_FLAGS = [
    "-p",
    # The message (with images and documents) arrives as JSON on stdin, so the process can start before it does.
    "--input-format", "stream-json",
    # The real Anthropic API stream events, one JSON object per line.
    "--output-format", "stream-json", "--verbose", "--include-partial-messages",
    # No built-in tools: prompt in, text (or a call to one of the client's tools) out.
    "--tools", "",
    "--disable-slash-commands",
    # Don't save gateway calls as resumable sessions.
    "--no-session-persistence",
]
# Without tools: skip CLAUDE.md, hooks, skills, plugins and MCP servers. The login still works.
PLAIN_FLAGS = ["--safe-mode", "--strict-mcp-config"]
# With tools --safe-mode cannot be used (it turns off MCP servers). --restricted and --setting-sources ""
# ignore user, project and local settings instead, and --strict-mcp-config loads only the gateway's server.
TOOL_FLAGS = ["--restricted", "--setting-sources", "", "--strict-mcp-config", "--mcp-config", "{dir}/mcp.json"]
# Not used: --bare. It ignores the subscription login and accepts only ANTHROPIC_API_KEY.

MODELS = [
    {"id": "claude-haiku-4-5-20251001", "label": "Haiku 4.5"},
    {"id": "claude-sonnet-5", "label": "Sonnet 5"},
    {"id": "claude-opus-5-5", "label": "Opus 5.5"},
    {"id": "claude-fable-5-1", "label": "Fable 5.1"},
]


def _usage(u: dict) -> Usage:
    cache_write = u.get("cache_creation_input_tokens") or 0
    cached = u.get("cache_read_input_tokens") or 0
    return Usage(input_tokens=(u.get("input_tokens") or 0) + cache_write + cached,
                 output_tokens=u.get("output_tokens") or 0, cached_tokens=cached, cache_write_tokens=cache_write,
                 reasoning_tokens=(u.get("output_tokens_details") or {}).get("thinking_tokens") or 0)


def _block(n: int, a: Attachment) -> List[dict]:
    """Claude reads images and PDFs itself; other documents go in as text documents."""
    label = {"type": "text", "text": f"Attachment {n}" + (f" ({a.filename}):" if a.filename else ":")}
    data = base64.b64encode(a.data).decode()
    if a.kind == "image":
        return [label, {"type": "image", "source": {"type": "base64", "media_type": a.media_type, "data": data}}]
    if a.media_type == "application/pdf":
        block = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}}
    else:
        block = {"type": "document",
                 "source": {"type": "text", "media_type": "text/plain", "data": attachments.document_text(a)}}
    if a.filename:
        block["title"] = a.filename
    return [label, block]


class ClaudeCode(Provider):
    id = "claude"
    name = "Claude Code"
    vendor = "Anthropic"
    apis = ["anthropic-messages"]
    default_model = "claude-haiku-4-5-20251001"
    honours = frozenset({"max_tokens", "thinking", "reasoning_effort"})
    tool_calling = "native"

    def check(self) -> ProviderStatus:
        status = ProviderStatus(models=MODELS)
        if not self.found():
            status.problem = f"`{self.binary}` is not installed."
            return status
        status.installed = True
        status.version = run_quick([*self.command, "--version"]).stdout.strip().replace(" (Claude Code)", "")
        out = run_quick([*self.command, "auth", "status", "--json"])
        try:
            auth = json.loads(out.stdout)
        except ValueError:
            auth = {}
        method = str(auth.get("authMethod") or "")
        if not auth.get("loggedIn"):
            status.problem = "Not logged in. Run `claude`, then /login."
        elif auth.get("apiProvider", "firstParty") != "firstParty" or "api" in method.lower() \
                or "console" in method.lower():
            status.problem = f"Logged in with '{method}', which bills the API. Log in with your Pro/Max plan."
        else:
            status.ready = True
            status.account = f"{method}, {auth.get('subscriptionType') or 'subscription'} plan"
        return status

    def launch(self, req: ChatRequest) -> Launch:
        tools = offered_tools(req)
        cmd = [*self.command, *COMMON_FLAGS, "--model", req.model]
        files = {}
        if tools:
            files["mcp.json"] = json.dumps({"mcpServers": {"gateway": {
                "command": bridge_command()[0], "args": bridge_command()[1:], "env": mcp_tools_env(tools)}}})
            cmd += TOOL_FLAGS + ["--allowedTools", ",".join(TOOL_PREFIX + t.name for t in tools)]
        else:
            cmd += PLAIN_FLAGS
        system = req.system + (tool_choice_note(req) if tools else "")
        if system:
            # A file, not an argument: big RAG system prompts would break the OS argument-length limit.
            files["system-prompt.txt"] = system
            cmd += ["--system-prompt-file", "{dir}/system-prompt.txt"]
        else:
            cmd += ["--system-prompt", ""]
        fmt = req.response_format
        if fmt:
            # Enforced by Claude Code: the model must deliver JSON matching the schema.
            schema = fmt.get("schema") if fmt["type"] == "json_schema" else {"type": "object"}
            cmd += ["--json-schema", json.dumps(schema)]
        if req.reasoning_effort in EFFORTS:
            cmd += ["--effort", req.reasoning_effort]
        extra = {}
        if req.max_tokens:
            extra["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(req.max_tokens)
        if req.thinking_budget is not None:
            extra["MAX_THINKING_TOKENS"] = str(req.thinking_budget)
        return Launch(self.id, cmd, extra, files, warm=True)

    def message(self, req: ChatRequest) -> bytes:
        """The one stdin message: the conversation as text, then each attachment as a native block."""
        found: List[Attachment] = []

        def mark(n: int, a: Attachment) -> str:
            found.append(a)
            return f"[Attachment {n}" + (f": {a.filename}]" if a.filename else "]")

        content = [{"type": "text", "text": render_prompt(req.turns, mark)}]
        for n, a in enumerate(found, 1):
            content += _block(n, a)
        line = {"type": "user", "message": {"role": "user", "content": content}}
        return (json.dumps(line) + "\n").encode("utf-8")

    async def run(self, req: ChatRequest, request_id: str) -> AsyncIterator[Event]:
        stdin = self.message(req)
        started = await self.pool.take(self.launch(req), request_id)
        structured = bool(req.response_format)
        async with CliRun(started, stdin=stdin, timeout=self.settings.request_timeout, request_id=request_id) as run:
            done = False  # the first model response has finished
            stop_reason = "end_turn"
            usage = {}
            tool = None  # [id, name, json so far] of the tool call being streamed
            result = None
            output = None  # structured output (JSON text) delivered through Claude Code's own tool
            held = []      # with structured output: text written before it, shown only if no output comes
            async for obj in run.json_lines():
                kind = obj.get("type")
                if kind == "system" and obj.get("subtype") == "init":
                    source = obj.get("apiKeySource")
                    if source not in (None, "none"):
                        log.error("[%s] claude would authenticate with %s; stopped it", request_id, source)
                        raise GatewayError("api_error", "The gateway refused this request: Claude Code would "
                                           "bill an API key instead of your subscription.", retry=False)
                elif kind == "stream_event" and obj.get("parent_tool_use_id") is None:
                    if done:
                        break  # a second model call has started
                    event = obj.get("event") or {}
                    if not structured:
                        yield Native(event)
                    etype = event.get("type")
                    if etype == "message_start":
                        message = event.get("message") or {}
                        usage = dict(message.get("usage") or {})
                        yield Start(message.get("model") or req.model)
                    elif etype == "content_block_start":
                        block = event.get("content_block") or {}
                        if block.get("type") == "tool_use":
                            name = block.get("name", "")
                            tool = [block.get("id", ""), name[len(TOOL_PREFIX):] if name.startswith(TOOL_PREFIX)
                                    else name, ""]
                    elif etype == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            if structured:
                                held.append(delta["text"])
                            else:
                                yield TextDelta(delta["text"])
                        elif delta.get("type") == "input_json_delta" and tool is not None:
                            tool[2] += delta.get("partial_json", "")
                    elif etype == "content_block_stop" and tool is not None:
                        arguments = json.loads(tool[2]) if tool[2].strip() else {}
                        if structured and tool[1] == STRUCTURED_TOOL:
                            output = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                        else:
                            yield ToolUse(ToolCall(tool[0], tool[1], arguments))
                        tool = None
                    elif etype == "message_delta":
                        stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
                        usage.update(event.get("usage") or {})
                    elif etype == "message_stop":
                        done = True
                        if output is not None:
                            # Like the API's structured output: the JSON is the answer.
                            yield TextDelta(output)
                            yield Finish("end_turn", _usage(usage))
                            break
                        if held:
                            yield TextDelta("".join(held))
                        yield Finish(stop_reason, _usage(usage))
                        if stop_reason == "tool_use":
                            break  # the client runs the tool, not Claude Code
                elif kind == "user" and done:
                    break  # after max_tokens Claude Code continues by itself; the API would stop here
                elif kind == "rate_limit_event":
                    self.rate_limit = {**(obj.get("rate_limit_info") or {}), "seen_at": time.time()}
                elif kind == "result":
                    result = obj
                    run.completed = True
                    break
            if not done:
                await run.finish()
                raise self._failure(result or {}, run.stderr, req.model)

    def _failure(self, result: dict, stderr: str, model: str) -> GatewayError:
        text = str(result.get("result") or "").strip()
        status = result.get("api_error_status")
        blob = f"{text}\n{stderr}".lower()
        if text:
            log.warning("claude reported: %s", text[:1000])
        if status == 401 or "not logged in" in blob or "/login" in blob:
            return GatewayError("authentication", "Claude Code on the gateway host is not logged in, or its login "
                                "expired. Run `claude`, then /login.", retry=False)
        if status == 404 or "unrecognized_model" in blob:
            return GatewayError("not_found", f"model: {model}")
        limit = self.rate_limit or {}
        if status == 429 or limit.get("status") == "rejected" or "usage limit" in blob or "hit your limit" in blob:
            message = "Your Claude subscription usage limit has been reached."
            if isinstance(limit.get("resetsAt"), (int, float)):
                message += " It resets at " + time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(limit["resetsAt"])) + "."
            return GatewayError("rate_limit", message, retry=False)
        if status == 529 or "overloaded" in blob:
            return GatewayError("overloaded", "Overloaded")
        if status == 413:
            return GatewayError("too_large", text or "Request exceeds the maximum allowed number of bytes.")
        if status == 400 or "prompt is too long" in blob:
            return GatewayError("invalid_request", text or "Invalid request.")
        if status == 403:
            return GatewayError("permission", text or "Permission denied.")
        if not result:
            return GatewayError("api_error", "Claude Code exited without a response.")
        return GatewayError("api_error", "Claude Code failed to produce a response.")
