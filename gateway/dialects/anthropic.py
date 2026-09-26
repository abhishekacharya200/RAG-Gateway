"""Anthropic Messages API: POST /v1/messages.

When Claude Code answers, the client gets the real Anthropic events and message that Claude Code
received, filtered to the fields of the official SDK. For other providers the same shapes are built.
"""

import copy
import json
from typing import AsyncIterator, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .. import attachments
from ..core import (ChatRequest, Event, Finish, GatewayError, Native, Part, Result, Start, TextDelta, ToolCall,
                    ToolDef, ToolResult, ToolUse, Turn, random_id)
from .base import (Dialect, FileLookup, check_tools, invalid, json_schema, no_files, question_and_context,
                   require_last_user, sse, tool_result_text)

# Claude Code sees the client's tools through the gateway's MCP server, under this prefix.
# The prefix is removed before anything reaches the client.
TOOL_PREFIX = "mcp__gateway__"


class MessageParam(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: Union[str, List[dict]]


class MessagesRequest(BaseModel):
    """Unknown fields are rejected, as the real API does."""

    model_config = ConfigDict(extra="forbid")
    model: str
    max_tokens: int = Field(ge=1)
    messages: List[MessageParam] = Field(min_length=1)
    system: Union[str, List[dict], None] = None
    stream: bool = False
    thinking: Optional[dict] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[dict] = None
    output_config: Optional[dict] = None   # {"format": {"type": "json_schema", "schema": ...}, "effort": ...}
    output_format: Optional[dict] = None   # the earlier (beta) name for output_config.format
    # Accepted but ignored: the CLIs have no way to set these.
    temperature: Optional[float] = Field(default=None, ge=0, le=1)
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[dict] = None
    service_tier: Optional[str] = None


IGNORED = ("temperature", "top_p", "top_k", "stop_sequences", "metadata", "service_tier")
SAMPLING = ("temperature", "top_p", "top_k", "stop_sequences")

# Field names of the official SDK types (anthropic 1.8.0): Message, Usage, MessageDeltaUsage, RawMessageDeltaEvent.Delta.
# Claude Code adds a few fields of its own (for example usage.iterations); those are dropped.
MESSAGE_FIELDS = {"id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "stop_details",
                  "container", "usage"}
USAGE_FIELDS = {"input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                "cache_creation", "output_tokens_details", "server_tool_use", "service_tier", "inference_geo"}
DELTA_USAGE_FIELDS = {"input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                      "output_tokens_details", "server_tool_use"}
DELTA_FIELDS = {"stop_reason", "stop_sequence", "stop_details", "container"}
THINKING_BLOCKS = ("thinking", "redacted_thinking")
PING = {"type": "ping"}

ERRORS = {"invalid_request": (400, "invalid_request_error"), "authentication": (401, "authentication_error"),
          "permission": (403, "permission_error"), "not_found": (404, "not_found_error"),
          "too_large": (413, "request_too_large"), "rate_limit": (429, "rate_limit_error"),
          "api_error": (500, "api_error"), "timeout": (504, "timeout_error"), "overloaded": (529, "overloaded_error")}
KEY_MESSAGES = {"missing_key": "x-api-key header is required", "bad_key": "invalid x-api-key"}


def message_id() -> str:
    """An id in the API's style, e.g. msg_011CfRy5YE4BMSUD8JaztiKS."""
    return "msg_01" + random_id(22)


def tool_use_id() -> str:
    return "toolu_01" + random_id(22)


def _only(data: dict, allowed: set) -> dict:
    return {k: v for k, v in data.items() if k in allowed}


def clean_message(message: dict) -> dict:
    out = _only(message, MESSAGE_FIELDS)
    if isinstance(out.get("usage"), dict):
        out["usage"] = _only(out["usage"], USAGE_FIELDS)
    return out


def _strip_prefix(name: str) -> str:
    return name[len(TOOL_PREFIX):] if name.startswith(TOOL_PREFIX) else name


def _source(source: dict, filename: Optional[str], where: str, files: FileLookup, default_type: str):
    kind = source.get("type")
    if kind == "base64":
        return attachments.from_base64(source.get("data", ""), source.get("media_type") or default_type,
                                       filename, where)
    if kind == "file":
        found = files(source.get("file_id", ""), where)
        return found if not filename else attachments.make(found.data, found.media_type, filename, where)
    if kind == "text":
        return attachments.make(str(source.get("data", "")).encode("utf-8"), source.get("media_type") or "text/plain",
                                filename, where)
    if kind == "content":
        return attachments.make(tool_result_text(source.get("content", "")).encode("utf-8"), "text/plain",
                                filename, where)
    if kind == "url":
        return attachments.from_data_url(source.get("url", ""), filename, where)
    raise invalid(f"{where}.source.type: '{kind}' is not supported by this gateway.")


def _parts(content, path: str, files: FileLookup) -> List[Part]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    parts: List[Part] = []
    for i, block in enumerate(content):
        where = f"{path}.{i}"
        kind = block.get("type")
        if kind == "text":
            if not isinstance(block.get("text"), str):
                raise invalid(f"{where}.text: Field required")
            parts.append(block["text"])
        elif kind == "image":
            parts.append(_source(block.get("source") or {}, None, where, files, "image/png"))
        elif kind == "document":
            parts.append(_source(block.get("source") or {}, block.get("title"), where, files, "application/pdf"))
        elif kind == "tool_use":
            parts.append(ToolCall(block.get("id", ""), _strip_prefix(block.get("name", "")), block.get("input") or {}))
        elif kind == "tool_result":
            parts.append(ToolResult(block.get("tool_use_id", ""), tool_result_text(block.get("content", "")),
                                    is_error=bool(block.get("is_error"))))
        elif kind in THINKING_BLOCKS:
            continue  # earlier thinking is dropped, as the API does for previous turns
        else:
            raise invalid(f"{where}.type: '{kind}' content blocks are not supported by this gateway.")
    return parts


def _thinking_budget(req: MessagesRequest) -> Optional[int]:
    thinking = req.thinking or {"type": "disabled"}
    kind = thinking.get("type")
    if kind == "disabled":
        return 0
    if kind == "adaptive":
        return None
    if kind == "enabled":
        budget = thinking.get("budget_tokens")
        if not isinstance(budget, int) or budget < 1024:
            raise invalid("thinking.enabled.budget_tokens: Input should be greater than or equal to 1024")
        if budget >= req.max_tokens:
            raise invalid("`max_tokens` must be greater than `thinking.budget_tokens`.")
        return budget
    raise invalid("thinking.type: Input should be 'enabled', 'disabled' or 'adaptive'")


def _tools(req: MessagesRequest) -> List[ToolDef]:
    tools = []
    for i, tool in enumerate(req.tools or []):
        if tool.get("type") not in (None, "custom"):
            raise invalid(f"tools.{i}: server tools such as '{tool.get('type')}' are not supported by this gateway.")
        tools.append(ToolDef(tool.get("name", ""), tool.get("description") or "",
                             json_schema(tool.get("input_schema") or {"type": "object"})))
    check_tools(tools)
    return tools


def _tool_choice(choice: Optional[dict]) -> Optional[dict]:
    if not choice:
        return None
    kind = choice.get("type")
    if kind not in ("auto", "any", "tool", "none"):
        raise invalid("tool_choice.type: Input should be 'auto', 'any', 'tool' or 'none'")
    return {"type": kind, "name": choice.get("name")}


def _output(req: MessagesRequest):
    """(response_format, effort) from output_config, or the older output_format."""
    config = req.output_config or {}
    fmt = config.get("format") or req.output_format
    effort = config.get("effort")
    if effort is not None and effort not in ("low", "medium", "high", "xhigh", "max"):
        raise invalid("output_config.effort: Input should be 'low', 'medium', 'high', 'xhigh' or 'max'")
    if not fmt:
        return None, effort
    if fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
        raise invalid("output_config.format: only {\"type\": \"json_schema\", \"schema\": {...}} is supported.")
    return {"type": "json_schema", "schema": fmt["schema"], "name": "output", "strict": True}, effort


def _validation_message(errors: list) -> str:
    parts = []
    for e in errors:
        loc = ".".join(str(p) for p in e.get("loc", ()))
        parts.append(f"{loc}: {e.get('msg')}" if loc else str(e.get("msg")))
    return "; ".join(parts) or "Invalid request."


class EventShaper:
    """Filters one raw Claude Code stream event to the SDK fields.

    When the client did not ask for thinking, thinking blocks are removed and the remaining blocks
    renumbered, so the client sees the same indexes a direct API call would give. Tool names lose the
    prefix Claude Code gives the gateway's tools.
    """

    def __init__(self, keep_thinking: bool):
        self.keep_thinking = keep_thinking
        self.index_map = {}

    def shape(self, event: dict) -> Optional[dict]:
        kind = event.get("type")
        if kind == "message_start":
            return {"type": kind, "message": clean_message(event["message"])}
        if kind == "content_block_start":
            block = dict(event["content_block"])
            if block.get("type") in THINKING_BLOCKS and not self.keep_thinking:
                self.index_map[event["index"]] = None
                return None
            if block.get("type") == "tool_use":
                block["name"] = _strip_prefix(block.get("name", ""))
            self.index_map[event["index"]] = sum(1 for v in self.index_map.values() if v is not None)
            return {"type": kind, "index": self.index_map[event["index"]], "content_block": block}
        if kind in ("content_block_delta", "content_block_stop"):
            index = self.index_map.get(event["index"])
            if index is None:
                return None
            return {**event, "index": index}
        if kind == "message_delta":
            return {"type": kind, "delta": _only(event.get("delta", {}), DELTA_FIELDS),
                    "usage": _only(event.get("usage", {}), DELTA_USAGE_FIELDS)}
        if kind == "message_stop":
            return {"type": kind}
        return event


def accumulate(events: List[dict]) -> dict:
    """Builds the final Message from stream events, the same way the SDK does."""
    message = None
    blocks = {}
    for event in events:
        kind = event["type"]
        if kind == "message_start":
            message = copy.deepcopy(event["message"])
            blocks = {}  # a new message_start (a retry) replaces anything before it
        elif kind == "content_block_start":
            blocks[event["index"]] = copy.deepcopy(event["content_block"])
        elif kind == "content_block_delta":
            block, delta = blocks[event["index"]], event["delta"]
            if delta["type"] == "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
            elif delta["type"] == "signature_delta":
                block["signature"] = delta["signature"]
            elif delta["type"] == "citations_delta":
                block.setdefault("citations", []).append(delta["citation"])
            elif delta["type"] == "input_json_delta":
                block["_json"] = block.get("_json", "") + delta.get("partial_json", "")
        elif kind == "message_delta":
            message.update(event["delta"])
            message["usage"] = {**message.get("usage", {}), **event["usage"]}
    if message is None:
        raise GatewayError("api_error", "Claude Code returned no message.")
    for block in blocks.values():
        if "_json" in block:
            raw = block.pop("_json")
            block["input"] = json.loads(raw) if raw.strip() else {}
    message["content"] = [blocks[i] for i in sorted(blocks)]
    return message


def native_message(natives: List[dict], keep_thinking: bool) -> dict:
    """The exact Anthropic message from Claude Code's raw events."""
    shaper = EventShaper(keep_thinking)
    return accumulate([e for e in (shaper.shape(n) for n in natives) if e is not None])


def message_to_events(message: dict) -> List[dict]:
    """Stream events for a stored message (used to replay cache hits)."""
    start = dict(message)
    start.update(content=[], stop_reason=None, stop_sequence=None)
    start["usage"] = {**message.get("usage", {}), "output_tokens": 1}
    events = [{"type": "message_start", "message": start}]
    for i, block in enumerate(message.get("content", [])):
        if block["type"] == "text":
            events.append({"type": "content_block_start", "index": i, "content_block": {**block, "text": ""}})
            events.append({"type": "content_block_delta", "index": i,
                           "delta": {"type": "text_delta", "text": block["text"]}})
        elif block["type"] == "thinking":
            events.append({"type": "content_block_start", "index": i,
                           "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            events.append({"type": "content_block_delta", "index": i,
                           "delta": {"type": "thinking_delta", "thinking": block["thinking"]}})
            events.append({"type": "content_block_delta", "index": i,
                           "delta": {"type": "signature_delta", "signature": block["signature"]}})
        elif block["type"] == "tool_use":
            events.append({"type": "content_block_start", "index": i, "content_block": {**block, "input": {}}})
            events.append({"type": "content_block_delta", "index": i,
                           "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}})
        else:
            events.append({"type": "content_block_start", "index": i, "content_block": block})
        events.append({"type": "content_block_stop", "index": i})
    events.append({"type": "message_delta", "delta": _only(message, DELTA_FIELDS),
                   "usage": _only(message.get("usage", {}), DELTA_USAGE_FIELDS)})
    events.append({"type": "message_stop"})
    return events


def _usage(result: Result) -> dict:
    u = result.usage
    return {"input_tokens": max(u.input_tokens - u.cached_tokens - u.cache_write_tokens, 0),
            "cache_creation_input_tokens": u.cache_write_tokens, "cache_read_input_tokens": u.cached_tokens,
            "output_tokens": u.output_tokens}


class AnthropicMessages(Dialect):
    id = "anthropic-messages"
    label = "Anthropic Messages"
    native_provider = "claude"

    def parse(self, body: dict, model: Optional[str] = None, options: Optional[dict] = None,
              files: Optional[FileLookup] = None) -> ChatRequest:
        try:
            req = MessagesRequest.model_validate(body)
        except ValidationError as exc:
            raise invalid(_validation_message(exc.errors()))
        files = files or no_files
        turns = [Turn(m.role, _parts(m.content, f"messages.{i}.content", files)) for i, m in enumerate(req.messages)]
        names = {p.id: p.name for t in turns for p in t.parts if isinstance(p, ToolCall)}
        for part in (p for t in turns for p in t.parts if isinstance(p, ToolResult)):
            part.name = names.get(part.call_id)
        require_last_user(turns, "messages")
        texts = [p for p in turns[-1].parts if isinstance(p, str)]
        question, context = question_and_context(texts)
        system_parts = _parts(req.system, "system", files)
        if any(not isinstance(p, str) for p in system_parts):
            raise invalid("system: only text blocks are supported.")
        response_format, effort = _output(req)
        return ChatRequest(
            api=self.id, provider="", model=req.model, system="\n\n".join(system_parts), turns=turns,
            max_tokens=req.max_tokens, stream=req.stream, body=body, question=question or "(tool results)",
            context=context, thinking_budget=_thinking_budget(req), tools=_tools(req),
            tool_choice=_tool_choice(req.tool_choice), response_format=response_format, reasoning_effort=effort,
            ignored=[name for name in IGNORED if getattr(req, name) is not None],
            sampling={name: getattr(req, name) for name in SAMPLING if getattr(req, name) is not None})

    def render(self, req: ChatRequest, result: Result) -> dict:
        if result.native is not None:
            return result.native
        content = [{"type": "text", "text": result.text}] if result.text or not result.tool_calls else []
        content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in result.tool_calls]
        return {"id": message_id(), "type": "message", "role": "assistant", "model": result.model,
                "content": content, "stop_reason": result.stop_reason, "stop_sequence": None,
                "usage": _usage(result)}

    async def stream(self, req: ChatRequest, events: AsyncIterator[Event]) -> AsyncIterator[bytes]:
        try:
            if req.provider == "claude" and not req.response_format:
                # Claude Code's own events, passed through. (Structured output is built: Claude Code produces
                # it through a tool of its own, which the client must not see.)
                shaper = EventShaper(keep_thinking=req.thinking_budget != 0)
                async for event in events:
                    if isinstance(event, Native):
                        out = shaper.shape(event.event)
                        if out is not None:
                            yield sse(out, out["type"])
                            if out["type"] == "message_start":
                                yield sse(PING, "ping")
                return
            # Another provider: the same event sequence, built from its text and tool calls.
            msg_id = message_id()
            index = -1
            text_open = False
            async for event in events:
                if isinstance(event, Start):
                    start = {"id": msg_id, "type": "message", "role": "assistant", "model": event.model,
                             "content": [], "stop_reason": None, "stop_sequence": None,
                             "usage": {"input_tokens": 0, "output_tokens": 0}}
                    yield sse({"type": "message_start", "message": start}, "message_start")
                    yield sse(PING, "ping")
                elif isinstance(event, TextDelta):
                    if not text_open:
                        index += 1
                        text_open = True
                        yield sse({"type": "content_block_start", "index": index,
                                   "content_block": {"type": "text", "text": ""}}, "content_block_start")
                    yield sse({"type": "content_block_delta", "index": index,
                               "delta": {"type": "text_delta", "text": event.text}}, "content_block_delta")
                elif isinstance(event, ToolUse):
                    if text_open:
                        text_open = False
                        yield sse({"type": "content_block_stop", "index": index}, "content_block_stop")
                    index += 1
                    call = event.call
                    yield sse({"type": "content_block_start", "index": index,
                               "content_block": {"type": "tool_use", "id": call.id, "name": call.name, "input": {}}},
                              "content_block_start")
                    yield sse({"type": "content_block_delta", "index": index,
                               "delta": {"type": "input_json_delta", "partial_json": json.dumps(call.arguments)}},
                              "content_block_delta")
                    yield sse({"type": "content_block_stop", "index": index}, "content_block_stop")
                elif isinstance(event, Finish):
                    if text_open or index < 0:
                        if index < 0:  # an empty answer still has one text block
                            index = 0
                            yield sse({"type": "content_block_start", "index": 0,
                                       "content_block": {"type": "text", "text": ""}}, "content_block_start")
                        yield sse({"type": "content_block_stop", "index": index}, "content_block_stop")
                    result = Result(model="", text="", stop_reason=event.stop_reason, usage=event.usage)
                    yield sse({"type": "message_delta",
                               "delta": {"stop_reason": event.stop_reason, "stop_sequence": None},
                               "usage": _usage(result)}, "message_delta")
                    yield sse({"type": "message_stop"}, "message_stop")
        except GatewayError as err:
            # The API reports errors after the stream has started as an `error` event.
            yield sse({"type": "error", "error": self._error_body(err)}, "error")
            raise

    def _error_body(self, err: GatewayError) -> dict:
        _, error_type = ERRORS.get(err.kind, ERRORS["api_error"])
        return {"type": error_type, "message": KEY_MESSAGES.get(err.code, err.message)}

    def error(self, err: GatewayError, request_id: str):
        status, _ = ERRORS.get(err.kind, ERRORS["api_error"])
        return status, {"type": "error", "error": self._error_body(err), "request_id": request_id}
