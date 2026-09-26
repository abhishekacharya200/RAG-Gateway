"""OpenAI Responses API: POST /v1/responses."""

import json
import secrets
import time
from typing import AsyncIterator, List, Optional

from .. import attachments
from ..core import (ChatRequest, Event, Finish, GatewayError, Part, Result, Start, TextDelta, ToolCall, ToolDef,
                    ToolResult, ToolUse, Turn, Usage)
from .base import (Dialect, FileLookup, check_tools, invalid, json_schema, no_files, question_and_context,
                   require_last_user, sse, tool_result_text)
from .openai_chat import error_body, error_status, response_format, tool_choice

KNOWN = {"model", "input", "instructions", "max_output_tokens", "stream", "stream_options", "reasoning", "text",
         "tools", "tool_choice", "previous_response_id", "store", "metadata", "temperature", "top_p", "truncation",
         "user", "include", "parallel_tool_calls", "service_tier", "background", "prompt", "conversation",
         "max_tool_calls", "safety_identifier", "prompt_cache_key", "top_logprobs"}
# Accepted but ignored: the CLIs have no way to set these.
IGNORED = ("temperature", "top_p", "store", "metadata", "truncation", "user", "include", "parallel_tool_calls",
           "service_tier", "max_tool_calls", "safety_identifier", "prompt_cache_key", "top_logprobs", "stream_options")
SAMPLING = ("temperature", "top_p")
TEXT_PARTS = ("input_text", "output_text", "text")


def _hex_id(prefix: str) -> str:
    return prefix + secrets.token_hex(24)


def _parts(content, path: str, files: FileLookup) -> List[Part]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        raise invalid(f"Invalid type for '{path}': expected a string or an array of content parts.")
    parts: List[Part] = []
    for i, part in enumerate(content):
        where = f"{path}[{i}]"
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in TEXT_PARTS and isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif kind == "input_image":
            if part.get("file_id"):
                parts.append(files(part["file_id"], where))
            else:
                parts.append(attachments.from_data_url(part.get("image_url") or "", None, where))
        elif kind == "input_file":
            if part.get("file_id"):
                parts.append(files(part["file_id"], where))
            elif part.get("file_data"):
                data, name = part["file_data"], part.get("filename")
                parts.append(attachments.from_data_url(data, name, where) if data.startswith("data:")
                             else attachments.from_base64(data, attachments.guess_type(name), name, where))
            else:
                raise invalid(f"'{where}': file_url is not supported by this gateway; send file_id or file_data.")
        elif kind == "refusal":
            continue
        else:
            raise invalid(f"Invalid content part at '{where}': type '{kind}' is not supported by this gateway.")
    return parts


def _usage(u: Usage) -> dict:
    return {"input_tokens": u.input_tokens, "input_tokens_details": {"cached_tokens": u.cached_tokens},
            "output_tokens": u.output_tokens, "output_tokens_details": {"reasoning_tokens": u.reasoning_tokens},
            "total_tokens": u.input_tokens + u.output_tokens}


def _part(text: str) -> dict:
    return {"type": "output_text", "annotations": [], "logprobs": [], "text": text}


def _call_item(call: ToolCall, status: str = "completed", arguments: Optional[str] = None) -> dict:
    return {"id": "fc_" + secrets.token_hex(24), "type": "function_call", "status": status,
            "arguments": json.dumps(call.arguments) if arguments is None else arguments,
            "call_id": call.id, "name": call.name}


class OpenAIResponses(Dialect):
    id = "openai-responses"
    label = "OpenAI Responses"
    native_provider = "codex"

    def parse(self, body: dict, model: Optional[str] = None, options: Optional[dict] = None,
              files: Optional[FileLookup] = None) -> ChatRequest:
        files = files or no_files
        unknown = sorted(set(body) - KNOWN)
        if unknown:
            raise invalid(f"Unknown parameter: '{unknown[0]}'.")
        if not isinstance(body.get("model"), str) or not body["model"]:
            raise invalid("Missing required parameter: 'model'.")
        if "input" not in body:
            raise invalid("Missing required parameter: 'input'.")
        for name in ("previous_response_id", "conversation", "prompt"):
            if body.get(name):
                raise invalid(f"'{name}' is not supported by this gateway: it does not store responses.")
        if body.get("background"):
            raise invalid("Background responses are not supported by this gateway.")
        tools = []
        for i, tool in enumerate(body.get("tools") or []):
            if tool.get("type") != "function":
                raise invalid(f"tools[{i}]: only 'function' tools are supported by this gateway, not "
                              f"'{tool.get('type')}'.")
            tools.append(ToolDef(tool.get("name", ""), tool.get("description") or "",
                                 json_schema(tool.get("parameters") or {"type": "object", "properties": {}})))
        check_tools(tools)

        system = [body["instructions"]] if isinstance(body.get("instructions"), str) and body["instructions"] else []
        turns: List[Turn] = []
        last_user: List[str] = []
        names = {}
        items = body["input"]
        if isinstance(items, str):
            items = [{"role": "user", "content": items}]
        if not isinstance(items, list) or not items:
            raise invalid("Invalid 'input': expected a string or a non-empty array of input items.")

        def add(role: str, parts: List[Part]) -> None:
            # Consecutive items of one side (a message and its function calls) form one turn.
            if turns and turns[-1].role == role:
                turns[-1].parts.extend(parts)
            else:
                turns.append(Turn(role, list(parts)))

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise invalid(f"Invalid type for 'input[{i}]': expected an object.")
            kind = item.get("type", "message")
            if kind == "message":
                role = item.get("role")
                parts = _parts(item.get("content"), f"input[{i}].content", files)
                if role in ("system", "developer"):
                    if any(not isinstance(p, str) for p in parts):
                        raise invalid(f"input[{i}]: {role} messages can only hold text.")
                    system += parts
                elif role in ("user", "assistant"):
                    add(role, parts)
                    if role == "user":
                        last_user = [p for p in parts if isinstance(p, str)]
                else:
                    raise invalid(f"Invalid value: '{role}'. Supported values are: 'assistant', 'system', "
                                  "'developer', and 'user'.")
            elif kind == "function_call":
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except ValueError:
                    raise invalid(f"input[{i}].arguments must be a JSON string.")
                names[item.get("call_id", "")] = item.get("name", "")
                add("assistant", [ToolCall(item.get("call_id", ""), item.get("name", ""), arguments)])
            elif kind == "function_call_output":
                add("user", [ToolResult(item.get("call_id", ""), tool_result_text(item.get("output", "")),
                                        name=names.get(item.get("call_id", "")))])
                last_user = []
            elif kind == "reasoning":
                continue  # earlier reasoning is not sent back to the CLIs
            else:
                raise invalid(f"input[{i}]: items of type '{kind}' are not supported by this gateway.")
        require_last_user(turns, "input")
        question, context = question_and_context(last_user)
        return ChatRequest(
            api=self.id, provider="", model=body["model"], system="\n\n".join(system), turns=turns,
            max_tokens=body.get("max_output_tokens"), stream=bool(body.get("stream")), body=body,
            question=question or "(tool results)", context=context,
            reasoning_effort=(body.get("reasoning") or {}).get("effort"), tools=tools,
            tool_choice=tool_choice(body.get("tool_choice")),
            response_format=response_format((body.get("text") or {}).get("format"), "text.format"),
            ignored=[name for name in IGNORED if body.get(name) is not None],
            sampling={name: body[name] for name in SAMPLING if body.get(name) is not None})

    def _response(self, req: ChatRequest, resp_id: str, created: int, model: str, status: str,
                  output: list, usage: Optional[Usage], stop_reason: Optional[str] = None) -> dict:
        body = req.body
        incomplete = stop_reason == "max_tokens"
        tools = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters,
                  "strict": False} for t in req.tools]
        return {
            "id": resp_id, "object": "response", "created_at": created,
            "status": "incomplete" if incomplete else status, "background": False, "error": None,
            "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
            "instructions": body.get("instructions"), "max_output_tokens": body.get("max_output_tokens"),
            "max_tool_calls": None, "model": model, "output": output, "parallel_tool_calls": True,
            "previous_response_id": None, "prompt_cache_key": None,
            "reasoning": {"effort": req.reasoning_effort, "summary": None}, "safety_identifier": None,
            "service_tier": "default", "store": body.get("store", True), "temperature": body.get("temperature", 1.0),
            "text": {"format": (body.get("text") or {}).get("format") or {"type": "text"}, "verbosity": "medium"},
            "tool_choice": body.get("tool_choice", "auto"), "tools": tools,
            "top_logprobs": 0, "top_p": body.get("top_p", 1.0), "truncation": "disabled",
            "usage": _usage(usage) if usage else None, "user": None, "metadata": body.get("metadata") or {}}

    def render(self, req: ChatRequest, result: Result) -> dict:
        output = []
        if result.text or not result.tool_calls:
            output.append({"id": _hex_id("msg_"), "type": "message", "status": "completed", "role": "assistant",
                           "content": [_part(result.text)]})
        output += [_call_item(c) for c in result.tool_calls]
        return self._response(req, _hex_id("resp_"), int(time.time()), result.model, "completed", output,
                              result.usage, result.stop_reason)

    async def stream(self, req: ChatRequest, events: AsyncIterator[Event]) -> AsyncIterator[bytes]:
        resp_id, msg_id, created, model = _hex_id("resp_"), _hex_id("msg_"), int(time.time()), req.model
        seq = 0
        text = ""
        output: list = []
        message_open = False

        def emit(kind: str, **fields) -> bytes:
            nonlocal seq
            data = {"type": kind, "sequence_number": seq, **fields}
            seq += 1
            return sse(data, kind)

        def close_message():
            item = {"id": msg_id, "type": "message", "status": "completed", "role": "assistant",
                    "content": [_part(text)]}
            output.append(item)
            index = len(output) - 1
            return [emit("response.output_text.done", item_id=msg_id, output_index=index, content_index=0,
                         text=text, logprobs=[]),
                    emit("response.content_part.done", item_id=msg_id, output_index=index, content_index=0,
                         part=_part(text)),
                    emit("response.output_item.done", output_index=index, item=item)]

        def open_message():
            index = len(output)
            return [emit("response.output_item.added", output_index=index,
                         item={"id": msg_id, "type": "message", "status": "in_progress", "role": "assistant",
                               "content": []}),
                    emit("response.content_part.added", item_id=msg_id, output_index=index, content_index=0,
                         part=_part(""))]

        try:
            async for event in events:
                if isinstance(event, Start):
                    model = event.model
                    pending = self._response(req, resp_id, created, model, "in_progress", [], None)
                    yield emit("response.created", response=pending)
                    yield emit("response.in_progress", response=pending)
                elif isinstance(event, TextDelta):
                    if not message_open:
                        message_open = True
                        for chunk in open_message():
                            yield chunk
                    text += event.text
                    yield emit("response.output_text.delta", item_id=msg_id, output_index=len(output),
                               content_index=0, delta=event.text, logprobs=[])
                elif isinstance(event, ToolUse):
                    if message_open:
                        message_open = False
                        for chunk in close_message():
                            yield chunk
                    arguments = json.dumps(event.call.arguments)
                    item = _call_item(event.call, status="in_progress", arguments="")
                    index = len(output)
                    yield emit("response.output_item.added", output_index=index, item=item)
                    yield emit("response.function_call_arguments.delta", item_id=item["id"], output_index=index,
                               delta=arguments)
                    yield emit("response.function_call_arguments.done", item_id=item["id"], output_index=index,
                               name=event.call.name, arguments=arguments)
                    item = {**item, "status": "completed", "arguments": arguments}
                    output.append(item)
                    yield emit("response.output_item.done", output_index=index, item=item)
                elif isinstance(event, Finish):
                    if message_open or not output:
                        if not message_open:
                            for chunk in open_message():
                                yield chunk
                        message_open = False
                        for chunk in close_message():
                            yield chunk
                    final = self._response(req, resp_id, created, model, "completed", output, event.usage,
                                           event.stop_reason)
                    kind = "response.incomplete" if final["status"] == "incomplete" else "response.completed"
                    yield emit(kind, response=final)
        except GatewayError as err:
            error = error_body(err)["error"]
            yield emit("error", code=error["code"], message=error["message"], param=None)
            raise

    def error(self, err: GatewayError, request_id: str):
        return error_status(err), error_body(err)
