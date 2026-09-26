"""OpenAI Chat Completions API: POST /v1/chat/completions."""

import json
import time
from typing import AsyncIterator, List, Optional

from .. import attachments
from ..core import (ALNUM, ChatRequest, Event, Finish, GatewayError, Part, Result, Start, TextDelta, ToolCall,
                    ToolDef, ToolResult, ToolUse, Turn, Usage, random_id)
from .base import (Dialect, FileLookup, check_tools, invalid, json_schema, no_files, question_and_context,
                   require_last_user, sse, tool_result_text)

KNOWN = {"model", "messages", "max_tokens", "max_completion_tokens", "stream", "stream_options", "temperature",
         "top_p", "n", "stop", "presence_penalty", "frequency_penalty", "seed", "user", "metadata", "store",
         "logprobs", "top_logprobs", "logit_bias", "parallel_tool_calls", "service_tier", "reasoning_effort",
         "prompt_cache_key", "safety_identifier", "verbosity", "tools", "tool_choice", "functions", "function_call",
         "response_format", "modalities", "audio", "prediction", "web_search_options"}
# Accepted but ignored: the CLIs have no way to set these.
IGNORED = ("temperature", "top_p", "stop", "presence_penalty", "frequency_penalty", "seed", "user", "metadata",
           "store", "logprobs", "top_logprobs", "logit_bias", "parallel_tool_calls", "service_tier",
           "prompt_cache_key", "safety_identifier", "verbosity")
SAMPLING = ("temperature", "top_p", "stop", "presence_penalty", "frequency_penalty", "seed")

FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "refusal": "content_filter",
          "tool_use": "tool_calls"}
ERRORS = {"invalid_request": (400, "invalid_request_error", None),
          "authentication": (401, "invalid_request_error", "invalid_api_key"),
          "permission": (403, "invalid_request_error", None),
          "not_found": (404, "invalid_request_error", "model_not_found"),
          "too_large": (413, "invalid_request_error", None),
          "rate_limit": (429, "requests", "rate_limit_exceeded"),
          "api_error": (500, "server_error", None), "overloaded": (503, "server_error", None),
          "timeout": (504, "server_error", None)}
KEY_MESSAGES = {
    "missing_key": "You didn't provide an API key. You need to provide your API key in an Authorization header "
                   "using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).",
    "bad_key": "Incorrect API key provided. Use the gateway key (GATEWAY_API_KEYS).",
}


def error_body(err: GatewayError) -> dict:
    """The {"error": {...}} body that both OpenAI APIs use."""
    _, error_type, code = ERRORS.get(err.kind, ERRORS["api_error"])
    return {"error": {"message": KEY_MESSAGES.get(err.code, err.message), "type": error_type, "param": None,
                      "code": code}}


def error_status(err: GatewayError) -> int:
    return ERRORS.get(err.kind, ERRORS["api_error"])[0]


def tool_choice(choice) -> Optional[dict]:
    """Both OpenAI APIs: "auto", "none", "required", or a specific function."""
    if choice is None:
        return None
    if choice in ("auto", "none"):
        return {"type": choice}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name") or choice.get("name")
        if name:
            return {"type": "tool", "name": name}
    raise invalid("Invalid value for 'tool_choice'.")


def response_format(fmt, where: str) -> Optional[dict]:
    """Chat's response_format and Responses' text.format, in the gateway's form."""
    if not fmt or fmt.get("type", "text") == "text":
        return None
    if fmt.get("type") == "json_object":
        return {"type": "json_object"}
    if fmt.get("type") == "json_schema":
        spec = fmt.get("json_schema") if "json_schema" in fmt else fmt  # Chat nests it; Responses does not
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise invalid(f"Missing required parameter: '{where}.schema'.")
        return {"type": "json_schema", "schema": spec["schema"], "name": spec.get("name") or "output",
                "strict": bool(spec.get("strict"))}
    raise invalid(f"Invalid value for '{where}.type': '{fmt.get('type')}'.")


def _parts(content, path: str, files: FileLookup) -> List[Part]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        raise invalid(f"Invalid type for '{path}': expected a string or an array of content parts.")
    parts: List[Part] = []
    for i, part in enumerate(content):
        where = f"{path}[{i}]"
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif kind == "image_url":
            parts.append(attachments.from_data_url((part.get("image_url") or {}).get("url", ""), None, where))
        elif kind == "file":
            file = part.get("file") or {}
            if file.get("file_id"):
                parts.append(files(file["file_id"], where))
            elif file.get("file_data"):
                data = file["file_data"]
                parts.append(attachments.from_data_url(data, file.get("filename"), where) if data.startswith("data:")
                             else attachments.from_base64(data, attachments.guess_type(file.get("filename")),
                                                          file.get("filename"), where))
            else:
                raise invalid(f"Missing 'file_id' or 'file_data' in '{where}.file'.")
        elif kind == "refusal":
            continue
        else:
            raise invalid(f"Invalid content part at '{where}': type '{kind}' is not supported by this gateway.")
    return parts


def _usage(u: Usage) -> dict:
    return {"prompt_tokens": u.input_tokens, "completion_tokens": u.output_tokens,
            "total_tokens": u.input_tokens + u.output_tokens,
            "prompt_tokens_details": {"cached_tokens": u.cached_tokens, "audio_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": u.reasoning_tokens, "audio_tokens": 0,
                                          "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}}


def completion_id() -> str:
    return "chatcmpl-" + random_id(29, ALNUM)


def _tool_calls_json(calls: List[ToolCall]) -> list:
    return [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in calls]


class OpenAIChat(Dialect):
    id = "openai-chat"
    label = "OpenAI Chat Completions"
    native_provider = "codex"

    def parse(self, body: dict, model: Optional[str] = None, options: Optional[dict] = None,
              files: Optional[FileLookup] = None) -> ChatRequest:
        files = files or no_files
        unknown = sorted(set(body) - KNOWN)
        if unknown:
            raise invalid(f"Unrecognized request argument supplied: {unknown[0]}")
        if not isinstance(body.get("model"), str) or not body["model"]:
            raise invalid("you must provide a model parameter")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise invalid("[] is too short - 'messages'" if isinstance(messages, list) else
                          "Missing required parameter: 'messages'.")
        if body.get("functions") or body.get("function_call"):
            raise invalid("The legacy 'functions' parameter is not supported by this gateway. Use 'tools'.")
        if body.get("n") not in (None, 1):
            raise invalid("This gateway returns one choice: 'n' must be 1.")
        if body.get("modalities") not in (None, ["text"]) or body.get("audio") or body.get("prediction") \
                or body.get("web_search_options"):
            raise invalid("Audio output, predictions and web search are not supported by this gateway.")
        tools = []
        for i, tool in enumerate(body.get("tools") or []):
            if tool.get("type") != "function":
                raise invalid(f"tools[{i}]: only 'function' tools are supported by this gateway.")
            fn = tool.get("function") or {}
            tools.append(ToolDef(fn.get("name", ""), fn.get("description") or "",
                                 json_schema(fn.get("parameters") or {"type": "object", "properties": {}})))
        check_tools(tools)

        system, turns, last_user = [], [], []
        names = {}
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise invalid(f"Invalid type for 'messages[{i}]': expected an object.")
            role = message.get("role")
            path = f"messages[{i}].content"
            if role in ("system", "developer"):
                texts = _parts(message.get("content"), path, files)
                if any(not isinstance(p, str) for p in texts):
                    raise invalid(f"{path}: system messages can only hold text.")
                system += texts
            elif role == "user":
                parts = _parts(message.get("content"), path, files)
                turns.append(Turn("user", parts))
                last_user = [p for p in parts if isinstance(p, str)]
            elif role == "assistant":
                parts = _parts(message.get("content"), path, files)
                for call in message.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    try:
                        arguments = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        raise invalid(f"messages[{i}].tool_calls: 'arguments' must be a JSON string.")
                    parts.append(ToolCall(call.get("id", ""), fn.get("name", ""), arguments))
                    names[call.get("id", "")] = fn.get("name", "")
                turns.append(Turn("assistant", parts))
            elif role == "tool":
                result = ToolResult(message.get("tool_call_id", ""), tool_result_text(message.get("content", "")),
                                    name=names.get(message.get("tool_call_id", "")))
                if turns and turns[-1].role == "user" and all(isinstance(p, ToolResult) for p in turns[-1].parts):
                    turns[-1].parts.append(result)  # several results for one assistant message
                else:
                    turns.append(Turn("user", [result]))
                last_user = []
            elif role == "function":
                raise invalid("'function' messages are not supported by this gateway. Use 'tool' messages.")
            else:
                raise invalid(f"Invalid value: '{role}'. Supported values are: 'system', 'assistant', 'user', "
                              "'tool', and 'function'.")
        require_last_user(turns, "messages")
        question, context = question_and_context(last_user)
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
        return ChatRequest(
            api=self.id, provider="", model=body["model"], system="\n\n".join(system), turns=turns,
            max_tokens=max_tokens, stream=bool(body.get("stream")), body=body, question=question or "(tool results)",
            context=context, reasoning_effort=body.get("reasoning_effort"), tools=tools,
            tool_choice=tool_choice(body.get("tool_choice")),
            response_format=response_format(body.get("response_format"), "response_format"),
            ignored=[name for name in IGNORED if body.get(name) is not None],
            sampling={name: body[name] for name in SAMPLING if body.get(name) is not None},
            options={"include_usage": bool((body.get("stream_options") or {}).get("include_usage"))})

    def render(self, req: ChatRequest, result: Result) -> dict:
        message = {"role": "assistant", "content": result.text if result.text or not result.tool_calls else None,
                   "refusal": None, "annotations": []}
        if result.tool_calls:
            message["tool_calls"] = _tool_calls_json(result.tool_calls)
        return {"id": completion_id(), "object": "chat.completion", "created": int(time.time()),
                "model": result.model,
                "choices": [{"index": 0, "message": message, "logprobs": None,
                             "finish_reason": FINISH.get(result.stop_reason, "stop")}],
                "usage": _usage(result.usage), "service_tier": "default", "system_fingerprint": None}

    async def stream(self, req: ChatRequest, events: AsyncIterator[Event]) -> AsyncIterator[bytes]:
        chunk_id, created, model = completion_id(), int(time.time()), req.model
        include_usage = req.options.get("include_usage")
        calls = 0

        def chunk(delta: dict, finish: Optional[str] = None, choices: bool = True, usage=None) -> bytes:
            data = {"id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "service_tier": "default", "system_fingerprint": None,
                    "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]
                    if choices else []}
            if include_usage:
                data["usage"] = usage
            return sse(data)

        try:
            async for event in events:
                if isinstance(event, Start):
                    model = event.model
                    yield chunk({"role": "assistant", "content": "", "refusal": None})
                elif isinstance(event, TextDelta):
                    yield chunk({"content": event.text})
                elif isinstance(event, ToolUse):
                    call = event.call
                    yield chunk({"tool_calls": [{"index": calls, "id": call.id, "type": "function",
                                                 "function": {"name": call.name,
                                                              "arguments": json.dumps(call.arguments)}}]})
                    calls += 1
                elif isinstance(event, Finish):
                    yield chunk({}, FINISH.get(event.stop_reason, "stop"))
                    if include_usage:
                        yield chunk({}, choices=False, usage=_usage(event.usage))
            yield b"data: [DONE]\n\n"
        except GatewayError as err:
            yield sse(error_body(err))
            raise

    def error(self, err: GatewayError, request_id: str):
        return error_status(err), error_body(err)
