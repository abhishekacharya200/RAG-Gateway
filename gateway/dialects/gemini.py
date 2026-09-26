"""Gemini API: POST /v1beta/models/{model}:generateContent and :streamGenerateContent."""

import json
import string
from typing import AsyncIterator, List, Optional

from .. import attachments
from ..core import (ChatRequest, Event, Finish, GatewayError, Part, Result, Start, TextDelta, ToolCall, ToolDef,
                    ToolResult, ToolUse, Turn, Usage, random_id)
from .base import Dialect, FileLookup, check_tools, invalid, json_schema, no_files, question_and_context, \
    require_last_user

# The API accepts both spellings of every field.
KNOWN = {"contents", "systemInstruction", "system_instruction", "generationConfig", "generation_config",
         "safetySettings", "safety_settings", "tools", "toolConfig", "tool_config", "cachedContent",
         "cached_content", "labels"}
# generationConfig fields that are accepted but ignored: the CLI has no way to set them.
IGNORED_CONFIG = (("temperature", "temperature"), ("topP", "top_p"), ("topK", "top_k"),
                  ("stopSequences", "stop_sequences"), ("seed", "seed"), ("presencePenalty", "presence_penalty"),
                  ("frequencyPenalty", "frequency_penalty"), ("thinkingConfig", "thinking_config"),
                  ("responseModalities", "response_modalities"), ("mediaResolution", "media_resolution"))
# The ignored fields that can still change the answer, so they belong in the cache key.
SAMPLING = (("temperature", "temperature"), ("topP", "top_p"), ("topK", "top_k"), ("stopSequences", "stop_sequences"),
            ("seed", "seed"), ("presencePenalty", "presence_penalty"), ("frequencyPenalty", "frequency_penalty"))

FINISH = {"end_turn": "STOP", "stop_sequence": "STOP", "max_tokens": "MAX_TOKENS", "refusal": "SAFETY",
          "tool_use": "STOP"}
ERRORS = {"invalid_request": (400, "INVALID_ARGUMENT"), "authentication": (401, "UNAUTHENTICATED"),
          "permission": (403, "PERMISSION_DENIED"), "not_found": (404, "NOT_FOUND"),
          "too_large": (400, "INVALID_ARGUMENT"), "rate_limit": (429, "RESOURCE_EXHAUSTED"),
          "api_error": (500, "INTERNAL"), "overloaded": (503, "UNAVAILABLE"), "timeout": (504, "DEADLINE_EXCEEDED")}
# What the Gemini API itself answers when the key is missing or wrong.
KEY_ERRORS = {
    "missing_key": (403, "PERMISSION_DENIED", "Method doesn't allow unregistered callers (callers without "
                    "established identity). Please use API Key or other form of API consumer identity to call "
                    "this API."),
    "bad_key": (400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."),
}


def _get(data: dict, camel: str, snake: str):
    return data.get(camel, data.get(snake))


def _parts(parts, path: str, files: FileLookup, results_by_name: dict) -> List[Part]:
    if not isinstance(parts, list):
        raise invalid(f"Invalid value at '{path}': expected a list of parts.")
    out: List[Part] = []
    for i, part in enumerate(parts):
        where = f"{path}[{i}]"
        if not isinstance(part, dict):
            raise invalid(f"Invalid value at '{where}': expected an object.")
        if part.get("thought"):
            continue  # earlier thinking is not sent back
        inline = _get(part, "inlineData", "inline_data")
        file_data = _get(part, "fileData", "file_data")
        call = _get(part, "functionCall", "function_call")
        response = _get(part, "functionResponse", "function_response")
        if isinstance(part.get("text"), str):
            out.append(part["text"])
        elif inline:
            out.append(attachments.from_base64(inline.get("data", ""), _get(inline, "mimeType", "mime_type") or "",
                                               _get(inline, "displayName", "display_name"), where))
        elif file_data:
            found = files(_get(file_data, "fileUri", "file_uri") or "", where)
            out.append(found)
        elif call:
            call_id = call.get("id") or f"call_{call.get('name', '')}_{i}"
            out.append(ToolCall(call_id, call.get("name", ""), call.get("args") or {}))
            results_by_name.setdefault(call.get("name", ""), []).append(call_id)
        elif response:
            name = response.get("name", "")
            waiting = results_by_name.get(name) or []
            call_id = response.get("id") or (waiting.pop(0) if waiting else f"call_{name}")
            out.append(ToolResult(call_id, json.dumps(response.get("response", {}), ensure_ascii=False), name=name))
        else:
            raise invalid(f"{where}: this kind of part is not supported by this gateway.")
    return out


def _usage(u: Usage) -> dict:
    usage = {"promptTokenCount": u.input_tokens, "candidatesTokenCount": max(u.output_tokens - u.reasoning_tokens, 0),
             "totalTokenCount": u.input_tokens + u.output_tokens,
             "promptTokensDetails": [{"modality": "TEXT", "tokenCount": u.input_tokens}]}
    if u.cached_tokens:
        usage["cachedContentTokenCount"] = u.cached_tokens
    if u.reasoning_tokens:
        usage["thoughtsTokenCount"] = u.reasoning_tokens
    return usage


def _response_id() -> str:
    return random_id(22, string.ascii_letters + string.digits + "-_")


def _chunk(parts: list, model: str, response_id: str, finish: Optional[str] = None,
           usage: Optional[Usage] = None) -> dict:
    candidate = {"content": {"parts": parts, "role": "model"}, "index": 0}
    if finish:
        candidate["finishReason"] = finish
    data = {"candidates": [candidate]}
    if usage is not None:
        data["usageMetadata"] = _usage(usage)
    data["modelVersion"] = model
    data["responseId"] = response_id
    return data


def _call_part(call: ToolCall) -> dict:
    return {"functionCall": {"name": call.name, "args": call.arguments, "id": call.id}}


def _sse(data: dict) -> bytes:
    return f"data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\r\n\r\n".encode()


def _tools(body: dict) -> List[ToolDef]:
    tools = []
    for i, tool in enumerate(body.get("tools") or []):
        declarations = _get(tool, "functionDeclarations", "function_declarations")
        if declarations is None:
            other = ", ".join(tool) or "empty"
            raise invalid(f"tools[{i}]: only functionDeclarations are supported by this gateway ({other}).")
        for fn in declarations:
            schema = _get(fn, "parametersJsonSchema", "parameters_json_schema") or fn.get("parameters") \
                or {"type": "object", "properties": {}}
            tools.append(ToolDef(fn.get("name", ""), fn.get("description") or "", json_schema(schema)))
    check_tools(tools)
    return tools


def _tool_choice(body: dict) -> Optional[dict]:
    config = _get(_get(body, "toolConfig", "tool_config") or {}, "functionCallingConfig", "function_calling_config")
    if not config:
        return None
    mode = (config.get("mode") or "AUTO").upper()
    allowed = _get(config, "allowedFunctionNames", "allowed_function_names") or []
    if mode == "NONE":
        return {"type": "none"}
    if mode in ("ANY", "VALIDATED") and len(allowed) == 1:
        return {"type": "tool", "name": allowed[0]}
    if mode == "ANY":
        return {"type": "any"}
    return {"type": "auto"}


class Gemini(Dialect):
    id = "gemini"
    label = "Gemini generateContent"
    native_provider = "antigravity"

    def parse(self, body: dict, model: Optional[str] = None, options: Optional[dict] = None,
              files: Optional[FileLookup] = None) -> ChatRequest:
        options = options or {}
        files = files or no_files
        unknown = sorted(set(body) - KNOWN)
        if unknown:
            raise invalid(f'Invalid JSON payload received. Unknown name "{unknown[0]}": Cannot find field.')
        contents = body.get("contents")
        if not isinstance(contents, list) or not contents:
            raise invalid("* GenerateContentRequest.contents: contents is not specified")
        if _get(body, "cachedContent", "cached_content"):
            raise invalid("cachedContent is not supported by this gateway.")
        config = _get(body, "generationConfig", "generation_config") or {}
        if (_get(config, "candidateCount", "candidate_count") or 1) != 1:
            raise invalid("This gateway returns one candidate: candidateCount must be 1.")
        mime = _get(config, "responseMimeType", "response_mime_type")
        schema = _get(config, "responseJsonSchema", "response_json_schema") or _get(config, "responseSchema", "response_schema")
        if mime not in (None, "text/plain", "application/json"):
            raise invalid(f"responseMimeType '{mime}' is not supported by this gateway (text/plain or application/json).")
        if schema and mime != "application/json":
            raise invalid("responseSchema needs responseMimeType 'application/json'.")
        output = None
        if mime == "application/json":
            output = {"type": "json_schema", "schema": json_schema(schema), "name": "output", "strict": True} \
                if schema else {"type": "json_object"}
        tools = _tools(body)

        system = []
        instruction = _get(body, "systemInstruction", "system_instruction")
        if isinstance(instruction, str):
            system = [instruction]
        elif isinstance(instruction, dict):
            system = _parts(instruction.get("parts") or [], "system_instruction.parts", files, {})
            if any(not isinstance(p, str) for p in system):
                raise invalid("system_instruction: only text parts are supported.")
        turns, last_texts, waiting = [], [], {}
        for i, content in enumerate(contents):
            if not isinstance(content, dict):
                raise invalid(f"Invalid value at 'contents[{i}]': expected an object.")
            role = content.get("role") or "user"
            if role not in ("user", "model", "function", "tool"):
                raise invalid(f"Please use a valid role: user, model. (contents[{i}].role is '{role}')")
            parts = _parts(content.get("parts") or [], f"contents[{i}].parts", files, waiting)
            turns.append(Turn("assistant" if role == "model" else "user", parts))
            last_texts = [p for p in parts if isinstance(p, str)]
        require_last_user(turns, "contents")
        question, context = question_and_context(last_texts)
        ignored = [camel for camel, snake in IGNORED_CONFIG if _get(config, camel, snake) is not None]
        ignored += [name for name in ("safetySettings", "labels") if body.get(name) is not None]
        return ChatRequest(
            api=self.id, provider="", model=(model or "").removeprefix("models/"), system="\n\n".join(system),
            turns=turns, max_tokens=_get(config, "maxOutputTokens", "max_output_tokens"),
            stream=bool(options.get("sse")), body=body, question=question or "(tool results)", context=context,
            tools=tools, tool_choice=_tool_choice(body), response_format=output, ignored=ignored,
            sampling={camel: _get(config, camel, snake) for camel, snake in SAMPLING
                      if _get(config, camel, snake) is not None},
            options={"array": bool(options.get("array"))})

    def render(self, req: ChatRequest, result: Result):
        parts = ([{"text": result.text}] if result.text or not result.tool_calls else []) + \
            [_call_part(c) for c in result.tool_calls]
        data = _chunk(parts, result.model, _response_id(), FINISH.get(result.stop_reason, "STOP"), result.usage)
        # streamGenerateContent without alt=sse answers with a JSON array of chunks.
        return [data] if req.options.get("array") else data

    async def stream(self, req: ChatRequest, events: AsyncIterator[Event]) -> AsyncIterator[bytes]:
        response_id, model = _response_id(), req.model
        try:
            async for event in events:
                if isinstance(event, Start):
                    model = event.model
                elif isinstance(event, TextDelta):
                    yield _sse(_chunk([{"text": event.text}], model, response_id))
                elif isinstance(event, ToolUse):
                    yield _sse(_chunk([_call_part(event.call)], model, response_id))
                elif isinstance(event, Finish):
                    yield _sse(_chunk([{"text": ""}], model, response_id, FINISH.get(event.stop_reason, "STOP"),
                                      event.usage))
        except GatewayError as err:
            yield _sse(self._body(err)[1])
            raise

    def _body(self, err: GatewayError):
        if err.code in KEY_ERRORS:
            status, rpc, message = KEY_ERRORS[err.code]
        else:
            (status, rpc), message = ERRORS.get(err.kind, ERRORS["api_error"]), err.message
        return status, {"error": {"code": status, "message": message, "status": rpc}}

    def error(self, err: GatewayError, request_id: str):
        return self._body(err)
