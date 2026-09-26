"""What every API format looks like to the gateway, and small helpers they share."""

import json
from typing import AsyncIterator, Callable, List, Optional, Tuple

from ..core import Attachment, ChatRequest, Event, GatewayError, Result, ToolDef, Turn

# Looks up an uploaded file by any API's reference (file-x, file_011x, files/x, a Gemini file URI).
FileLookup = Callable[[str, str], Attachment]  # (reference, where in the request) -> attachment


class Dialect:
    """One API format: parses its requests, renders its responses, streams and errors."""

    id = ""               # e.g. "openai-chat"
    label = ""            # e.g. "OpenAI Chat Completions"
    native_provider = ""  # the provider that answers when the model name has no "provider/" prefix

    def parse(self, body: dict, model: Optional[str] = None, options: Optional[dict] = None,
              files: Optional[FileLookup] = None) -> ChatRequest:
        """Checks a request body and turns it into a ChatRequest. Raises GatewayError("invalid_request")."""
        raise NotImplementedError

    def render(self, req: ChatRequest, result: Result):
        """The full (non-streaming) response body."""
        raise NotImplementedError

    def stream(self, req: ChatRequest, events: AsyncIterator[Event]) -> AsyncIterator[bytes]:
        """The streaming response. On a GatewayError it sends the format's error event, then re-raises."""
        raise NotImplementedError

    def error(self, err: GatewayError, request_id: str) -> Tuple[int, dict]:
        """HTTP status and body for an error, in this format's shape."""
        raise NotImplementedError


def sse(data, event: Optional[str] = None) -> bytes:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n".encode()


def invalid(message: str) -> GatewayError:
    return GatewayError("invalid_request", message, retry=False)


def no_files(reference: str, where: str) -> Attachment:
    raise invalid(f"{where}: uploaded files are not available here.")


def question_and_context(texts: List[str]) -> Tuple[str, Optional[str]]:
    """RAG apps usually send the retrieved passages first and the question last."""
    if not texts:
        return "", None
    return texts[-1], ("\n\n".join(texts[:-1]) if len(texts) > 1 else None)


def require_last_user(turns: List[Turn], where: str) -> None:
    if not turns or turns[-1].role != "user":
        raise invalid(f"This gateway needs the last message in {where} to come from the user or carry tool "
                      "results (prefilling the answer is not supported).")
    last = turns[-1]
    if not last.text.strip() and len(last.parts) == len([p for p in last.parts if isinstance(p, str)]):
        raise invalid(f"The last message in {where} is empty.")


def json_schema(schema) -> dict:
    """A tool's parameters as plain JSON Schema (Gemini sends OpenAPI-style upper-case type names)."""
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if value is None:
                continue
            if key == "type" and isinstance(value, str):
                out[key] = value.lower()
            else:
                out[key] = json_schema(value)
        return out
    if isinstance(schema, list):
        return [json_schema(v) for v in schema]
    return schema


def check_tools(tools: List[ToolDef]) -> None:
    names = set()
    for tool in tools:
        if not tool.name or not all(c.isalnum() or c in "_-" for c in tool.name) or len(tool.name) > 50:
            raise invalid(f"Tool name '{tool.name}' must be 1-50 letters, digits, '_' or '-' on this gateway.")
        if tool.name in names:
            raise invalid(f"Tool names must be unique: '{tool.name}' appears twice.")
        names.add(tool.name)


def tool_result_text(content) -> str:
    """Tool results can be a string, a list of blocks, or any JSON value; the CLIs get text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(b, dict) and b.get("type") == "text" for b in content):
        return "\n\n".join(b.get("text", "") for b in content)
    return json.dumps(content, ensure_ascii=False)
