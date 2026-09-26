"""Minimal POC: Anthropic Messages API -> local FastAPI -> `claude -p` (Claude Code subscription auth)."""

import json
import logging
import os
import shlex
import subprocess
import tempfile
import uuid
from typing import List, Literal, Optional, Union

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateway")

CLAUDE_TIMEOUT_SECONDS = 300

# Flags verified against `claude --help` (Claude Code 2.1.283):
#   -p / --output-format json   non-interactive, one JSON result object on stdout
#   --tools ""                  disable every built-in tool (plain prompt -> text)
#   --safe-mode                 skip CLAUDE.md, hooks, skills, plugins, MCP servers; auth still works
#   --strict-mcp-config         no MCP servers (none passed via --mcp-config)
#   --disable-slash-commands    no skills
#   --no-session-persistence    don't save these calls as resumable sessions
# Not used: --bare, because it ignores OAuth and only accepts ANTHROPIC_API_KEY.
CLAUDE_CMD = [
    "claude", "-p",
    "--output-format", "json",
    "--tools", "",
    "--safe-mode",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--no-session-persistence",
]


# ---------- startup auth check ----------

if os.environ.get("ANTHROPIC_API_KEY"):
    bar = "!" * 78
    print(f"\n{bar}\n"
          "WARNING: ANTHROPIC_API_KEY is set in this shell.\n"
          "If Claude Code saw it, it would bill the API key instead of your subscription.\n"
          "This gateway REMOVES it from the environment of the `claude` subprocess only.\n"
          "Your own shell environment is not changed. Unset it to silence this warning:\n"
          "    unset ANTHROPIC_API_KEY\n"
          f"{bar}\n", flush=True)


# ---------- request models ----------

class TextBlock(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[TextBlock]]


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    system: Optional[Union[str, List[TextBlock]]] = None
    messages: List[Message]


# ---------- helpers ----------

def to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(block.text for block in content)


def flatten(req: MessagesRequest) -> str:
    parts = []
    system = to_text(req.system).strip()
    if system:
        parts.append(f"SYSTEM:\n{system}")
    for msg in req.messages:
        parts.append(f"{msg.role.upper()}:\n{to_text(msg.content).strip()}")
    parts.append("Reply as ASSISTANT to the last USER message. Output only the reply text.")
    return "\n\n".join(parts)


def api_error(status: int, err_type: str, message: str) -> JSONResponse:
    # Same shape as real Anthropic errors, so the SDK raises the matching exception class.
    return JSONResponse(status_code=status,
                        content={"type": "error", "error": {"type": err_type, "message": message}})


def looks_unauthenticated(text: str) -> bool:
    text = text.lower()
    return any(s in text for s in ("not logged in", "please run /login", "invalid api key",
                                   "oauth token", "authentication_error"))


# ---------- app ----------

app = FastAPI(title="rag-gateway-poc")


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    log.warning("request validation failed: %s", exc.errors())
    return api_error(400, "invalid_request_error", f"Invalid request: {exc.errors()}")


@app.post("/v1/messages")
def create_message(req: MessagesRequest):
    prompt = flatten(req)

    child_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    log.info("COMMAND: %s   (prompt sent on stdin, %d chars)", shlex.join(CLAUDE_CMD), len(prompt))
    log.info("PROMPT:\n%s", prompt)

    try:
        with tempfile.TemporaryDirectory(prefix="claude-poc-") as empty_dir:
            proc = subprocess.run(CLAUDE_CMD, input=prompt, text=True, capture_output=True,
                                  cwd=empty_dir, env=child_env, timeout=CLAUDE_TIMEOUT_SECONDS)
    except FileNotFoundError:
        log.error("`claude` CLI not found on PATH")
        return api_error(500, "api_error", "Claude Code CLI is not installed on the gateway host.")
    except subprocess.TimeoutExpired:
        log.error("claude timed out after %ss", CLAUDE_TIMEOUT_SECONDS)
        return api_error(504, "timeout_error", "Claude Code timed out.")

    log.info("EXIT CODE: %s", proc.returncode)
    log.info("STDOUT:\n%s", proc.stdout)
    log.info("STDERR:\n%s", proc.stderr)

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        data = None

    # Claude Code reports most failures inside the JSON (is_error=true), sometimes with exit code != 0.
    if proc.returncode != 0 or data is None or data.get("is_error"):
        detail = f"{proc.stderr}\n{(data or {}).get('result', '')}"
        if looks_unauthenticated(detail):
            log.error("claude is not authenticated")
            return api_error(401, "authentication_error",
                             "Claude Code on the gateway host is not logged in. Run `claude` and /login.")
        if data is None:
            log.error("claude returned invalid JSON")
            return api_error(502, "api_error", "Claude Code returned output that is not valid JSON.")
        log.error("claude failed (exit %s)", proc.returncode)
        return api_error(502, "api_error", "Claude Code failed to produce a response.")

    text = (data.get("result") or "").strip()
    if not text:
        log.error("claude returned an empty result")
        return api_error(502, "api_error", "Claude Code returned an empty response.")

    # Real model name comes from modelUsage keys; fall back to what the client asked for.
    model_usage = data.get("modelUsage") or {}
    model = next(iter(model_usage), req.model)

    # Real token counts from Claude Code. Note: input includes Claude Code's own system prompt.
    usage = data.get("usage") or {}

    return {
        "id": f"msg_local_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": data.get("stop_reason") or "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        },
    }
