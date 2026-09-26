"""OpenAI models through the Codex CLI (`codex exec`), using your ChatGPT login."""

import json
import logging
import os
import re
from typing import AsyncIterator, Optional

from .. import attachments
from ..core import (Attachment, ChatRequest, Event, Finish, GatewayError, Start, TextDelta, ToolCall, ToolUse,
                    Usage, render_prompt)
from ..process import CliRun, Launch, run_quick
from .base import (Provider, ProviderStatus, bridge_command, call_id, clean_json, json_note, mcp_tools_env,
                   offered_tools, tool_choice_note)

log = logging.getLogger("gateway.codex")

# Features that give Codex tools. Turned off so a request is prompt in, text out (checked with `codex features list`).
DISABLED_FEATURES = ("shell_tool", "unified_exec", "apps", "plugins", "browser_use", "computer_use",
                     "image_generation", "multi_agent", "hooks", "skill_search", "tool_suggest", "sleep_tool", "goals")

# Codex refuses an empty instructions file, so requests without a system prompt get this neutral one.
NEUTRAL_INSTRUCTIONS = "You are a helpful assistant."


def _usage(u: dict) -> Usage:
    return Usage(input_tokens=u.get("input_tokens") or 0, output_tokens=u.get("output_tokens") or 0,
                 cached_tokens=u.get("cached_input_tokens") or 0,
                 cache_write_tokens=u.get("cache_write_input_tokens") or 0,
                 reasoning_tokens=u.get("reasoning_output_tokens") or 0)


def _error_from(message: str, model: str) -> Optional[GatewayError]:
    """Maps a Codex error message to an API error. None when it is only a notice (for example a retry)."""
    status = None
    detail = message
    try:  # Codex sometimes forwards the upstream error as JSON
        data = json.loads(message)
        status = data.get("status")
        detail = (data.get("error") or {}).get("message") or message
        if (data.get("error") or {}).get("type") == "invalid_request_error" and not status:
            status = 400  # for example an invalid schema for structured output
    except (ValueError, AttributeError):
        match = re.search(r"status (\d{3})", message) or re.search(r"\b([45]\d\d) [A-Z][a-z]", message)
        status = int(match.group(1)) if match else None
    low = message.lower()
    if "model" in low and (status == 404 or "does not exist" in low or "not supported" in low):
        return GatewayError("not_found", f"The model `{model}` does not exist or you do not have access to it.",
                            retry=False)
    if status == 401 or "not logged in" in low or "unauthorized" in low:
        return GatewayError("authentication", "Codex on the gateway host is not logged in, or its login expired. "
                            "Run `codex login` and sign in with ChatGPT.", retry=False)
    if status == 429 or "usage limit" in low:
        return GatewayError("rate_limit", detail[:500], retry=False)
    if status == 403:
        return GatewayError("permission", detail[:500], retry=False)
    if status == 400:
        return GatewayError("invalid_request", detail[:500], retry=False)
    if status and status >= 500:
        return GatewayError("overloaded" if status in (503, 529) else "api_error", detail[:500])
    return None


class Codex(Provider):
    id = "codex"
    name = "Codex"
    vendor = "OpenAI"
    apis = ["openai-chat", "openai-responses"]
    default_model = "gpt-6-luna"
    honours = frozenset({"reasoning_effort"})
    tool_calling = "native"

    def check(self) -> ProviderStatus:
        status = ProviderStatus()
        if not self.found():
            status.problem = f"`{self.binary}` is not installed. Install it with: npm install -g @openai/codex"
            return status
        status.installed = True
        status.version = run_quick([*self.command, "--version"]).stdout.strip().replace("codex-cli ", "")
        out = run_quick([*self.command, "login", "status"])
        login = (out.stdout + out.stderr).strip()
        if "chatgpt" in login.lower():
            status.ready = True
            status.account = "ChatGPT login"
        elif "api key" in login.lower():
            status.problem = "Logged in with an API key, which bills the OpenAI API. Run `codex login` and choose ChatGPT."
        else:
            status.problem = "Not logged in. Run `codex login` and sign in with ChatGPT."
        try:
            catalog = json.loads(run_quick([*self.command, "debug", "models"], timeout=60).stdout)
            status.models = [{"id": m["slug"], "label": m.get("display_name") or m["slug"]}
                             for m in catalog.get("models", []) if m.get("visibility") == "list"]
        except (ValueError, KeyError, OSError):
            pass
        if status.models:
            self.default_model = status.models[0]["id"]
        return status

    def launch(self, req: ChatRequest):
        """Codex reads the prompt before it starts, so it gains nothing from being started early."""
        tools = offered_tools(req)
        images = {}

        def show(n: int, a: Attachment) -> str:
            if a.kind == "image":
                images[f"attachments/{n}{attachments.extension(a.media_type)}"] = a.data
                return f"[Image {n}" + (f": {a.filename}" if a.filename else "") + " (attached)]"
            name = f' name="{a.filename}"' if a.filename else ""
            return f"<document{name}>\n{attachments.document_text(a)}\n</document>"

        prompt = render_prompt(req.turns, show)
        instructions = (req.system or NEUTRAL_INSTRUCTIONS) + (tool_choice_note(req) if tools else "")
        fmt = req.response_format
        schema = None
        if fmt and fmt["type"] == "json_schema" and fmt.get("strict"):
            schema = json.dumps(fmt["schema"])  # enforced by Codex, with OpenAI's strict-schema rules
        else:
            instructions += json_note(req)  # like the API's non-strict mode: asked for, not enforced
        cmd = [*self.command, "exec", "--json", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
               "--ignore-rules", "--sandbox", "read-only", "--cd", "{dir}"]
        for feature in DISABLED_FEATURES:
            cmd += ["--disable", feature]
        cmd += ["-c", 'web_search="disabled"', "-c", "include_environment_context=false",
                # Refuse API-key logins: requests must use the ChatGPT subscription.
                "-c", 'forced_login_method="chatgpt"',
                # The request's system prompt replaces Codex's built-in coding-agent instructions.
                "-c", "model_instructions_file=" + json.dumps("{dir}/instructions.md"),
                "--model", req.model]
        if req.reasoning_effort:
            cmd += ["-c", f"model_reasoning_effort={json.dumps(req.reasoning_effort)}"]
        if tools:
            command = bridge_command()
            cmd += ["-c", f"mcp_servers.gateway.command={json.dumps(command[0])}",
                    "-c", f"mcp_servers.gateway.args={json.dumps(command[1:])}",
                    "-c", "mcp_servers.gateway.env={GATEWAY_TOOLS=" + json.dumps(mcp_tools_env(tools)["GATEWAY_TOOLS"]) + "}"]
        if schema:
            cmd += ["--output-schema", "{dir}/schema.json"]
        for path in images:
            cmd += ["--image", "{dir}/" + path]
        cmd.append("-")  # prompt on stdin
        # A home folder of its own, so Codex does not list your personal skills in every prompt.
        # CODEX_HOME still points at the real one, where the login is.
        extra = {"HOME": "{dir}/home", "CODEX_HOME": os.environ.get("CODEX_HOME")
                 or os.path.join(os.path.expanduser("~"), ".codex")}
        files = {"instructions.md": instructions, "home/.keep": "", **images}
        if schema:
            files["schema.json"] = schema
        return Launch(self.id, cmd, extra, files, warm=False), prompt

    async def run(self, req: ChatRequest, request_id: str) -> AsyncIterator[Event]:
        launch, prompt = self.launch(req)
        started = await self.pool.take(launch, request_id)
        async with CliRun(started, stdin=prompt.encode("utf-8"), timeout=self.settings.request_timeout,
                          request_id=request_id) as run:
            began = False
            messages = 0
            async for obj in run.json_lines():
                kind = obj.get("type")
                item = obj.get("item") or {}
                if kind in ("turn.started", "item.started", "item.completed") and not began:
                    began = True
                    yield Start(req.model)
                if kind == "item.started" and item.get("type") == "mcp_tool_call" and item.get("server") == "gateway":
                    # The model called one of the client's tools: hand the call back, do not run it.
                    arguments = item.get("arguments")
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments or "{}")
                    yield ToolUse(ToolCall(call_id(), item.get("tool", ""), arguments or {}))
                    yield Finish("tool_use", Usage())  # Codex reports usage only for a finished turn
                    return
                if kind == "item.completed":
                    if item.get("type") == "agent_message":
                        # Codex gives each message whole, so a stream gets it in one piece.
                        text = item.get("text") or ""
                        yield TextDelta(("\n\n" if messages else "") + (clean_json(text) if req.response_format else text))
                        messages += 1
                    elif item.get("type") == "error":
                        log.info("[%s] codex notice: %s", request_id, str(item.get("message"))[:300])
                elif kind == "error":
                    message = str(obj.get("message") or "")
                    err = _error_from(message, req.model)
                    if err is not None and not err.retry:
                        raise err  # a client error: Codex's own retries cannot fix it
                    log.info("[%s] codex: %s", request_id, message[:300])
                elif kind == "turn.completed":
                    run.completed = True
                    yield Finish("end_turn", _usage(obj.get("usage") or {}))
                    return
                elif kind == "turn.failed":
                    message = str((obj.get("error") or {}).get("message") or "")
                    raise _error_from(message, req.model) or GatewayError("api_error", message[:500] or
                                                                          "Codex failed to produce a response.")
            await run.finish()
            raise _error_from(run.stderr, req.model) or GatewayError("api_error", "Codex exited without a response.")
