#!/usr/bin/env python3
"""Stand-in for the Codex CLI in tests. Prints `codex exec --json` lines shaped like codex-cli 0.157.1's.

FAKE_CODEX_SCENARIO picks the behaviour; FAKE_CODEX_LOGIN the login status. Each exec run appends its argv,
environment, working directory, stdin and instructions file to FAKE_CLAUDE_RECORD.
"""

import json
import os
import sys
import time

SCENARIO = os.environ.get("FAKE_CODEX_SCENARIO", "ok")


def out(obj):
    print(json.dumps(obj), flush=True)


def main():
    args = sys.argv[1:]
    if args == ["--version"]:
        print("codex-cli 0.157.1")
        return 0
    if args[:2] == ["login", "status"]:
        login = os.environ.get("FAKE_CODEX_LOGIN", "chatgpt")
        print({"chatgpt": "Logged in using ChatGPT", "apikey": "Logged in using an API key - sk-proj-***"}
              .get(login, "Not logged in"), file=sys.stderr)
        return 0 if login in ("chatgpt", "apikey") else 1
    if args[:2] == ["debug", "models"]:
        print(json.dumps({"models": [
            {"slug": "gpt-6-luna", "display_name": "GPT-6-Luna", "visibility": "list"},
            {"slug": "gpt-reserve", "display_name": "GPT-Reserve", "visibility": "hide"},
            {"slug": "gpt-5.6-terra", "display_name": "GPT-5.6-Terra", "visibility": "list"}]}))
        return 0
    if args[:1] != ["exec"]:
        print(f"unexpected arguments {args}", file=sys.stderr)
        return 2

    prompt = sys.stdin.read()
    instructions = None
    config = [args[i + 1] for i, arg in enumerate(args) if arg == "-c"]
    for value in config:
        if value.startswith("model_instructions_file="):
            with open(json.loads(value.split("=", 1)[1]), encoding="utf-8") as f:
                instructions = f.read()
    images = [args[i + 1] for i, arg in enumerate(args) if arg == "--image"]
    schema = None
    if "--output-schema" in args:
        with open(args[args.index("--output-schema") + 1], encoding="utf-8") as f:
            schema = json.load(f)
    record = os.environ.get("FAKE_CLAUDE_RECORD")
    if record:
        with open(record, "a", encoding="utf-8") as f:
            f.write(json.dumps({"cli": "codex", "argv": args, "env": dict(os.environ), "cwd": os.getcwd(),
                                "cwd_files": sorted(os.listdir(".")), "stdin": prompt, "system": instructions,
                                "config": config, "images": [[p, os.path.getsize(p)] for p in images],
                                "schema": schema}) + "\n")

    out({"type": "thread.started", "thread_id": "01a0ddab-0000-7000-8000-000000000000"})
    out({"type": "turn.started"})
    if SCENARIO == "ok":
        out({"type": "item.completed", "item": {"id": "item_0", "type": "error",
                                                "message": "Codex is ignoring 1 unrecognized configuration setting."}})
        out({"type": "item.completed", "item": {"id": "item_1", "type": "agent_message",
                                                "text": "Codex echo: " + prompt[:40]}})
        out({"type": "turn.completed", "usage": {"input_tokens": 4511, "cached_input_tokens": 1792,
                                                 "cache_write_input_tokens": 0, "output_tokens": 9,
                                                 "reasoning_output_tokens": 2}})
        return 0
    if SCENARIO == "json_fenced":
        out({"type": "item.completed", "item": {"id": "item_0", "type": "agent_message",
                                                "text": '```json\n{"city": "Paris"}\n```'}})
        out({"type": "turn.completed", "usage": {"input_tokens": 4400, "output_tokens": 12}})
        return 0
    if SCENARIO == "bad_schema":
        upstream = {"type": "error", "error": {"type": "invalid_request_error", "code": "invalid_json_schema",
                    "message": "Invalid schema for response_format 'codex_output_schema': 'additionalProperties' "
                               "is required to be supplied and to be false."}}
        out({"type": "error", "message": json.dumps(upstream, indent=2)})
        out({"type": "turn.failed", "error": {"message": json.dumps(upstream, indent=2)}})
        return 1
    if SCENARIO == "tool_call":
        out({"type": "item.started", "item": {"id": "item_0", "type": "mcp_tool_call", "server": "gateway",
                                              "tool": "get_weather", "arguments": {"city": "Paris"},
                                              "result": None, "error": None, "status": "in_progress"}})
        time.sleep(8)  # waiting for approval / the tool; the gateway must not wait
        return 0
    if SCENARIO == "bad_model":
        # Codex retries a 404 many times; the gateway must stop at the first one.
        out({"type": "error", "message": "Reconnecting... 1/5 (unexpected status 404 Not Found: The model "
                                         "`gpt-nope` does not exist or you do not have access to it.)"})
        time.sleep(8)
        out({"type": "turn.failed", "error": {"message": "unexpected status 404 Not Found"}})
        return 1
    if SCENARIO == "usage_limit":
        out({"type": "turn.failed", "error": {"message": "You've hit your usage limit. Try again later."}})
        return 1
    if SCENARIO == "garbage":
        print("this is not json")
        return 1
    print(f"unknown scenario {SCENARIO}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
