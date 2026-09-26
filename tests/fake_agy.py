#!/usr/bin/env python3
"""Stand-in for the Antigravity CLI (agy) in tests. Prints stream-json lines shaped like agy 1.2.7's.

FAKE_AGY_SCENARIO picks the behaviour; FAKE_AGY_LOGIN=none makes `agy models` fail. Each run appends its
argv, environment, working directory, stdin and agent file to FAKE_CLAUDE_RECORD.
"""

import json
import os
import sys
import time

SCENARIO = os.environ.get("FAKE_AGY_SCENARIO", "ok")


def out(obj):
    print(json.dumps(obj), flush=True)


def main():
    args = sys.argv[1:]
    if args == ["--version"]:
        print("1.2.7")
        return 0
    if args == ["models"]:
        if os.environ.get("FAKE_AGY_LOGIN") == "none":
            print("error: you are not signed in", file=sys.stderr)
            return 1
        print("Fetching available models...")
        print("gemini-3.8-flash-low\tGemini 3.8 Flash (Low)")
        print("gemini-3.1-pro-high\tGemini 3.1 Pro (High)")
        return 0
    if "-p=" not in args:
        print(f"unexpected arguments {args}", file=sys.stderr)
        return 2

    started_at = time.time()
    message = json.loads(sys.stdin.readline())
    read_at = time.time()
    agent_path = os.path.join(".agents", "agents", "gateway.md")
    agent = open(agent_path, encoding="utf-8").read() if os.path.exists(agent_path) else None
    record = os.environ.get("FAKE_CLAUDE_RECORD")
    if record:
        with open(record, "a", encoding="utf-8") as f:
            f.write(json.dumps({"cli": "agy", "argv": args, "env": dict(os.environ), "cwd": os.getcwd(),
                                "stdin": message, "agent": agent, "pid": os.getpid(), "started_at": started_at,
                                "read_at": read_at}) + "\n")

    model = args[args.index("--model") + 1]
    prompt = message["message"]["content"]
    if SCENARIO == "ok":
        out({"event": "init", "conversation_id": "c1", "init": {"model": model, "cwd": os.getcwd(), "agent": "gateway"}})
        out({"event": "step_update", "step_update": {"step_index": 0, "state": "DONE", "step_type": "user_input"}})
        out({"event": "step_update", "step_update": {"step_index": 1, "state": "ACTIVE", "step_type": "agent_response",
                                                     "text_delta": "Gemini echo: "}})
        out({"event": "step_update", "step_update": {"step_index": 1, "state": "ACTIVE", "step_type": "agent_response",
                                                     "text_delta": prompt[:40]}})
        usage = {"input_tokens": 564, "output_tokens": 128, "thinking_tokens": 124, "cache_read_tokens": 0,
                 "total_tokens": 692}
        out({"event": "step_update", "step_update": {"step_index": 1, "state": "DONE", "step_type": "agent_response",
                                                     "text_delta": "\n", "usage": usage}})
        out({"event": "result", "result": {"conversation_id": "c1", "status": "SUCCESS",
                                           "response": "Gemini echo: " + prompt[:40] + "\n", "usage": usage}})
        return 0
    if SCENARIO == "json_fenced":
        out({"event": "init", "conversation_id": "c1", "init": {"model": model}})
        for piece in ("```json\n", '{"city": ', '"Paris"}', "\n```\n"):
            out({"event": "step_update", "step_update": {"step_index": 1, "state": "ACTIVE",
                                                         "step_type": "agent_response", "text_delta": piece}})
        usage = {"input_tokens": 600, "output_tokens": 10, "thinking_tokens": 0, "cache_read_tokens": 0}
        out({"event": "result", "result": {"conversation_id": "c1", "status": "SUCCESS", "response": "", "usage": usage}})
        return 0
    if SCENARIO == "tool_call":
        out({"event": "init", "conversation_id": "c1", "init": {"model": model}})
        for piece in ("<tool_", 'calls>\n[{"name": "get_weather", "arguments": {"city": "Paris"}}]\n', "</tool_calls>\n"):
            out({"event": "step_update", "step_update": {"step_index": 1, "state": "ACTIVE",
                                                         "step_type": "agent_response", "text_delta": piece}})
        usage = {"input_tokens": 735, "output_tokens": 40, "thinking_tokens": 0, "cache_read_tokens": 0}
        out({"event": "result", "result": {"conversation_id": "c1", "status": "SUCCESS", "response": "", "usage": usage}})
        return 0
    if SCENARIO == "bad_model":
        error = f'invalid model selection (--model "{model}"): model {model} is not recognized as a known model'
        print("Error: " + error, file=sys.stderr)
        out({"event": "result", "result": {"conversation_id": "", "status": "ERROR", "response": "", "error": error}})
        return 1
    if SCENARIO == "quota":
        out({"event": "init", "conversation_id": "c1", "init": {"model": model}})
        out({"event": "result", "result": {"conversation_id": "c1", "status": "ERROR", "response": "",
                                           "error": "RESOURCE_EXHAUSTED: quota exceeded for this account"}})
        return 3
    print(f"unknown scenario {SCENARIO}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
