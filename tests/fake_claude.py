#!/usr/bin/env python3
"""Stand-in for the `claude` CLI in tests. It prints stream-json lines shaped like Claude Code 2.1.283's.

FAKE_CLAUDE_SCENARIO picks the behaviour. FAKE_CLAUDE_RECORD is a file where each run appends
its argv, environment, working directory, stdin and system prompt, so tests can inspect them.
"""

import json
import os
import sys
import time

SCENARIO = os.environ.get("FAKE_CLAUDE_SCENARIO", "ok")
MODEL = "claude-haiku-4-5-20251001"


def out(obj):
    print(json.dumps(obj), flush=True)


def event(e):
    out({"type": "stream_event", "event": e, "session_id": "s", "parent_tool_use_id": None, "uuid": "u"})


def usage(input_tokens, output_tokens):
    # Includes the extra fields Claude Code adds (iterations, speed), which the gateway must drop.
    return {"input_tokens": input_tokens, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": output_tokens, "service_tier": "standard", "inference_geo": "not_available",
            "iterations": [{"input_tokens": input_tokens, "output_tokens": output_tokens}], "speed": "standard"}


def message(msg_id, text, stop_reason, thinking=True):
    event({"type": "message_start", "message": {
        "model": MODEL, "id": msg_id, "type": "message", "role": "assistant", "content": [], "container": None,
        "stop_reason": None, "stop_sequence": None, "stop_details": None, "usage": usage(350, 1)}})
    index = 0
    if thinking:
        event({"type": "content_block_start", "index": 0,
               "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
        event({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}})
        event({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}})
        event({"type": "content_block_stop", "index": 0})
        index = 1
    event({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})
    for i in range(0, len(text), 7):
        event({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text[i:i + 7]}})
    event({"type": "content_block_stop", "index": index})
    event({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None,
                                              "stop_details": None, "container": None},
           "usage": usage(350, 12)})
    event({"type": "message_stop"})


def init(api_key_source="none"):
    out({"type": "system", "subtype": "init", "model": MODEL, "apiKeySource": api_key_source, "tools": []})


def rate_limit(status="allowed"):
    out({"type": "rate_limit_event", "rate_limit_info": {
        "status": status, "resetsAt": 1790654400, "rateLimitType": "seven_day", "utilization": 0.5,
        "unifiedWindows": {"five_hour": {"utilization": 0.1, "resetsAt": 1790431200},
                           "seven_day": {"utilization": 0.5, "resetsAt": 1790654400}}}})


def result(text, is_error=False, api_error_status=None):
    out({"type": "result", "subtype": "success", "is_error": is_error, "api_error_status": api_error_status,
         "result": text, "stop_reason": "end_turn", "usage": usage(350, 12)})


def main():
    args = sys.argv[1:]
    if args == ["--version"]:
        print("2.1.283 (Claude Code)")
        return 0
    if args[:2] == ["auth", "status"]:
        auth = os.environ.get("FAKE_AUTH", "ok")
        info = {"loggedIn": auth != "out", "authMethod": "api_key" if auth == "api" else "claude.ai",
                "apiProvider": "firstParty", "subscriptionType": "pro", "email": "someone@example.com"}
        print(json.dumps(info, indent=2))
        return 0

    started_at = time.time()
    raw = sys.stdin.readline()  # --input-format stream-json: one user message
    read_at = time.time()
    message_in = json.loads(raw)["message"]
    content = message_in["content"]
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    prompt = blocks[0]["text"]
    mcp = None
    if "--mcp-config" in args:
        with open(args[args.index("--mcp-config") + 1], encoding="utf-8") as f:
            mcp = json.load(f)
    system = None
    if "--system-prompt-file" in args:
        with open(args[args.index("--system-prompt-file") + 1], encoding="utf-8") as f:
            system = f.read()
    elif "--system-prompt" in args:
        system = args[args.index("--system-prompt") + 1]
    record = os.environ.get("FAKE_CLAUDE_RECORD")
    if record:
        with open(record, "a", encoding="utf-8") as f:
            f.write(json.dumps({"cli": "claude", "argv": args, "env": dict(os.environ), "cwd": os.getcwd(),
                                "cwd_files": sorted(os.listdir(".")), "stdin": prompt, "system": system,
                                "blocks": [{k: v for k, v in b.items() if k != "source"} | (
                                    {"source_type": b["source"]["type"], "media_type": b["source"].get("media_type"),
                                     "size": len(b["source"].get("data", ""))} if "source" in b else {})
                                           for b in blocks],
                                "mcp": mcp, "pid": os.getpid(), "started_at": started_at, "read_at": read_at}) + "\n")

    answer = "Echo: " + prompt[:40]
    if SCENARIO == "ok":
        init()
        message("msg_01FakeAAAAAAAAAAAAAAAAAAA", answer, "end_turn")
        out({"type": "assistant", "message": {"id": "msg_01FakeAAAAAAAAAAAAAAAAAAA", "content": []}})
        rate_limit()
        result(answer)
        return 0
    if SCENARIO == "structured":
        # Like `claude --json-schema`: some text, then the output through Claude Code's StructuredOutput tool.
        init()
        event({"type": "message_start", "message": {
            "model": MODEL, "id": "msg_01JsonAAAAAAAAAAAAAAAAAAA", "type": "message", "role": "assistant", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": usage(700, 1)}})
        event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Here it is:"}})
        event({"type": "content_block_stop", "index": 0})
        event({"type": "content_block_start", "index": 1, "content_block": {
            "type": "tool_use", "id": "toolu_01Structured", "name": "StructuredOutput", "input": {}}})
        event({"type": "content_block_delta", "index": 1,
               "delta": {"type": "input_json_delta", "partial_json": '{"city": "Paris", "country": "France"}'}})
        event({"type": "content_block_stop", "index": 1})
        event({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": usage(700, 20)})
        event({"type": "message_stop"})
        time.sleep(8)
        return 0
    if SCENARIO == "tool_use":
        # The model calls the client's tool (offered through the gateway's MCP server), then Claude Code
        # would run it; the gateway must stop here and hand the call to the client.
        init()
        event({"type": "message_start", "message": {
            "model": MODEL, "id": "msg_01ToolAAAAAAAAAAAAAAAAAAA", "type": "message", "role": "assistant", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": usage(900, 1)}})
        event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Checking."}})
        event({"type": "content_block_stop", "index": 0})
        event({"type": "content_block_start", "index": 1, "content_block": {
            "type": "tool_use", "id": "toolu_01FakeToolAAAAAAAAAAAAA", "name": "mcp__gateway__get_weather", "input": {}}})
        event({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"city": '}})
        event({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'}})
        event({"type": "content_block_stop", "index": 1})
        event({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
               "usage": usage(900, 30)})
        event({"type": "message_stop"})
        time.sleep(8)  # waiting for the MCP tool, which never answers
        return 0
    if SCENARIO == "max_tokens":
        # Like Claude Code: after max_tokens it adds a user turn and calls the model again.
        init()
        message("msg_01FirstAAAAAAAAAAAAAAAAA", "first part", "max_tokens", thinking=False)
        rate_limit()
        out({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Output token limit hit."}]}})
        message("msg_01SecondAAAAAAAAAAAAAAAA", "second part", "end_turn", thinking=False)
        result("first part second part")
        return 0
    if SCENARIO == "not_logged_in":
        init()
        result("Not logged in · Please run /login", is_error=True)
        return 1
    if SCENARIO == "bad_model":
        init()
        print('[claude-code:unrecognized_model] {"model":"nope"}', file=sys.stderr)
        result("There's an issue with the selected model (nope).", is_error=True, api_error_status=404)
        return 1
    if SCENARIO == "rate_limited":
        init()
        rate_limit("rejected")
        result("You've hit your limit", is_error=True, api_error_status=429)
        return 1
    if SCENARIO == "api_key_source":
        init(api_key_source="ANTHROPIC_API_KEY")
        message("msg_01ShouldNotBeSeenAAAAAAA", answer, "end_turn")
        result(answer)
        return 0
    if SCENARIO == "garbage":
        print("this is not json")
        return 1
    if SCENARIO == "midstream_error":
        init()
        event({"type": "message_start", "message": {
            "model": MODEL, "id": "msg_01MidAAAAAAAAAAAAAAAAAAAA", "type": "message", "role": "assistant",
            "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage(350, 1)}})
        event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}})
        result("Overloaded", is_error=True, api_error_status=529)
        return 1
    print(f"unknown scenario {SCENARIO}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
