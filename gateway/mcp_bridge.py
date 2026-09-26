"""A tiny MCP server (stdio, JSON-RPC) that offers the client's tools to Claude Code or Codex.

Usage: GATEWAY_TOOLS='[...]' python mcp_bridge.py      (the tools as MCP tool definitions, in JSON)

It only lists the tools. When the model calls one, the gateway sees the call in the CLI's output,
returns it to the client (whose app runs the tool) and stops the CLI, so tools/call is never answered.
This file runs on its own, without the gateway package.
"""

import json
import os
import sys


def main() -> None:
    tools = json.loads(os.environ.get("GATEWAY_TOOLS") or "[]")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        method, message_id = message.get("method"), message.get("id")
        if message_id is None:
            continue  # a notification
        if method == "initialize":
            result = {"protocolVersion": (message.get("params") or {}).get("protocolVersion", "2025-06-18"),
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "gateway", "version": "1.0"}}
        elif method == "tools/list":
            result = {"tools": tools}
        elif method == "ping":
            result = {}
        elif method == "tools/call":
            continue  # handled by the client's app; the gateway stops the CLI before this matters
        else:
            reply = {"jsonrpc": "2.0", "id": message_id, "error": {"code": -32601, "message": "Method not found"}}
            print(json.dumps(reply), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result}), flush=True)


if __name__ == "__main__":
    main()
