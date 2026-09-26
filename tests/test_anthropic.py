"""Anthropic Messages format, answered by the fake Claude Code."""

from pathlib import Path

import anthropic
import pytest

from conftest import KEY, history, sse_data
from gateway.dialects.anthropic import MESSAGE_FIELDS, USAGE_FIELDS

BODY = {"model": "claude-haiku-4-5-20251001", "max_tokens": 256,
        "messages": [{"role": "user", "content": "What is RAG?"}]}


def sdk(client):
    return anthropic.Anthropic(api_key=KEY, base_url="http://testserver", http_client=client, max_retries=0)


def post(client, body=None, **headers):
    return client.post("/v1/messages", json=body or BODY, headers={"x-api-key": KEY, **headers})


# ---------- format ----------

def test_non_streaming_matches_sdk_types(gateway):
    client = gateway()
    msg = sdk(client).messages.create(**BODY)
    assert msg.type == "message" and msg.role == "assistant"
    assert msg.id == "msg_01FakeAAAAAAAAAAAAAAAAAAA"  # the upstream id is passed through
    assert [b.type for b in msg.content] == ["text"]  # thinking was not requested, so it is removed
    assert msg.content[0].text == "Echo: What is RAG?"
    assert msg.stop_reason == "end_turn"
    assert msg.usage.input_tokens == 350 and msg.usage.output_tokens == 12

    raw = post(client).json()
    assert set(raw) <= MESSAGE_FIELDS
    assert set(raw["usage"]) <= USAGE_FIELDS  # Claude Code's extra usage fields are dropped


def test_streaming_matches_sdk_types(gateway):
    client = gateway(cache_enabled=False)
    with sdk(client).messages.stream(**BODY) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert text == "Echo: What is RAG?"
    assert final.content[0].text == text and final.stop_reason == "end_turn"

    events = sse_data(post(client, {**BODY, "stream": True}).text)
    kinds = [e["type"] for e in events]
    assert kinds[:3] == ["message_start", "ping", "content_block_start"]
    assert kinds[-2:] == ["message_delta", "message_stop"]
    assert {e["index"] for e in events if "index" in e} == {0}  # renumbered after removing thinking


def test_thinking_kept_when_requested(gateway):
    client = gateway()
    msg = sdk(client).messages.create(**{**BODY, "max_tokens": 4000},
                                      thinking={"type": "enabled", "budget_tokens": 2000})
    assert [b.type for b in msg.content] == ["thinking", "text"]
    assert gateway.runs("claude")[-1]["env"]["MAX_THINKING_TOKENS"] == "2000"


def test_max_tokens_stops_after_first_model_call(gateway):
    client = gateway(claude="max_tokens")
    msg = sdk(client).messages.create(**BODY)
    assert msg.stop_reason == "max_tokens"
    assert msg.content[0].text == "first part"
    assert msg.id == "msg_01FirstAAAAAAAAAAAAAAAAA"


# ---------- how Claude Code is run ----------

def test_claude_is_isolated_and_uses_subscription(gateway):
    client = gateway()
    post(client, {**BODY, "system": "Answer from the context only."})
    run = gateway.runs("claude")[-1]
    argv, env = run["argv"], run["env"]
    for flag in ("-p", "--safe-mode", "--strict-mcp-config", "--disable-slash-commands",
                 "--no-session-persistence", "--include-partial-messages"):
        assert flag in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--model") + 1] == BODY["model"]
    assert run["system"] == "Answer from the context only."  # sent as a file, not as an argument
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "CLAUDECODE"):
        assert name not in env
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "256" and env["MAX_THINKING_TOKENS"] == "0"
    assert run["stdin"] == "What is RAG?"
    assert Path(run["cwd"]).name.startswith("gateway-claude-") and run["cwd_files"] == ["system-prompt.txt"]
    assert not Path(run["cwd"]).exists()  # temp folder removed afterwards


def test_multi_turn_and_content_blocks(gateway):
    client = gateway()
    body = {**BODY, "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        {"role": "user", "content": [{"type": "text", "text": "Context:\nRAG retrieves documents."},
                                     {"type": "text", "text": "Question:\nWhat is RAG?"}]},
    ]}
    assert post(client, body).status_code == 200
    prompt = gateway.runs("claude")[-1]["stdin"]
    assert prompt.startswith("<conversation_history>\n<user>\nHi\n</user>\n\n<assistant>\nHello!\n</assistant>")
    assert prompt.endswith("Context:\nRAG retrieves documents.\n\nQuestion:\nWhat is RAG?")
    item = history(client)[0]
    assert item["question"] == "Question:\nWhat is RAG?" and item["context"] == "Context:\nRAG retrieves documents."
    assert item["provider"] == "claude" and item["api"] == "anthropic-messages"


def test_ignored_params_are_logged(gateway, caplog):
    r = post(gateway(), {**BODY, "temperature": 0.2, "top_p": 0.9})
    assert r.status_code == 200
    assert "ignored, Claude Code cannot set them: temperature, top_p" in caplog.text


def test_only_real_api_headers(gateway):
    r = post(gateway(), {**BODY, "temperature": 0.2})
    assert r.headers["request-id"].startswith("req_01")
    assert not [name for name in r.headers if name.startswith("x-gateway")]


def test_cache_hit_skips_claude_and_works_for_streaming(gateway):
    client = gateway()
    first = post(client)
    second = post(client)
    assert len(gateway.runs("claude")) == 1  # the second answer came from the cache
    assert second.json()["content"] == first.json()["content"] and second.json()["id"] != first.json()["id"]

    with sdk(client).messages.stream(**BODY) as stream:  # served from the cache as a stream
        assert "".join(stream.text_stream) == "Echo: What is RAG?"
    assert len(gateway.runs("claude")) == 1

    assert post(client, **{"cache-control": "no-cache"}).status_code == 200
    assert len(gateway.runs("claude")) == 2
    assert [item["cache"] for item in history(client)] == ["bypass", "hit", "hit", "miss"]


# ---------- errors ----------

@pytest.mark.parametrize("scenario, status, error_type", [
    ("not_logged_in", 401, "authentication_error"),
    ("bad_model", 404, "not_found_error"),
    ("rate_limited", 429, "rate_limit_error"),
    ("api_key_source", 500, "api_error"),
    ("garbage", 500, "api_error"),
    ("midstream_error", 529, "overloaded_error"),
])
def test_claude_failures_map_to_api_errors(gateway, scenario, status, error_type):
    client = gateway(claude=scenario, cache_enabled=False)
    r = post(client)
    assert r.status_code == status
    body = r.json()
    assert body["type"] == "error" and body["error"]["type"] == error_type
    assert body["request_id"] == r.headers["request-id"]
    with pytest.raises(anthropic.APIStatusError) as caught:
        sdk(client).messages.create(**BODY)
    assert caught.value.status_code == status


def test_error_after_stream_started_is_an_error_event(gateway):
    client = gateway(claude="midstream_error", cache_enabled=False)
    events = sse_data(post(client, {**BODY, "stream": True}).text)
    assert events[-1] == {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    with pytest.raises(anthropic.APIStatusError):
        with sdk(client).messages.stream(**BODY) as stream:
            "".join(stream.text_stream)


def test_api_key_source_is_refused_before_any_output(gateway):
    r = post(gateway(claude="api_key_source"))
    assert r.status_code == 500 and "API key" in r.json()["error"]["message"]
    assert r.headers["x-should-retry"] == "false"


@pytest.mark.parametrize("body, message", [
    ({"model": "m", "messages": BODY["messages"]}, "max_tokens: Field required"),
    ({**BODY, "unknown_field": 1}, "unknown_field: Extra inputs are not permitted"),
    ({**BODY, "messages": [{"role": "user", "content": [{"type": "server_tool_use", "id": "x"}]}]}, "'server_tool_use' content"),
    ({**BODY, "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}, "prefill"),
    ({**BODY, "tools": [{"type": "web_search_20250305", "name": "web_search"}]}, "server tools"),
])
def test_bad_requests(gateway, body, message):
    r = post(gateway(), body)
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert message in r.json()["error"]["message"]


def test_gateway_key_is_required(gateway):
    client = gateway()
    assert client.post("/v1/messages", json=BODY).json()["error"]["message"] == "x-api-key header is required"
    wrong = client.post("/v1/messages", json=BODY, headers={"x-api-key": "nope"})
    assert wrong.status_code == 401 and wrong.json()["error"]["message"] == "invalid x-api-key"
    bearer = client.post("/v1/messages", json=BODY, headers={"authorization": f"Bearer {KEY}"})
    assert bearer.status_code == 200
    assert len(gateway.runs("claude")) == 1  # only the authorised request reached Claude Code
