"""OpenAI Chat Completions and Responses formats, answered by the fake Codex CLI."""

import time
from pathlib import Path

import openai
import pytest

from conftest import KEY, history, sse_data

MESSAGES = [{"role": "system", "content": "Answer from the context only."},
            {"role": "user", "content": "What is RAG?"}]


def sdk(client):
    return openai.OpenAI(api_key=KEY, base_url="http://testserver/v1", http_client=client, max_retries=0)


# ---------- Chat Completions ----------

def test_chat_completion_matches_sdk_types(gateway):
    client = gateway()
    completion = sdk(client).chat.completions.create(model="gpt-6-luna", messages=MESSAGES)
    assert completion.object == "chat.completion" and completion.id.startswith("chatcmpl-")
    assert completion.model == "gpt-6-luna"
    choice = completion.choices[0]
    assert choice.message.role == "assistant" and choice.message.content == "Codex echo: What is RAG?"
    assert choice.finish_reason == "stop"
    assert completion.usage.prompt_tokens == 4511 and completion.usage.completion_tokens == 9
    assert completion.usage.prompt_tokens_details.cached_tokens == 1792
    assert completion.usage.completion_tokens_details.reasoning_tokens == 2


def test_chat_streaming_with_usage(gateway):
    client = gateway(cache_enabled=False)
    stream = sdk(client).chat.completions.create(model="gpt-6-luna", messages=MESSAGES, stream=True,
                                                 stream_options={"include_usage": True})
    chunks = list(stream)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Codex echo: What is RAG?"
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "stop"
    assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens == 4520

    raw = client.post("/v1/chat/completions", headers={"authorization": f"Bearer {KEY}"},
                      json={"model": "gpt-6-luna", "messages": MESSAGES, "stream": True})
    assert raw.text.rstrip().endswith("data: [DONE]")
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in sse_data(raw.text))


def test_codex_is_isolated_and_uses_chatgpt_login(gateway):
    client = gateway()
    sdk(client).chat.completions.create(model="gpt-6-luna", messages=MESSAGES, reasoning_effort="low")
    run = gateway.runs("codex")[-1]
    argv, env = run["argv"], run["env"]
    for flag in ("exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules"):
        assert flag in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("--model") + 1] == "gpt-6-luna"
    for feature in ("shell_tool", "unified_exec", "apps", "browser_use", "computer_use"):
        assert feature in argv
    assert 'forced_login_method="chatgpt"' in argv and 'model_reasoning_effort="low"' in argv
    assert argv[-1] == "-"  # prompt on stdin
    assert run["system"] == "Answer from the context only." and run["stdin"] == "What is RAG?"
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        assert name not in env
    assert Path(run["cwd"]).name.startswith("gateway-codex-") and not Path(run["cwd"]).exists()


def test_codex_gets_neutral_instructions_without_system_prompt(gateway):
    client = gateway()
    sdk(client).chat.completions.create(model="gpt-6-luna", messages=[{"role": "user", "content": "hi"}])
    assert gateway.runs("codex")[-1]["system"] == "You are a helpful assistant."


def test_unknown_model_fails_fast(gateway):
    client = gateway(codex="bad_model")
    started = time.monotonic()
    with pytest.raises(openai.NotFoundError) as caught:
        sdk(client).chat.completions.create(model="gpt-nope", messages=MESSAGES)
    assert time.monotonic() - started < 5  # did not wait for Codex's own retries
    assert caught.value.body["code"] == "model_not_found"


def test_usage_limit_is_a_rate_limit_error(gateway):
    client = gateway(codex="usage_limit")
    with pytest.raises(openai.RateLimitError) as caught:
        sdk(client).chat.completions.create(model="gpt-6-luna", messages=MESSAGES)
    assert "usage limit" in caught.value.body["message"]


@pytest.mark.parametrize("extra, message", [
    ({"foo": 1}, "Unrecognized request argument supplied: foo"),
    ({"tools": [{"type": "custom", "custom": {"name": "f"}}]}, "only 'function' tools"),
    ({"n": 2}, "'n' must be 1"),
    ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]},
     "only data: URLs or uploaded files"),
])
def test_chat_bad_requests(gateway, extra, message):
    client = gateway()
    r = client.post("/v1/chat/completions", headers={"authorization": f"Bearer {KEY}"},
                    json={"model": "gpt-6-luna", "messages": MESSAGES, **extra})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert message in r.json()["error"]["message"]


def test_openai_key_errors(gateway):
    client = gateway()
    missing = client.post("/v1/chat/completions", json={"model": "gpt-6-luna", "messages": MESSAGES})
    assert missing.status_code == 401 and "didn't provide an API key" in missing.json()["error"]["message"]
    wrong = openai.OpenAI(api_key="nope", base_url="http://testserver/v1", http_client=client, max_retries=0)
    with pytest.raises(openai.AuthenticationError) as caught:
        wrong.chat.completions.create(model="gpt-6-luna", messages=MESSAGES)
    assert caught.value.body["code"] == "invalid_api_key"


# ---------- Responses ----------

def test_responses_matches_sdk_types(gateway):
    client = gateway()
    response = sdk(client).responses.create(model="gpt-6-luna", instructions="Answer from the context only.",
                                            input="What is RAG?")
    assert response.object == "response" and response.status == "completed" and response.id.startswith("resp_")
    assert response.output_text == "Codex echo: What is RAG?"
    assert response.output[0].type == "message" and response.output[0].content[0].type == "output_text"
    assert response.usage.input_tokens == 4511 and response.usage.output_tokens_details.reasoning_tokens == 2
    assert gateway.runs("codex")[-1]["system"] == "Answer from the context only."


def test_responses_streaming(gateway):
    client = gateway(cache_enabled=False)
    with sdk(client).responses.stream(model="gpt-6-luna", input=[
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": [{"type": "input_text", "text": "Context:\nRAG retrieves documents."},
                                         {"type": "input_text", "text": "What is RAG?"}]}]) as stream:
        deltas = [e.delta for e in stream if e.type == "response.output_text.delta"]
        final = stream.get_final_response()
    assert "".join(deltas) == final.output_text and final.status == "completed"
    item = history(client)[0]
    assert item["provider"] == "codex" and item["api"] == "openai-responses"
    assert item["question"] == "What is RAG?" and item["context"] == "Context:\nRAG retrieves documents."

    raw = client.post("/v1/responses", headers={"authorization": f"Bearer {KEY}"},
                      json={"model": "gpt-6-luna", "input": "hi", "stream": True})
    events = [e["type"] for e in sse_data(raw.text)]
    assert events[:4] == ["response.created", "response.in_progress", "response.output_item.added",
                          "response.content_part.added"]
    assert events[-1] == "response.completed"


def test_responses_bad_requests(gateway):
    client = gateway()
    for extra, message in (({"previous_response_id": "resp_1"}, "does not store responses"),
                           ({"foo": 1}, "Unknown parameter: 'foo'.")):
        r = client.post("/v1/responses", headers={"authorization": f"Bearer {KEY}"},
                        json={"model": "gpt-6-luna", "input": "hi", **extra})
        assert r.status_code == 400 and message in r.json()["error"]["message"]
