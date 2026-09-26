"""Structured (JSON) output, model lists and token counts (fake CLIs, official SDKs)."""

import json

import anthropic
import openai
import pytest
from google import genai
from google.genai import errors, types

from conftest import KEY
from gateway import counting

CITY = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"], "additionalProperties": False}
ASK = [{"role": "user", "content": "Capital of France?"}]


def claude_sdk(client):
    return anthropic.Anthropic(api_key=KEY, base_url="http://testserver", http_client=client, max_retries=0)


def openai_sdk(client):
    return openai.OpenAI(api_key=KEY, base_url="http://testserver/v1", http_client=client, max_retries=0)


def gemini_sdk(client):
    return genai.Client(api_key=KEY, http_options=types.HttpOptions(base_url="http://testserver", httpx_client=client))


# ---------- structured output ----------

def test_anthropic_structured_output(gateway):
    client = gateway(claude="structured")
    msg = claude_sdk(client).messages.create(model="claude-haiku-4-5-20251001", max_tokens=200, messages=ASK,
                                             output_config={"format": {"type": "json_schema", "schema": CITY},
                                                            "effort": "low"})
    # Like the API: the JSON is the answer; Claude Code's preamble and its StructuredOutput tool are not shown.
    assert [b.type for b in msg.content] == ["text"] and msg.stop_reason == "end_turn"
    assert json.loads(msg.content[0].text) == {"city": "Paris", "country": "France"}
    argv = gateway.runs("claude")[-1]["argv"]
    assert json.loads(argv[argv.index("--json-schema") + 1]) == CITY
    assert argv[argv.index("--effort") + 1] == "low"


def test_anthropic_structured_output_streams(gateway):
    client = gateway(claude="structured", cache_enabled=False)
    with claude_sdk(client).messages.stream(model="claude-haiku-4-5-20251001", max_tokens=200, messages=ASK,
                                            output_config={"format": {"type": "json_schema", "schema": CITY}}) as s:
        text = "".join(s.text_stream)
    assert json.loads(text) == {"city": "Paris", "country": "France"}


def test_openai_strict_schema_is_enforced_by_codex(gateway):
    client = gateway(codex="json_fenced")
    completion = openai_sdk(client).chat.completions.create(model="gpt-6-luna", messages=ASK, response_format={
        "type": "json_schema", "json_schema": {"name": "city", "schema": CITY, "strict": True}})
    assert json.loads(completion.choices[0].message.content) == {"city": "Paris"}  # code fences removed
    run = gateway.runs("codex")[-1]
    assert run["schema"] == CITY and "Reply with a single JSON" not in run["system"]


def test_openai_loose_schema_is_asked_for(gateway):
    client = gateway(codex="json_fenced")
    response = openai_sdk(client).responses.create(model="gpt-6-luna", input="Capital of France?", text={
        "format": {"type": "json_schema", "name": "city", "schema": {"type": "object"}}})
    assert json.loads(response.output_text) == {"city": "Paris"}
    run = gateway.runs("codex")[-1]
    assert run["schema"] is None and "Reply with a single JSON value that matches this JSON Schema" in run["system"]
    assert response.text.format.type == "json_schema"


def test_codex_schema_errors_are_bad_requests(gateway):
    client = gateway(codex="bad_schema")
    with pytest.raises(openai.BadRequestError) as caught:
        openai_sdk(client).chat.completions.create(model="gpt-6-luna", messages=ASK, response_format={
            "type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}, "strict": True}})
    assert "additionalProperties" in caught.value.body["message"]


def test_gemini_json_output(gateway):
    client = gateway(agy="json_fenced")
    response = gemini_sdk(client).models.generate_content(
        model="gemini-3.8-flash-low", contents="Capital of France?",
        config=types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=CITY))
    assert json.loads(response.text) == {"city": "Paris"}
    assert "Reply with a single JSON value" in gateway.runs("agy")[-1]["agent"]


# ---------- model lists ----------

def test_model_lists(gateway):
    client = gateway()
    assert "claude-haiku-4-5-20251001" in [m.id for m in claude_sdk(client).models.list()]
    assert claude_sdk(client).models.retrieve("claude-sonnet-5").display_name == "Sonnet 5"
    assert [m.id for m in openai_sdk(client).models.list()] == ["gpt-6-luna", "gpt-5.6-terra"]
    assert openai_sdk(client).models.retrieve("gpt-6-luna").owned_by == "openai"
    with pytest.raises(openai.NotFoundError):
        openai_sdk(client).models.retrieve("gpt-nope")
    names = [m.name for m in gemini_sdk(client).models.list()]
    assert names == ["models/gemini-3.8-flash-low", "models/gemini-3.1-pro-high"]
    assert gemini_sdk(client).models.get(model="gemini-3.8-flash-low").display_name == "Gemini 3.8 Flash (Low)"
    with pytest.raises(errors.ClientError):
        gemini_sdk(client).models.get(model="gemini-nope")


# ---------- token counts ----------

def test_anthropic_count_tokens_runs_claude_with_one_token(gateway):
    client = gateway()
    counted = claude_sdk(client).messages.count_tokens(model="claude-haiku-4-5-20251001", messages=ASK)
    assert counted.input_tokens == 350
    run = gateway.runs("claude")[-1]
    assert run["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "1"
    assert claude_sdk(client).messages.count_tokens(model="claude-haiku-4-5-20251001", messages=ASK).input_tokens == 350
    assert len(gateway.runs("claude")) == 1  # the second count came from the cache
    with pytest.raises(anthropic.BadRequestError):
        claude_sdk(client).messages.count_tokens(model="codex/gpt-6-luna", messages=ASK)


class WordTokenizer:
    """Stands in for the real tokenizers: one token per word."""

    def encode(self, text):
        return text.split()

    def count_tokens(self, contents, config=None):
        words = sum(len((p.text or "").split()) for c in contents for p in c.parts)
        words += len(config.system_instruction.split()) if config and config.system_instruction else 0
        return type("Counted", (), {"total_tokens": words})()


def test_openai_and_gemini_counts_use_local_tokenizers(gateway, monkeypatch):
    monkeypatch.setitem(counting._tokenizers, "openai", WordTokenizer())
    monkeypatch.setitem(counting._tokenizers, "gemini", WordTokenizer())
    counting._ready.set()
    client = gateway()
    counted = openai_sdk(client).responses.input_tokens.count(model="gpt-6-luna", input="What is RAG exactly?",
                                                              instructions="Be brief.")
    assert counted.input_tokens == (4 + 2) + 3 * 2 + 3  # words, plus the per-message framing
    assert gateway.runs("codex") == []  # no model call
    total = gemini_sdk(client).models.count_tokens(model="gemini-3.8-flash-low", contents="What is RAG exactly?")
    assert total.total_tokens == 4
    assert gateway.runs("agy") == []
