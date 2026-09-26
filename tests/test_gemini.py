"""Gemini generateContent format, answered by the fake Antigravity CLI."""

from pathlib import Path

import pytest
from google import genai
from google.genai import errors, types

from conftest import KEY, history, sse_data

MODEL = "gemini-3.8-flash-low"


def sdk(client, key=KEY):
    return genai.Client(api_key=key, http_options=types.HttpOptions(base_url="http://testserver", httpx_client=client))


def test_generate_content_matches_sdk_types(gateway):
    client = gateway()
    response = sdk(client).models.generate_content(
        model=MODEL, contents="What is RAG?",
        config=types.GenerateContentConfig(system_instruction="Answer from the context only.", max_output_tokens=100))
    assert response.text == "Gemini echo: What is RAG?"  # agy's closing newline is not part of the answer
    assert response.candidates[0].finish_reason == types.FinishReason.STOP
    assert response.model_version == MODEL
    usage = response.usage_metadata
    assert usage.prompt_token_count == 564 and usage.thoughts_token_count == 124
    assert usage.candidates_token_count == 4 and usage.total_token_count == 692


def test_stream_generate_content(gateway):
    client = gateway(cache_enabled=False)
    chunks = list(sdk(client).models.generate_content_stream(model=MODEL, contents="What is RAG?"))
    assert "".join(c.text or "" for c in chunks) == "Gemini echo: What is RAG?"
    assert chunks[-1].candidates[0].finish_reason == types.FinishReason.STOP
    assert chunks[-1].usage_metadata.prompt_token_count == 564

    raw = client.post(f"/v1beta/models/{MODEL}:streamGenerateContent", headers={"x-goog-api-key": KEY},
                      json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    assert isinstance(raw.json(), list)  # without alt=sse the API answers with a JSON array


def test_antigravity_is_isolated(gateway):
    client = gateway()
    sdk(client).models.generate_content(
        model=MODEL, contents=[types.Content(role="user", parts=[types.Part(text="Context:\nRAG retrieves."),
                                                                 types.Part(text="What is RAG?")])],
        config=types.GenerateContentConfig(system_instruction="Answer from the context only."))
    run = gateway.runs("agy")[-1]
    argv, env = run["argv"], run["env"]
    assert argv[argv.index("--agent") + 1] == "gateway" and argv[argv.index("--model") + 1] == MODEL
    for flag in ("--sandbox", "--disable-slash-commands"):
        assert flag in argv
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert "excludeDefaultComponents: true" in run["agent"] and run["agent"].endswith("Answer from the context only.")
    assert run["stdin"] == {"event": "user", "message": {"role": "user",
                                                         "content": "Context:\nRAG retrieves.\n\nWhat is RAG?"}}
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert name not in env
    assert Path(run["cwd"]).name.startswith("gateway-antigravity-") and not Path(run["cwd"]).exists()
    item = history(client)[0]
    assert item["provider"] == "antigravity" and item["api"] == "gemini"
    assert item["question"] == "What is RAG?" and item["context"] == "Context:\nRAG retrieves."


def test_unknown_model_is_not_found(gateway):
    client = gateway(agy="bad_model")
    with pytest.raises(errors.ClientError) as caught:
        sdk(client).models.generate_content(model="gemini-nope", contents="hi")
    assert caught.value.code == 404 and caught.value.status == "NOT_FOUND"


def test_quota_is_resource_exhausted(gateway):
    client = gateway(agy="quota")
    with pytest.raises(errors.ClientError) as caught:
        sdk(client).models.generate_content(model=MODEL, contents="hi")
    assert caught.value.code == 429 and caught.value.status == "RESOURCE_EXHAUSTED"


def test_gemini_key_errors_match_the_real_api(gateway):
    client = gateway()
    body = {"contents": [{"parts": [{"text": "hi"}]}]}
    missing = client.post(f"/v1beta/models/{MODEL}:generateContent", json=body)
    assert missing.status_code == 403 and missing.json()["error"]["status"] == "PERMISSION_DENIED"
    with pytest.raises(errors.ClientError) as caught:
        sdk(client, key="nope").models.generate_content(model=MODEL, contents="hi")
    assert caught.value.code == 400 and "API key not valid" in caught.value.message
    by_query = client.post(f"/v1beta/models/{MODEL}:generateContent?key={KEY}", json=body)
    assert by_query.status_code == 200


def test_gemini_bad_requests(gateway):
    client = gateway()
    headers = {"x-goog-api-key": KEY}
    unknown = client.post(f"/v1beta/models/{MODEL}:generateContent", headers=headers,
                          json={"contents": [{"parts": [{"text": "hi"}]}], "foo": 1})
    assert unknown.status_code == 400 and 'Unknown name "foo"' in unknown.json()["error"]["message"]
    image = client.post(f"/v1beta/models/{MODEL}:generateContent", headers=headers,
                        json={"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": "x"}}]}]})
    assert image.status_code == 400 and image.json()["error"]["status"] == "INVALID_ARGUMENT"
    method = client.post(f"/v1beta/models/{MODEL}:embedContent", headers=headers, json={})
    assert method.status_code == 404
