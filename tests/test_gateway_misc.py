"""Routing between providers, startup checks, shared cache, safety, and the web page's API."""

import sys

import anthropic
import openai
import pytest
from fastapi.testclient import TestClient

from conftest import FAKE_AGY, FAKE_CLAUDE, FAKE_CODEX, KEY, history
from gateway.config import Settings
from gateway.main import create_app

MESSAGES = [{"role": "user", "content": "What is RAG?"}]


def test_provider_prefix_routes_across_formats(gateway):
    client = gateway()
    # Anthropic format answered by Codex: the answer comes back in Anthropic's shape.
    msg = anthropic.Anthropic(api_key=KEY, base_url="http://testserver", http_client=client, max_retries=0) \
        .messages.create(model="codex/gpt-6-luna", max_tokens=100, messages=MESSAGES)
    assert msg.content[0].text == "Codex echo: What is RAG?" and msg.model == "gpt-6-luna"
    assert msg.usage.input_tokens == 4511 - 1792 and msg.usage.cache_read_input_tokens == 1792
    # OpenAI format answered by Antigravity.
    completion = openai.OpenAI(api_key=KEY, base_url="http://testserver/v1", http_client=client, max_retries=0) \
        .chat.completions.create(model="agy/gemini-3.8-flash-low", messages=MESSAGES)
    assert completion.choices[0].message.content == "Gemini echo: What is RAG?"
    assert [(i["provider"], i["api"]) for i in history(client)] == [("antigravity", "openai-chat"),
                                                                     ("codex", "anthropic-messages")]


def test_cache_is_shared_between_formats_of_one_provider(gateway):
    client = gateway()
    sdk = openai.OpenAI(api_key=KEY, base_url="http://testserver/v1", http_client=client, max_retries=0)
    sdk.chat.completions.create(model="gpt-6-luna", messages=MESSAGES)
    response = sdk.responses.create(model="gpt-6-luna", input="What is RAG?")
    assert response.output_text == "Codex echo: What is RAG?"
    assert len(gateway.runs("codex")) == 1
    assert [i["cache"] for i in history(client)] == ["hit", "miss"]


def test_unavailable_provider_does_not_stop_the_others(gateway):
    client = gateway(auth="out")  # Claude Code logged out; Codex and Antigravity fine
    r = client.post("/v1/messages", headers={"x-api-key": KEY},
                    json={"model": "claude-haiku-4-5-20251001", "max_tokens": 10, "messages": MESSAGES})
    assert r.status_code == 401 and "Claude Code is not available" in r.json()["error"]["message"]
    status = client.get("/ui/api/status", headers={"x-api-key": KEY}).json()
    ready = {p["id"]: p["ready"] for p in status["providers"]}
    assert ready == {"claude": False, "codex": True, "antigravity": True}


@pytest.mark.parametrize("overrides", [
    {"codex_login": "apikey"},      # an API-key login would bill the OpenAI API
    {"agy_login": "none"},
])
def test_provider_problems_are_reported(gateway, overrides):
    client = gateway(**overrides)
    providers = {p["id"]: p for p in client.get("/ui/api/status", headers={"x-api-key": KEY}).json()["providers"]}
    broken = providers["codex"] if "codex_login" in overrides else providers["antigravity"]
    assert not broken["ready"] and broken["problem"]


def test_startup_refused_when_no_provider_is_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_AUTH", "out")
    app = create_app(Settings(claude_bin=(sys.executable, FAKE_CLAUDE), codex_bin="/nonexistent/codex",
                              agy_bin="/nonexistent/agy", db_path=str(tmp_path / "g.db"), load_tokenizers=False))
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


def test_other_computers_are_refused(gateway):
    client = gateway(client_addr=("192.168.1.20", 50000))
    r = client.post("/v1/messages", headers={"x-api-key": KEY}, json={})
    assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"


def test_other_host_names_are_refused(gateway):
    # A web page whose domain resolves to 127.0.0.1 must not be able to read the page or call the API.
    client = gateway()
    for path in ("/", "/v1/messages", "/v1/chat/completions"):
        r = client.get(path, headers={"host": "evil.example:8787"})
        assert r.status_code == 403
    assert client.get("/health", headers={"host": "localhost:8787"}).status_code == 200


def test_web_page_api(gateway):
    client = gateway()
    client.post("/v1/messages", headers={"x-api-key": KEY, "x-gateway-source": "ui"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 10, "messages": MESSAGES})
    headers = {"x-api-key": KEY}
    items = history(client)
    assert items[0]["question"] == "What is RAG?" and items[0]["answer"] == "Echo: What is RAG?"
    assert items[0]["source"] == "ui" and items[0]["input_tokens"] == 350
    item = client.get(f"/ui/api/history/{items[0]['id']}", headers=headers).json()
    assert item["response"]["type"] == "message" and item["request"]["model"] == "claude-haiku-4-5-20251001"
    assert client.get("/ui/api/history?provider=codex", headers=headers).json()["items"] == []

    status = client.get("/ui/api/status", headers=headers).json()
    claude = next(p for p in status["providers"] if p["id"] == "claude")
    assert claude["account"] == "claude.ai, pro plan" and claude["rate_limit"]["status"] == "allowed"
    assert {a["id"] for a in status["apis"]} == {"anthropic-messages", "openai-chat", "openai-responses", "gemini"}
    restarted = gateway()  # same database, new app: the last plan-usage reading is still there
    claude = next(p for p in restarted.get("/ui/api/status", headers=headers).json()["providers"]
                  if p["id"] == "claude")
    assert claude["rate_limit"]["status"] == "allowed"
    assert client.get("/ui/api/history").status_code == 401
    page = client.get("/")
    assert page.status_code == 200 and 'const apiKey = "test-key";' in page.text  # key filled in, nothing to type
