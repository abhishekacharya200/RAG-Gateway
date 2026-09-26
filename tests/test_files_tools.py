"""Files API, images and documents, tools, and ready-started processes (fake CLIs, official SDKs)."""

import base64
import io
import json
import os
import time

import anthropic
import openai
import pytest
from google import genai
from google.genai import types

from conftest import KEY, history, sse_data, tiny_pdf, tiny_png

WEATHER = {"name": "get_weather", "description": "Get the current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}


def claude_sdk(client):
    return anthropic.Anthropic(api_key=KEY, base_url="http://testserver", http_client=client, max_retries=0)


def openai_sdk(client):
    return openai.OpenAI(api_key=KEY, base_url="http://testserver/v1", http_client=client, max_retries=0)


def gemini_sdk(client):
    return genai.Client(api_key=KEY, http_options=types.HttpOptions(base_url="http://testserver", httpx_client=client))


# ---------- Files API ----------

def test_openai_files_api(gateway):
    client = gateway()
    sdk = openai_sdk(client)
    uploaded = sdk.files.create(file=("notes.txt", b"The launch code is 42.", "text/plain"), purpose="user_data")
    assert uploaded.id.startswith("file-") and uploaded.bytes == 22 and uploaded.filename == "notes.txt"
    assert os.path.exists(uploaded.model_extra["path"])  # where the gateway stored it
    assert sdk.files.retrieve(uploaded.id).filename == "notes.txt"
    assert [f.id for f in sdk.files.list()] == [uploaded.id]
    assert sdk.files.content(uploaded.id).read() == b"The launch code is 42."
    # Referenced in a request, Codex gets the file's text.
    sdk.chat.completions.create(model="gpt-6-luna", messages=[{"role": "user", "content": [
        {"type": "file", "file": {"file_id": uploaded.id}}, {"type": "text", "text": "What is the code?"}]}])
    assert 'name="notes.txt">\nThe launch code is 42.\n</document>' in gateway.runs("codex")[-1]["stdin"]
    assert sdk.files.delete(uploaded.id).deleted
    assert not os.path.exists(uploaded.model_extra["path"])


def test_anthropic_files_api_and_pdf_to_claude(gateway):
    client = gateway()
    sdk = claude_sdk(client)
    uploaded = sdk.beta.files.upload(file=("secret.pdf", tiny_pdf("The secret word is pineapple."), "application/pdf"))
    assert uploaded.id.startswith("file_011") and uploaded.mime_type == "application/pdf"
    sdk.messages.create(model="claude-haiku-4-5-20251001", max_tokens=100, messages=[{"role": "user", "content": [
        {"type": "document", "source": {"type": "file", "file_id": uploaded.id}},
        {"type": "text", "text": "What is the secret word?"}]}])
    blocks = gateway.runs("claude")[-1]["blocks"]
    # Claude Code reads PDFs itself: it gets the real PDF as a document block.
    assert [b["type"] for b in blocks] == ["text", "text", "document"]
    assert blocks[2]["source_type"] == "base64" and blocks[2]["media_type"] == "application/pdf"
    assert blocks[0]["text"].startswith("[Attachment 1: secret.pdf]")
    assert sdk.beta.files.retrieve_metadata(uploaded.id).filename == "secret.pdf"
    sdk.beta.files.delete(uploaded.id)
    assert client.get("/ui/api/files", headers={"x-api-key": KEY}).json()["files"] == []


def test_gemini_files_api_and_image_to_antigravity(gateway):
    client = gateway()
    sdk = gemini_sdk(client)
    uploaded = sdk.files.upload(file=io.BytesIO(tiny_png()), config={"mime_type": "image/png", "display_name": "red.png"})
    assert uploaded.name.startswith("files/") and uploaded.uri.endswith(uploaded.name.split("/")[1])
    assert uploaded.state.name == "ACTIVE"
    response = sdk.models.generate_content(model="gemini-3.8-flash-low", contents=[
        types.Part.from_uri(file_uri=uploaded.uri, mime_type="image/png"), "What colour is this?"])
    assert response.text.startswith("Gemini echo:")
    # Antigravity attaches an image when the prompt mentions its path with @.
    prompt = gateway.runs("agy")[-1]["stdin"]["message"]["content"]
    assert "[Image 1: red.png] @" in prompt and ".png" in prompt
    assert sdk.files.get(name=uploaded.name).display_name == "red.png"
    sdk.files.delete(name=uploaded.name)
    assert [f.name for f in sdk.files.list()] == []


def test_files_need_the_gateway_key(gateway):
    client = gateway()
    r = client.post("/v1/files", files={"file": ("a.txt", b"x", "text/plain")}, data={"purpose": "user_data"})
    assert r.status_code == 401
    r = client.post("/v1/files", files={"file": ("a.exe", b"MZ", "application/x-msdownload")},
                    headers={"authorization": f"Bearer {KEY}"})
    assert r.status_code == 400 and "not supported" in r.json()["error"]["message"]


# ---------- inline images and documents ----------

def test_inline_image_reaches_each_cli(gateway):
    client = gateway()
    png = base64.b64encode(tiny_png()).decode()
    claude_sdk(client).messages.create(model="claude-haiku-4-5-20251001", max_tokens=50, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
        {"type": "text", "text": "Colour?"}]}])
    assert gateway.runs("claude")[-1]["blocks"][2] == {"type": "image", "source_type": "base64",
                                                       "media_type": "image/png", "size": len(png)}
    openai_sdk(client).chat.completions.create(model="gpt-6-luna", messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
        {"type": "text", "text": "Colour?"}]}])
    run = gateway.runs("codex")[-1]
    assert len(run["images"]) == 1 and run["images"][0][1] == len(tiny_png())  # passed with --image
    assert run["stdin"].startswith("[Image 1 (attached)]")
    gemini_sdk(client).models.generate_content(model="gemini-3.8-flash-low", contents=[
        types.Part.from_bytes(data=tiny_png(), mime_type="image/png"), "Colour?"])
    assert "[Image 1] @" in gateway.runs("agy")[-1]["stdin"]["message"]["content"]


def test_inline_pdf_from_google_sdk(gateway):
    # The Google SDK sends inline bytes as URL-safe base64.
    client = gateway()
    secret = tiny_pdf("The secret word is pineapple? Yes> of course/")
    assert b"-" in base64.urlsafe_b64encode(secret) or b"_" in base64.urlsafe_b64encode(secret)
    gemini_sdk(client).models.generate_content(model="gemini-3.8-flash-low", contents=[
        types.Part.from_bytes(data=secret, mime_type="application/pdf"), "Secret word?"])
    assert "The secret word is pineapple?" in gateway.runs("agy")[-1]["stdin"]["message"]["content"]


def test_inline_pdf_becomes_text_for_codex(gateway):
    client = gateway()
    pdf = base64.b64encode(tiny_pdf("The secret word is pineapple.")).decode()
    openai_sdk(client).responses.create(model="gpt-6-luna", input=[{"role": "user", "content": [
        {"type": "input_file", "filename": "secret.pdf", "file_data": f"data:application/pdf;base64,{pdf}"},
        {"type": "input_text", "text": "What is the secret word?"}]}])
    assert "The secret word is pineapple." in gateway.runs("codex")[-1]["stdin"]
    item = history(client)[0]
    assert item["attachments"] == [{"name": "secret.pdf", "media_type": "application/pdf", "size": len(tiny_pdf(
        "The secret word is pineapple."))}]
    stored = client.get(f"/ui/api/history/{item['id']}", headers={"x-api-key": KEY}).json()
    assert "characters of file data" in json.dumps(stored["request"])  # the history does not keep file contents


# ---------- tools ----------

def test_claude_tool_call_is_native(gateway):
    client = gateway(claude="tool_use")
    started = time.monotonic()
    msg = claude_sdk(client).messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=200, messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[{"name": "get_weather", "description": WEATHER["description"], "input_schema": WEATHER["parameters"]}])
    assert time.monotonic() - started < 5  # did not wait for Claude Code to run the tool
    assert msg.stop_reason == "tool_use"
    assert [b.type for b in msg.content] == ["text", "tool_use"]
    tool = msg.content[1]
    assert tool.name == "get_weather" and tool.input == {"city": "Paris"} and tool.id == "toolu_01FakeToolAAAAAAAAAAAAA"
    run = gateway.runs("claude")[-1]
    argv = run["argv"]
    assert "--restricted" in argv and "--safe-mode" not in argv
    assert argv[argv.index("--allowedTools") + 1] == "mcp__gateway__get_weather"
    server = run["mcp"]["mcpServers"]["gateway"]
    assert server["args"][0].endswith("mcp_bridge.py")
    assert json.loads(server["env"]["GATEWAY_TOOLS"])[0]["name"] == "get_weather"


def test_claude_tool_call_streams(gateway):
    client = gateway(claude="tool_use", cache_enabled=False)
    with claude_sdk(client).messages.stream(
            model="claude-haiku-4-5-20251001", max_tokens=200, messages=[{"role": "user", "content": "Weather?"}],
            tools=[{"name": "get_weather", "input_schema": WEATHER["parameters"]}]) as stream:
        final = stream.get_final_message()
    assert final.stop_reason == "tool_use" and final.content[-1].input == {"city": "Paris"}


def test_tool_results_go_back_as_text(gateway):
    client = gateway()
    claude_sdk(client).messages.create(model="claude-haiku-4-5-20251001", max_tokens=100, messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                                           "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "22 degrees"}]}],
        tools=[{"name": "get_weather", "input_schema": WEATHER["parameters"]}])
    prompt = gateway.runs("claude")[-1]["stdin"]
    assert '<tool_call id="toolu_1" name="get_weather">{"city": "Paris"}</tool_call>' in prompt
    assert prompt.endswith('<tool_result call_id="toolu_1" name="get_weather">\n22 degrees\n</tool_result>')


def test_openai_chat_tool_call(gateway):
    client = gateway(codex="tool_call")
    started = time.monotonic()
    completion = openai_sdk(client).chat.completions.create(
        model="gpt-6-luna", messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[{"type": "function", "function": WEATHER}], tool_choice="required")
    assert time.monotonic() - started < 5
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.content is None
    call = choice.message.tool_calls[0]
    assert call.function.name == "get_weather" and json.loads(call.function.arguments) == {"city": "Paris"}
    run = gateway.runs("codex")[-1]
    assert any(c.startswith("mcp_servers.gateway.command=") for c in run["config"])
    assert "you must call at least one" in run["system"]
    # The follow-up with the tool's result.
    gateway_ok = gateway()
    openai_sdk(gateway_ok).chat.completions.create(model="gpt-6-luna", tools=[{"type": "function", "function": WEATHER}],
                                                   messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": call.id, "type": "function", "function": {
            "name": "get_weather", "arguments": call.function.arguments}}]},
        {"role": "tool", "tool_call_id": call.id, "content": "22 degrees"}])
    assert f'<tool_result call_id="{call.id}" name="get_weather">\n22 degrees' in gateway.runs("codex")[-1]["stdin"]


def test_openai_responses_tool_call_streaming(gateway):
    client = gateway(codex="tool_call", cache_enabled=False)
    with openai_sdk(client).responses.stream(model="gpt-6-luna", input="Weather in Paris?",
                                             tools=[{"type": "function", **WEATHER}]) as stream:
        kinds = [e.type for e in stream]
        final = stream.get_final_response()
    assert "response.function_call_arguments.done" in kinds
    call = final.output[-1]
    assert call.type == "function_call" and call.name == "get_weather" and json.loads(call.arguments) == {"city": "Paris"}


def test_gemini_function_call(gateway):
    client = gateway(agy="tool_call")
    response = gemini_sdk(client).models.generate_content(
        model="gemini-3.8-flash-low", contents="Weather in Paris?",
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
                name="get_weather", description=WEATHER["description"],
                parameters=types.Schema(type="OBJECT", properties={"city": types.Schema(type="STRING")},
                                        required=["city"]))])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
    call = response.function_calls[0]
    assert call.name == "get_weather" and call.args == {"city": "Paris"}
    agent = gateway.runs("agy")[-1]["agent"]
    assert "<functions>" in agent and '"type": "object"' in agent  # Gemini's OBJECT became JSON Schema's object


def test_tool_choice_none_offers_no_tools(gateway):
    client = gateway()
    claude_sdk(client).messages.create(model="claude-haiku-4-5-20251001", max_tokens=50,
                                       messages=[{"role": "user", "content": "hi"}], tool_choice={"type": "none"},
                                       tools=[{"name": "get_weather", "input_schema": WEATHER["parameters"]}])
    assert "--safe-mode" in gateway.runs("claude")[-1]["argv"]


# ---------- ready-started processes ----------

def test_second_request_uses_a_ready_process(gateway):
    client = gateway(cache_enabled=False)
    body = {"model": "claude-haiku-4-5-20251001", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/messages", json=body, headers={"x-api-key": KEY}).status_code == 200
    time.sleep(1.0)  # the gateway starts the next process in the background
    sent = time.time()
    assert client.post("/v1/messages", json=body, headers={"x-api-key": KEY}).status_code == 200
    run = gateway.runs("claude")[-1]
    assert run["started_at"] < sent - 0.5  # the process was already running before the request came
    assert client.get("/ui/api/status", headers={"x-api-key": KEY}).json()["warm_pool"]["enabled"]


def test_codex_gets_its_own_home_folder(gateway):
    client = gateway()
    openai_sdk(client).chat.completions.create(model="gpt-6-luna", messages=[{"role": "user", "content": "hi"}])
    env = gateway.runs("codex")[-1]["env"]
    assert env["HOME"].endswith("/home") and "gateway-codex-" in env["HOME"]
    assert env["CODEX_HOME"].endswith(".codex")
    assert "include_environment_context=false" in gateway.runs("codex")[-1]["config"]


def test_antigravity_tool_block_keeps_only_offered_tools():
    from gateway.providers.antigravity import parse_tool_calls
    reply = ('<tool_calls>\n[{"name": "default_api:manage_task", "arguments": {"Action": "status"}}, '
             '{"name": "get_weather", "arguments": {"city": "Paris"}}]\n</tool_calls>')
    calls = parse_tool_calls(reply, {"get_weather"})
    assert [(c.name, c.arguments) for c in calls] == [("get_weather", {"city": "Paris"})]
    assert parse_tool_calls('<tool_calls>[{"name": "other", "arguments": {}}]</tool_calls>', {"get_weather"}) is None
    assert parse_tool_calls("It is sunny.", {"get_weather"}) is None
