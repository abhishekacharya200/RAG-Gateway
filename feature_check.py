"""Live check of files, images, PDFs, tools and ready-started processes, with the official SDKs.

Start the gateway first (python -m gateway), then:

    python feature_check.py                 # everything: about 20 small real calls
    python feature_check.py tools warm      # only some sections

Sections: claude-pdf, codex-image, codex-pdf, agy-image, agy-pdf, claude-tools, codex-tools, agy-tools,
claude-json, codex-json, agy-json, models, counts, warm.
A name like "tools" or "agy" picks every section it appears in. pytest does not run this file.
"""

import base64
import io
import struct
import sys
import time
import zlib

import anthropic
import openai
from google import genai
from google.genai import types

BASE = "http://127.0.0.1:8787"
KEY = "local-test-key"
CLAUDE, GPT, GEMINI = "claude-haiku-4-5-20251001", "gpt-6-luna", "gemini-3.8-flash-low"
WEATHER = {"name": "get_weather", "description": "Get the current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
RUN = int(time.time())  # makes every question new, so nothing comes from the gateway's cache

claude = anthropic.Anthropic(api_key=KEY, base_url=BASE)
oai = openai.OpenAI(api_key=KEY, base_url=f"{BASE}/v1")
gem = genai.Client(api_key=KEY, http_options=types.HttpOptions(base_url=BASE))
NO_AFC = types.AutomaticFunctionCallingConfig(disable=True)


def png(color=(220, 20, 20), size=16) -> bytes:
    raw = b"".join(b"\x00" + bytes(color) * size for _ in range(size))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def pdf(text: str) -> bytes:
    content = f"BT /F1 18 Tf 50 700 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)


SECRET = pdf("The secret word is pineapple.")


def claude_pdf():
    uploaded = claude.beta.files.upload(file=("secret.pdf", SECRET, "application/pdf"))
    print("uploaded:", uploaded.id, "->", uploaded.model_extra["path"])
    msg = claude.messages.create(model=CLAUDE, max_tokens=50, messages=[{"role": "user", "content": [
        {"type": "document", "source": {"type": "file", "file_id": uploaded.id}},
        {"type": "text", "text": f"(check {RUN}) What is the secret word? One word."}]}])
    print("answer:", msg.content[0].text)
    claude.beta.files.delete(uploaded.id)


def codex_image():
    data_url = "data:image/png;base64," + base64.b64encode(png()).decode()
    chat = oai.chat.completions.create(model=GPT, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": f"(check {RUN}) Main colour of the image? One word."}]}])
    print("answer:", chat.choices[0].message.content)


def codex_pdf():
    uploaded = oai.files.create(file=("secret.pdf", SECRET, "application/pdf"), purpose="user_data")
    print("uploaded:", uploaded.id, "->", uploaded.model_extra["path"])
    response = oai.responses.create(model=GPT, input=[{"role": "user", "content": [
        {"type": "input_file", "file_id": uploaded.id},
        {"type": "input_text", "text": f"(check {RUN}) What is the secret word? One word."}]}])
    print("answer:", response.output_text)
    oai.files.delete(uploaded.id)


def agy_image():
    uploaded = gem.files.upload(file=io.BytesIO(png()), config={"mime_type": "image/png", "display_name": "red.png"})
    print("uploaded:", uploaded.name, uploaded.uri)
    reply = gem.models.generate_content(model=GEMINI, contents=[
        types.Part.from_uri(file_uri=uploaded.uri, mime_type="image/png"),
        f"(check {RUN}) Main colour of the image? One word."],
        config=types.GenerateContentConfig(automatic_function_calling=NO_AFC))
    print("answer:", reply.text)
    gem.files.delete(name=uploaded.name)


def agy_pdf():
    reply = gem.models.generate_content(model=GEMINI, contents=[
        types.Part.from_bytes(data=SECRET, mime_type="application/pdf"),
        f"(check {RUN}) What is the secret word? One word."],
        config=types.GenerateContentConfig(automatic_function_calling=NO_AFC))
    print("answer:", reply.text)


def claude_tools():
    tools = [{"name": WEATHER["name"], "description": WEATHER["description"], "input_schema": WEATHER["parameters"]}]
    question = {"role": "user", "content": f"(check {RUN}) What is the weather in Paris?"}
    msg = claude.messages.create(model=CLAUDE, max_tokens=200, tools=tools, messages=[question])
    call = next(b for b in msg.content if b.type == "tool_use")
    print("stop_reason:", msg.stop_reason, "| call:", call.name, call.input, call.id)
    msg = claude.messages.create(model=CLAUDE, max_tokens=200, tools=tools, messages=[
        question, {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in msg.content]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "content": "22 degrees, sunny"}]}])
    print("answer:", msg.content[0].text)


def codex_tools():
    messages = [{"role": "user", "content": f"(check {RUN}) What is the weather in Paris?"}]
    tools = [{"type": "function", "function": WEATHER}]
    chat = oai.chat.completions.create(model=GPT, messages=messages, tools=tools)
    call = chat.choices[0].message.tool_calls[0]
    print("finish_reason:", chat.choices[0].finish_reason, "| call:", call.function.name, call.function.arguments)
    chat = oai.chat.completions.create(model=GPT, tools=tools, messages=messages + [
        {"role": "assistant", "content": None, "tool_calls": [call.model_dump()]},
        {"role": "tool", "tool_call_id": call.id, "content": "22 degrees, sunny"}])
    print("answer:", chat.choices[0].message.content)


def agy_tools():
    weather = types.Tool(function_declarations=[types.FunctionDeclaration(
        name="get_weather", description=WEATHER["description"],
        parameters=types.Schema(type="OBJECT", properties={"city": types.Schema(type="STRING")}, required=["city"]))])
    config = types.GenerateContentConfig(tools=[weather], automatic_function_calling=NO_AFC)
    ask = types.Content(role="user", parts=[types.Part(text=f"(check {RUN}) What is the weather in Paris?")])
    reply = gem.models.generate_content(model=GEMINI, contents=[ask], config=config)
    call = reply.function_calls[0]
    print("call:", call.name, call.args)
    reply = gem.models.generate_content(model=GEMINI, config=config, contents=[
        ask, types.Content(role="model", parts=[types.Part(function_call=call)]),
        types.Content(role="user", parts=[types.Part.from_function_response(
            name="get_weather", response={"result": "22 degrees, sunny"})])])
    print("answer:", reply.text)


CITY = {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"], "additionalProperties": False}


def claude_json():
    msg = claude.messages.create(model=CLAUDE, max_tokens=200, messages=[
        {"role": "user", "content": f"(check {RUN}) Which city is the capital of Japan?"}],
        output_config={"format": {"type": "json_schema", "schema": CITY}})
    print("answer:", msg.content[0].text, "| stop_reason:", msg.stop_reason)


def codex_json():
    chat = oai.chat.completions.create(model=GPT, messages=[
        {"role": "user", "content": f"(check {RUN}) Which city is the capital of Japan?"}],
        response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": CITY, "strict": True}})
    print("answer:", chat.choices[0].message.content)


def agy_json():
    reply = gem.models.generate_content(model=GEMINI, contents=f"(check {RUN}) Which city is the capital of Japan?",
                                        config=types.GenerateContentConfig(response_mime_type="application/json",
                                                                           response_json_schema=CITY,
                                                                           automatic_function_calling=NO_AFC))
    print("answer:", reply.text)


def models():
    print("anthropic:", [m.id for m in claude.models.list()])
    print("openai:   ", [m.id for m in oai.models.list()])
    print("gemini:   ", [m.name for m in gem.models.list()][:5], "...")


def counts():
    question = [{"role": "user", "content": "What is retrieval-augmented generation, in one paragraph?"}]
    print("anthropic count_tokens (a one-token Claude Code run):",
          claude.messages.count_tokens(model=CLAUDE, messages=question).input_tokens)
    print("openai input_tokens (local tokenizer):",
          oai.responses.input_tokens.count(model=GPT, input=question[0]["content"]).input_tokens)
    print("gemini countTokens (local tokenizer):",
          gem.models.count_tokens(model=GEMINI, contents=question[0]["content"]).total_tokens)


def warm():
    """The first request with a new setup starts Antigravity; the second finds a process already running."""
    config = types.GenerateContentConfig(system_instruction=f"Be terse. ({RUN})", automatic_function_calling=NO_AFC)
    for attempt in (1, 2):
        started = time.monotonic()
        gem.models.generate_content(model=GEMINI, contents=f"(check {RUN}-{attempt}) Say ok.", config=config)
        print(f"request {attempt}: {time.monotonic() - started:.1f} s")
        if attempt == 1:
            time.sleep(15)  # the gateway starts the next process in the background


SECTIONS = {"claude-pdf": ("Claude reads an uploaded PDF itself", claude_pdf),
            "codex-image": ("Codex sees an inline image (--image)", codex_image),
            "codex-pdf": ("Codex reads an uploaded PDF's text (Responses API)", codex_pdf),
            "agy-image": ("Antigravity sees an uploaded image (@path)", agy_image),
            "agy-pdf": ("Antigravity reads an inline PDF's text", agy_pdf),
            "claude-tools": ("Tools: Claude (native, through MCP)", claude_tools),
            "codex-tools": ("Tools: Codex (native, through MCP)", codex_tools),
            "agy-tools": ("Tools: Antigravity (reply format in its instructions)", agy_tools),
            "claude-json": ("Structured output: Claude (enforced, --json-schema)", claude_json),
            "codex-json": ("Structured output: Codex (strict schema, --output-schema)", codex_json),
            "agy-json": ("Structured output: Antigravity (asked for in its instructions)", agy_json),
            "models": ("Model lists through each SDK", models),
            "counts": ("Token counts through each SDK", counts),
            "warm": ("Ready-started processes: two Antigravity requests with one new setup", warm)}


def main() -> None:
    wanted = set(sys.argv[1:])
    for name, (title, check) in SECTIONS.items():
        if wanted and not (name in wanted or set(name.split("-")) & wanted):
            continue
        print(f"\n=== {title}")
        check()
    print("\nFeature checks finished.")


if __name__ == "__main__":
    main()
