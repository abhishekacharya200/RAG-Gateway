"""Live end-to-end check of every API format with its official SDK. Start the gateway first: python -m gateway

Run it yourself (python test_client.py); pytest does not collect it, because it spends real plan usage.

Makes 7 small real model calls: Claude (Haiku) once, Codex three times, Antigravity (Gemini Flash) three times.
The unknown-model checks do not reach any model.
"""

import os
import time

import anthropic
import openai
from google import genai
from google.genai import errors as genai_errors, types

BASE = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8787")
KEY = "local-test-key"
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash-low")

def main() -> None:
    run = str(int(time.time()))  # makes every question new, so nothing is answered from the cache
    context = "RAG is a technique where relevant documents are retrieved before sending a question to an LLM."
    system = "Answer strictly from the supplied context in one sentence. If the context is not enough, say so."
    question = f"(run {run}) What is RAG?"

    claude = anthropic.Anthropic(api_key=KEY, base_url=BASE)
    oai = openai.OpenAI(api_key=KEY, base_url=f"{BASE}/v1")
    gemini = genai.Client(api_key=KEY, http_options=types.HttpOptions(base_url=BASE))


    def section(title):
        print(f"\n=== {title}")


    # 1. Anthropic Messages -> Claude Code
    section(f"Anthropic Messages -> Claude Code ({CLAUDE_MODEL})")
    msg = claude.messages.create(model=CLAUDE_MODEL, max_tokens=200, system=system, messages=[{"role": "user", "content": [
        {"type": "text", "text": f"Context:\n{context}"}, {"type": "text", "text": question}]}])
    print(f"{msg.id}  {msg.model}  stop={msg.stop_reason}  in={msg.usage.input_tokens} out={msg.usage.output_tokens}")
    print(msg.content[0].text)

    # 2. OpenAI Chat Completions -> Codex
    section(f"OpenAI Chat Completions -> Codex ({OPENAI_MODEL})")
    chat = oai.chat.completions.create(model=OPENAI_MODEL, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\n{question}"}])
    print(f"{chat.id}  {chat.model}  finish={chat.choices[0].finish_reason}  "
          f"prompt={chat.usage.prompt_tokens} completion={chat.usage.completion_tokens}")
    print(chat.choices[0].message.content)

    section("OpenAI Chat Completions, streaming, with usage")
    stream = oai.chat.completions.create(model=OPENAI_MODEL, stream=True, stream_options={"include_usage": True},
                                         messages=[{"role": "user", "content": f"(run {run}) Count from 1 to 5."}])
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.usage:
            print(f"\nusage: {chunk.usage.total_tokens} tokens")

    # 3. OpenAI Responses -> Codex
    section(f"OpenAI Responses -> Codex ({OPENAI_MODEL})")
    response = oai.responses.create(model=OPENAI_MODEL, instructions=system,
                                    input=f"Context:\n{context}\n\n(run {run}) Why retrieve documents first?")
    print(f"{response.id[:24]}...  {response.model}  status={response.status}  in={response.usage.input_tokens} "
          f"out={response.usage.output_tokens}")
    print(response.output_text)

    # 4. Gemini -> Antigravity
    section(f"Gemini generateContent -> Antigravity ({GEMINI_MODEL})")
    reply = gemini.models.generate_content(model=GEMINI_MODEL, contents=f"Context:\n{context}\n\n{question}",
                                           config=types.GenerateContentConfig(system_instruction=system))
    meta = reply.usage_metadata
    print(f"{reply.response_id}  {reply.model_version}  finish={reply.candidates[0].finish_reason.name}  "
          f"prompt={meta.prompt_token_count} answer={meta.candidates_token_count} thoughts={meta.thoughts_token_count}")
    print(reply.text)

    section("Gemini streamGenerateContent")
    for chunk in gemini.models.generate_content_stream(model=GEMINI_MODEL, contents=f"(run {run}) Count from 1 to 5."):
        print(chunk.text or "", end="", flush=True)
    print()

    section("Gemini, multi-turn")
    chat_session = gemini.models.generate_content(model=GEMINI_MODEL, contents=[
        types.Content(role="user", parts=[types.Part(text=f"(run {run}) My favourite colour is teal.")]),
        types.Content(role="model", parts=[types.Part(text="Noted: teal.")]),
        types.Content(role="user", parts=[types.Part(text="What is my favourite colour? One word.")])])
    print(chat_session.text)

    # 5. Errors come back as each SDK's normal exception
    section("Unknown models")
    for name, call, error in (
            ("anthropic", lambda: claude.messages.create(model="claude-nope", max_tokens=5,
                                                         messages=[{"role": "user", "content": "hi"}]), anthropic.NotFoundError),
            ("openai", lambda: oai.chat.completions.create(model="gpt-nope",
                                                           messages=[{"role": "user", "content": "hi"}]), openai.NotFoundError),
            ("gemini", lambda: gemini.models.generate_content(model="gemini-nope", contents="hi"), genai_errors.ClientError)):
        try:
            call()
        except error as err:
            print(f"{name}: {type(err).__name__} {getattr(err, 'status_code', getattr(err, 'code', ''))}")

    print("\nAll checks finished.")


if __name__ == "__main__":
    main()
