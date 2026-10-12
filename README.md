# RAG Gateway

A local server that speaks the **Anthropic**, **OpenAI** and **Gemini** APIs and answers through the CLIs you are
already logged in to: **Claude Code** (Claude Pro/Max), **Codex** (ChatGPT plan) and **Antigravity** (Google
account). Point your RAG app at it while you build and test. In production, change the SDK's base URL and API key
environment variables and the same code talks to the real API.

It supports text, streaming, images, PDFs and text files (inline or through each API's Files API), tool
(function) calling, structured JSON output, model lists and token counts, in each API's own format.

```
 your app, official SDKs                  gateway (FastAPI, 127.0.0.1:8787)                     CLIs you are logged in to
 ───────────────────────                  ─────────────────────────────────                     ─────────────────────────
 anthropic  POST /v1/messages          ┐  API formats          providers                     ┌─ claude -p      Claude models
 openai     POST /v1/chat/completions  ├─ parse request ──►  route ─► cache ─► run the CLI ─┼─ codex exec     OpenAI models
 openai     POST /v1/responses         │  render response ◄── same event stream  ◄──────────┘  agy -p          Gemini models
 google     POST /v1beta/models/..     ┘  (JSON or SSE)            │
                                                                   └─► history (gateway.db) ─► web page at /
```

Each API format goes to its own provider: `/v1/messages` to Claude Code, the two OpenAI endpoints to Codex, and
the Gemini endpoints to Antigravity. A model name with a provider prefix sends a request elsewhere, for example
`codex/gpt-6-luna` through the Anthropic format or `agy/gemini-3.8-flash-low` through the OpenAI format.

The web page at `http://127.0.0.1:8787/` lets you pick a provider, API format and model, ask a question with
optional RAG context, attach files, try tools, watch the answer stream in, and see every question and answer
(including the ones your apps send), coloured by provider, with the exact API request and response.

## Quick start

1. Log in to the CLIs you want to use. Each one is optional; the gateway starts with any that are ready.
   ```bash
   claude auth status        # "loggedIn": true, "authMethod": "claude.ai"   (fix: run claude, then /login)
   codex login status        # "Logged in using ChatGPT"                      (fix: codex login)
   agy models                # prints a model list                            (fix: run agy and sign in)
   ```
   Codex is installed with `npm install -g @openai/codex`. Do not log any CLI in with an API key: the gateway
   refuses API-key logins, because they bill the API instead of your plan.

2. Get the code and install:
   ```bash
   git clone https://github.com/abhishekacharya200/RAG-Gateway.git
   cd RAG-Gateway
   python3 -m venv .venv            # on Ubuntu without python3-venv: uv venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt  # with uv: uv pip install -r requirements.txt
   ```

3. Start the gateway:
   ```bash
   python -m gateway
   ```
   The log shows one line per provider, for example
   `Codex ready: .../codex 0.157.1, ChatGPT login, 4 models`, or why one is not available.

4. Open `http://127.0.0.1:8787/`, or run the live check in another terminal:
   ```bash
   python test_client.py      # every API format through its official SDK (7 small real calls)
   ```

## Connecting your app

| SDK | base URL | API key |
|---|---|---|
| `anthropic` | `http://127.0.0.1:8787` | `local-test-key` |
| `openai` | `http://127.0.0.1:8787/v1` | `local-test-key` |
| `google-genai` | `http://127.0.0.1:8787` | `local-test-key` |

```python
import anthropic, openai
from google import genai
from google.genai import types

claude = anthropic.Anthropic(base_url="http://127.0.0.1:8787", api_key="local-test-key")
claude.messages.create(model="claude-haiku-4-5-20251001", max_tokens=500,
                       messages=[{"role": "user", "content": "What is RAG?"}])

gpt = openai.OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="local-test-key")
gpt.chat.completions.create(model="gpt-6-luna", messages=[{"role": "user", "content": "What is RAG?"}])
gpt.responses.create(model="gpt-6-luna", input="What is RAG?")

gemini = genai.Client(api_key="local-test-key", http_options=types.HttpOptions(base_url="http://127.0.0.1:8787"))
gemini.models.generate_content(model="gemini-3.8-flash-low", contents="What is RAG?")
```

Streaming works with each SDK's usual calls (`messages.stream`, `stream=True`, `responses.stream`,
`generate_content_stream`).

**Production:** point each SDK back at the real API with its usual variables and a real key, for example
`ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY`, `OPENAI_BASE_URL` + `OPENAI_API_KEY`, or the Gemini key. Set those only
in your app's terminal: the gateway removes every provider key from the CLIs' environment anyway.

### Sample RAG app

`rag_app.py` is a complete small RAG app in one file: sample help-center documents, keyword search (BM25), and
questions through the official SDK of the provider you pick. For every question it prints only the exact request
the SDK sent and the exact response it got back.

```bash
python rag_app.py                              # Claude, Anthropic Messages API
python rag_app.py --provider codex             # OpenAI models, Chat Completions API
python rag_app.py --provider antigravity       # Gemini models, Gemini API
python rag_app.py "Can I get a refund?"        # one question and exit
python rag_app.py --docs ~/notes               # your own .txt / .md files
```

## Files, images and documents

Send them the way each API normally takes them: inline (base64 or a data: URL), or upload once through the
Files API and refer to the file by its id. Images (PNG, JPEG, GIF, WebP), PDFs and text files (txt, md, csv,
json, code...) are supported, up to 32 MB each.

| Files API | Upload | Refer to it in a request |
|---|---|---|
| Anthropic (`client.beta.files.upload`) | `POST /v1/files` | `{"type": "document", "source": {"type": "file", "file_id": "file_011..."}}` |
| OpenAI (`client.files.create`) | `POST /v1/files` | Chat: `{"type": "file", "file": {"file_id": "file-..."}}`, Responses: `input_file` / `input_image` with `file_id` |
| Google (`client.files.upload`) | `POST /upload/v1beta/files` (resumable) | `{"fileData": {"fileUri": "<the file's uri>"}}` |

List, get, download and delete work too (`GET /v1/files`, `GET /v1/files/{id}`, `/content`, `DELETE`, and the
same under `/v1beta/files`). Every file object has one extra field, `path`: where the gateway stored the file
(in the `files/` folder, named by its id). A file uploaded through one API can be used from the others.

What each CLI receives:

| | Claude Code | Codex | Antigravity |
|---|---|---|---|
| Images | the image itself | the image itself (`--image`) | the image itself (an `@path` mention in the prompt) |
| PDFs | the PDF itself (Claude reads the pages) | the PDF's text | the PDF's text |
| Text files | the text | the text | the text |

Web URLs for images and files are not fetched; send the data or upload the file.

## Tools (function calling)

Define tools in each API's usual way (`tools` with `input_schema`, `function` tools, or `functionDeclarations`).
When the model decides to call one, the response is a normal tool call in that API's format
(`tool_use` block with `stop_reason: "tool_use"`, `tool_calls` with `finish_reason: "tool_calls"`, a
`function_call` output item, or a `functionCall` part). Your app runs the tool and sends the result back in the
next request (`tool_result`, a `tool` message, `function_call_output`, or `functionResponse`), exactly as with
the real API. The gateway never runs your tools.

- **Claude Code and Codex call tools natively.** The gateway offers your tools to the CLI through a small MCP
  server (`gateway/mcp_bridge.py`), watches for the model's call, returns it to your app and stops the CLI before
  it tries to run the tool. With Claude Code the tool call is the real one from the API (real `toolu_…` id).
- **Antigravity** cannot use MCP tools with the plain agent the gateway uses, so your tools are described in its
  instructions and the model replies in a fixed format that the gateway turns into a normal function call. Its
  own background-task tool cannot be switched off; now and then the model uses it and mentions it in an answer.
- `tool_choice` (`auto`, `any`/`required`, a named tool, `none`) is passed on as an instruction; the CLIs have no
  switch for it. Earlier tool calls and results in the conversation are sent to the CLI as text.
- Codex reports token usage only for finished turns, so a Codex tool-call response has usage 0.

## Structured output (JSON)

Ask for JSON the way each API does it: Anthropic `output_config.format` (`json_schema`), OpenAI
`response_format` / Responses `text.format` (`json_schema` or `json_object`), or Gemini
`responseMimeType: "application/json"` with `responseSchema` / `responseJsonSchema`. The answer text is the JSON.

| | How | Guaranteed? |
|---|---|---|
| Claude Code | `--json-schema` (Claude Code delivers the output through a tool of its own; the gateway returns just the JSON) | yes |
| Codex, `strict: true` | `--output-schema`, with OpenAI's strict-schema rules (`additionalProperties: false`, all properties required); a schema that breaks them gets the same 400 error as the API | yes |
| Codex, not strict, or `json_object` | asked for in its instructions, as the API's non-strict mode does | no |
| Antigravity | asked for in its instructions; code fences around the JSON are removed | no |

Anthropic's `output_config.effort` (`low` ... `max`) is passed to Claude Code's `--effort`.

## Model lists and token counts

- `GET /v1/models` and `GET /v1/models/{id}` answer in Anthropic's shape for the Anthropic SDK (Claude Code's
  models) and in OpenAI's shape for the OpenAI SDK (Codex's models). `GET /v1beta/models` lists Antigravity's
  models for the Google SDK. Creation dates are not known, so they are 1970-01-01 / 0.
- `POST /v1/messages/count_tokens` (Anthropic): Claude has no public tokenizer, so the gateway runs the request
  through Claude Code with a one-token answer and returns the input count it reports. That count includes the
  ~340 tokens Claude Code adds to every request, so it is higher than the real API's. Counts are cached.
- `POST /v1/responses/input_tokens` (OpenAI) and `:countTokens` (Gemini) use the official local tokenizers
  (`tiktoken` o200k_base; Google's local tokenizer for Gemini 2.5, the newest it knows). No model call is made.
  Only text is counted. The tokenizer files are downloaded once (into `~/.cache/rag-gateway`) and loaded in the
  background when the gateway starts; `GATEWAY_TOKENIZERS=0` skips that.

## Ready-started processes

Claude Code and Antigravity read the prompt from stdin, so the gateway can start them before a request arrives.
After a request, it starts the next process for the same setup (same model, system prompt, tools and
max_tokens) in the background. Measured on this machine: Antigravity 11.3 s cold, 2.2 s with a ready process;
Claude Code about 1 s faster. Codex reads its prompt before doing anything, so it gains nothing and always
starts fresh. A ready process is dropped after 10 minutes unused (`GATEWAY_WARM_IDLE_SECONDS`), at most 4 are
kept, and `GATEWAY_WARM_POOL=0` turns it off.

## How close is this to the real APIs?

Claude Code streams the real Anthropic events it receives, so Anthropic-format answers from Claude are the real
messages (real `msg_…` id, model and usage), filtered to the official SDK's fields. Codex and Antigravity do not
expose their raw API traffic, so the gateway builds those responses in the exact shape of each API. All four
formats are tested with the official SDKs: if the SDK parses it, it has that API's format.

| | Anthropic Messages | OpenAI Chat Completions | OpenAI Responses | Gemini |
|---|---|---|---|---|
| Endpoint | `/v1/messages` | `/v1/chat/completions` | `/v1/responses` | `/v1beta/models/{m}:generateContent`, `:streamGenerateContent` |
| Answered by | Claude Code | Codex | Codex | Antigravity |
| Streaming | real events | `chat.completion.chunk` + `[DONE]`, usage chunk | `response.*` events with sequence numbers | SSE (`alt=sse`) or JSON array |
| Errors | `{"type":"error",...}` + `request_id` | `{"error":{message,type,param,code}}` | same | `{"error":{code,message,status}}` |
| Gateway key | `x-api-key` | `Authorization: Bearer` | same | `x-goog-api-key` or `?key=` |

Differences to know about:

- **Token counts are real but include each CLI's own framing:** about 340 input tokens for Claude Code, about
  4,300 for Codex (its fixed instructions and tool framing), about 550 for Antigravity. Tools add their
  definitions on top.
- **max_tokens** is enforced for Claude only. Codex and Antigravity have no way to set it; the log says so.
- **Codex gives each answer whole**, so a Codex stream arrives in one piece. Claude and Antigravity stream as they write.
- **Sampling settings** (`temperature`, `top_p`, `stop`...) are accepted and ignored; the log lists them.
- **Multi-turn** works; earlier turns are sent as a `<conversation_history>` block inside the last message.
- **Not supported** (clear 400 errors): server tools (web search, code execution), audio, web URLs for files,
  assistant prefill, `previous_response_id`, more than one choice. Endpoints such as batches and embeddings
  return 404.
- Codex needs instructions, so a request without a system prompt gets "You are a helpful assistant.".

## Safety checks

- **Subscription only.** At startup each CLI's login is checked: Claude Code must use a claude.ai plan, Codex a
  ChatGPT login (API-key logins are refused), Antigravity a Google sign-in. Claude Code's `apiKeySource` is checked
  on every request, and Codex runs with `forced_login_method="chatgpt"`.
- **Clean environment for every CLI.** All `ANTHROPIC_*`, `OPENAI_*`, `CODEX_*`, `GEMINI_*`, `GOOGLE_*` and
  `CLAUDE*` variables are removed (API keys, base URLs, Bedrock/Vertex switches, parent Claude Code sessions).
  Kept: `CLAUDE_CODE_OAUTH_TOKEN`, `CLAUDE_CONFIG_DIR`, `CODEX_HOME`.
- **No tools, no projects.** Each request runs in a new empty temp folder that is deleted afterwards.
  Claude Code: `--tools "" --safe-mode --strict-mcp-config`; with tools, `--restricted --setting-sources ""
  --strict-mcp-config` and only the gateway's MCP server. Codex: read-only sandbox, `--ignore-user-config`,
  tool features disabled, `--ephemeral`, and a home folder of its own so your personal skills stay out of its
  prompt. Antigravity: a custom agent with `excludeDefaultComponents: true` (no default prompt, no built-in
  tools except its background-task tool), `--sandbox`.
- **This computer only.** The gateway listens on 127.0.0.1, refuses other addresses and other host names, and
  `python -m gateway` refuses a public address unless `GATEWAY_ALLOW_REMOTE=1`.
- **Gateway key.** Every API call needs `GATEWAY_API_KEYS` in the place its SDK sends keys.

## Privacy

- Claude Code adds a short block to every request that includes your **Claude account email**, your OS and the
  date. There is no supported switch to turn it off.
- Antigravity keeps each conversation in its own history (`~/.gemini/antigravity-cli/`); it has no flag to turn
  that off. Claude Code and Codex are run without saving sessions.
- Every question and answer is saved in `gateway.db` for the web page and the cache (file contents are not kept
  there, only their names and sizes). Delete it to clear, or set `GATEWAY_HISTORY=0` and `GATEWAY_CACHE=0`.
- Uploaded files stay in `files/` until you delete them (Files API `DELETE`, or the page's Uploaded files list).

## Cache

Identical requests are answered from `gateway.db` for 24 hours without running a CLI. The key covers the provider,
model, system prompt, messages and settings, not the API format, so the same question through OpenAI Chat and
OpenAI Responses shares one answer. Skip it for one request with the `cache-control: no-cache` header, or untick
"Reuse saved answers" on the page.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `GATEWAY_API_KEYS` | `local-test-key` | Comma-separated keys your apps may send |
| `GATEWAY_HOST` / `GATEWAY_PORT` | `127.0.0.1` / `8787` | Where to listen |
| `CLAUDE_BIN` / `CODEX_BIN` / `AGY_BIN` | `claude` / `codex` / `agy` | The CLIs; a missing one only disables its provider |
| `GATEWAY_MAX_CONCURRENCY` | `2` | CLI runs at the same time, per provider; others wait |
| `GATEWAY_QUEUE_TIMEOUT_SECONDS` | `120` | Wait for a free slot before an "overloaded" error |
| `GATEWAY_TIMEOUT_SECONDS` | `600` | Longest time one request may take |
| `GATEWAY_CACHE` / `GATEWAY_CACHE_TTL_SECONDS` | `1` / `86400` | Response cache on/off and lifetime |
| `GATEWAY_HISTORY` | `1` | Save questions and answers for the web page |
| `GATEWAY_DB` | `gateway.db` in the project folder | SQLite file |
| `GATEWAY_FILES_DIR` | `files/` in the project folder | Where uploaded files are stored |
| `GATEWAY_WARM_POOL` / `GATEWAY_WARM_IDLE_SECONDS` | `1` / `600` | Ready-started processes on/off, and how long an unused one is kept |
| `GATEWAY_TOKENIZERS` / `GATEWAY_TOKENIZER_CACHE` | `1` / `~/.cache/rag-gateway` | Load the OpenAI and Gemini tokenizers at startup, and where their files are kept |
| `GATEWAY_LOG_LEVEL` | `INFO` | `DEBUG` also logs prompts and every raw line a CLI prints |
| `GATEWAY_ALLOW_REMOTE` | `0` | Allow connections from other computers (not recommended) |

## Logs

Each request logs its `request-id`, the API format and provider, the exact CLI command, the exit code, stderr,
parameters that were ignored, the stop reason and token counts.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Fake `claude`, `codex` and `agy` scripts in `tests/` stand in for the real CLIs, so the 79 tests spend nothing.
They drive every API format with its official SDK, including files, images, tools, structured output, model
lists, token counts and ready-started processes. `pytest.ini` limits pytest to `tests/`, so the live checks below
never run by accident.

GitHub Actions (`.github/workflows/tests.yml`) runs these tests on Linux, macOS and Windows with Python 3.10 and
3.12 on every push. The gateway itself has been used live on Linux only.

Live checks against the real CLIs (they spend a little of your plans):

```bash
python test_client.py                      # every API format, text and streaming (7 small calls)
python feature_check.py                    # files, images, PDFs, tools, JSON, models, counts, ready processes (~20 calls)
python feature_check.py tools json         # only some sections: claude, codex, agy, pdf, image, tools, json,
                                           # models, counts, warm
```

## Files

```
gateway/core.py                 provider-neutral request (text, attachments, tool calls), events, result, errors
gateway/dialects/               the API formats: anthropic.py, openai_chat.py, openai_responses.py, gemini.py
gateway/providers/              the CLIs: claude_code.py, codex.py, antigravity.py (+ routing in __init__.py)
gateway/process.py              running a CLI: clean environment, temp folder, JSON lines, ready-started processes
gateway/attachments.py          images and documents: decoding, type checks, PDF and text extraction
gateway/files.py                Files API ids and file objects for each API
gateway/mcp_bridge.py           the MCP server that offers your tools to Claude Code and Codex
gateway/counting.py             token counts with the local tokenizers (OpenAI and Gemini)
.github/workflows/tests.yml     the offline tests on Linux, macOS and Windows
gateway/service.py              one request end to end: route, cache, run, respond, record
gateway/store.py                SQLite cache, history and saved plan-usage readings
gateway/main.py                 HTTP routes, gateway key, this-computer-only checks, the web page's data
gateway/static/index.html       the web page
rag_app.py                      a sample RAG app in one file, for any of the three providers
test_client.py                  live check of every format with the official SDKs
feature_check.py                live check of files, images, PDFs, tools and ready-started processes
poc/app.py                      the first proof of concept, kept for reference
```

## Use it for your own testing only

Your Claude, ChatGPT and Google plans are for your own use. Use this gateway on your own computer while you build.
Do not share it, expose it to other people, or serve real users through it. For production, use API keys.

## License

MIT. See [LICENSE](LICENSE).
