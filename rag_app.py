"""A small RAG app in one file. It uses the gateway exactly the way a real app would, through the official
SDK of whichever provider you pick.

For every question it prints two things and nothing else: the exact API request the SDK sends and the exact
response that comes back (both captured from the SDK's own HTTP client).

Start the gateway first (python -m gateway), then:

    python rag_app.py                                  # Claude models, Anthropic Messages API
    python rag_app.py --provider codex                 # OpenAI models, OpenAI Chat Completions API
    python rag_app.py --provider antigravity           # Gemini models, Gemini API
    python rag_app.py "Can I get a refund?"            # ask one question and exit
    python rag_app.py --docs ~/notes                   # use your own .txt and .md files
    python rag_app.py --provider codex --model gpt-5.6-terra

Production: point each SDK at the real API with its usual variables, for example
ANTHROPIC_BASE_URL=https://api.anthropic.com plus a real ANTHROPIC_API_KEY. Nothing in this file changes.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import anthropic
import httpx
import openai
from google import genai
from google.genai import errors as genai_errors, types

GATEWAY = "http://127.0.0.1:8787"
GATEWAY_KEY = "local-test-key"
TOP_K = 3
MAX_TOKENS = 800

# Each SDK talks to the gateway by default, or to the real API when its usual variables say so.
PROVIDERS = {
    "claude": {"model": "claude-haiku-4-5-20251001",
               "base_url": os.environ.get("ANTHROPIC_BASE_URL", GATEWAY),
               "api_key": os.environ.get("ANTHROPIC_API_KEY", GATEWAY_KEY)},
    "codex": {"model": "gpt-6-luna",
              "base_url": os.environ.get("OPENAI_BASE_URL", f"{GATEWAY}/v1"),
              "api_key": os.environ.get("OPENAI_API_KEY", GATEWAY_KEY)},
    "antigravity": {"model": "gemini-3.8-flash-low",
                    "base_url": os.environ.get("GOOGLE_GEMINI_BASE_URL", GATEWAY),
                    "api_key": os.environ.get("GEMINI_API_KEY", GATEWAY_KEY)},
}

SYSTEM_PROMPT = (
    "You are the help-center assistant. Answer using only the passages in <documents>. "
    "Cite the passages you used as [1], [2] and so on. If the passages do not contain the answer, "
    "say that the documents don't cover it instead of guessing. Keep answers short."
)

# A made-up product's help center, so there is something to search without any setup.
SAMPLE_DOCS = {
    "plans.md": "Nimbus Drive has three plans. Free includes 5 GB of storage. Plus costs $4 per month for "
                "200 GB. Pro costs $12 per month for 2 TB. Paying yearly gives two months free on Plus and Pro.",
    "refunds.md": "Yearly plans can be refunded within 30 days of purchase. Monthly plans are not refunded, "
                  "but you can cancel at any time and keep access until the end of the paid month. Refunds "
                  "reach the original payment method within 5 to 7 business days.",
    "sharing.md": "Share links expire after 7 days by default. On Pro you can choose any expiry up to one year "
                  "or no expiry at all. Password-protected share links are available on Pro only.",
    "security.md": "Files are encrypted with AES-256 at rest and TLS 1.3 in transit. Two-factor authentication "
                   "works with any authenticator app. Nimbus Drive holds a SOC 2 Type II report.",
    "support.md": "Support answers email Monday to Friday, 9:00 to 18:00 CET. Pro customers also get live chat "
                  "around the clock. The average first reply time for email is 4 hours.",
    "trash.md": "Deleted files stay in the trash for 30 days on Free and Plus, and for 180 days on Pro. Free "
                "accounts that are inactive for 12 months are deleted after two warning emails.",
    "apps.md": "The desktop app runs on Windows 10 or later, macOS 12 or later, and Ubuntu 22.04 or later. "
               "Selective sync lets you choose which folders are kept on each computer.",
}

# ---------- retrieval: simple BM25 keyword search, no extra packages ----------

STOPWORDS = set("a an and are as at be by can do does for from how i if in is it of on or so the to "
                "what when where which who why will with you your my me we our".split())


def tokenize(text: str) -> list:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


@dataclass
class Chunk:
    source: str
    text: str


def load_docs(folder: str) -> list:
    """Splits every .txt and .md file into chunks of about 1,000 characters along paragraph breaks."""
    chunks = []
    for path in sorted(Path(folder).expanduser().rglob("*")):
        if path.suffix.lower() not in (".txt", ".md") or not path.is_file():
            continue
        current = ""
        for para in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace")):
            if current and len(current) + len(para) > 1000:
                chunks.append(Chunk(path.name, current.strip()))
                current = ""
            current += para + "\n\n"
        if current.strip():
            chunks.append(Chunk(path.name, current.strip()))
    return chunks


class BM25:
    def __init__(self, chunks: list, k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.counts = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = sum(self.lengths) / max(len(chunks), 1)
        df = Counter(term for counts in self.counts for term in counts)
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int) -> list:
        terms = tokenize(query)
        scored = []
        for i, counts in enumerate(self.counts):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scored.append((score, i))
        return [self.chunks[i] for _, i in sorted(scored, reverse=True)[:k]]


# ---------- asking the model ----------

def show(label: str, body) -> None:
    print(f"\n{label}\n{json.dumps(body, indent=2, ensure_ascii=False)}")


class Wire:
    """The SDK's HTTP client, with hooks that keep the last request and response so they print exactly."""

    def __init__(self):
        self.request = None
        self.response = None
        self.http = httpx.Client(timeout=600, event_hooks={"request": [self._on_request],
                                                          "response": [self._on_response]})

    def _on_request(self, request: httpx.Request) -> None:
        self.request = request
        self.response = None

    def _on_response(self, response: httpx.Response) -> None:
        response.read()
        self.response = response

    def show(self) -> None:
        if self.request is not None:
            show(f"{self.request.method} {self.request.url}", json.loads(self.request.content))
        if self.response is not None:
            show(f"{self.response.status_code} {self.response.reason_phrase}", json.loads(self.response.text))


def make_client(provider: str, wire: Wire):
    settings = PROVIDERS[provider]
    if provider == "claude":
        return anthropic.Anthropic(base_url=settings["base_url"], api_key=settings["api_key"], http_client=wire.http)
    if provider == "codex":
        return openai.OpenAI(base_url=settings["base_url"], api_key=settings["api_key"], http_client=wire.http)
    return genai.Client(api_key=settings["api_key"],
                        http_options=types.HttpOptions(base_url=settings["base_url"], httpx_client=wire.http))


def call(provider: str, client, model: str, documents: str, question: str, history: list) -> str:
    """One question in the provider's own API format. history holds earlier (role, text) turns."""
    context = f"<documents>\n{documents}\n</documents>"
    if provider == "claude":
        message = client.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
            messages=[*({"role": role, "content": text} for role, text in history),
                      {"role": "user", "content": [{"type": "text", "text": context},
                                                   {"type": "text", "text": question}]}])
        return message.content[0].text
    if provider == "codex":
        completion = client.chat.completions.create(
            model=model, max_completion_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      *({"role": role, "content": text} for role, text in history),
                      {"role": "user", "content": [{"type": "text", "text": context},
                                                   {"type": "text", "text": question}]}])
        return completion.choices[0].message.content
    response = client.models.generate_content(
        model=model,
        contents=[*(types.Content(role="model" if role == "assistant" else "user", parts=[types.Part(text=text)])
                    for role, text in history),
                  types.Content(role="user", parts=[types.Part(text=context), types.Part(text=question)])],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, max_output_tokens=MAX_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="A small RAG app that talks to a real API or to the gateway.")
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="claude",
                        help="claude (Anthropic API), codex (OpenAI API) or antigravity (Gemini API)")
    parser.add_argument("--model", help="model name (default: a light model of the provider)")
    parser.add_argument("--docs", help="folder with .txt or .md files (default: built-in sample documents)")
    args = parser.parse_args()

    if args.docs:
        chunks = load_docs(args.docs)
        if not chunks:
            sys.exit(f"No .txt or .md files found in {args.docs}")
    else:
        chunks = [Chunk(name, text) for name, text in SAMPLE_DOCS.items()]
    index = BM25(chunks)
    wire = Wire()
    client = make_client(args.provider, wire)
    model = args.model or PROVIDERS[args.provider]["model"]

    questions = [args.question] if args.question else None
    history = []
    while True:
        if questions is not None:
            if not questions:
                break
            question = questions.pop()
        else:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/reset":  # forget the conversation
            history = []
            continue
        # Follow-ups like "what about Pro?" search better together with the previous question.
        last_question = history[-2][1] if history else ""
        hits = index.search(f"{last_question} {question}", TOP_K)
        documents = "\n".join(f'<document index="{i}" source="{c.source}">\n{c.text}\n</document>'
                              for i, c in enumerate(hits, 1)) or "(no matching passages)"
        try:
            answer = call(args.provider, client, model, documents, question, history)
        except (anthropic.APIConnectionError, openai.APIConnectionError, httpx.ConnectError):
            sys.exit(f"\nCan't reach {PROVIDERS[args.provider]['base_url']}. Is the gateway running? "
                     "Start it with: python -m gateway")
        except (anthropic.APIStatusError, openai.APIStatusError, genai_errors.APIError):
            wire.show()  # the error, exactly as the API returned it
            continue
        wire.show()
        # Earlier turns are kept without their documents, to keep requests small.
        history = (history + [("user", question), ("assistant", answer)])[-6:]


if __name__ == "__main__":
    main()
