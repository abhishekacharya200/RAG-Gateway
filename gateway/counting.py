"""Token counts for the count endpoints of the OpenAI and Gemini formats, with the official local tokenizers.

Claude has no public tokenizer, so Anthropic's count_tokens runs Claude Code instead (see service.py).
The tokenizer files are downloaded once (slowly on some connections), cached, and loaded in the background
when the gateway starts. Only text is counted: images and files add nothing here.
"""

import json
import logging
import os
import threading
from typing import List

from .core import ChatRequest, GatewayError, ToolCall, ToolResult

log = logging.getLogger("gateway.counting")

CACHE_DIR = os.environ.get("GATEWAY_TOKENIZER_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "rag-gateway")
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(CACHE_DIR, "tiktoken"))

# o200k_base is the tokenizer of OpenAI's GPT-4o and GPT-5 families. Google's local tokenizer knows the
# Gemini 2.5 models; newer Gemini models may count slightly differently.
OPENAI_ENCODING = "o200k_base"
GEMINI_TOKENIZER_MODEL = "gemini-2.5-flash"

_tokenizers = {}
_ready = threading.Event()


def _load() -> None:
    try:
        import tiktoken
        _tokenizers["openai"] = tiktoken.get_encoding(OPENAI_ENCODING)
    except Exception as exc:
        log.warning("OpenAI tokenizer not available: %s", exc)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "experimental" notice
            from google.genai.local_tokenizer import LocalTokenizer
            _tokenizers["gemini"] = LocalTokenizer(model_name=GEMINI_TOKENIZER_MODEL)
    except Exception as exc:
        log.warning("Gemini tokenizer not available: %s", exc)
    _ready.set()
    log.info("tokenizers ready: %s", ", ".join(sorted(_tokenizers)) or "none")


def warm_up() -> None:
    """Loads the tokenizers in the background (the first time this downloads them)."""
    threading.Thread(target=_load, name="tokenizers", daemon=True).start()


def _tokenizer(name: str, timeout: float = 120):
    _ready.wait(timeout)
    if name not in _tokenizers:
        raise GatewayError("api_error", f"The {name} tokenizer is not available on the gateway host "
                           "(it downloads its data once; check the internet connection and the gateway log).",
                           retry=False)
    return _tokenizers[name]



def _texts_of(turn) -> List[str]:
    out = []
    for part in turn.parts:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, ToolCall):
            out.append(f"{part.name} {json.dumps(part.arguments)}")
        elif isinstance(part, ToolResult):
            out.append(part.content)
    return out


def count_openai(req: ChatRequest) -> int:
    encoding = _tokenizer("openai")
    texts = [req.system] + [x for turn in req.turns for x in _texts_of(turn)]
    tokens = sum(len(encoding.encode(t)) for t in texts if t)
    tokens += sum(len(encoding.encode(json.dumps({"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}))) for t in req.tools)
    # Each message costs a few tokens of framing (OpenAI's documented rule for chat models).
    return tokens + 3 * (len(req.turns) + (1 if req.system else 0)) + 3


def count_gemini(req: ChatRequest) -> int:
    from google.genai import types
    tokenizer = _tokenizer("gemini")
    contents = [types.Content(role="model" if t.role == "assistant" else "user",
                              parts=[types.Part(text=x) for x in _texts_of(t)] or [types.Part(text="")])
                for t in req.turns]
    config = types.CountTokensConfig(system_instruction=req.system) if req.system else None
    return tokenizer.count_tokens(contents, config=config).total_tokens

