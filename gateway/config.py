"""Gateway settings. Every value can be changed with an environment variable."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

DEFAULT_DB = str(Path(__file__).resolve().parent.parent / "gateway.db")
DEFAULT_FILES = str(Path(__file__).resolve().parent.parent / "files")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8787
    # Keys your app must send as x-api-key. They only protect the gateway; they are never sent to Anthropic.
    api_keys: tuple = ("local-test-key",)
    # The CLIs that answer requests. A missing one only disables that provider. A tuple is a command with
    # a prefix, e.g. (python, fake_claude.py) in the tests.
    claude_bin: Union[str, Sequence[str]] = "claude"
    codex_bin: Union[str, Sequence[str]] = "codex"
    agy_bin: Union[str, Sequence[str]] = "agy"
    # CLI processes allowed at the same time, per provider. Extra requests wait for a free slot.
    max_concurrency: int = 2
    queue_timeout: float = 120.0
    request_timeout: float = 600.0
    cache_enabled: bool = True
    cache_ttl: int = 24 * 3600
    history_enabled: bool = True
    db_path: str = DEFAULT_DB
    # Uploaded files (Files API) are stored here, named by their id.
    files_dir: str = DEFAULT_FILES
    # Keep a Claude Code / Antigravity process started ahead of time for recently used setups.
    warm_pool: bool = True
    warm_idle_seconds: float = 600.0
    # Load the OpenAI and Gemini tokenizers at startup, for their token-count endpoints.
    load_tokenizers: bool = True
    # Refuse connections from other machines unless this is set.
    allow_remote: bool = False
    log_level: str = "INFO"


def load_settings() -> Settings:
    keys = os.environ.get("GATEWAY_API_KEYS", "local-test-key")
    return Settings(
        host=os.environ.get("GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("GATEWAY_PORT", "8787")),
        api_keys=tuple(k.strip() for k in keys.split(",") if k.strip()) or ("local-test-key",),
        claude_bin=os.environ.get("CLAUDE_BIN", "claude"),
        codex_bin=os.environ.get("CODEX_BIN", "codex"),
        agy_bin=os.environ.get("AGY_BIN", "agy"),
        max_concurrency=int(os.environ.get("GATEWAY_MAX_CONCURRENCY", "2")),
        queue_timeout=float(os.environ.get("GATEWAY_QUEUE_TIMEOUT_SECONDS", "120")),
        request_timeout=float(os.environ.get("GATEWAY_TIMEOUT_SECONDS", "600")),
        cache_enabled=_env_bool("GATEWAY_CACHE", True),
        cache_ttl=int(os.environ.get("GATEWAY_CACHE_TTL_SECONDS", str(24 * 3600))),
        history_enabled=_env_bool("GATEWAY_HISTORY", True),
        db_path=os.environ.get("GATEWAY_DB", DEFAULT_DB),
        files_dir=os.environ.get("GATEWAY_FILES_DIR", DEFAULT_FILES),
        warm_pool=_env_bool("GATEWAY_WARM_POOL", True),
        warm_idle_seconds=float(os.environ.get("GATEWAY_WARM_IDLE_SECONDS", "600")),
        load_tokenizers=_env_bool("GATEWAY_TOKENIZERS", True),
        allow_remote=_env_bool("GATEWAY_ALLOW_REMOTE", False),
        log_level=os.environ.get("GATEWAY_LOG_LEVEL", "INFO").upper(),
    )
