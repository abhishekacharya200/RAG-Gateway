"""Shared test setup. Fake CLIs stand in for claude, codex and agy, so tests spend nothing from any plan.

The official SDKs are used as clients wherever possible: if an SDK parses a response, it has that API's format.
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.main import create_app

HERE = Path(__file__).parent
FAKE_CLAUDE = str(HERE / "fake_claude.py")
FAKE_CODEX = str(HERE / "fake_codex.py")
FAKE_AGY = str(HERE / "fake_agy.py")
KEY = "test-key"


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """Returns start(claude=..., codex=..., agy=..., **settings) -> TestClient, plus start.runs(cli)."""
    record = tmp_path / "runs.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_RECORD", str(record))
    # Keys and session variables that must never reach a CLI.
    for name, value in {"ANTHROPIC_API_KEY": "sk-ant-x", "OPENAI_API_KEY": "sk-proj-x", "GEMINI_API_KEY": "AIza-x",
                        "CLAUDECODE": "1"}.items():
        monkeypatch.setenv(name, value)
    clients = []

    def start(claude="ok", codex="ok", agy="ok", auth="ok", codex_login="chatgpt", agy_login="ok",
              client_addr=("testclient", 50000), **overrides):
        monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", claude)
        monkeypatch.setenv("FAKE_CODEX_SCENARIO", codex)
        monkeypatch.setenv("FAKE_AGY_SCENARIO", agy)
        monkeypatch.setenv("FAKE_AUTH", auth)
        monkeypatch.setenv("FAKE_CODEX_LOGIN", codex_login)
        monkeypatch.setenv("FAKE_AGY_LOGIN", agy_login)
        # The fakes run through Python, so the tests work on Windows too (no shebang there).
        options = dict(claude_bin=(sys.executable, FAKE_CLAUDE), codex_bin=(sys.executable, FAKE_CODEX),
                       agy_bin=(sys.executable, FAKE_AGY), api_keys=(KEY,), db_path=str(tmp_path / "gateway.db"),
                       files_dir=str(tmp_path / "files"), load_tokenizers=False)
        options.update(overrides)
        client = TestClient(create_app(Settings(**options)), client=client_addr)
        client.__enter__()
        clients.append(client)
        return client

    def runs(cli=None):
        if not record.exists():
            return []
        rows = [json.loads(line) for line in record.read_text().splitlines()]
        return [r for r in rows if cli is None or r["cli"] == cli]

    start.runs = runs
    yield start
    for client in clients:
        client.__exit__(None, None, None)


def sse_data(text: str) -> list:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


def history(client) -> list:
    return client.get("/ui/api/history", headers={"x-api-key": KEY}).json()["items"]


def tiny_png(color=(220, 20, 20), size=8) -> bytes:
    import struct
    import zlib
    raw = b"".join(b"\x00" + bytes(color) * size for _ in range(size))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def tiny_pdf(text: str) -> bytes:
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
