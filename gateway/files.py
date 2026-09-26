"""Uploaded files: ids in each API's style, and each API's file object.

One store holds the files. Every API format can use a file uploaded through any other one:
OpenAI ids look like file-<id>, Anthropic ids like file_011<id>, Gemini names like files/<id>.
"""

import base64
import hashlib
import re
import secrets
import string
import time
from typing import Optional

_ID_CHARS = string.ascii_lowercase + string.digits


def new_core_id() -> str:
    return "".join(secrets.choice(_ID_CHARS) for _ in range(16))


def core_id(ref: str) -> Optional[str]:
    """The stored id from any API's reference: file-x, file_011x, files/x or a Gemini file URI."""
    ref = (ref or "").strip()
    for pattern in (r"^file-([a-z0-9]{16})$", r"^file_011([a-z0-9]{16})$", r"(?:^|/)files/([a-z0-9]{16})$"):
        match = re.search(pattern, ref)
        if match:
            return match.group(1)
    return None


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int(ts % 1 * 1_000_000):06d}Z"


def openai_object(row: dict) -> dict:
    return {"id": f"file-{row['id']}", "object": "file", "bytes": row["size"], "created_at": int(row["created_at"]),
            "filename": row["filename"], "purpose": row["purpose"] or "user_data", "status": "processed",
            "expires_at": None, "status_details": None, "path": row["path"]}


def anthropic_object(row: dict) -> dict:
    return {"id": f"file_011{row['id']}", "type": "file", "filename": row["filename"],
            "mime_type": row["media_type"], "size_bytes": row["size"], "created_at": _iso(row["created_at"]),
            "downloadable": False, "path": row["path"]}


def gemini_object(row: dict, base_url: str) -> dict:
    created = _iso(row["created_at"])
    return {"name": f"files/{row['id']}", "displayName": row["filename"], "mimeType": row["media_type"],
            "sizeBytes": str(row["size"]), "createTime": created, "updateTime": created,
            "sha256Hash": base64.b64encode(bytes.fromhex(row["sha256"])).decode(),
            "uri": f"{base_url.rstrip('/')}/v1beta/files/{row['id']}", "state": "ACTIVE", "source": "UPLOADED",
            "path": row["path"]}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
