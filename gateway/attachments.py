"""Images and documents in requests: decoding, type checks, and turning documents into text."""

import base64
import binascii
import io
import mimetypes
import os
import re
from typing import Optional

from .core import Attachment, GatewayError

IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
TEXT_TYPES = {"application/json", "application/xml", "application/yaml", "application/x-yaml", "application/csv",
              "application/javascript", "application/x-python", "application/x-sh", "application/sql",
              "application/x-ndjson", "application/toml"}
# Extensions that are text even when the system does not know their type.
TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml",
                   ".xml", ".html", ".htm", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c",
                   ".h", ".cpp", ".cs", ".rb", ".php", ".sh", ".sql", ".log", ".ini", ".cfg", ".rst", ".tex"}
MAX_BYTES = 32 * 1024 * 1024


def _invalid(message: str) -> GatewayError:
    return GatewayError("invalid_request", message, retry=False)


def guess_type(filename: Optional[str], fallback: str = "application/octet-stream") -> str:
    if not filename:
        return fallback
    if os.path.splitext(filename)[1].lower() in TEXT_EXTENSIONS:
        known = mimetypes.guess_type(filename)[0]
        return known if known and (known.startswith("text/") or known in TEXT_TYPES) else "text/plain"
    return mimetypes.guess_type(filename)[0] or fallback


def is_text(media_type: str) -> bool:
    return media_type.startswith("text/") or media_type in TEXT_TYPES


def check_supported(media_type: str, where: str) -> None:
    if media_type in IMAGE_TYPES or media_type == "application/pdf" or is_text(media_type):
        return
    raise _invalid(f"{where}: files of type '{media_type}' are not supported by this gateway. "
                   "Send images (PNG, JPEG, GIF, WebP), PDFs or text files.")


def make(data: bytes, media_type: str, filename: Optional[str], where: str) -> Attachment:
    if len(data) > MAX_BYTES:
        raise GatewayError("too_large", f"{where}: the file is larger than {MAX_BYTES // (1024 * 1024)} MB.",
                           retry=False)
    media_type = (media_type or "application/octet-stream").split(";")[0].strip().lower()
    if media_type == "application/octet-stream":
        media_type = guess_type(filename)
    check_supported(media_type, where)
    return Attachment("image" if media_type in IMAGE_TYPES else "document", media_type, data, filename)


def from_base64(data: str, media_type: str, filename: Optional[str], where: str) -> Attachment:
    cleaned = re.sub(r"\s+", "", data or "")
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError, TypeError):
        # The Google SDK sends URL-safe base64 (- and _), which the Gemini API accepts too.
        try:
            if not re.fullmatch(r"[A-Za-z0-9_-]*=*", cleaned):
                raise ValueError
            raw = base64.urlsafe_b64decode(cleaned + "=" * (-len(cleaned) % 4))
        except (binascii.Error, ValueError):
            raise _invalid(f"{where}: the data is not valid base64.")
    return make(raw, media_type, filename, where)


def from_data_url(url: str, filename: Optional[str], where: str) -> Attachment:
    match = re.match(r"data:([^;,]*)((?:;[^;,]*)*),(.*)", url or "", re.S)
    if not match:
        raise _invalid(f"{where}: only data: URLs or uploaded files are supported by this gateway, not web URLs.")
    media_type, params, payload = match.groups()
    if ";base64" in params:
        return from_base64(payload, media_type, filename, where)
    from urllib.parse import unquote
    return make(unquote(payload).encode("utf-8"), media_type or "text/plain", filename, where)


def document_text(attachment: Attachment) -> str:
    """The text of a document, for CLIs that cannot read the file itself."""
    if attachment.media_type == "application/pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(attachment.data))
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
        except Exception:
            raise _invalid(f"{attachment.filename or 'The PDF'} could not be read.")
        text = "\n\n".join(f"--- page {i} ---\n{p}" for i, p in enumerate(pages, 1) if p)
        return text or "(This PDF has no text layer; it may be scanned images.)"
    return attachment.data.decode("utf-8", "replace")


def extension(media_type: str) -> str:
    return {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp",
            "application/pdf": ".pdf"}.get(media_type) or mimetypes.guess_extension(media_type) or ".bin"
