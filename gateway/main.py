"""FastAPI app: the four API formats, the web page, and the page's small JSON API.

    POST /v1/messages                                   Anthropic Messages      -> Claude Code
    POST /v1/chat/completions                           OpenAI Chat Completions -> Codex
    POST /v1/responses                                  OpenAI Responses        -> Codex
    POST /v1beta/models/{model}:generateContent         Gemini                  -> Antigravity
    POST /v1beta/models/{model}:streamGenerateContent   Gemini (streaming)      -> Antigravity

    POST/GET/DELETE /v1/files ...                       Files API (Anthropic or OpenAI shape, by the SDK's headers)
    POST /upload/v1beta/files, GET/DELETE /v1beta/files Files API (Gemini shape)

A model name like "codex/gpt-6-luna" or "agy/gemini-3.8-flash-low" sends a request to another provider.
"""

import hmac
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import attachments, counting, files
from .config import Settings, load_settings
from .core import GatewayError, request_id
from .dialects import ANTHROPIC, DIALECTS, GEMINI, OPENAI_CHAT, OPENAI_RESPONSES, Dialect
from .providers import build_providers
from .service import Gateway
from .store import Store

log = logging.getLogger("gateway")

STATIC = Path(__file__).parent / "static"
LOCAL_CLIENTS = {"127.0.0.1", "::1", "testclient"}
# Host names a browser uses for this computer. Anything else (a web page's own domain pointed at
# 127.0.0.1, known as DNS rebinding) could read the web page, which carries the gateway key.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "testserver"}


def _gateway_key(request: Request) -> Optional[str]:
    """Wherever each SDK puts its key: x-api-key (Anthropic), Bearer (OpenAI), x-goog-api-key or ?key= (Gemini)."""
    bearer = request.headers.get("authorization", "")
    return (request.headers.get("x-api-key") or (bearer[7:].strip() if bearer.lower().startswith("bearer ") else None)
            or request.headers.get("x-goog-api-key") or request.query_params.get("key"))


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    store = Store(settings.db_path, settings.files_dir)
    gateway = Gateway(settings, store, build_providers(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            gateway.startup()
            if settings.load_tokenizers:
                counting.warm_up()
        except RuntimeError as exc:
            log.error("\n\nGATEWAY NOT STARTED: %s\n", exc)
            raise
        log.info("Gateway ready at http://%s:%s  (web page at /; cache %s, history %s)", settings.host, settings.port,
                 "on" if settings.cache_enabled else "off", "on" if settings.history_enabled else "off")
        yield
        await gateway.shutdown()

    app = FastAPI(title="rag-gateway", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.store = store
    app.state.gateway = gateway

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if settings.allow_remote:
            return await call_next(request)
        client = request.client.host if request.client else ""
        host = request.headers.get("host", "").lower()
        host = host[:host.index("]") + 1] if host.startswith("[") and "]" in host else host.split(":")[0]
        if client not in LOCAL_CLIENTS or host not in LOCAL_HOSTS:
            log.warning("refused request from %s for host %r", client, host)
            return _error(ANTHROPIC, GatewayError("permission", "This gateway only accepts requests from this "
                                                  "computer, addressed to 127.0.0.1 or localhost."), request_id())
        return await call_next(request)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        kind = {404: "not_found", 405: "invalid_request"}.get(exc.status_code, "api_error")
        return _error(ANTHROPIC, GatewayError(kind, str(exc.detail)), request_id())

    def check_key(request: Request) -> None:
        key = _gateway_key(request)
        if not key:
            raise GatewayError("authentication", "The gateway key is missing.", retry=False, code="missing_key")
        if not any(hmac.compare_digest(key.encode(), k.encode()) for k in settings.api_keys):
            raise GatewayError("authentication", "The gateway key is wrong.", retry=False, code="bad_key")

    async def api_call(dialect: Dialect, request: Request, model: Optional[str] = None,
                       options: Optional[dict] = None):
        rid = request_id()
        request.state.request_id = rid
        try:
            check_key(request)  # before reading the body, as the real APIs do
            try:
                body = await request.json()
            except ValueError:
                raise GatewayError("invalid_request", "The request body is not valid JSON.", retry=False)
            if not isinstance(body, dict):
                raise GatewayError("invalid_request", "The request body must be a JSON object.", retry=False)
            return await gateway.handle(dialect, request, body, model=model, options=options)
        except GatewayError as err:
            return _error(dialect, err, rid)

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        return await api_call(ANTHROPIC, request)

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        return await api_call(OPENAI_CHAT, request)

    @app.post("/v1/responses")
    async def openai_responses(request: Request):
        return await api_call(OPENAI_RESPONSES, request)

    @app.post("/v1beta/models/{target:path}")
    @app.post("/v1/models/{target:path}")
    async def gemini(request: Request, target: str):
        model, _, method = target.rpartition(":")
        if method == "countTokens":
            return await gemini_count(request, model)
        if method == "generateContent":
            options = {}
        elif method == "streamGenerateContent":
            sse = request.query_params.get("alt") == "sse"
            options = {"sse": sse, "array": not sse}
        else:
            return _error(GEMINI, GatewayError("not_found", f"Method not found: {method or target}"), request_id())
        return await api_call(GEMINI, request, model=model, options=options)

    # ---------- token counts ----------

    async def counted(dialect: Dialect, request: Request, count):
        rid = request_id()
        request.state.request_id = rid
        try:
            check_key(request)
            try:
                body = await request.json()
            except ValueError:
                raise GatewayError("invalid_request", "The request body is not valid JSON.", retry=False)
            if not isinstance(body, dict):
                raise GatewayError("invalid_request", "The request body must be a JSON object.", retry=False)
            return JSONResponse(await count(body), headers={"request-id": rid})
        except GatewayError as err:
            return _error(dialect, err, rid)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count(request: Request):
        return await counted(ANTHROPIC, request, lambda body: gateway.count_claude(ANTHROPIC, request, body))

    @app.post("/v1/responses/input_tokens")
    async def openai_count(request: Request):
        async def count(body):
            req = OPENAI_RESPONSES.parse(body, files=gateway.file)
            return {"object": "response.input_tokens", "input_tokens": counting.count_openai(req)}
        return await counted(OPENAI_RESPONSES, request, count)

    async def gemini_count(request: Request, model: str):
        async def count(body):
            inner = body.get("generateContentRequest") or body.get("generate_content_request") or body
            req = GEMINI.parse({k: v for k, v in inner.items() if k != "model"}, model=model, files=gateway.file)
            total = counting.count_gemini(req)
            return {"totalTokens": total, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": total}]}
        return await counted(GEMINI, request, count)

    # ---------- model lists ----------

    def models_of(provider_id: str) -> list:
        provider = gateway.providers[provider_id]
        return provider.status.models if provider.status.ready else []

    def anthropic_model(m: dict) -> dict:
        return {"type": "model", "id": m["id"], "display_name": m["label"], "created_at": "1970-01-01T00:00:00Z"}

    def openai_model(m: dict) -> dict:
        return {"id": m["id"], "object": "model", "created": 0, "owned_by": "openai"}

    def gemini_model(m: dict) -> dict:
        return {"name": f"models/{m['id']}", "version": "", "displayName": m["label"],
                "description": f"{m['label']}, answered through Antigravity",
                "supportedGenerationMethods": ["generateContent", "streamGenerateContent", "countTokens"]}

    def wants_gemini(request: Request) -> bool:
        return bool(request.headers.get("x-goog-api-key") or request.query_params.get("key"))

    @app.get("/v1/models")
    async def list_models(request: Request):
        dialect = GEMINI if wants_gemini(request) else files_dialect(request)

        async def action():
            if dialect is GEMINI:
                return {"models": [gemini_model(m) for m in models_of("antigravity")]}
            if dialect is ANTHROPIC:
                data = [anthropic_model(m) for m in models_of("claude")]
                return {"data": data, "has_more": False, "first_id": data[0]["id"] if data else None,
                        "last_id": data[-1]["id"] if data else None}
            return {"object": "list", "data": [openai_model(m) for m in models_of("codex")]}
        return await file_call(dialect, request, action)

    @app.get("/v1/models/{model_id:path}")
    async def get_model(request: Request, model_id: str):
        dialect = GEMINI if wants_gemini(request) else files_dialect(request)

        async def action():
            provider, shape = {GEMINI: ("antigravity", gemini_model), ANTHROPIC: ("claude", anthropic_model)}.get(
                dialect, ("codex", openai_model))
            wanted = model_id.removeprefix("models/")
            for m in models_of(provider):
                if m["id"] == wanted:
                    return shape(m)
            raise GatewayError("not_found", f"model: {model_id}" if dialect is ANTHROPIC else
                               f"The model `{model_id}` does not exist or you do not have access to it.", retry=False)
        return await file_call(dialect, request, action)

    @app.get("/v1beta/models")
    async def gemini_models(request: Request):
        async def action():
            return {"models": [gemini_model(m) for m in models_of("antigravity")]}
        return await file_call(GEMINI, request, action)

    @app.get("/v1beta/models/{model_id:path}")
    async def gemini_get_model(request: Request, model_id: str):
        async def action():
            wanted = model_id.removeprefix("models/")
            for m in models_of("antigravity"):
                if m["id"] == wanted:
                    return gemini_model(m)
            raise GatewayError("not_found", f"models/{wanted} is not found for API version v1beta.", retry=False)
        return await file_call(GEMINI, request, action)

    # ---------- Files API ----------

    def files_dialect(request: Request) -> Dialect:
        """The Anthropic SDK sends anthropic-version; the OpenAI SDK does not."""
        return ANTHROPIC if request.headers.get("anthropic-version") else OPENAI_CHAT

    def file_object(dialect: Dialect, row: dict) -> dict:
        return files.anthropic_object(row) if dialect is ANTHROPIC else files.openai_object(row)

    def save_file(filename: str, media_type: str, data: bytes, purpose: Optional[str]) -> dict:
        attachment = attachments.make(data, media_type, filename, "file")  # size and type checks
        return store.file_add(files.new_core_id(), filename, attachment.media_type, data, files.sha256(data),
                              purpose, attachments.extension(attachment.media_type))

    async def file_call(dialect: Dialect, request: Request, action):
        rid = request_id()
        try:
            check_key(request)
            return await action()
        except GatewayError as err:
            return _error(dialect, err, rid)

    def find_file(reference: str) -> dict:
        core = files.core_id(reference)
        row = store.file_get(core) if core else None
        if row is None:
            raise GatewayError("not_found", f"File not found: {reference}", retry=False)
        return row

    @app.post("/v1/files")
    async def upload_file(request: Request):
        dialect = files_dialect(request)

        async def action():
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise GatewayError("invalid_request", "Missing required parameter: 'file'.", retry=False)
            data = await upload.read()
            row = save_file(upload.filename or "upload", upload.content_type or "", data, form.get("purpose"))
            log.info("file uploaded: %s (%s, %d bytes) -> %s", row["filename"], row["media_type"], row["size"],
                     row["path"])
            return JSONResponse(file_object(dialect, row))
        return await file_call(dialect, request, action)

    @app.get("/v1/files")
    async def list_files(request: Request):
        dialect = files_dialect(request)

        async def action():
            data = [file_object(dialect, row) for row in store.file_list()]
            first, last = (data[0]["id"], data[-1]["id"]) if data else (None, None)
            if dialect is ANTHROPIC:
                return {"data": data, "first_id": first, "last_id": last, "has_more": False}
            return {"object": "list", "data": data, "first_id": first, "last_id": last, "has_more": False}
        return await file_call(dialect, request, action)

    @app.get("/v1/files/{file_id}")
    async def get_file(request: Request, file_id: str):
        dialect = files_dialect(request)

        async def action():
            return file_object(dialect, find_file(file_id))
        return await file_call(dialect, request, action)

    @app.get("/v1/files/{file_id}/content")
    async def get_file_content(request: Request, file_id: str):
        dialect = files_dialect(request)

        async def action():
            row = find_file(file_id)
            with open(row["path"], "rb") as f:
                return Response(f.read(), media_type=row["media_type"])
        return await file_call(dialect, request, action)

    @app.delete("/v1/files/{file_id}")
    async def delete_file(request: Request, file_id: str):
        dialect = files_dialect(request)

        async def action():
            row = find_file(file_id)
            store.file_delete(row["id"])
            if dialect is ANTHROPIC:
                return {"id": file_id, "type": "file_deleted"}
            return {"id": file_id, "object": "file", "deleted": True}
        return await file_call(dialect, request, action)

    # Gemini uploads use Google's resumable protocol: a "start" call returns an upload URL, then the bytes
    # go to that URL ("upload, finalize"). The random upload_id in the URL stands in for the key there.
    uploads = {}

    @app.post("/upload/v1beta/files")
    async def gemini_upload(request: Request):
        rid = request_id()
        command = request.headers.get("x-goog-upload-command", "").lower()
        upload_id = request.query_params.get("upload_id")
        try:
            if not upload_id:
                check_key(request)
                if request.headers.get("x-goog-upload-protocol", "").lower() != "resumable" or "start" not in command:
                    raise GatewayError("invalid_request", "Only resumable uploads are supported by this gateway.",
                                       retry=False)
                raw = await request.body()
                meta = (json.loads(raw) if raw.strip() else {}).get("file") or {}
                upload_id = secrets.token_urlsafe(18)
                uploads[upload_id] = {
                    "name": meta.get("displayName") or meta.get("display_name") or "upload",
                    "media_type": request.headers.get("x-goog-upload-header-content-type") or meta.get("mimeType")
                    or meta.get("mime_type") or "", "data": bytearray(), "created": time.time()}
                url = f"{str(request.base_url).rstrip('/')}/upload/v1beta/files?upload_id={upload_id}" \
                      "&upload_protocol=resumable"
                return Response(status_code=200, headers={"x-goog-upload-url": url, "x-goog-upload-status": "active",
                                                          "x-goog-upload-chunk-granularity": "8388608"})
            session = uploads.get(upload_id)
            if session is None:
                raise GatewayError("not_found", "Upload session not found.", retry=False)
            offset = int(request.headers.get("x-goog-upload-offset", len(session["data"])))
            if offset != len(session["data"]):
                raise GatewayError("invalid_request", "Upload offset does not match the bytes received.", retry=False)
            session["data"] += await request.body()
            if "finalize" not in command:
                return Response(status_code=200, headers={"x-goog-upload-status": "active",
                                                          "x-goog-upload-size-received": str(len(session["data"]))})
            del uploads[upload_id]
            row = save_file(session["name"], session["media_type"], bytes(session["data"]), None)
            log.info("file uploaded: %s (%s, %d bytes) -> %s", row["filename"], row["media_type"], row["size"],
                     row["path"])
            return JSONResponse({"file": files.gemini_object(row, str(request.base_url))},
                                headers={"x-goog-upload-status": "final"})
        except (ValueError, KeyError):
            return _error(GEMINI, GatewayError("invalid_request", "Invalid upload request.", retry=False), rid)
        except GatewayError as err:
            return _error(GEMINI, err, rid)

    @app.get("/v1beta/files")
    async def gemini_list_files(request: Request):
        async def action():
            return {"files": [files.gemini_object(row, str(request.base_url)) for row in store.file_list()]}
        return await file_call(GEMINI, request, action)

    @app.get("/v1beta/files/{file_id}")
    async def gemini_get_file(request: Request, file_id: str):
        async def action():
            return files.gemini_object(find_file(f"files/{file_id}"), str(request.base_url))
        return await file_call(GEMINI, request, action)

    @app.delete("/v1beta/files/{file_id}")
    async def gemini_delete_file(request: Request, file_id: str):
        async def action():
            store.file_delete(find_file(f"files/{file_id}")["id"])
            return {}
        return await file_call(GEMINI, request, action)

    # ---------- web page ----------

    @app.get("/", include_in_schema=False)
    async def index():
        # The page only reaches this computer, so it can carry the gateway key: nobody has to type it.
        key = json.dumps(settings.api_keys[0]).replace("<", "\\u003c")
        html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__GATEWAY_KEY__", key)
        return HTMLResponse(html, headers={"cache-control": "no-store"})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    def ui_auth(request: Request) -> Optional[JSONResponse]:
        try:
            check_key(request)
        except GatewayError as err:
            return _error(ANTHROPIC, err, request_id())
        return None

    @app.get("/ui/api/status")
    async def ui_status(request: Request):
        if (denied := ui_auth(request)) is not None:
            return denied
        return {"providers": [p.describe() for p in gateway.providers.values()],
                "apis": [{"id": d.id, "label": d.label, "provider": d.native_provider} for d in DIALECTS.values()],
                "stats": store.stats(),
                "settings": {"cache_enabled": settings.cache_enabled, "cache_ttl": settings.cache_ttl,
                             "max_concurrency": settings.max_concurrency},
                "warm_pool": gateway.pool.describe()}

    @app.get("/ui/api/history")
    async def ui_history(request: Request, limit: int = 50, provider: Optional[str] = None):
        if (denied := ui_auth(request)) is not None:
            return denied
        return {"items": store.history(max(1, min(limit, 200)), provider or None)}

    @app.get("/ui/api/history/{item_id}")
    async def ui_history_item(request: Request, item_id: int):
        if (denied := ui_auth(request)) is not None:
            return denied
        item = store.history_item(item_id)
        if item is None:
            return _error(ANTHROPIC, GatewayError("not_found", "No such history item."), request_id())
        return item

    @app.get("/ui/api/files")
    async def ui_files(request: Request):
        if (denied := ui_auth(request)) is not None:
            return denied
        return {"files": [{**row, "openai_id": f"file-{row['id']}", "anthropic_id": f"file_011{row['id']}",
                           "gemini_name": f"files/{row['id']}",
                           "gemini_uri": f"{str(request.base_url).rstrip('/')}/v1beta/files/{row['id']}"}
                          for row in store.file_list()]}

    @app.delete("/ui/api/files/{file_id}")
    async def ui_delete_file(request: Request, file_id: str):
        if (denied := ui_auth(request)) is not None:
            return denied
        return {"deleted": store.file_delete(file_id)}

    @app.post("/ui/api/cache/clear")
    async def ui_cache_clear(request: Request):
        if (denied := ui_auth(request)) is not None:
            return denied
        return {"deleted": store.cache_clear()}

    return app


def _error(dialect: Dialect, err: GatewayError, rid: str) -> JSONResponse:
    status, body = dialect.error(err, rid)
    return JSONResponse(body, status_code=status, headers={"request-id": rid, **err.headers()})


logging.basicConfig(level=load_settings().log_level, format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
app = create_app()
