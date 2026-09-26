"""Runs one API request end to end: pick the provider, check the cache, run the CLI, answer in the
client's API format, and record it in the history."""

import asyncio
import json
import logging
import time
from contextlib import aclosing
from typing import AsyncIterator, Dict, Optional

import anyio
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import attachments, files
from .config import Settings
from .core import (Attachment, ChatRequest, Collector, Event, Finish, GatewayError, Native, Result, Start, TextDelta,
                   ToolUse)
from .dialects import Dialect
from .dialects.anthropic import message_id, message_to_events, native_message
from .process import WarmPool, billing_vars_set
from .providers import Provider, route
from .store import Store, cache_key

log = logging.getLogger("gateway")

CACHEABLE_STOPS = {"end_turn", "max_tokens", "stop_sequence", "tool_use"}


def _replay(result: Result) -> list:
    """Events for a cached answer, so a cache hit can be streamed like a live one."""
    events = [Native(e) for e in message_to_events(result.native)] if result.native else []
    events += [Start(result.model)]
    if result.text:
        events.append(TextDelta(result.text))
    events += [ToolUse(call) for call in result.tool_calls]
    return events + [Finish(result.stop_reason, result.usage)]


async def _iterate(events: list) -> AsyncIterator[Event]:
    for event in events:
        yield event


async def _prepend(first: Event, rest: AsyncIterator[Event]) -> AsyncIterator[Event]:
    yield first
    async for event in rest:
        yield event


def _shrink(value, key: str = ""):
    """The request body for the history, with file contents replaced by their size."""
    if isinstance(value, dict):
        return {k: _shrink(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_shrink(v) for v in value]
    if isinstance(value, str) and len(value) > 300 and (key in ("data", "file_data") or value.startswith("data:")):
        return f"[{len(value):,} characters of file data]"
    return value


def _answer_text(result: Result) -> str:
    calls = [f"[tool call] {c.name}({json.dumps(c.arguments, ensure_ascii=False)})" for c in result.tool_calls]
    return "\n\n".join(([result.text] if result.text else []) + calls)


class Gateway:
    def __init__(self, settings: Settings, store: Store, providers: Dict[str, Provider]):
        self.settings = settings
        self.store = store
        self.providers = providers
        self.slots = {pid: asyncio.Semaphore(settings.max_concurrency) for pid in providers}
        self.pool = WarmPool(enabled=settings.warm_pool, idle_seconds=settings.warm_idle_seconds)
        for provider in providers.values():
            provider.pool = self.pool

    # ---------- startup and shutdown ----------

    def startup(self) -> None:
        risky = billing_vars_set()
        if risky:
            bar = "!" * 78
            log.warning("\n%s\nWARNING: %s set in the gateway's environment.\n"
                        "If a CLI saw them, it could bill an API key instead of your subscription. The gateway\n"
                        "removes them before it starts any CLI. To silence this warning:  unset %s\n%s",
                        bar, ", ".join(risky), " ".join(risky), bar)
        for provider in self.providers.values():
            provider.status = provider.check()
            provider.rate_limit = self.store.get_state(f"rate_limit:{provider.id}") or \
                (self.store.get_state("rate_limit") if provider.id == "claude" else None)
            s = provider.status
            if s.ready:
                log.info("%-12s ready: %s %s, %s, %d models", provider.name, provider.binary, s.version, s.account,
                         len(s.models))
            else:
                log.warning("%-12s not available: %s", provider.name, s.problem)
        if not any(p.status.ready for p in self.providers.values()):
            raise RuntimeError("No provider is ready. " + " ".join(
                f"{p.name}: {p.status.problem}" for p in self.providers.values()))

    async def shutdown(self) -> None:
        await self.pool.close()

    # ---------- token counts ----------

    async def count_claude(self, dialect: Dialect, request: Request, body: dict) -> dict:
        """Anthropic's count_tokens. Claude has no public tokenizer, so Claude Code answers the request with
        one token and the input count it reports is returned (it includes Claude Code's own framing)."""
        request_id = request.state.request_id
        req = dialect.parse({**body, "max_tokens": 4096}, files=self.file)  # count_tokens has no max_tokens
        req.provider, req.model = route(req.model, dialect.native_provider)
        if req.provider != "claude":
            raise GatewayError("invalid_request", "count_tokens counts through Claude Code, so it works for Claude "
                               "models only.", retry=False)
        provider = self.providers["claude"]
        if not provider.status.ready:
            raise GatewayError("authentication", f"Claude Code is not available: {provider.status.problem}",
                               retry=False)
        req.max_tokens, req.thinking_budget, req.stream = 1, 0, False
        key = cache_key({"count_tokens": req.cache_material()})
        cached = self.store.cache_get(key, self.settings.cache_ttl)
        if cached is not None:
            return cached
        collector = Collector()
        async with aclosing(self._run(provider, req, request_id, collector)) as events:
            async for _ in events:
                pass
        counted = {"input_tokens": collector.result(req.model).usage.input_tokens}
        self.store.cache_put(key, counted)
        log.info("[%s] count_tokens through Claude Code: %d", request_id, counted["input_tokens"])
        return counted

    # ---------- uploaded files ----------

    def file(self, reference: str, where: str) -> Attachment:
        """An uploaded file, by any API's reference to it."""
        core = files.core_id(reference)
        row = self.store.file_get(core) if core else None
        if row is None:
            raise GatewayError("invalid_request", f"{where}: file '{reference}' was not found on this gateway.",
                               retry=False)
        with open(row["path"], "rb") as f:
            data = f.read()
        return attachments.make(data, row["media_type"], row["filename"], where)

    # ---------- one request ----------

    async def handle(self, dialect: Dialect, request: Request, body: dict, model: Optional[str] = None,
                     options: Optional[dict] = None):
        request_id = request.state.request_id
        req = dialect.parse(body, model=model, options=options, files=self.file)
        req.provider, req.model = route(req.model, dialect.native_provider)
        provider = self.providers[req.provider]
        if not provider.status.ready:
            kind = "authentication" if provider.status.installed else "api_error"
            raise GatewayError(kind, f"{provider.name} is not available on the gateway host: "
                               f"{provider.status.problem}", retry=False)
        if req.max_tokens and "max_tokens" not in provider.honours:
            req.ignored.append("max_tokens")
        if req.reasoning_effort and "reasoning_effort" not in provider.honours:
            req.ignored.append("reasoning_effort")
        source = request.headers.get("x-gateway-source", "api")[:20]
        found = req.attachments
        log.info("[%s] %s -> %s model=%s stream=%s turns=%d attachments=%d tools=%d source=%s", request_id,
                 dialect.label, provider.name, req.model, req.stream, len(req.turns), len(found), len(req.tools),
                 source)
        if req.ignored:
            log.warning("[%s] ignored, %s cannot set them: %s", request_id, provider.name, ", ".join(req.ignored))

        started = time.monotonic()
        headers = {"request-id": request_id}  # only headers the real APIs send

        # ---- cache ----
        key = None
        cache_state = "off"
        if self.settings.cache_enabled:
            if "no-cache" in request.headers.get("cache-control", "").lower():
                cache_state = "bypass"
            else:
                cache_state = "miss"
                key = cache_key(req.cache_material())
                cached = self._cached(key)
                if cached is not None:
                    log.info("[%s] cache hit, %s not called", request_id, provider.name)
                    self._record(req, request_id, source, "hit", started, dialect, result=cached)
                    if req.stream:
                        return StreamingResponse(dialect.stream(req, _iterate(_replay(cached))),
                                                 media_type="text/event-stream", headers=headers)
                    return JSONResponse(dialect.render(req, cached), headers=headers)

        # ---- provider ----
        collector = Collector()
        events = self._run(provider, req, request_id, collector)

        def finish(result: Optional[Result] = None, err: Optional[GatewayError] = None, status: int = 0):
            if provider.rate_limit:
                self.store.set_state(f"rate_limit:{provider.id}", provider.rate_limit)
            if result is not None and key and result.stop_reason in CACHEABLE_STOPS:
                self.store.cache_put(key, result.to_json())
            self._record(req, request_id, source, cache_state, started, dialect, result=result, err=err,
                         status=status)

        if not req.stream:
            try:
                async with aclosing(events):
                    async for _ in events:
                        pass
            except GatewayError as err:
                finish(err=err)
                raise
            result = self._result(req, collector)
            finish(result=result)
            return JSONResponse(dialect.render(req, result), headers=headers)

        # Streaming: wait for the first event, so a failure before any output is still a normal HTTP error.
        try:
            first = await events.__anext__()
        except StopAsyncIteration:
            err = GatewayError("api_error", f"{provider.name} returned no answer.")
            finish(err=err)
            raise err
        except GatewayError as err:
            finish(err=err)
            raise

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in dialect.stream(req, _prepend(first, events)):
                    yield chunk
            except GatewayError as err:
                finish(err=err)  # the format's error event was already sent
                return
            except (asyncio.CancelledError, GeneratorExit):
                finish(err=GatewayError("api_error", "The client disconnected before the answer finished."),
                       status=499)
                raise
            finally:
                with anyio.CancelScope(shield=True):
                    await events.aclose()
            finish(result=self._result(req, collector))

        return StreamingResponse(body(), media_type="text/event-stream",
                                 headers={**headers, "cache-control": "no-cache"})

    async def _run(self, provider: Provider, req: ChatRequest, request_id: str,
                   collector: Collector) -> AsyncIterator[Event]:
        slots = self.slots[provider.id]
        try:
            await asyncio.wait_for(slots.acquire(), self.settings.queue_timeout)
        except asyncio.TimeoutError:
            raise GatewayError("overloaded", "Overloaded")
        try:
            async with aclosing(provider.run(req, request_id)) as events:
                async for event in events:
                    collector.add(event)
                    yield event
        finally:
            slots.release()

    def _result(self, req: ChatRequest, collector: Collector) -> Result:
        result = collector.result(req.model)
        if collector.natives:  # Claude Code: keep the exact Anthropic message too
            result.native = native_message(collector.natives, keep_thinking=req.thinking_budget != 0)
        return result

    def _cached(self, key: str) -> Optional[Result]:
        data = self.store.cache_get(key, self.settings.cache_ttl)
        try:
            result = Result.from_json(data) if data else None
        except (TypeError, KeyError):
            return None  # an entry from an older version
        if result is not None and result.native:
            result.native = {**result.native, "id": message_id()}  # every answer gets its own id
        return result

    def _record(self, req: ChatRequest, request_id: str, source: str, cache: str, started: float, dialect: Dialect,
                result: Optional[Result] = None, err: Optional[GatewayError] = None, status: int = 0) -> None:
        if not self.settings.history_enabled:
            return
        if err is not None:
            http_status, response = dialect.error(err, request_id)
            status = status or http_status
        else:
            status, response = 200, dialect.render(req, result)
        usage = result.usage if result else None
        found = [{"name": a.filename or a.media_type, "media_type": a.media_type, "size": len(a.data)}
                 for a in req.attachments]
        self.store.add_history(
            request_id=request_id, created_at=time.time(), source=source, provider=req.provider, api=req.api,
            model=result.model if result else req.model, stream=int(req.stream), cache=cache, status=status,
            latency_ms=int((time.monotonic() - started) * 1000),
            input_tokens=usage.input_tokens if usage else None, output_tokens=usage.output_tokens if usage else None,
            question=req.question, context=req.context, attachments=found or None,
            answer=_answer_text(result) if result else None,
            error=f"{err.kind}: {err.message}" if err else None, request=_shrink(req.body), response=response)
        if result is not None:
            log.info("[%s] done in %.1fs: %s, stop=%s, tokens in=%d out=%d%s", request_id, time.monotonic() - started,
                     result.model, result.stop_reason, result.usage.input_tokens, result.usage.output_tokens,
                     f", {len(result.tool_calls)} tool call(s)" if result.tool_calls else "")
        elif err is not None:
            log.warning("[%s] failed: %s: %s", request_id, err.kind, err.message)
