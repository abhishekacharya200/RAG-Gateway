"""Running a CLI as a child process: clean environment, its own temp folder, JSON lines out, always cleaned up.

Some CLIs (Claude Code, Antigravity) wait for the prompt on stdin, so they can be started before a request
arrives. The WarmPool keeps one such process ready per launch setup, which saves their start-up time.
"""

import asyncio
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Union

import anyio

from .core import GatewayError

log = logging.getLogger("gateway.process")

# No child process ever gets another provider's API key or endpoint, or the variables of a parent
# Claude Code session. With any of these, a CLI could bill an API key instead of your subscription.
_STRIP_PREFIXES = ("ANTHROPIC_", "CLAUDE", "OPENAI_", "CODEX_", "GEMINI_", "GOOGLE_", "VERTEX")
_STRIP = {"MAX_THINKING_TOKENS"}
# Kept: subscription tokens and config folders, which are not API billing.
_KEEP = {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR", "CODEX_HOME"}

# Set in the gateway's own environment, these are worth a warning (they are removed for the children anyway).
BILLING_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                "CODEX_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI",
                "GOOGLE_APPLICATION_CREDENTIALS")

# A long answer arrives as one JSON line. asyncio's default line limit (64 KB) is too small.
_LINE_LIMIT = 64 * 1024 * 1024


def child_env(extra: Optional[dict] = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k in _KEEP or not (k.startswith(_STRIP_PREFIXES) or k in _STRIP)}
    env.update(extra or {})
    return env


def billing_vars_set() -> List[str]:
    return [name for name in BILLING_VARS if os.environ.get(name)]


def resolve(binary: str) -> Optional[str]:
    """Full path of a CLI. shutil.which also finds claude.cmd / codex.cmd on Windows."""
    return shutil.which(binary)


def run_quick(cmd: List[str], timeout: float = 30) -> subprocess.CompletedProcess:
    """For short checks at startup (version, login status, model list)."""
    return subprocess.run(cmd, capture_output=True, text=True, env=child_env(), timeout=timeout,
                          stdin=subprocess.DEVNULL)


@dataclass
class Launch:
    """Everything fixed when a CLI process starts. "{dir}" in the command or the environment stands for the
    run's own temp folder (file contents are written exactly as given)."""

    provider: str
    cmd: List[str]
    extra_env: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, Union[str, bytes]] = field(default_factory=dict)  # path in the folder -> content
    warm: bool = False  # the CLI waits for its prompt on stdin, so it can be started ahead of time

    def key(self) -> str:
        files = sorted((k, v if isinstance(v, str) else hashlib.sha256(v).hexdigest()) for k, v in self.files.items())
        return hashlib.sha256(json.dumps([self.provider, self.cmd, sorted(self.extra_env.items()), files])
                              .encode()).hexdigest()


class Started:
    """A running CLI process with its temp folder."""

    def __init__(self, proc, folder: str, key: str, cmd: List[str]):
        self.proc = proc
        self.folder = folder
        self.key = key
        self.cmd = cmd
        self.started_at = time.monotonic()
        self.stderr_task = asyncio.ensure_future(proc.stderr.read())

    def alive(self) -> bool:
        return self.proc.returncode is None

    def write(self, relative: str, content: Union[str, bytes]) -> str:
        path = os.path.join(self.folder, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content.encode("utf-8") if isinstance(content, str) else content)
        return path

    async def kill(self) -> None:
        with anyio.CancelScope(shield=True):
            if self.proc.returncode is None:
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
            await self.proc.wait()
            await self.stderr_task
        shutil.rmtree(self.folder, ignore_errors=True)


async def start(launch: Launch) -> Started:
    folder = tempfile.mkdtemp(prefix=f"gateway-{launch.provider}-")
    # Forward slashes, so a Windows path cannot break the TOML or JSON strings it ends up in.
    slashed = folder.replace(os.sep, "/")

    def fill(text: str) -> str:
        return text.replace("{dir}", slashed)

    for relative, content in launch.files.items():
        path = os.path.join(folder, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content.encode("utf-8") if isinstance(content, str) else content)
    cmd = [fill(part) for part in launch.cmd]
    env = child_env({k: fill(v) for k, v in launch.extra_env.items()})
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=folder, env=env, limit=_LINE_LIMIT)
    except FileNotFoundError:
        shutil.rmtree(folder, ignore_errors=True)
        raise GatewayError("api_error", f"`{cmd[0]}` is not installed on the gateway host.", retry=False)
    except NotImplementedError:  # Windows event loop without subprocess support (uvicorn --reload)
        shutil.rmtree(folder, ignore_errors=True)
        raise GatewayError("api_error", "This event loop cannot start processes. On Windows, start the gateway "
                           "with `python -m gateway`.", retry=False)
    return Started(proc, folder, launch.key(), cmd)


class WarmPool:
    """Keeps a started process ready for each launch setup that was used recently."""

    def __init__(self, enabled: bool = True, idle_seconds: float = 600, max_idle: int = 4):
        self.enabled = enabled
        self.idle_seconds = idle_seconds
        self.max_idle = max_idle
        self._idle: Dict[str, Started] = {}
        self._starting: set = set()
        self._tasks: set = set()

    async def take(self, launch: Launch, request_id: str) -> Started:
        await self._reap()
        use_pool = self.enabled and launch.warm
        started = self._idle.pop(launch.key(), None) if use_pool else None
        if started is not None and not started.alive():
            await started.kill()
            started = None
        if started is not None:
            log.info("[%s] using a %s process started %.1fs ago", request_id, launch.provider,
                     time.monotonic() - started.started_at)
        else:
            started = await start(launch)
        log.info("[%s] run: %s %s", request_id, " ".join(f"{k}={v}" for k, v in launch.extra_env.items()),
                 shlex.join(started.cmd))
        if use_pool:
            self._refill(launch)  # so the next request with this setup finds a process ready
        return started

    def _refill(self, launch: Launch) -> None:
        key = launch.key()
        if key in self._idle or key in self._starting:
            return
        self._starting.add(key)

        async def run():
            try:
                if len(self._idle) >= self.max_idle:  # make room: drop the oldest ready process
                    oldest = min(self._idle, key=lambda k: self._idle[k].started_at)
                    await self._idle.pop(oldest).kill()
                self._idle[key] = await start(launch)
            except Exception as exc:  # a failed warm start only costs the next request its start-up time
                log.warning("could not start a ready %s process: %s", launch.provider, exc)
            finally:
                self._starting.discard(key)

        task = asyncio.ensure_future(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _reap(self) -> None:
        now = time.monotonic()
        for key in [k for k, s in self._idle.items() if now - s.started_at > self.idle_seconds or not s.alive()]:
            await self._idle.pop(key).kill()

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for started in list(self._idle.values()):
            await started.kill()
        self._idle.clear()

    def describe(self) -> dict:
        return {"enabled": self.enabled, "ready": len(self._idle)}


class CliRun:
    """One CLI run that prints JSON lines.

        async with CliRun(started, stdin=..., timeout=..., request_id=...) as run:
            async for obj in run.json_lines():
                ...
            run.completed = True   # the final event was seen: let it exit by itself

    On exit the process is stopped if it is still running, even when the request was cancelled,
    and its temp folder is removed.
    """

    def __init__(self, started: Started, *, stdin: bytes, timeout: float, request_id: str):
        self.started = started
        self.stdin = stdin
        self.timeout = timeout
        self.request_id = request_id
        self.completed = False
        self.exit_code = None
        self.stderr = ""

    async def __aenter__(self) -> "CliRun":
        proc = self.started.proc
        try:
            proc.stdin.write(self.stdin)
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            log.warning("[%s] the CLI closed stdin early", self.request_id)  # its output says why
        return self

    async def json_lines(self) -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        stdout = self.started.proc.stdout
        while True:
            try:
                line = await asyncio.wait_for(stdout.readline(), max(deadline - loop.time(), 0.001))
            except asyncio.TimeoutError:
                raise GatewayError("timeout", f"The CLI did not finish within {int(self.timeout)} seconds.")
            if not line:
                return
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            log.debug("[%s] stdout: %s", self.request_id, text if len(text) <= 4000 else text[:4000] + " ...")
            try:
                obj = json.loads(text)
            except ValueError:
                log.info("[%s] stdout (not JSON): %s", self.request_id, text[:500])
                continue
            if isinstance(obj, dict):
                yield obj

    async def __aexit__(self, *exc) -> None:
        await self.finish()

    async def finish(self) -> None:
        """Stops the process and records its exit code and stderr. Safe to call twice."""
        if self.exit_code is not None:
            return
        proc = self.started.proc
        # Shielded: this must run to the end even when the client disconnected and the request is cancelled.
        with anyio.CancelScope(shield=True):
            if proc.returncode is None and self.completed:
                try:
                    await asyncio.wait_for(proc.wait(), 5)  # it is already exiting by itself
                except asyncio.TimeoutError:
                    pass
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            await proc.wait()
            self.exit_code = proc.returncode
            self.stderr = (await self.started.stderr_task).decode("utf-8", "replace").strip()
        shutil.rmtree(self.started.folder, ignore_errors=True)
        log.info("[%s] exit=%s", self.request_id, self.exit_code)
        if self.stderr:
            log.info("[%s] stderr: %s", self.request_id, self.stderr[:4000])
