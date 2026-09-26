"""SQLite storage: the response cache, the request history shown in the web page, and small saved values."""

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    response   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id         TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    filename   TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size       INTEGER NOT NULL,
    sha256     TEXT NOT NULL,
    purpose    TEXT,
    path       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT NOT NULL,
    created_at    REAL NOT NULL,
    source        TEXT NOT NULL,
    provider      TEXT NOT NULL DEFAULT 'claude',
    api           TEXT NOT NULL DEFAULT 'anthropic-messages',
    model         TEXT,
    stream        INTEGER NOT NULL,
    cache         TEXT NOT NULL,
    status        INTEGER NOT NULL,
    latency_ms    INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    question      TEXT,
    context       TEXT,
    attachments   TEXT,
    answer        TEXT,
    error         TEXT,
    request       TEXT,
    response      TEXT
);
"""

# Columns added after the first version, with how to add them to an older database.
_ADDED_COLUMNS = {
    "context": "ALTER TABLE history ADD COLUMN context TEXT",
    "provider": "ALTER TABLE history ADD COLUMN provider TEXT NOT NULL DEFAULT 'claude'",
    "api": "ALTER TABLE history ADD COLUMN api TEXT NOT NULL DEFAULT 'anthropic-messages'",
    "attachments": "ALTER TABLE history ADD COLUMN attachments TEXT",
}

_LIST_COLUMNS = ("id", "request_id", "created_at", "source", "provider", "api", "model", "stream", "cache",
                 "status", "latency_ms", "input_tokens", "output_tokens", "question", "context", "attachments",
                 "answer", "error")


def cache_key(material: dict) -> str:
    return hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Store:
    def __init__(self, path: str, files_dir: Optional[str] = None):
        self.path = path
        self.files_dir = files_dir or os.path.join(os.path.dirname(os.path.abspath(path)), "files")
        self._db: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.executescript(_SCHEMA)
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(history)")}
            for name, statement in _ADDED_COLUMNS.items():
                if name not in columns:  # a database made by an earlier version
                    self._db.execute(statement)
            self._db.commit()
        return self._db

    # ---------- cache ----------

    def cache_get(self, key: str, ttl: int) -> Optional[dict]:
        with self._lock:
            row = self._conn().execute("SELECT created_at, response FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None or time.time() - row["created_at"] > ttl:
            return None
        return json.loads(row["response"])

    def cache_put(self, key: str, value: dict) -> None:
        with self._lock, self._conn() as db:
            db.execute("INSERT OR REPLACE INTO cache (key, created_at, response) VALUES (?, ?, ?)",
                       (key, time.time(), json.dumps(value)))

    def cache_clear(self) -> int:
        with self._lock, self._conn() as db:
            return db.execute("DELETE FROM cache").rowcount

    # ---------- uploaded files ----------

    def file_add(self, file_id: str, filename: str, media_type: str, data: bytes, sha256: str,
                 purpose: Optional[str], extension: str) -> dict:
        os.makedirs(self.files_dir, exist_ok=True)
        path = os.path.join(self.files_dir, file_id + extension)  # the id, never the client's file name
        with open(path, "wb") as f:
            f.write(data)
        row = {"id": file_id, "created_at": time.time(), "filename": filename, "media_type": media_type,
               "size": len(data), "sha256": sha256, "purpose": purpose, "path": path}
        with self._lock, self._conn() as db:
            db.execute("INSERT INTO files (id, created_at, filename, media_type, size, sha256, purpose, path) "
                       "VALUES (:id, :created_at, :filename, :media_type, :size, :sha256, :purpose, :path)", row)
        return row

    def file_get(self, file_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn().execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return dict(row) if row else None

    def file_list(self) -> list:
        with self._lock:
            rows = self._conn().execute("SELECT * FROM files ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def file_delete(self, file_id: str) -> bool:
        row = self.file_get(file_id)
        if row is None:
            return False
        with self._lock, self._conn() as db:
            db.execute("DELETE FROM files WHERE id = ?", (file_id,))
        try:
            os.remove(row["path"])
        except OSError:
            pass
        return True

    # ---------- small saved values (the last plan-usage reading of each provider) ----------

    def set_state(self, key: str, value: dict) -> None:
        with self._lock, self._conn() as db:
            db.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, json.dumps(value)))

    def get_state(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._conn().execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    # ---------- history ----------

    def add_history(self, **row) -> None:
        for name in ("request", "response", "attachments"):
            if row.get(name) is not None:
                row[name] = json.dumps(row[name])
        names = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock, self._conn() as db:
            db.execute(f"INSERT INTO history ({names}) VALUES ({marks})", tuple(row.values()))

    def history(self, limit: int = 50, provider: Optional[str] = None) -> list:
        return [self._decoded(r) for r in self._history(limit, provider)]

    @staticmethod
    def _decoded(row: dict) -> dict:
        if row.get("attachments"):
            row["attachments"] = json.loads(row["attachments"])
        return row

    def _history(self, limit: int, provider: Optional[str]) -> list:
        query = f"SELECT {', '.join(_LIST_COLUMNS)} FROM history"
        args = []
        if provider:
            query += " WHERE provider = ?"
            args.append(provider)
        query += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn().execute(query, args).fetchall()
        return [dict(r) for r in rows]

    def history_item(self, item_id: int) -> Optional[dict]:
        with self._lock:
            row = self._conn().execute("SELECT * FROM history WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        item = self._decoded(dict(row))
        for name in ("request", "response"):
            item[name] = json.loads(item[name]) if item[name] else None
        return item

    def stats(self) -> dict:
        with self._lock:
            db = self._conn()
            requests, hits = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(cache = 'hit'), 0) FROM history").fetchone()
            entries = db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            per_provider = {row[0]: row[1] for row in
                            db.execute("SELECT provider, COUNT(*) FROM history GROUP BY provider")}
        return {"requests": requests, "cache_hits": hits, "cache_entries": entries, "per_provider": per_provider}
