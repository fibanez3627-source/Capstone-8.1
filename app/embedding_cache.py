"""Cache: text -> vector and drawing -> description. Tokens are paid only once.

A SQLite file next to the manifest (<RAG_DB_DIR>/manifest.json.cache.sqlite). No database
server is needed. A `rebuild` first copies the vectors already stored in Chroma into this
cache, so moving from the old MySQL cache costs nothing.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from array import array
from collections import OrderedDict
from typing import Any

from langchain_core.embeddings import Embeddings

from app.config import settings


class EmbeddingCache:
    """SQLite file next to the manifest. Thread-safe (vision workers + API threads)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
            self._db.execute("CREATE TABLE IF NOT EXISTS cap (k TEXT PRIMARY KEY, t TEXT NOT NULL)")
            self._db.commit()

    @staticmethod
    def text_key(text: str, model: str | None = None) -> str:
        return hashlib.sha256(f"{model or settings.embed_model}\n{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        uniq = list(dict.fromkeys(keys))
        with self._lock:
            for i in range(0, len(uniq), 900):          # SQLite variable limit
                part = uniq[i:i + 900]
                sql = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(part))})"
                for k, v in self._db.execute(sql, part):
                    a = array("f")
                    a.frombytes(v)
                    out[k] = a.tolist()
        return out

    def put_many(self, items: dict[str, Any]) -> None:
        if not items:
            return
        rows = [(k, array("f", map(float, v)).tobytes()) for k, v in items.items()]
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self._db.commit()

    def get_caption(self, key: str) -> str:
        with self._lock:
            row = self._db.execute("SELECT t FROM cap WHERE k = ?", (key,)).fetchone()
        return row[0] if row else ""

    def put_caption(self, key: str, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cap VALUES (?, ?)", (key, text))
            self._db.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM emb").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


def open_embedding_cache(sqlite_path: str) -> EmbeddingCache:
    return EmbeddingCache(sqlite_path)


class CachedEmbeddings(Embeddings):
    """Wraps the OpenRouter embeddings: texts already in the cache cost nothing; only new
    texts go to the API. Query vectors are kept in a small thread-safe LRU, so a repeated
    question (common in support) skips the embedding round-trip entirely."""

    def __init__(self, inner: Embeddings, cache: EmbeddingCache,
                 query_cache_size: int = 2000) -> None:
        self.inner, self.cache = inner, cache
        self._queries: OrderedDict[str, list[float]] = OrderedDict()
        self._qsize = query_cache_size
        self._qlock = threading.Lock()
        self._slock = threading.Lock()
        self.stats: dict[str, int] = {}
        self.reset_stats()

    def fork(self) -> "CachedEmbeddings":
        """Same API client, cache and query LRU; its own indexing statistics."""
        twin = CachedEmbeddings.__new__(CachedEmbeddings)
        twin.inner, twin.cache = self.inner, self.cache
        twin._queries, twin._qsize, twin._qlock = self._queries, self._qsize, self._qlock
        twin._slock = threading.Lock()
        twin.stats = {}
        twin.reset_stats()
        return twin

    def reset_stats(self) -> None:
        with self._slock:
            self.stats = {"hits": 0, "misses": 0, "chars_saved": 0, "chars_sent": 0}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = list(texts)
        keys = [EmbeddingCache.text_key(t) for t in texts]
        found = self.cache.get_many(keys)
        missing: dict[str, str] = {}
        hits = chars_saved = 0
        for k, t in zip(keys, texts):
            if k in found:
                hits += 1
                chars_saved += len(t)
            elif k not in missing:
                missing[k] = t
        items = list(missing.items())
        sent = chars_sent = 0
        for i in range(0, len(items), settings.embed_api_batch):   # cache after EVERY call
            part = items[i:i + settings.embed_api_batch]
            vecs = self.inner.embed_documents([t for _, t in part])
            new = {k: list(map(float, v)) for (k, _), v in zip(part, vecs)}
            self.cache.put_many(new)
            found.update(new)
            sent += len(part)
            chars_sent += sum(len(t) for _, t in part)
        with self._slock:
            self.stats["hits"] += hits
            self.stats["chars_saved"] += chars_saved
            self.stats["misses"] += sent
            self.stats["chars_sent"] += chars_sent
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        with self._qlock:
            v = self._queries.get(text)
            if v is not None:
                self._queries.move_to_end(text)
                return v
        v = list(map(float, self.inner.embed_query(text)))
        with self._qlock:
            self._queries[text] = v
            if len(self._queries) > self._qsize:
                self._queries.popitem(last=False)
        return v
