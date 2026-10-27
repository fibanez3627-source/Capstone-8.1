"""HybridEvaluationRetriever — the 5.7 core (BM25 + Chroma, incremental, cached,
resumable indexing):

* No shared mutable query state: every search method takes an optional `hits` list owned
  by the question, so the sources of one answer never mix with another.
* Lock-free reads. The in-memory search structures live in an immutable IndexSnapshot.
  Re-indexing builds a NEW snapshot and swaps the reference in one assignment.
* Only indexing is exclusive (`_write_lock` in-process; app/jobs.py adds a lock file so
  two consoles never index at the same time).
* No indexing in __init__: call load() (fast start from disk), then update_index() /
  rebuild_index() from the `update` / `rebuild` commands.
* maybe_reload(): a `chat` session notices a new manifest written by another console and
  reloads its snapshot in a background thread (no API calls).
* Files inside RAG_EXCLUDE_DIRS (e.g. Importaciones) are never indexed.
* BM25 engine unchanged (rank_bm25, built-in fallback if missing).

Production 1.0:
* BM25 analyzer = textutils.bm25_tokenize (accents folded, ES/EN stop words + stems, part
  codes as extra tokens). Only the in-memory snapshot changes: no re-indexing, no API cost.
* Fusion = Reciprocal Rank Fusion (app/fusion.py) instead of min-max + weights.
* Cross-encoder reranker over the top fused candidates (app/reranker.py), then code
  pinning and table expansion.
* Kept from the live 14-Oct version: search_query() (query_lang glossary + typo fix against
  the corpus words) for the vector AND BM25 search, vocab(), and focus_doc_tables() after
  the table expansion. The reranker scores the question as typed.
"""
from __future__ import annotations

import base64
import difflib
import hashlib
import heapq
import json
import logging
import math
import os
import pickle
import re
import threading
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.catalog import DOC_TYPE_LABEL, build_catalog_entry, canon_type
from app.config import (TABLE_EXTS, IMAGE_EXTS, IMAGE_SECTIONS, OTHER_SECTION, PDF_EXTS,
                        SECTION_LABEL, TEXT_EXTS, settings)
from app.embedding_cache import CachedEmbeddings, EmbeddingCache, open_embedding_cache
from app.errors import IndexingAborted, api_error_message, classify_api_error, is_fatal_api_error
from app.prompts import IMAGE_PROMPT
from app.pdf_tables import extract_pdf_chunks
from app.fusion import rerank_and_select, rrf_fuse
from app.query_lang import build_vocab, expand_query, spell_fix
from app.textutils import (bm25_tokenize, caption_from_text, extract_codes, file_hash,
                           fmt_secs, load_image_png, merge_chunks, mkey, norm,
                           read_csv_table, simple_tokenize)

log = logging.getLogger("rag.retriever")
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

try:
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None
# (no load_dotenv here: app.config reads the ONE .env of the project — RAG_ENV_FILE)

# ---------------------------------------------------------------------------------------
# BM25 — left exactly as in 5.7
# ---------------------------------------------------------------------------------------
class _FallbackBM25:
    """BM25 used only when rank_bm25 is missing."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if corpus else 0.0
        self.doc_freqs = [Counter(d) for d in corpus]
        n = len(corpus)
        df = Counter(t for d in self.doc_freqs for t in d)
        self.idf = {t: max(0.0, math.log((n - c + 0.5) / (c + 0.5) + 1.0)) for t, c in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        q = [t for t in query if t in self.idf]
        scores = []
        for freqs, length in zip(self.doc_freqs, self.doc_len):
            nrm = self.k1 * (1 - self.b + self.b * length / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for t in q:
                tf = freqs.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + nrm)
            scores.append(s)
        return scores


try:
    from rank_bm25 import BM25Okapi  # type: ignore
    BM25_ENGINE = "rank_bm25"
except ImportError:  # pragma: no cover
    BM25Okapi = _FallbackBM25  # type: ignore
    BM25_ENGINE = "built-in fallback"


# ---------------------------------------------------------------------------------------
# Immutable search snapshot + search signals
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IndexSnapshot:
    chunks: dict[str, dict] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)
    kind_idx: dict[str, list[int]] = field(default_factory=dict)
    bm25: Any = None
    catalog: dict[str, dict] = field(default_factory=dict)      # rel -> catalog entry
    file_idx: dict[str, list[int]] = field(default_factory=dict)  # rel -> chunk indices
    chunk_rel: list[str] = field(default_factory=list)          # chunk index -> rel
    section_idx: dict[str, list[int]] = field(default_factory=dict)  # section -> chunk indices
    files: list[tuple[str, str]] = field(default_factory=list)  # (path, rel)
    n_pending: int = 0
    manifest_mtime: float = 0.0
    built_at: float = 0.0
    vocab: frozenset = frozenset()                      # corpus words (query spell-fix)
    vocab_idx: dict = field(default_factory=dict)       # (first letter, length) -> words

    @property
    def empty(self) -> bool:
        return not self.chunks or self.bm25 is None


@dataclass
class SearchSignals:
    """What the router needs to decide 'fast path is enough' vs 'escalate'."""
    n_results: int = 0
    vector_ok: bool = True
    max_vector: float = 0.0     # best raw vector relevance in [0, 1]
    max_bm25: float = 0.0       # best raw BM25 score
    max_rerank: float = 0.0     # best cross-encoder score in [0, 1] (0 = not reranked)


# ---------------------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------------------
class HybridEvaluationRetriever:
    def __init__(self, data_dir: str | None = None, top_k: int | None = None,
                 recursive: bool | None = None, vision_llm: Any = None,
                 embeddings: Embeddings | None = None, chroma_dir: str | None = None,
                 manifest_file: str | None = None, namespace: str = "",
                 cached_embeddings: CachedEmbeddings | None = None) -> None:
        """namespace: multi-RAG prefix. Files of a non-default knowledge base are named
        '@<namespace>/<path>' so ids, citations and /files links never collide between
        bases. The default base keeps namespace '' (its existing index stays valid).
        cached_embeddings: share one embeddings wrapper (and its query-vector LRU) between
        bases, so a question searched in N bases is embedded once."""
        self.namespace = namespace.strip().strip("@/")
        self._data_dir = Path(data_dir or settings.data_dir).resolve()
        self._top_k = top_k or settings.top_k
        self._recursive = settings.recursive if recursive is None else recursive
        self._vision_llm = vision_llm
        # index layout: <RAG_DB_DIR>/chroma, manifest.json, state.json
        self._chroma_dir = chroma_dir or settings.chroma_dir
        self._manifest_file = manifest_file or settings.manifest_file
        self._state_file = (os.path.join(os.path.dirname(os.path.abspath(manifest_file)), "state.json")
                            if manifest_file else settings.state_file)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""])

        self._snap = IndexSnapshot()                    # swapped atomically, never mutated
        self._write_lock = threading.Lock()             # one indexing at a time (in-process)
        self._reload_lock = threading.Lock()
        self._last_reload_check = 0.0
        self._table_cache: dict[str, tuple] = {}
        self._table_lock = threading.Lock()
        self._vector_warned = False

        # indexing-only state (touched only while holding _write_lock)
        self._vision_ok = True
        self._vision_lock = threading.Lock()
        self._retry_later: set[str] = set()
        self._captions: dict[str, str] = {}
        self._hashes: dict[str, tuple[float, str]] = {}
        self._n_vision_calls = 0
        self._n_vision_cached = 0
        self.last_sync: dict = {}
        self.startup_warning = ""
        log.info("index: chroma=%s manifest=%s state=%s", self._chroma_dir,
                 self._manifest_file, self._state_file)
        os.makedirs(self._chroma_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self._manifest_file)), exist_ok=True)
        if cached_embeddings is not None:
            self._cache = cached_embeddings.cache
            self.raw_embeddings = cached_embeddings.inner
            self.embeddings = cached_embeddings.fork()
        else:
            self._cache = open_embedding_cache(self._manifest_file + ".cache.sqlite")
            if embeddings is None:
                from app.llm import make_embeddings
                embeddings = make_embeddings()
            self.raw_embeddings = embeddings
            self.embeddings = CachedEmbeddings(self.raw_embeddings, self._cache)
        self.vector_db = self._open_chroma()

    # ======================== snapshot access ========================
    @property
    def snapshot(self) -> IndexSnapshot:
        return self._snap

    def _swap(self, snap: IndexSnapshot) -> None:
        self._snap = snap                                # single reference assignment
        with self._table_lock:
            self._table_cache = {}

    def load(self) -> None:
        """Fast start: build the in-memory snapshot from the manifest. No folder scan, no
        API calls. Safe to call while serving (reads keep using the old snapshot)."""
        t0 = time.time()
        manifest = self._load_manifest(quiet=True)
        self._swap(self._build_snapshot(manifest))
        try:
            n_vec = self.vector_db._collection.count()
        except Exception:
            n_vec = -1
        log.info("Index loaded in %s: %d files, %d chunks, %s vectors",
                 fmt_secs(time.time() - t0), len(manifest), len(self._snap.chunks),
                 n_vec if n_vec >= 0 else "?")
        if n_vec == 0 and self._snap.chunks:
            log.warning("Chroma is empty but the manifest has chunks -> BM25 only. "
                        "Run a rebuild to regenerate the vectors.")

    def maybe_reload(self) -> None:
        """Cheap per-question check: if another process re-indexed (manifest changed),
        reload the snapshot in a background thread. Never blocks the request."""
        now = time.time()
        if now - self._last_reload_check < settings.reload_check_secs:
            return
        self._last_reload_check = now
        try:
            mtime = os.path.getmtime(self._manifest_file)
        except OSError:
            return
        if mtime <= self._snap.manifest_mtime or self._write_lock.locked():
            return
        if not self._reload_lock.acquire(blocking=False):
            return

        def _bg() -> None:
            try:
                log.info("Manifest changed on disk -> reloading index snapshot")
                self.load()
            except Exception as e:
                log.warning("reload failed: %s", e)
            finally:
                self._reload_lock.release()

        threading.Thread(target=_bg, name="index-reload", daemon=True).start()

    # ======================== state file ========================
    def _open_chroma(self) -> Chroma:
        return Chroma(persist_directory=self._chroma_dir, embedding_function=self.embeddings)

    def _load_state(self) -> dict:
        try:
            with open(self._state_file, encoding="utf-8") as f:
                st = json.load(f)
            return st if isinstance(st, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_state(self, st: dict) -> None:
        try:
            tmp = self._state_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(st, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._state_file)
        except OSError as e:
            log.warning("could not save index state: %s", e)

    def _save_state(self, complete: bool) -> None:
        st = self._load_state()
        st.update(complete=complete, pdf_dir=str(self._data_dir), embed_model=settings.embed_model)
        if complete:
            st["indexed_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_state(st)

    def index_complete(self) -> bool:
        st = self._load_state()
        if "complete" in st:
            return bool(st["complete"])
        if os.path.exists(self._manifest_file) and self._load_manifest(quiet=True):
            self._save_state(True)
            return True
        return False

    def stats(self) -> dict:
        """Small health summary for the API (local reads only)."""
        snap, st = self._snap, self._load_state()
        try:
            n_vec = self.vector_db._collection.count()
        except Exception:
            n_vec = -1
        return {"files": len(snap.files), "chunks": len(snap.chunks),
                "chunks_by_kind": {k: len(v) for k, v in snap.kind_idx.items()},
                "catalog": len(snap.catalog), "vectors": n_vec,
                "pending_drawings": snap.n_pending, "complete": bool(st.get("complete")),
                "indexed_at": st.get("indexed_at", ""), "bm25_engine": BM25_ENGINE,
                "last_sync": self.last_sync or (st.get("history") or [{}])[-1],
                "startup_warning": self.startup_warning}

    # ======================== content hashes + cache seeding ========================
    def _hash_of(self, path: str) -> str:
        m = os.path.getmtime(path)
        got = self._hashes.get(path)
        if got and got[0] == m:
            return got[1]
        h = file_hash(path)
        self._hashes[path] = (m, h)
        return h

    def _caption_key(self, path: str) -> str:
        return hashlib.sha256(f"{settings.effective_vision_model}\n{IMAGE_PROMPT}\n"
                              f"{self._hash_of(path)}".encode("utf-8")).hexdigest()

    def _seed_cache(self, manifest: dict) -> None:
        """Once per index: copy the vectors already stored in Chroma (and the drawing
        descriptions already in the manifest) into the local cache. Local, free."""
        st = self._load_state()
        if st.get("cache_seeded") == settings.embed_model:
            return
        if st.get("embed_model", settings.embed_model) != settings.embed_model:
            log.info("index uses another embedding model; its vectors are not reused")
        else:
            n = 0
            try:
                col = self.vector_db._collection
                total = col.count()
                for off in range(0, total, settings.chroma_batch):
                    r = col.get(include=["embeddings", "documents"],
                                limit=settings.chroma_batch, offset=off)
                    docs, embs = r.get("documents"), r.get("embeddings")
                    if docs is None or embs is None:
                        continue
                    items = {EmbeddingCache.text_key(d): e for d, e in zip(docs, embs)
                             if d and e is not None and len(e)}
                    self._cache.put_many(items)
                    n += len(items)
            except Exception as e:
                log.warning("could not copy existing vectors to the cache: %s", e)
                return
            nc = 0
            for path, e in manifest.items():
                if Path(path).suffix.lower() not in IMAGE_EXTS or e.get("mtime") == -1:
                    continue
                try:
                    if os.path.getmtime(path) != e["mtime"]:
                        continue
                    cap = caption_from_text(e["chunks"][0]["text"]) if e.get("chunks") else ""
                    if cap:
                        self._cache.put_caption(self._caption_key(path), cap)
                        nc += 1
                except (OSError, KeyError, IndexError):
                    continue
            if n or nc:
                log.info("cache seeded: %d vectors, %d drawing descriptions", n, nc)
        st = self._load_state()
        st["cache_seeded"] = settings.embed_model
        self._write_state(st)

    # ======================== the two indexing entry points ========================
    def update_index(self) -> dict:
        """New/changed files are indexed, deleted ones removed. Exclusive."""
        with self._write_lock:
            self._retry_later.clear()
            self._captions.clear()
            self._vision_ok = True
            try:
                self._sync_and_index()
                self.startup_warning = ""
            except IndexingAborted as e:
                # no credits / network down: keep serving what is already indexed
                self.startup_warning = str(e)
                self._swap(self._build_snapshot(self._load_manifest(quiet=True)))
                raise
            return self.last_sync

    def rebuild_index(self) -> dict:
        """Wipe vectors + manifest and index everything again (cache keeps paid vectors).
        While it runs, queries keep using the OLD snapshot; vector search may fall back to
        BM25 for a moment because the collection is being recreated."""
        with self._write_lock:
            self._seed_cache(self._load_manifest(quiet=True))
            self._save_state(False)
            try:
                self.vector_db.delete_collection()
            except Exception as e:
                log.warning("could not delete collection: %s", e)
            self.vector_db = self._open_chroma()
            self._vector_warned = False
            for f in (self._manifest_file, self._manifest_file + ".tmp"):
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass
            self._retry_later.clear()
            self._captions.clear()
            self._vision_ok = True
            self._sync_and_index()
            return self.last_sync

    # ======================== discovery ========================
    def _image_allowed(self, f: Path) -> bool:
        """Images are indexed only inside the drawing folders declared in the .env
        (RAG_DIR_DRAWINGS / RAG_DIR_FOTOS / RAG_DIR_EXPLOSIONADOS)."""
        try:
            rel = f.relative_to(self._data_dir).as_posix()
        except ValueError:
            return False
        return settings.section_of(rel) in IMAGE_SECTIONS

    @staticmethod
    def _has_sibling_image(f: Path) -> bool:
        return any(f.with_suffix(e).exists() or f.with_suffix(e.upper()).exists()
                   for e in IMAGE_EXTS)

    def _iter_files(self) -> Iterator[Path]:
        if not Path(self._data_dir).exists():
            log.warning("data folder '%s' does not exist", self._data_dir)
            return
        globber = self._data_dir.rglob if self._recursive else self._data_dir.glob
        docs = PDF_EXTS + TEXT_EXTS + TABLE_EXTS
        seen = set()
        for f in sorted(globber("*")):
            try:
                if not f.is_file() or f.name.startswith("~$"):
                    continue
            except OSError:
                continue
            try:                                  # RAG_EXCLUDE_DIRS (e.g. Importaciones)
                if settings.is_excluded(f.relative_to(self._data_dir).as_posix()):
                    continue
            except ValueError:
                pass
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                ok = self._image_allowed(f)
            elif ext in docs:
                ok = not (ext == ".txt" and self._has_sibling_image(f))
            else:
                ok = False
            key = str(f.resolve())
            if ok and key not in seen:
                seen.add(key)
                yield f

    def _rel_name(self, path: str) -> str:
        try:
            rel = Path(path).relative_to(self._data_dir).as_posix()
        except ValueError:
            rel = Path(path).name
        return f"@{self.namespace}/{rel}" if self.namespace else rel

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    # ======================== per-type chunking ========================
    @staticmethod
    def _extract_text(path: str) -> str:
        if path.lower().endswith(".pdf"):
            if fitz is None:
                raise RuntimeError("No hay librería PDF: pip install pymupdf")
            with fitz.open(path) as doc:
                return " ".join(t for t in (pg.get_text() for pg in doc) if t.strip())
        return Path(path).read_text(encoding="utf-8", errors="ignore")

    def _text_chunks(self, path: str, rel: str) -> list[dict]:
        if path.lower().endswith(".pdf"):
            # page-aware, table-aware chunks (printed page numbers, headers repeated)
            return extract_pdf_chunks(path, rel, self._splitter)
        return [{"id": f"{rel}_chunk{i}", "source": rel, "text": t, "kind": "text", "path": path}
                for i, t in enumerate(self._splitter.split_text(self._extract_text(path)))]

    def _csv_chunks(self, path: str, rel: str) -> list[dict]:
        headers, rows, preamble = read_csv_table(path)
        title = Path(path).stem
        overview = (f"[TABLE] {title}\nFile: {rel}\n"
                    + (f"Title: {' / '.join(preamble)}\n" if preamble else "")
                    + f"Columns: {', '.join(headers)}\nNumber of rows: {len(rows)}")
        chunks = [{"id": f"{rel}_table", "source": rel, "text": overview, "kind": "table", "path": path}]
        step = settings.csv_rows_per_chunk
        for start in range(0, len(rows), step):
            lines = []
            for n, r in enumerate(rows[start:start + step], start=start + 1):
                pairs = [f"{h}: {v}" for h, v in zip(headers, r) if v]
                lines.append(f"Row {n}: " + " | ".join(pairs))
            chunks.append({"id": f"{rel}_row{start + 1}", "source": rel,
                           "text": f"[TABLE ROW] {title}\n" + "\n".join(lines),
                           "kind": "table", "path": path})
        return chunks

    def _describe_image(self, path: str) -> str:
        """One vision call. Any failure -> indexed by name now, described on the next run."""
        if not settings.describe_images:
            return ""
        try:
            ck = self._caption_key(path)
            cached = self._cache.get_caption(ck)
        except OSError:
            ck, cached = "", ""
        if cached:
            self._n_vision_cached += 1
            return cached
        if self._vision_llm is None or not self._vision_ok:
            self._retry_later.add(path)
            return ""
        try:
            b64 = base64.b64encode(load_image_png(path)).decode()
            msg = HumanMessage(content=[
                {"type": "text", "text": IMAGE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ])
            self._n_vision_calls += 1
            caption = str(self._vision_llm.invoke([msg]).content).strip()
            if ck:
                self._cache.put_caption(ck, caption)
            return caption
        except Exception as e:
            self._retry_later.add(path)
            if is_fatal_api_error(e):
                with self._vision_lock:
                    if self._vision_ok:
                        self._vision_ok = False
                        log.error("vision stopped: %s (drawings indexed by name, described "
                                  "next run)", api_error_message(e))
            else:
                log.warning("vision: could not describe %s: %s", Path(path).name, str(e)[:200])
            return ""

    def _describe_images_parallel(self, paths: list[str]) -> None:
        if not paths or not settings.describe_images or self._vision_llm is None:
            return
        log.info("vision: describing %d drawing(s) with %d workers", len(paths),
                 settings.vision_workers)
        with ThreadPoolExecutor(max_workers=settings.vision_workers) as ex:
            futs = {ex.submit(self._describe_image, p): p for p in paths}
            for fut in as_completed(futs):
                self._captions[futs[fut]] = fut.result()

    def _image_chunks(self, path: str, rel: str) -> list[dict]:
        p = Path(path)
        name_words = re.sub(r"[_\-.]+", " ", p.stem)
        sidecar = ""
        for cand in (p.with_suffix(".txt"), p.with_suffix(".TXT")):
            if cand.exists():
                sidecar = cand.read_text(encoding="utf-8", errors="ignore").strip()
                break
        caption = self._captions[path] if path in self._captions else self._describe_image(path)
        text = ("[DRAWING] Technical drawing / exploded view (dibujo técnico, vista explosionada, "
                f"despiece)\nFile: {rel}\nName: {p.stem} ({name_words})\nFolder: {p.parent.name}\n"
                + (f"Notes: {sidecar}\n" if sidecar else "")
                + (f"Description:\n{caption}\n" if caption else ""))
        return [{"id": f"{rel}_drawing", "source": rel, "text": text, "kind": "image", "path": path}]

    def _build_chunks(self, path: str) -> list[dict]:
        rel, ext = self._rel_name(path), Path(path).suffix.lower()
        if ext in TABLE_EXTS:
            return self._csv_chunks(path, rel)
        if ext in IMAGE_EXTS:
            return self._image_chunks(path, rel)
        return self._text_chunks(path, rel)

    # ======================== persistence ========================
    def _load_manifest(self, quiet: bool = False) -> dict:
        if not os.path.exists(self._manifest_file):
            if not quiet:
                log.info("no manifest at %s -> everything will be indexed", self._manifest_file)
            return {}
        try:
            with open(self._manifest_file, "rb") as f:
                m = pickle.load(f)
            if not isinstance(m, dict):
                raise ValueError("unexpected format")
            return {p: e for p, e in m.items()
                    if isinstance(e, dict) and "mtime" in e and "chunks" in e}
        except Exception as e:
            log.warning("unreadable manifest, rebuilding: %s", e)
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        tmp = self._manifest_file + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(manifest, f)
        os.replace(tmp, self._manifest_file)

    def _embed(self, docs: list[Document]) -> None:
        for i in range(0, len(docs), settings.chroma_batch):
            batch = docs[i:i + settings.chroma_batch]
            self.vector_db.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])

    def _check_embeddings(self) -> None:
        """One tiny call before parsing anything: fail in 1 s, not after 10 minutes."""
        try:
            self.embeddings.inner.embed_query("ping")
        except Exception as e:
            raise IndexingAborted(api_error_message(e, f"Embeddings failed: {e}")
                                  + "\nNothing new indexed; the existing index is intact.") from e

    # ======================== sync ========================
    def _reuse_unchanged(self, candidates: list[str], current_files: dict[str, float],
                         manifest: dict, deleted_files: list[str]) -> tuple[list, list]:
        """Files that only LOOK new (same content, new date; or folder moved)."""
        touched, moved = [], []
        for p in candidates:
            e = manifest.get(p)
            if e and e["mtime"] != -1 and e.get("hash"):
                try:
                    if self._hash_of(p) == e["hash"]:
                        e["mtime"] = current_files[p]
                        touched.append(p)
                except OSError:
                    pass
        by_rel: dict[str, str] = {}
        for old in deleted_files:
            e = manifest[old]
            rel = e.get("rel") or (e["chunks"][0]["source"] if e.get("chunks") else "")
            if rel and e.get("mtime") != -1:
                by_rel.setdefault(rel, old)
        for p in candidates:
            if p in manifest or not by_rel:
                continue
            old = by_rel.get(self._rel_name(p))
            if not old:
                continue
            e = manifest[old]
            try:
                same = (e["hash"] == self._hash_of(p)) if e.get("hash") \
                    else (e["mtime"] == current_files[p])
            except OSError:
                same = False
            if same:
                for c in e["chunks"]:
                    c["path"] = p
                e["mtime"] = current_files[p]
                manifest[p] = manifest.pop(old)
                deleted_files.remove(old)
                del by_rel[self._rel_name(p)]
                moved.append(p)
        return touched, moved

    def _sync_and_index(self) -> None:
        t_all = time.time()
        cached_manifest = self._load_manifest()
        self._seed_cache(cached_manifest)
        self.embeddings.reset_stats()
        self._n_vision_calls = self._n_vision_cached = 0

        current_files: dict[str, float] = {}
        for p in self._iter_files():
            try:
                current_files[str(p)] = os.path.getmtime(p)
            except OSError as e:
                log.warning("cannot read %s: %s", p, e)

        candidates = [p for p, m in current_files.items()
                      if p not in cached_manifest or cached_manifest[p]["mtime"] != m]
        deleted_files = [p for p in cached_manifest if p not in current_files]
        touched, moved = self._reuse_unchanged(candidates, current_files, cached_manifest,
                                               deleted_files)
        reused = set(touched) | set(moved)
        new_or_modified = [p for p in candidates if p not in reused]
        n_new = sum(1 for p in new_or_modified if p not in cached_manifest)
        if reused:
            self._save_manifest(cached_manifest)

        sync = {"at": datetime.now().isoformat(timespec="seconds"),
                "found": len(current_files), "new": n_new,
                "changed": len(new_or_modified) - n_new, "deleted": len(deleted_files),
                "reused_files": len(current_files) - len(new_or_modified),
                "failed": 0, "chunks_added": 0, "complete": False}
        log.info("sync: %d found, %d new, %d changed, %d deleted, %d reused",
                 sync["found"], n_new, sync["changed"], len(deleted_files), sync["reused_files"])
        try:
            if new_or_modified or deleted_files:
                self._save_state(False)
                if new_or_modified:
                    self._check_embeddings()
                self._index_files(new_or_modified, deleted_files, cached_manifest,
                                  current_files, sync)
            sync["complete"] = True
        finally:
            self._record_sync(sync, time.time() - t_all)
        self._save_state(True)
        self._swap(self._build_snapshot(cached_manifest))

    def _index_files(self, new_or_modified: list[str], deleted_files: list[str],
                     cached_manifest: dict, current_files: dict[str, float],
                     sync: dict) -> None:
        stale_ids = [c["id"] for p in (deleted_files + new_or_modified) if p in cached_manifest
                     for c in cached_manifest[p]["chunks"]]
        if stale_ids:
            try:
                for i in range(0, len(stale_ids), settings.chroma_batch):
                    self.vector_db.delete(ids=stale_ids[i:i + settings.chroma_batch])
            except Exception as e:
                log.warning("could not delete %d old vectors: %s (retrieve ignores them)",
                            len(stale_ids), e)
        for path in deleted_files:
            del cached_manifest[path]
        if deleted_files:
            self._save_manifest(cached_manifest)

        self._describe_images_parallel(
            [p for p in new_or_modified if Path(p).suffix.lower() in IMAGE_EXTS])

        pending_docs: list[Document] = []
        pending_entries: dict = {}
        done = [0]

        def flush() -> None:
            if pending_docs:
                try:
                    self._embed(pending_docs)
                except Exception as e:
                    self._save_manifest(cached_manifest)     # only committed files
                    raise IndexingAborted(
                        api_error_message(e, f"Embedding error: {e}")
                        + f"\nSaved: {done[0]} file(s) indexed; the rest resumes next run.") from e
                sync["chunks_added"] += len(pending_docs)
            cached_manifest.update(pending_entries)
            done[0] += len(pending_entries)
            self._save_manifest(cached_manifest)
            pending_docs.clear()
            pending_entries.clear()

        total = len(new_or_modified)
        for n, path in enumerate(new_or_modified, 1):
            try:
                chunk_dicts = self._build_chunks(path)
                fhash = self._hash_of(path)
            except Exception as e:
                log.warning("[%d/%d] %s FAILED: %s", n, total, self._rel_name(path), e)
                sync["failed"] += 1
                continue
            to_embed = chunk_dicts
            if (Path(path).suffix.lower() in TABLE_EXTS
                    and len(chunk_dicts) - 1 > settings.vector_max_table_rows):
                to_embed = chunk_dicts[:1]                 # big table: rows BM25-only
            if not chunk_dicts:
                log.warning("%s: no text (scanned PDF?)", self._rel_name(path))
            pending_docs.extend(Document(
                page_content=c["text"],
                metadata={"chunk_id": c["id"], "source": c["source"], "kind": c["kind"]})
                for c in to_embed if c["text"].strip())
            mtime = -1 if path in self._retry_later else current_files[path]
            pending_entries[path] = {"mtime": mtime, "hash": fhash,
                                     "rel": self._rel_name(path), "chunks": chunk_dicts}
            log.info("[%d/%d] %s: %d chunks", n, total, self._rel_name(path), len(chunk_dicts))
            if n % settings.save_every == 0 or len(pending_docs) >= settings.embed_flush_docs:
                flush()
        flush()

    def _record_sync(self, sync: dict, secs: float) -> None:
        es = self.embeddings.stats
        sync.update(secs=round(secs, 1), api_texts=es["misses"], cached_texts=es["hits"],
                    chars_sent=es["chars_sent"], chars_saved=es["chars_saved"],
                    vision_calls=self._n_vision_calls, vision_cached=self._n_vision_cached)
        self.last_sync = sync
        st = self._load_state()
        tot = st.get("totals") or {}
        for k in ("api_texts", "cached_texts", "chars_sent", "chars_saved", "vision_calls",
                  "vision_cached", "reused_files"):
            tot[k] = int(tot.get(k, 0)) + int(sync.get(k, 0))
        st["totals"] = tot
        st["history"] = (st.get("history") or [])[-19:] + [sync]
        self._write_state(st)
        log.info("sync done in %s (embeddings: %d cached / %d via API; drawings: %d cached / "
                 "%d via API)", fmt_secs(secs), es["hits"], es["misses"],
                 self._n_vision_cached, self._n_vision_calls)

    def _build_snapshot(self, manifest: dict) -> IndexSnapshot:
        """In-memory BM25 + chunk table + catalog from the manifest (no API calls)."""
        tokenized_corpus = []
        chunks: dict[str, dict] = {}
        chunk_ids: list[str] = []
        kind_idx: dict[str, list[int]] = {}
        catalog: dict[str, dict] = {}
        file_idx: dict[str, list[int]] = {}
        chunk_rel: list[str] = []
        section_idx: dict[str, list[int]] = {}
        files: list[tuple[str, str]] = []
        vocab: set[str] = set()
        for path, item in manifest.items():
            rel = item.get("rel") or (item["chunks"][0]["source"] if item.get("chunks")
                                      else self._rel_name(path))
            if settings.is_excluded(rel):            # RAG_EXCLUDE_DIRS, even before `update`
                continue
            files.append((path, rel))
            section = settings.section_of(rel)       # from the folders declared in the .env
            for chunk in item["chunks"]:
                chunk.setdefault("kind", "text")
                chunk["section"] = section
                section_idx.setdefault(section, []).append(len(chunk_ids))
                kind_idx.setdefault(chunk["kind"], []).append(len(chunk_ids))
                file_idx.setdefault(rel, []).append(len(chunk_ids))
                chunk_rel.append(rel)
                chunks[chunk["id"]] = chunk
                chunk_ids.append(chunk["id"])
                tokenized_corpus.append(bm25_tokenize(chunk["text"]))
                if settings.query_spell_fix:
                    vocab.update(simple_tokenize(chunk["text"]))
            try:
                catalog[rel] = build_catalog_entry(path, rel, item["chunks"])
            except Exception as e:
                log.warning("catalog: could not describe %s: %s", rel, e)
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        try:
            mtime = os.path.getmtime(self._manifest_file)
        except OSError:
            mtime = 0.0
        return IndexSnapshot(
            chunks=chunks, chunk_ids=chunk_ids, kind_idx=kind_idx, bm25=bm25,
            catalog=catalog, file_idx=file_idx, chunk_rel=chunk_rel, files=files,
            section_idx=section_idx,
            n_pending=sum(1 for e in manifest.values() if e.get("mtime") == -1),
            manifest_mtime=mtime, built_at=time.time(),
            vocab=frozenset(vocab), vocab_idx=build_vocab(vocab) if vocab else {})

    # ======================== search ========================
    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if lo == hi:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def _vector_search(self, query: str, kinds: set[str] | None,
                       pool: int | None = None) -> tuple[dict[str, float], bool]:
        flt = {"kind": {"$in": sorted(kinds)}} if kinds else None
        db = self.vector_db                                  # local ref (rebuild may swap it)
        k = pool or settings.candidate_pool
        try:
            try:
                vec = db.similarity_search_with_relevance_scores(query, k=k, filter=flt)
            except Exception as e:
                if classify_api_error(e):
                    raise
                vec = db.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            if not self._vector_warned:
                log.warning("vector search failed (%s); using BM25 only", api_error_message(e))
                self._vector_warned = True
            return {}, False
        return {d.metadata["chunk_id"]: float(score) for d, score in vec
                if "chunk_id" in d.metadata}, True

    def retrieve(self, query: str, top_k: int | None = None, kinds: set[str] | None = None,
                 hits: list[dict] | None = None, sections: set[str] | None = None) -> list[dict]:
        """Hybrid search; `kinds` restricts to chunk types ({"image"} = drawings only),
        `sections` to the folders declared in the .env ({"manuales"}, {"hojas_tecnicas",
        "productos"} ...). Results are appended to `hits` (the caller's list) if given."""
        return self.retrieve_with_signals(query, top_k, kinds, hits, sections)[0]

    def vocab(self, snap: IndexSnapshot | None = None) -> tuple[frozenset, dict]:
        """Words of the index (plain, unstemmed corpus words) + their typo index. Built with
        the snapshot. (The live version read the BM25 idf keys; since 1.0 those are STEMS,
        which would make spell_fix suggest stems, so the words are kept separately.)"""
        snap = snap or self._snap
        return snap.vocab, snap.vocab_idx

    def search_query(self, query: str, snap: IndexSnapshot | None = None) -> str:
        """Glossary translation + typo correction against the documents' own words:
        'Düsen' -> + 'nozzle nozzles', 'nozles' -> + 'nozzles' (app/query_lang.py)."""
        if getattr(settings, "query_translate", "glossary") == "off":
            return query
        snap = snap or self._snap
        q = expand_query(query)
        if settings.query_spell_fix and snap.vocab:
            q = spell_fix(q, snap.vocab, snap.vocab_idx)
        return q

    def _candidates(self, query: str, kinds: set[str] | None = None,
                    sections: set[str] | None = None, snap: IndexSnapshot | None = None,
                    expanded: bool = False,
                    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]], SearchSignals]:
        """Raw candidate lists, best first: (vector [(id, relevance)], BM25 [(id, score)],
        signals). Used by retrieve_with_signals() and by MultiRetriever, which fuses the
        lists of every base in ONE Reciprocal Rank Fusion. `expanded`: the query already
        went through search_query() (MultiRetriever expands once for all bases)."""
        snap = snap or self._snap
        sig = SearchSignals()
        if snap.empty or not (query or "").strip():
            return [], [], sig
        if not expanded:
            query = self.search_query(query, snap)
        sections = set(sections) if sections else None

        def ok(c: str) -> bool:
            ch = snap.chunks.get(c)
            return (ch is not None and (not kinds or ch["kind"] in kinds)
                    and (not sections or ch.get("section", OTHER_SECTION) in sections))

        if sections and not any(snap.section_idx.get(x) for x in sections):
            return [], [], sig                               # that source has no files
        # with a section filter the vector pool is widened, then filtered here
        pool = settings.candidate_pool * (5 if sections else 1)
        vec_scores, sig.vector_ok = self._vector_search(query, kinds, pool)
        vec = sorted(((c, v) for c, v in vec_scores.items() if ok(c)),
                     key=lambda x: x[1], reverse=True)[:settings.candidate_pool]
        raw = snap.bm25.get_scores(bm25_tokenize(query))
        idx: Any = range(len(raw))
        if kinds:
            idx = {i for kd in kinds for i in snap.kind_idx.get(kd, [])}
        if sections:
            sec = {i for x in sections for i in snap.section_idx.get(x, [])}
            idx = sec if not kinds else (set(idx) & sec)
        top = heapq.nlargest(settings.candidate_pool, idx, key=lambda i: raw[i])
        # != 0 (not > 0): rank_bm25 gives NEGATIVE idf to very common words (small corpora)
        bm = [(snap.chunk_ids[i], float(raw[i])) for i in top if raw[i] != 0]
        sig.max_vector = max((v for _, v in vec), default=0.0)
        sig.max_bm25 = max((v for _, v in bm), default=0.0)
        return vec, bm, sig

    def retrieve_with_signals(self, query: str, top_k: int | None = None,
                              kinds: set[str] | None = None,
                              hits: list[dict] | None = None,
                              sections: set[str] | None = None,
                              ) -> tuple[list[dict], SearchSignals]:
        """Hybrid search: vector + BM25 candidates -> RRF -> cross-encoder -> code pinning
        -> top_k -> whole tables. `kinds` restricts to chunk types ({"image"} = drawings
        only), `sections` to the folders declared in the .env. The signals keep the RAW
        best vector / BM25 scores, so the router's fast-path thresholds mean the same as
        in 5.7."""
        snap = self._snap
        k = top_k or self._top_k
        search_q = self.search_query(query or "", snap)
        vec, bm, sig = self._candidates(search_q, kinds, sections, snap, expanded=True)
        if not vec and not bm:
            return [], sig
        fused = rrf_fuse([[c for c, _ in vec], [c for c, _ in bm]],
                         [settings.rrf_vector_weight, settings.rrf_bm25_weight])
        results, sig.max_rerank = rerank_and_select(query, fused, snap.chunks, k, max_extra=6,
                                                    focus_query=search_q)
        sig.n_results = len(results)
        if hits is not None:
            hits.extend(results)
        return results, sig

    def sections_present(self) -> set[str]:
        """Sections (declared folders) that have at least one indexed chunk."""
        return {s for s, v in self._snap.section_idx.items() if v}

    def drawings(self) -> list[dict]:
        snap = self._snap
        return [snap.chunks[snap.chunk_ids[i]] for i in snap.kind_idx.get("image", [])]

    # ======================== document-level access ========================
    def resolve_doc(self, raw: str, snap: IndexSnapshot | None = None) -> tuple[str | None, list[str]]:
        """File name, partial name, title or chunk id -> (rel of one file, other candidates)."""
        snap = snap or self._snap
        r = str(raw or "").strip().strip("[]\"'` ").strip()
        if not r or not snap.catalog:
            return None, []
        if r in snap.catalog:
            return r, []
        if r in snap.chunks:
            return snap.chunks[r].get("source"), []
        m = re.match(r"(.+?)_(?:chunk\d+|row\d+|table|drawing)$", r)
        if m and m.group(1) in snap.catalog:
            return m.group(1), []
        rl = r.lower().replace("\\", "/")
        low = {k.lower(): k for k in snap.catalog}
        if rl in low:
            return low[rl], []
        exact = [k for k in snap.catalog
                 if Path(k).name.lower() == rl or Path(k).stem.lower() == rl]
        if exact:
            return exact[0], exact[1:5]
        key = mkey(r)
        if len(key) >= 3:
            part = [k for k, e in snap.catalog.items()
                    if key in mkey(k) or key in mkey(e["title"])]
            if part:
                part.sort(key=len)
                return part[0], part[1:5]
        names = {Path(k).name.lower(): k for k in snap.catalog}
        titles = {e["title"].lower(): k for k, e in snap.catalog.items()}
        close = difflib.get_close_matches(rl, list(names) + list(titles), n=5, cutoff=0.6)
        found = list(dict.fromkeys(names.get(c) or titles[c] for c in close))
        return (found[0], found[1:]) if found else (None, [])

    @staticmethod
    def _file_bm25(snap: IndexSnapshot, query: str) -> dict[str, float]:
        best: dict[str, float] = {}
        if snap.bm25 is None or not query.strip():
            return best
        raw = snap.bm25.get_scores(bm25_tokenize(query))
        for i, sc in enumerate(raw):
            if sc > 0:
                rel = snap.chunk_rel[i]
                if sc > best.get(rel, 0.0):
                    best[rel] = float(sc)
        return best

    def list_documents(self, keyword: str = "", manufacturer: str = "", doc_type: str = "",
                       limit: int | None = None) -> tuple[list[dict], int]:
        snap = self._snap
        limit = limit or settings.list_docs_max
        items = list(snap.catalog.values())
        if manufacturer:
            mk = mkey(manufacturer)
            items = [e for e in items if mk in mkey(e["manufacturer"]) or mk in mkey(e["rel"])]
        if doc_type:
            t = canon_type(doc_type)
            items = [e for e in items if e["type"] == t or (t == "document" and e["type"] in
                                                            ("manual", "datasheet", "document"))]
        keyword = str(keyword or "").strip()
        if keyword:
            q_codes = extract_codes(keyword)
            q_words = {w for w in simple_tokenize(keyword) if len(w) >= 3}
            text_score = self._file_bm25(snap, keyword)
            top = max(text_score.values(), default=0.0) or 1.0
            scored = []
            for e in items:
                sc = 10.0 * len(q_codes & e["codes"])
                name_words = set(simple_tokenize(f"{e['rel']} {e['title']} {e['manufacturer']}"))
                sc += 2.0 * len(q_words & name_words)
                sc += 5.0 * text_score.get(e["rel"], 0.0) / top
                if sc > 0:
                    scored.append((sc, e))
            scored.sort(key=lambda x: x[0], reverse=True)
            items = [e for _, e in scored]
        else:
            items.sort(key=lambda e: (e["manufacturer"].lower(), e["type"], e["rel"].lower()))
        return items[:max(1, int(limit))], len(items)

    def catalog_summary(self) -> str:
        cat = self._snap.catalog
        by_type = Counter(DOC_TYPE_LABEL.get(e["type"], e["type"]) for e in cat.values())
        return f"{len(cat)} files: " + ", ".join(f"{v} {k}" for k, v in by_type.most_common())

    def read_document(self, doc: str, part: int = 1, query: str = "",
                      max_chars: int | None = None, hits: list[dict] | None = None) -> str:
        snap = self._snap
        max_chars = max_chars or settings.read_doc_max_chars
        rel, alts = self.resolve_doc(doc, snap)
        if not rel or rel not in snap.catalog:
            sugg = ", ".join(alts) if alts else ""
            return (f"No document matches '{doc}'. Call list_documents to see the file names"
                    + (f" (similar: {sugg})" if sugg else "") + ".")
        e = snap.catalog[rel]
        chunks = [snap.chunks[snap.chunk_ids[i]] for i in snap.file_idx.get(rel, [])]
        header = (f"[DOCUMENT {rel}] {{{SECTION_LABEL.get(e.get('section', OTHER_SECTION), '?')}}} "
                  f"{DOC_TYPE_LABEL.get(e['type'], e['type'])} | "
                  f"{e['manufacturer'] or '?'} | {e['title'][:100]}")
        if alts:
            header += f"\n(other files with a similar name: {', '.join(alts)})"
        note = ""
        if e["type"] == "table":
            rows = [c["text"].split("\n", 1)[-1] for c in chunks[1:]]
            shown = rows[:settings.read_table_max_rows]
            full = (chunks[0]["text"] if chunks else "") + "\n" + "\n".join(shown)
            if len(rows) > len(shown):
                note = (f"\n... {len(rows) - len(shown)} more rows. Use table_lookup(code=...) "
                        "to find exact rows.")
        else:
            full = merge_chunks([c["text"] for c in chunks])
        if not full.strip():
            return header + "\n(this file has no readable text — scanned PDF?)"

        query = str(query or "").strip()
        if query and len(full) > max_chars and e["type"] not in ("table", "drawing"):
            idx = snap.file_idx.get(rel, [])
            raw = snap.bm25.get_scores(bm25_tokenize(query)) if snap.bm25 is not None else []
            order = sorted(range(len(idx)), key=lambda j: raw[idx[j]] if len(raw) else 0,
                           reverse=True)
            picked, used = [], 0
            for j in order:
                t = chunks[j]["text"]
                if used + len(t) > max_chars and picked:
                    break
                picked.append(j)
                used += len(t)
            body = merge_chunks([chunks[j]["text"] for j in sorted(picked)])[:max_chars]
            n_parts = math.ceil(len(full) / max_chars)
            text = (f"{header}\n(passages of this file most relevant to '{query}'; the whole "
                    f"file has {n_parts} parts — use part=N without query to read it in "
                    f"order)\n{body}")
        else:
            n_parts = max(1, math.ceil(len(full) / max_chars))
            p = min(max(1, int(part or 1)), n_parts)
            body = full[(p - 1) * max_chars: p * max_chars]
            text = f"{header} — part {p} of {n_parts}\n{body}{note}"
            if p < n_parts:
                text += (f"\n... (continues: read_document(doc='{rel}', part={p + 1}), or give "
                         "query to jump to the relevant passages)")
        if hits is not None:
            hits.append({"id": rel, "source": rel, "text": body[:2000],
                         "kind": "image" if e["type"] == "drawing" else "document",
                         "section": e.get("section", OTHER_SECTION), "path": e["path"]})
        return text

    def _table_rows(self, path: str) -> tuple[list[str], list[list[str]], list[list[str]]]:
        """(headers, rows, normalized cells) of one CSV, cached until the file changes."""
        m = os.path.getmtime(path)
        with self._table_lock:
            got = self._table_cache.get(path)
        if got and got[0] == m:
            return got[1], got[2], got[3]
        headers, rows, _ = read_csv_table(path)
        normed = [[norm(c) for c in r] for r in rows]
        with self._table_lock:
            self._table_cache[path] = (m, headers, rows, normed)
        return headers, rows, normed

    def table_lookup(self, code: str, columns: str = "", table: str = "",
                     limit: int | None = None, hits: list[dict] | None = None) -> str:
        """Exact search of article / model codes in every CSV table (no API calls)."""
        snap = self._snap
        limit = limit or settings.table_lookup_max_rows
        codes = [c.strip() for c in re.split(r"[,;]+", str(code or "")) if c.strip()]
        codes = [c for c in codes if len(norm(c)) >= 2]
        if not codes:
            return "table_lookup needs a code, e.g. code='123456' or code='VDV-25'."
        tables = [e for e in snap.catalog.values() if e["type"] == "table"]
        if table:
            tk = mkey(table)
            tables = [e for e in tables if tk in mkey(e["rel"]) or tk in mkey(e["title"])]
        if not tables:
            return ("No CSV table" + (f" matches '{table}'" if table else " in the index")
                    + ". Use semantic_search instead.")
        wanted_cols = [c.strip() for c in re.split(r"[,;]+", str(columns or "")) if c.strip()]
        found: list[tuple[int, str, int, str]] = []
        cache: dict[str, tuple[list[str], list[list[str]]]] = {}
        problems = []
        for e in tables:
            try:
                headers, rows, normed = self._table_rows(e["path"])
            except Exception as ex:
                problems.append(f"{e['rel']}: {ex}")
                continue
            cache[e["rel"]] = (headers, rows)
            for c in codes:
                nc = norm(c)
                for i, cells in enumerate(normed):
                    sc = 0
                    for cell, raw in zip(cells, rows[i]):
                        if not cell:
                            continue
                        if cell == nc:
                            sc = 3
                            break
                        if nc in cell:
                            if nc in {norm(x) for x in re.split(r"[\s/,;|()]+", raw)}:
                                sc = max(sc, 2)
                            elif len(nc) >= 4:
                                sc = max(sc, 1)
                    if sc:
                        found.append((sc, e["rel"], i, c))
        if not found:
            msg = (f"No row contains {', '.join(repr(c) for c in codes)} in the "
                   f"{len(tables)} CSV table(s). Check the spelling, or use semantic_search "
                   "/ drawing_search.")
            return msg + (f"\n(unreadable tables: {'; '.join(problems)})" if problems else "")
        found.sort(key=lambda x: (-x[0], x[1], x[2]))
        seen, lines = set(), []
        label = {3: "exact", 2: "code in cell", 1: "partial"}
        n_exact = sum(1 for f in found if f[0] == 3)
        col_note = ""
        step = settings.csv_rows_per_chunk
        for sc, rel, i, c in found:
            if (rel, i) in seen:
                continue
            seen.add((rel, i))
            if len(lines) >= limit:
                continue
            headers, rows = cache[rel]
            row = rows[i]
            pairs = [(h, v) for h, v in zip(headers, row) if v]
            if wanted_cols:
                keys = [mkey(w) for w in wanted_cols]
                sel = [(h, v) for h, v in pairs
                       if any(k and (k in mkey(h) or mkey(h) in k) for k in keys)
                       or norm(c) in norm(v)]
                if len(sel) > sum(1 for h, v in pairs if norm(c) in norm(v)):
                    pairs = sel
                else:
                    col_note = (f"\n(columns '{columns}' not found; available in {rel}: "
                                f"{', '.join(headers)[:300]})")
            start = (i // step) * step
            cid = f"{rel}_row{start + 1}"
            src = SECTION_LABEL.get(snap.catalog[rel].get("section", OTHER_SECTION), "?")
            lines.append(f"[{cid}] {{{src}}} ({label[sc]} match for {c}) Row {i + 1}: "
                         + " | ".join(f"{h}: {v}" for h, v in pairs))
            if hits is not None:
                chunk = snap.chunks.get(cid)
                hits.append(dict(chunk, match=label[sc]) if chunk else
                            {"id": cid, "source": rel, "text": lines[-1], "kind": "table",
                             "path": snap.catalog[rel]["path"], "match": label[sc]})
        head = (f"table_lookup({', '.join(codes)}) -> {len(seen)} matching row(s) "
                f"({n_exact} exact) in {len({f[1] for f in found})} table(s); showing "
                f"{len(lines)}")
        return head + "\n" + "\n".join(lines) + col_note
