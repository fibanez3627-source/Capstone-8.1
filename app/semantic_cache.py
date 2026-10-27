"""Caché semántica de RESPUESTAS en SQLite (Lab 7.2 del Capstone 7.1, adaptada a Relant).

Una pregunta que ya se respondió no vuelve a pagar el router, la búsqueda ni el LLM:

    ProductionRouter.answer(q)
      0. q -> pregunta autónoma (seguimientos de la conversación)
      1. CACHÉ  ── exacta  (texto normalizado igual)                -> 0 tokens
               └─ semántica (coseno >= SEMANTIC_CACHE_THRESHOLD)
                     + guardas deterministas (gratis):
                         · mismos códigos de parte  (VDV-25 ≠ VDV-32)
                         · mismos números           (20:1 ≠ 10:1, 250 ≠ 300)
                         · misma fuente pedida      (manual ≠ hoja técnica)
                     + validación con LLM barato (opcional)          -> ~1 llamada pequeña
      2. si no hay hit: router normal (fast path / agente) y se GUARDA la respuesta.

Archivo: <RAG_DB_DIR>/semantic_cache.sqlite (o RAG_SEMANTIC_CACHE_FILE). Sin servidor.

Tablas
    answers  una fila por respuesta guardada: pregunta, vector, respuesta completa (JSON
             con fuentes y dibujos), uso original (tokens / costo), huella de las fuentes,
             contadores de hits y de tokens ahorrados.
    events   una fila por consulta (hit exacto / semántico / rechazado / miss / guardado /
             no guardado): base de `python -m app.cli cache-stats` y de la comparativa.

Invalidación (SEMANTIC_CACHE_INVALIDATION)
    sources  (defecto) la respuesta guarda una huella (hash del texto indexado) de cada
             archivo que usó; si uno cambió o se borró tras un `update`, la entrada se
             descarta. Lo demás de la caché sigue valiendo.
    index    cualquier re-indexación invalida toda la caché.
    off      sólo TTL (SEMANTIC_CACHE_TTL_DAYS).

Nunca se guardan: preguntas de aclaración, respuestas "no encontré / INSUFFICIENT_CONTEXT",
listados del catálogo (cambian al agregar archivos) ni respuestas que usan las secciones de
SEMANTIC_CACHE_SKIP_SECTIONS (por defecto precios/existencias: cambian a diario).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.textutils import extract_codes, fold, parse_json, query_numbers

log = logging.getLogger("rag.semantic_cache")

SCHEMA_VERSION = 1
MAX_VALIDATIONS_PER_LOOKUP = 2          # worst case of a miss: 2 small validator calls

CACHE_VALIDATE_SYSTEM = (
    "You validate a semantic-cache hit for a technical assistant (spare parts, pumps, valves, "
    "dispensing equipment). You get the NEW question, the CACHED question it matched and the "
    "CACHED answer.\n"
    "Answer valid=true ONLY if (1) the new question asks for exactly the same information as "
    "the cached question (same product / model / part code, same quantity or property, same "
    "kind of document), and (2) the cached answer fully answers the new question.\n"
    "Different products, different values asked (pressure vs flow), a different scope "
    "(one part vs the whole list) or a different requested format -> valid=false.\n"
    'Respond with ONLY valid JSON, no other text: {"valid": true} or '
    '{"valid": false, "reason": "..."}'
)

# Answers that only say the documents do not contain the information (ES / EN / DE).
# Cheap to recompute and useless to repeat: never stored (same rule as Capstone 7.1).
_NO_ANSWER_RE = re.compile(
    r"INSUFFICIENT[_ ]CONTEXT|"
    r"no (encontr\w+|hay informaci\w+|tengo informaci\w+|se encontr\w+|dispongo)|"
    r"(los|las) (documentos|fragmentos|fuentes) (proporcionad\w+ )?no (contienen|mencionan|incluyen|indican)|"
    r"no (se menciona|se indica|aparece|figura|est[aá] (disponible|indicad\w+))|"
    r"do(es)? not (contain|mention|include|provide)|could ?n'?t find|cannot (find|answer|determine)|"
    r"no (information|passage|mention) (about|on|of)|nicht (enthalten|gefunden)",
    re.IGNORECASE)


def is_no_answer(answer: str) -> bool:
    head = (answer or "")[:350].replace("’", "'").replace("**", "")
    return bool(_NO_ANSWER_RE.search(head))


def normalize_question(q: str) -> str:
    """Key of the exact match: lower-case, no accents, no punctuation, single spaces."""
    return fold(q or "")


def question_features(q: str, intent_fn: Callable[[str], list[str]] | None = None) -> dict:
    """What MUST be equal between two questions for a semantic hit (free, deterministic)."""
    return {"codes": sorted(extract_codes(q or "")),
            "numbers": sorted(query_numbers(q or "")),
            "intent": sorted(intent_fn(q or "")) if intent_fn else []}


def scope_key(instructions: str = "") -> str:
    """Answers are only reused with the same models, prompts and extra instructions."""
    raw = "|".join([settings.llm_model, settings.effective_agent_model,
                    settings.embed_model, (instructions or "").strip(),
                    settings.source_routing, f"v{SCHEMA_VERSION}"])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class CacheLookup:
    kind: str                       # exact | semantic | miss | rejected | disabled
    similarity: float = 0.0
    entry: dict | None = None       # stored row (with "result" decoded) on a hit
    best_question: str = ""         # nearest stored question (also on a miss)
    note: str = ""                  # human-readable explanation (log / console)
    validated: bool = False         # an LLM validated the semantic hit
    candidates: int = 0             # entries compared

    @property
    def hit(self) -> bool:
        return self.kind in ("exact", "semantic")


@dataclass
class _ScopeIndex:
    ids: list[int] = field(default_factory=list)
    mat: np.ndarray | None = None   # (n, dim) float32, rows L2-normalised


class SemanticAnswerCache:
    """Thread-safe SQLite answer cache with an in-memory cosine index per scope."""

    def __init__(self, path: str, embed_fn: Callable[[str], list[float]] | None, *,
                 threshold: float | None = None, ttl_days: float | None = None,
                 max_entries: int | None = None, validate: bool | None = None,
                 invalidation: str | None = None) -> None:
        self.path = path
        self.embed_fn = embed_fn
        self.threshold = settings.semantic_cache_threshold if threshold is None else threshold
        self.ttl_days = settings.semantic_cache_ttl_days if ttl_days is None else ttl_days
        self.max_entries = settings.semantic_cache_max_entries if max_entries is None else max_entries
        self.validate = settings.semantic_cache_validate if validate is None else validate
        self.invalidation = (invalidation or settings.semantic_cache_invalidation
                             or "sources").strip().lower()
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._index: dict[str, _ScopeIndex] = {}
        self._data_version = -1
        self._fp_cache: dict[str, str] = {}
        self._fp_snapshot: Any = None               # fingerprints belong to THIS snapshot
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    question TEXT NOT NULL,
                    norm TEXT NOT NULL,
                    features TEXT NOT NULL,
                    emb BLOB,
                    dim INTEGER NOT NULL DEFAULT 0,
                    result TEXT NOT NULL,
                    route TEXT NOT NULL DEFAULT '',
                    sources_fp TEXT NOT NULL DEFAULT '{}',
                    index_sig TEXT NOT NULL DEFAULT '',
                    tokens_in INTEGER NOT NULL DEFAULT 0,
                    tokens_out INTEGER NOT NULL DEFAULT 0,
                    llm_calls INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    created REAL NOT NULL,
                    last_hit REAL NOT NULL DEFAULT 0,
                    hits INTEGER NOT NULL DEFAULT 0,
                    tokens_saved INTEGER NOT NULL DEFAULT 0,
                    cost_saved REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS answers_scope_norm ON answers(scope, norm);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    question TEXT NOT NULL,
                    similarity REAL,
                    entry_id INTEGER,
                    tokens_used INTEGER NOT NULL DEFAULT 0,
                    tokens_saved INTEGER NOT NULL DEFAULT 0,
                    cost_saved REAL NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    note TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
            """)
            self._db.execute("INSERT OR IGNORE INTO meta VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
            self._db.commit()

    # ------------------------------------------------------------------ helpers
    def _embed(self, text: str) -> np.ndarray | None:
        if self.embed_fn is None:
            return None
        try:
            v = np.asarray(self.embed_fn(text), dtype=np.float32)
        except Exception as e:                      # embeddings down -> exact match only
            log.warning("semantic cache: embedding failed (%s) -> exact match only", str(e)[:120])
            return None
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def _refresh_if_changed(self) -> None:
        """Another process (a second console) wrote to the file -> rebuild the vectors."""
        dv = self._db.execute("PRAGMA data_version").fetchone()[0]
        if dv != self._data_version:
            self._data_version = dv
            self._index.clear()

    def _scope_index(self, scope: str) -> _ScopeIndex:
        idx = self._index.get(scope)
        if idx is not None:
            return idx
        ids, vecs, dim = [], [], 0
        for row in self._db.execute("SELECT id, emb, dim FROM answers WHERE scope=? AND emb IS NOT NULL",
                                    (scope,)):
            v = np.frombuffer(row["emb"], dtype=np.float32)
            if dim and v.shape[0] != dim:
                continue                             # embedding model changed: ignore old rows
            dim = v.shape[0]
            ids.append(int(row["id"]))
            vecs.append(v)
        idx = _ScopeIndex(ids, np.vstack(vecs) if vecs else None)
        self._index[scope] = idx
        return idx

    def _expired(self, row: sqlite3.Row) -> bool:
        return bool(self.ttl_days) and (time.time() - float(row["created"])) > self.ttl_days * 86400

    # ---- source fingerprints (invalidation="sources") ----
    def file_fingerprint(self, snapshot: Any, rel: str) -> str:
        """sha1 of the indexed text of one file ("" = not in the index any more)."""
        if snapshot is not self._fp_snapshot:      # strong ref: no id() reuse after a reload
            self._fp_snapshot, self._fp_cache = snapshot, {}
        if rel in self._fp_cache:
            return self._fp_cache[rel]
        idxs = (getattr(snapshot, "file_idx", None) or {}).get(rel)
        if not idxs:
            fp = ""
        else:
            h = hashlib.sha1()
            ids = snapshot.chunk_ids
            for i in idxs:
                h.update(snapshot.chunks[ids[i]].get("text", "").encode("utf-8", "ignore"))
            fp = h.hexdigest()[:16]
        self._fp_cache[rel] = fp
        return fp

    def sources_fingerprint(self, snapshot: Any, sources: list[dict]) -> dict[str, str]:
        out = {}
        for s in sources or []:
            rel = s.get("source") or ""
            if not rel or s.get("kind") == "web" or rel in out:
                continue
            out[rel] = self.file_fingerprint(snapshot, rel)
        return out

    def _still_valid(self, row: sqlite3.Row, snapshot: Any) -> tuple[bool, str]:
        if self._expired(row):
            return False, f"expired (TTL {self.ttl_days} days)"
        if snapshot is None or self.invalidation == "off":
            return True, ""
        if self.invalidation == "index":
            sig = index_signature(snapshot)
            return (row["index_sig"] == sig, "" if row["index_sig"] == sig else "index re-built")
        stored = json.loads(row["sources_fp"] or "{}")
        for rel, fp in stored.items():
            if self.file_fingerprint(snapshot, rel) != fp:
                return False, f"source changed: {rel}"
        return True, ""

    def _delete(self, entry_id: int, scope: str) -> None:
        self._db.execute("DELETE FROM answers WHERE id=?", (entry_id,))
        self._db.commit()
        self._index.pop(scope, None)

    # ------------------------------------------------------------------ lookup
    def lookup(self, question: str, *, scope: str, features: dict, snapshot: Any = None,
               validator: Callable[[str, str, str], tuple[bool, str]] | None = None) -> CacheLookup:
        norm = normalize_question(question)
        if not norm:
            return CacheLookup("miss", note="empty question")
        with self._lock:
            self._refresh_if_changed()
            # 1) exact (normalised) match: trusted, zero tokens
            for row in self._db.execute("SELECT * FROM answers WHERE scope=? AND norm=? "
                                        "ORDER BY created DESC", (scope, norm)).fetchall():
                ok, why = self._still_valid(row, snapshot)
                if not ok:
                    self._delete(int(row["id"]), scope)
                    log.info("semantic cache: dropped entry %s (%s)", row["id"], why)
                    continue
                return CacheLookup("exact", 1.0, _decode(row), row["question"],
                                   note=f"exact match (entry {row['id']})", candidates=1)
            # 2) semantic: cosine over the stored question vectors
            qv = self._embed(question)
            if qv is None:
                return CacheLookup("miss", note="no embedding -> exact match only")
            idx = self._scope_index(scope)
            if idx.mat is None or idx.mat.shape[1] != qv.shape[0]:
                return CacheLookup("miss", note="miss (cache empty)")
            sims = idx.mat @ qv
            order = np.argsort(-sims)[:5]
            best_q, best_sim, reasons = "", float(sims[order[0]]), []
            seen_answers: set[str] = set()          # an alias shares its original's answer
            validations = 0
            for j in order:
                sim = float(sims[j])
                row = self._db.execute("SELECT * FROM answers WHERE id=?", (idx.ids[j],)).fetchone()
                if row is None:
                    continue
                best_q = best_q or row["question"]
                if sim < self.threshold:
                    break
                ok, why = self._still_valid(row, snapshot)
                if not ok:
                    self._delete(int(row["id"]), scope)
                    reasons.append(why)
                    continue
                stored = json.loads(row["features"] or "{}")
                diff = [k for k in ("codes", "numbers", "intent")
                        if stored.get(k, []) != features.get(k, [])]
                if diff:
                    reasons.append(f"guard: different {'/'.join(diff)} vs '{row['question'][:50]}' "
                                   f"(sim={sim:.3f})")
                    continue
                akey = hashlib.sha1(row["result"].encode("utf-8")).hexdigest()
                if akey in seen_answers:
                    continue
                seen_answers.add(akey)
                if self.validate and validator is not None:
                    if validations >= MAX_VALIDATIONS_PER_LOOKUP:
                        break
                    validations += 1
                    entry = _decode(row)
                    valid, why = validator(question, row["question"], entry["result"].get("answer", ""))
                    if not valid:
                        reasons.append(f"validator rejected sim={sim:.3f}: {why or 'no reason'}")
                        continue
                    return CacheLookup("semantic", sim, entry, row["question"], validated=True,
                                       note=f"sim={sim:.3f} vs '{row['question'][:60]}' (validated)",
                                       candidates=len(idx.ids))
                return CacheLookup("semantic", sim, _decode(row), row["question"],
                                   note=f"sim={sim:.3f} vs '{row['question'][:60]}'",
                                   candidates=len(idx.ids))
            if reasons:
                return CacheLookup("rejected", best_sim, None, best_q, note="; ".join(reasons)[:400],
                                   candidates=len(idx.ids))
            return CacheLookup("miss", best_sim, None, best_q,
                               note=f"miss (best sim={best_sim:.3f})", candidates=len(idx.ids))

    # ------------------------------------------------------------------ store
    def store(self, question: str, result: dict, *, scope: str, features: dict,
              snapshot: Any = None, latency_ms: float = 0.0) -> int | None:
        norm = normalize_question(question)
        if not norm:
            return None
        qv = self._embed(question)
        u = result.get("usage") or {}
        keep = {k: result.get(k) for k in ("answer", "route", "escalation", "sources", "drawings",
                                           "agent_steps", "source_route", "usage")}
        fp = self.sources_fingerprint(snapshot, result.get("sources") or []) if snapshot else {}
        with self._lock:
            self._refresh_if_changed()
            self._db.execute("DELETE FROM answers WHERE scope=? AND norm=?", (scope, norm))
            cur = self._db.execute(
                "INSERT INTO answers (scope, question, norm, features, emb, dim, result, route, "
                "sources_fp, index_sig, tokens_in, tokens_out, llm_calls, cost_usd, latency_ms, "
                "created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (scope, question, norm, json.dumps(features, ensure_ascii=False),
                 qv.tobytes() if qv is not None else None, int(qv.shape[0]) if qv is not None else 0,
                 json.dumps(keep, ensure_ascii=False, default=str), str(result.get("route") or ""),
                 json.dumps(fp, ensure_ascii=False), index_signature(snapshot),
                 int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0),
                 int(u.get("llm_calls") or 0), float(u.get("est_cost_usd") or 0.0),
                 float(latency_ms), time.time()))
            self._db.commit()
            self._index.pop(scope, None)
            self._prune()
            return int(cur.lastrowid)

    def add_alias(self, question: str, entry_id: int, *, scope: str, features: dict) -> int | None:
        """A validated paraphrase becomes its own entry pointing to the same answer, so the
        next time it is an EXACT hit (no validator call). Keeps the original's age (TTL),
        usage and source fingerprints."""
        norm = normalize_question(question)
        qv = self._embed(question)
        if not norm:
            return None
        with self._lock:
            if self._db.execute("SELECT 1 FROM answers WHERE scope=? AND norm=?",
                                (scope, norm)).fetchone():
                return None
            cur = self._db.execute(
                "INSERT INTO answers (scope, question, norm, features, emb, dim, result, route, "
                "sources_fp, index_sig, tokens_in, tokens_out, llm_calls, cost_usd, latency_ms, "
                "created) SELECT scope, ?, ?, ?, ?, ?, result, route, sources_fp, index_sig, "
                "tokens_in, tokens_out, llm_calls, cost_usd, latency_ms, created "
                "FROM answers WHERE id=?",
                (question, norm, json.dumps(features, ensure_ascii=False),
                 qv.tobytes() if qv is not None else None,
                 int(qv.shape[0]) if qv is not None else 0, int(entry_id)))
            self._db.commit()
            self._index.pop(scope, None)
            return int(cur.lastrowid) if cur.rowcount else None

    def _prune(self) -> None:
        if self.ttl_days:
            self._db.execute("DELETE FROM answers WHERE created < ?",
                             (time.time() - self.ttl_days * 86400,))
        n = self._db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        if self.max_entries and n > self.max_entries:
            # least recently useful first: never hit and oldest
            self._db.execute("DELETE FROM answers WHERE id IN (SELECT id FROM answers ORDER BY "
                             "MAX(last_hit, created) ASC LIMIT ?)", (n - self.max_entries,))
            self._index.clear()
        self._db.commit()

    def record_hit(self, entry_id: int, tokens_saved: int, cost_saved: float) -> None:
        with self._lock:
            self._db.execute("UPDATE answers SET hits = hits + 1, last_hit = ?, "
                             "tokens_saved = tokens_saved + ?, cost_saved = cost_saved + ? "
                             "WHERE id = ?", (time.time(), int(tokens_saved), float(cost_saved),
                                              int(entry_id)))
            self._db.commit()

    def record_event(self, outcome: str, question: str, *, similarity: float | None = None,
                     entry_id: int | None = None, tokens_used: int = 0, tokens_saved: int = 0,
                     cost_saved: float = 0.0, latency_ms: float = 0.0, note: str = "") -> None:
        with self._lock:
            self._db.execute("INSERT INTO events (ts, outcome, question, similarity, entry_id, "
                             "tokens_used, tokens_saved, cost_saved, latency_ms, note) "
                             "VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (time.time(), outcome, question[:500], similarity, entry_id,
                              int(tokens_used), int(tokens_saved), float(cost_saved),
                              float(latency_ms), note[:500]))
            self._db.commit()

    # ------------------------------------------------------------------ admin
    def stats(self) -> dict:
        with self._lock:
            a = self._db.execute("SELECT COUNT(*) n, COALESCE(SUM(hits),0) hits, "
                                 "COALESCE(SUM(tokens_saved),0) tok, COALESCE(SUM(cost_saved),0) usd "
                                 "FROM answers").fetchone()
            ev = {r["outcome"]: r["n"] for r in self._db.execute(
                "SELECT outcome, COUNT(*) n FROM events GROUP BY outcome")}
            sav = self._db.execute("SELECT COALESCE(SUM(tokens_saved),0) tok, "
                                   "COALESCE(SUM(cost_saved),0) usd, COALESCE(SUM(tokens_used),0) "
                                   "used FROM events WHERE outcome IN ('hit-exact','hit-semantic')"
                                   ).fetchone()
            top = [dict(r) for r in self._db.execute(
                "SELECT id, question, hits, tokens_saved, route FROM answers "
                "ORDER BY hits DESC, created DESC LIMIT 10")]
        lookups = sum(v for k, v in ev.items() if k.startswith(("hit", "miss", "rejected")))
        hits = ev.get("hit-exact", 0) + ev.get("hit-semantic", 0)
        return {"file": self.path, "entries": a["n"], "threshold": self.threshold,
                "validate_semantic_hits": self.validate, "invalidation": self.invalidation,
                "ttl_days": self.ttl_days, "events": ev, "lookups": lookups,
                "hit_rate": round(hits / lookups, 3) if lookups else None,
                "tokens_saved": int(sav["tok"]), "tokens_spent_on_validation": int(sav["used"]),
                "usd_saved_est": round(float(sav["usd"]), 6), "top_entries": top}

    def clear(self) -> int:
        with self._lock:
            n = self._db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
            self._db.execute("DELETE FROM answers")
            self._db.execute("DELETE FROM events")
            self._db.commit()
            self._index.clear()
            return int(n)

    def close(self) -> None:
        with self._lock:
            self._db.close()


# ---------------------------------------------------------------------------------------
def _decode(row: sqlite3.Row) -> dict:
    d = dict(row)
    d.pop("emb", None)
    d["result"] = json.loads(d.get("result") or "{}")
    return d


def index_signature(snapshot: Any) -> str:
    if snapshot is None:
        return ""
    return f"{getattr(snapshot, 'manifest_mtime', 0.0):.3f}:{len(getattr(snapshot, 'chunks', {}) or {})}"


def cacheable(result: dict) -> tuple[bool, str]:
    """(store it?, reason when not). Only complete, reusable answers are stored."""
    ans = str(result.get("answer") or "")
    if result.get("clarify") or result.get("route") == "clarify":
        return False, "clarifying question"
    if not ans.strip() or ans.startswith("ERROR"):
        return False, "empty / error"
    if result.get("route") == "cache":
        return False, "already from cache"
    if result.get("escalation") == "catalog":
        return False, "catalog listing (changes when files are added)"
    if is_no_answer(ans):
        return False, "'not found' answer"
    skip = {s.strip() for s in settings.semantic_cache_skip_sections if s.strip()}
    if skip:
        used = {s.get("section") for s in result.get("sources") or [] if s.get("cited")}
        used = used or {s.get("section") for s in result.get("sources") or []}
        hit = sorted(used & skip)
        if hit:
            return False, f"uses volatile section(s): {', '.join(hit)}"
    return True, ""


def make_llm_validator(llm: Any, meter: Any) -> Callable[[str, str, str], tuple[bool, str]]:
    """Cheap LLM check of a semantic hit (FAIL CLOSED: an error means 'recompute')."""
    def _validate(new_q: str, cached_q: str, cached_a: str) -> tuple[bool, str]:
        try:
            raw = meter.invoke(llm, [SystemMessage(content=CACHE_VALIDATE_SYSTEM),
                                     HumanMessage(content=f"NEW question: {new_q}\n\nCACHED question: "
                                                          f"{cached_q}\n\nCACHED answer: {cached_a[:2000]}")],
                               stage="cache_validate")
            d = parse_json(raw)
            if not isinstance(d, dict) or "valid" not in d:
                return False, "validator reply was not valid JSON"
            return bool(d.get("valid")), str(d.get("reason", "") or "")
        except Exception as e:
            return False, f"validator error: {type(e).__name__}"
    return _validate


def open_semantic_cache(retriever: Any = None, path: str | None = None,
                        **kw: Any) -> SemanticAnswerCache:
    """The answer cache of the default index, sharing the retriever's query-embedding LRU
    (a repeated question is embedded once)."""
    from pathlib import Path
    p = Path(path or settings.semantic_cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    embed_fn = None
    if retriever is not None and getattr(retriever, "embeddings", None) is not None:
        embed_fn = retriever.embeddings.embed_query
    return SemanticAnswerCache(str(p), embed_fn, **kw)
