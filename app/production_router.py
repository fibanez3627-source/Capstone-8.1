"""ProductionRouter: QueryRouter + conversation context + semantic answer cache + free
heuristic pre-route + trace log.

    ProductionRouter.answer(question, history)
      0. follow-up -> standalone question from the conversation (app/conversation_context.py)
      1. SEMANTIC ANSWER CACHE (app/semantic_cache.py, SQLite): exact or validated semantic
         hit -> the stored answer, no router / search / answer LLM call
      2. heuristic pre-route (app/preroute.py): obviously multi-step -> force the agent
      3. QueryRouter.answer()  (fast path -> signals / INSUFFICIENT_CONTEXT -> agent)
      4. the answer is stored in the cache when it is reusable (semantic_cache.cacheable)
      5. one trace entry per question in the log (app/qa_log.py, Capstone 7.1 format)

The result keeps the QueryRouter .to_dict() contract; it adds "cache" (what the cache did),
"standalone_query" and "rewrite" when a follow-up was rewritten, and "preroute" when the
pre-route fired. A cache hit has route == "cache".
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.config import settings
from app.conversation_context import contextualize
from app.llm import UsageMeter
from app.preroute import heuristic_preroute
from app.qa_log import format_answer_trace, path_label, trace
from app.router import QueryRouter

log = logging.getLogger("rag.router.production")


class RouterResult:
    """Dict-backed result with the same .to_dict() as the QueryRouter result."""

    def __init__(self, data: dict, inner: Any = None) -> None:
        self._data, self._inner = data, inner

    def to_dict(self) -> dict:
        return dict(self._data)

    def __getattr__(self, name: str) -> Any:
        data = self.__dict__.get("_data") or {}
        if name in data:
            return data[name]
        inner = self.__dict__.get("_inner")
        if inner is not None:
            return getattr(inner, name)
        raise AttributeError(name)


class ProductionRouter(QueryRouter):
    def __init__(self, retriever: Any, llm: Any, agent_llm: Any | None = None,
                 cache: Any = None, validator_llm: Any = None) -> None:
        """cache: a SemanticAnswerCache (None = no answer cache).
        validator_llm: cheap model that confirms semantic hits (default: llm)."""
        super().__init__(retriever, llm, agent_llm)
        self.cache = cache
        self.validator_llm = validator_llm or llm

    def answer(self, query: str, force_agent: bool = False, debug: bool = False,
               retriever: Any = None, instructions: str | None = None,
               history: list[dict] | None = None, use_cache: bool = True,
               trace_label: str = "ASK", **kwargs: Any) -> RouterResult:
        """history: previous turns of this console session, oldest first, as
        [{"role": "user" | "assistant", "content": "..."}].
        use_cache=False skips the answer cache for this question (lookup AND store)."""
        t0 = time.perf_counter()
        if instructions is None:
            instructions = settings.instructions

        # 0) conversation context: a follow-up becomes a standalone question
        typed, rewrite_meter = query, UsageMeter()
        query, method = contextualize(query, history or [], getattr(self.agent, "llm", None),
                                      rewrite_meter)

        # 1) semantic answer cache (only on the default index)
        cache_on = self.cache is not None and use_cache and retriever is None
        cache_info: dict = {"kind": "off", "note": "cache off"}
        cache_meter = UsageMeter()
        if cache_on and force_agent:
            cache_info = {"kind": "bypass", "note": "forced agent -> lookup skipped (answer still stored)"}
        elif cache_on:
            hit = self._cache_lookup(query, instructions, cache_meter, t0, debug)
            if isinstance(hit, dict) and hit.get("route") == "cache":
                self._add_rewrite(hit, typed, query, method, rewrite_meter)
                self._trace(trace_label, hit, typed, t0, hit.get("transcript"))
                if not debug:
                    hit["transcript"] = None
                return RouterResult(hit)
            cache_info = hit

        # 2) free heuristic pre-route (never takes a question away from router.py's
        #    dedicated paths: catalog listing, a named document / page, part codes)
        reason = None if force_agent or self._has_own_path(query) else heuristic_preroute(query)
        call = dict(force_agent=force_agent or bool(reason), debug=True,   # transcript -> log
                    instructions=instructions, **kwargs)
        if retriever is not None:
            call["retriever"] = retriever

        # 3) the router (fast path -> signals -> agent)
        res = super().answer(query, **call)
        d = res.to_dict() if hasattr(res, "to_dict") else dict(res)
        transcript = d.get("transcript")
        if not debug:
            d["transcript"] = None
        if reason:
            d["preroute"] = reason
            d["escalation"] = reason[:64]
        self._add_rewrite(d, typed, query, method, rewrite_meter)
        self._merge_usage(d, cache_meter)            # validator calls of a rejected hit
        d["cache"] = cache_info

        # 4) store the reusable answers
        if cache_on:
            self._cache_store(query, instructions, d, t0)
        tm = dict(d.get("timings_ms") or {})
        tm["total"] = round((time.perf_counter() - t0) * 1000, 1)
        d["timings_ms"] = tm
        self._trace(trace_label, d, typed, t0, transcript)
        return RouterResult(d, res)

    # ------------------------------------------------------------------ cache
    def _cache_lookup(self, query: str, instructions: str, meter: UsageMeter, t0: float,
                      debug: bool) -> dict:
        """A full result dict (route == 'cache') on a hit, else the cache info dict."""
        from app.semantic_cache import make_llm_validator, question_features, scope_key
        self.retriever.maybe_reload()
        t_l = time.perf_counter()
        try:
            lk = self.cache.lookup(query, scope=scope_key(instructions),
                                   features=question_features(query, self.source_intent),
                                   snapshot=self.retriever.snapshot,
                                   validator=make_llm_validator(self.validator_llm, meter))
        except Exception as e:                      # the cache must never break an answer
            from app.errors import is_fatal_api_error
            if is_fatal_api_error(e):
                raise
            log.warning("semantic cache lookup failed: %s", e)
            return {"kind": "error", "note": f"lookup error: {type(e).__name__}"}
        lookup_ms = (time.perf_counter() - t_l) * 1000
        used = meter.input_tokens + meter.output_tokens
        if not lk.hit:
            self.cache.record_event(lk.kind, query, similarity=lk.similarity or None,
                                    tokens_used=used, latency_ms=lookup_ms, note=lk.note)
            return {"kind": lk.kind, "note": lk.note, "similarity": round(lk.similarity, 4),
                    "nearest_question": lk.best_question, "lookup_ms": round(lookup_ms, 1)}

        e, stored = lk.entry, lk.entry["result"]
        orig_tokens = int(e["tokens_in"]) + int(e["tokens_out"])
        usage = meter.as_dict()
        saved_tok = max(0, orig_tokens - used)
        saved_usd = max(0.0, float(e["cost_usd"]) - float(usage["est_cost_usd"]))
        total_ms = (time.perf_counter() - t0) * 1000
        info = {"kind": lk.kind, "note": lk.note, "similarity": round(lk.similarity, 4),
                "matched_question": lk.best_question, "entry_id": int(e["id"]),
                "validated": lk.validated, "original_route": e["route"],
                "original_llm_calls": int(e["llm_calls"]), "original_tokens": orig_tokens,
                "tokens_saved": saved_tok, "cost_saved": round(saved_usd, 6),
                "original_latency_ms": round(float(e["latency_ms"]), 1),
                "lookup_ms": round(lookup_ms, 1), "hits_before": int(e["hits"])}
        self.cache.record_hit(int(e["id"]), saved_tok, saved_usd)
        if lk.kind == "semantic":                   # next time this wording is an exact hit
            try:
                info["alias_entry"] = self.cache.add_alias(
                    query, int(e["id"]), scope=scope_key(instructions),
                    features=question_features(query, self.source_intent))
            except Exception as ex:
                log.warning("semantic cache alias failed: %s", ex)
        self.cache.record_event(f"hit-{lk.kind}", query, similarity=lk.similarity,
                                entry_id=int(e["id"]), tokens_used=used, tokens_saved=saved_tok,
                                cost_saved=saved_usd, latency_ms=total_ms, note=lk.note)
        with self._stats_lock:
            self._stats["queries"] += 1
            self._stats["route:cache"] += 1
            self._stats["llm_calls"] += usage["llm_calls"]
        log.info("route=cache(%s) sim=%.3f saved=%d tokens q=%r", lk.kind, lk.similarity,
                 saved_tok, query[:80])
        return {"answer": stored.get("answer", ""), "route": "cache", "escalation": "",
                "clarify": False, "sources": stored.get("sources") or [],
                "drawings": stored.get("drawings") or [], "agent_steps": 0, "usage": usage,
                "timings_ms": {"retrieval": 0.0, "total": round(total_ms, 1),
                               "llm": round(meter.llm_seconds * 1000, 1),
                               "cache_lookup": round(lookup_ms, 1)},
                "baseline": None, "source_route": stored.get("source_route", ""),
                "transcript": [f"[cache] {lk.kind} hit: {lk.note}"],
                "cache": info}

    def _cache_store(self, query: str, instructions: str, d: dict, t0: float) -> None:
        from app.semantic_cache import cacheable, question_features, scope_key
        info = d.setdefault("cache", {})
        ok, why = cacheable(d)
        if not ok:
            info.update(stored=0, not_stored=why)
            try:
                self.cache.record_event("not-stored", query, note=why)
            except Exception:
                pass
            return
        try:
            eid = self.cache.store(query, d, scope=scope_key(instructions),
                                   features=question_features(query, self.source_intent),
                                   snapshot=self.retriever.snapshot,
                                   latency_ms=(time.perf_counter() - t0) * 1000)
            info["stored"] = eid or 0
            self.cache.record_event("stored", query, entry_id=eid)
        except Exception as e:
            log.warning("semantic cache store failed: %s", e)
            info.update(stored=0, not_stored=f"store error: {type(e).__name__}")

    # ------------------------------------------------------------------ helpers
    def _trace(self, label: str, d: dict, typed: str, t0: float,
               transcript: list[str] | None) -> None:
        if not settings.trace_log or not label:
            return
        ms = (time.perf_counter() - t0) * 1000
        trace(f"{label} | {path_label(d)}", format_answer_trace(d, typed, ms, transcript))

    def _has_own_path(self, q: str) -> bool:
        """router.py answers these better than the agent: catalog listing, a named
        document / page, part codes (exact table lookup)."""
        from app.conversation_context import CLARIFY_JOIN
        q = q.split(CLARIFY_JOIN)[0]
        if self.is_catalog_question(q) or self.detect_codes(q):
            return True
        return bool(self.detect_doc_page(q)[0])

    @staticmethod
    def _merge_usage(d: dict, meter: UsageMeter) -> None:
        """Add the calls of another meter (rewrite, cache validation) to the answer's usage."""
        if not meter.calls:
            return
        u = dict(d.get("usage") or {})
        extra = meter.as_dict()
        for k in ("llm_calls", "input_tokens", "output_tokens"):
            u[k] = int(u.get(k) or 0) + int(extra.get(k) or 0)
        for k in ("llm_seconds", "est_cost_usd"):
            u[k] = round(float(u.get(k) or 0) + float(extra.get(k) or 0), 6)
        stages = dict(u.get("by_stage") or {})
        for k, v in (extra.get("by_stage") or {}).items():
            stages[k] = stages.get(k, 0) + v
        u["by_stage"] = stages
        d["usage"] = u

    @classmethod
    def _add_rewrite(cls, d: dict, typed: str, standalone: str, method: str,
                     meter: UsageMeter) -> None:
        """Report the rewrite and add its tokens to the answer's usage."""
        if not method:
            return
        d["standalone_query"] = standalone
        d["rewrite"] = {"method": method, "typed": typed}
        cls._merge_usage(d, meter)

    def stats(self) -> dict:
        base = dict(super().stats() or {})
        from app.reranker import get_reranker
        base["reranker"] = get_reranker().status()
        if self.cache is not None:
            base["semantic_cache"] = self.cache.stats()
        return base
