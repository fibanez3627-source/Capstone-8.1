"""Production 1.0 — Reciprocal Rank Fusion + reranking + code pinning.

Shared by HybridEvaluationRetriever (one knowledge base) and MultiRetriever (several).

Why RRF instead of the 5.7 min-max fusion:
* min-max rescales every list to [0, 1], so ONE weak candidate becomes 1.0 and one
  outlier squashes all the others; the scores of different bases are not comparable,
  which is why MultiRetriever needed a "damping" heuristic.
* RRF uses only positions:  score(c) = sum over lists of  w / (k + rank_in_list(c)).
  No normalisation, no damping, robust to score scales (Cormack et al., 2009; k = 60).

Pipeline after the candidate lists are ready:
    rrf_fuse()  ->  top RERANKER_CANDIDATES  ->  cross-encoder  ->  code pinning
                ->  top_k  ->  expand_table_hits()  (whole tables, as before)
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.config import settings
from app.pdf_tables import expand_table_hits
from app.textutils import extract_codes

RankList = Sequence[str]          # chunk ids, best first


def rrf_fuse(lists: Sequence[RankList], weights: Sequence[float] | None = None,
             k: int | None = None) -> list[tuple[str, float]]:
    """[(chunk_id, score)] best first. Score normalised to [0, 1]: 1.0 = ranked first in
    every list. A chunk absent from a list simply gets nothing from it."""
    k = settings.rrf_k if k is None else k
    weights = list(weights) if weights is not None else [1.0] * len(lists)
    acc: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    n = 0
    for lst, w in zip(lists, weights):
        if not w:
            continue
        for rank, cid in enumerate(lst, start=1):
            acc[cid] = acc.get(cid, 0.0) + w / (k + rank)
            if cid not in first_seen:
                first_seen[cid] = n
                n += 1
    # 1.0 = first in EVERY list (empty lists included: a chunk only one method found, or
    # BM25-only mode after a vector failure, tops out at that method's share)
    best_possible = sum(w for w in weights if w) / (k + 1) or 1.0
    # ties -> the order in which the candidate was first seen (stable, deterministic)
    ranked = sorted(acc.items(), key=lambda x: (-x[1], first_seen[x[0]]))
    return [(cid, s / best_possible) for cid, s in ranked]


def _codes_of(chunk: Mapping[str, Any]) -> set[str]:
    return extract_codes(str(chunk.get("text") or "")[:4000])


def rerank_and_select(query: str, fused: list[tuple[str, float]],
                      chunks: Mapping[str, dict], top_k: int,
                      max_extra: int, focus_query: str | None = None) -> tuple[list[dict], float]:
    """fused candidates -> final results (dicts) + best rerank score (0 if not reranked).

    * The first RERANKER_CANDIDATES fused candidates are re-scored by the cross-encoder.
    * Code pinning: a candidate whose text contains a part code of the question is kept in
      the top_k even if the cross-encoder ranks it low (cross-encoders are weak on codes).
    * expand_table_hits() then adds the other pieces of matched tables, as in 5.7, and
      focus_doc_tables() the tables of the manual of the model named in the question
      (live 14-Oct behaviour; it gets the glossary-expanded query)."""
    from app.reranker import get_reranker

    cand = [(cid, s) for cid, s in fused if cid in chunks]
    rr = get_reranker()
    best_rr = 0.0
    if rr.enabled and len(cand) > 1:
        pool = cand[:max(top_k, settings.reranker_candidates)]
        scores = rr.score(query, [chunks[cid].get("text", "") for cid, _ in pool])
        if scores is not None:
            order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
            results = [dict(chunks[pool[i][0]], score=round(scores[i], 3),
                            rerank_score=round(scores[i], 4),
                            fused_score=round(pool[i][1], 4)) for i in order]
            best_rr = max(scores)
        else:
            results = [dict(chunks[cid], score=round(s, 3), fused_score=round(s, 4))
                       for cid, s in cand]
    else:
        results = [dict(chunks[cid], score=round(s, 3), fused_score=round(s, 4))
                   for cid, s in cand]

    final = results[:top_k]
    q_codes = extract_codes(query) if settings.code_pinning else set()
    if q_codes and final:
        # up to half of the top_k slots are guaranteed to chunks that contain a code of the
        # question; they replace the lowest-ranked chunks without a code
        quota = max(1, top_k // 2)
        has = [r for r in final if q_codes & _codes_of(r)]
        missing = [r for r in results[top_k:] if q_codes & _codes_of(r)][:max(0, quota - len(has))]
        if missing:
            keep_ids = {r["id"] for r in has}
            droppable = [r for r in reversed(final) if r["id"] not in keep_ids]
            drop = {r["id"] for r in droppable[:len(missing)]}
            final = [r for r in final if r["id"] not in drop] + \
                [dict(r, pinned=True) for r in missing[:len(drop)]]
    out = expand_table_hits(final, chunks, max_extra=max_extra)
    try:
        from app.pdf_tables import focus_doc_tables
    except ImportError:                                   # older pdf_tables.py
        return out, best_rr
    return focus_doc_tables(focus_query or query, out, chunks), best_rr
