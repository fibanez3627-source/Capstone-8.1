"""Trace log per question, same format as Capstone 7.1 (capstone_7_1_agent.log):

    [2026-10-23T15:30:12] ASK | cache-hit (semantic)
    Q: ¿Cuál es la presión de trabajo de la VDV-25?
    STANDALONE: (same)
    CACHE: sim=0.941 vs 'presión máxima de la VDV-25' (validated)
    ROUTE: cache  ESCALATION: -  SOURCE_ROUTE: datasheet
    ACTIONS: [...]
    STAGES: {'cache_validate': 1}
    TOKENS: 412  (in 398 / out 14)  CALLS: 1  COST: 0.000127  LATENCY_MS: 820
    SAVED: ~5 210 tokens / $0.0021 vs the original answer
    SOURCES: hojas_tecnicas/...pdf, manuales/...pdf
    A: ...
    ------------------------------------------------------------------------

File: RAG_TRACE_LOG (default <RAG_LOG_DIR or ./logs>/relant_rag_agent.log). TRACE_LOG=false
turns it off. Every write is appended and flushed; a failure to write never breaks a reply.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

from app.config import settings

log = logging.getLogger("rag.trace")
_lock = threading.Lock()
SEP = "-" * 72


def trace(label: str, text: str, path: str | None = None) -> None:
    """Capstone 7.1 `log(label, text)`."""
    if not settings.trace_log and path is None:
        return
    p = Path(path or settings.trace_log_path)
    ts = datetime.now().isoformat(timespec="seconds")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with _lock, p.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {label}\n{text}\n{SEP}\n")
    except OSError as e:                                 # never break an answer for a log
        log.warning("trace log not written (%s): %s", p, e)


def path_label(d: dict) -> str:
    """'cache-hit (semantic)', 'fast', 'agent (insufficient_context)', ..."""
    c = d.get("cache") or {}
    if d.get("route") == "cache":
        return f"cache-hit ({c.get('kind', '?')})"
    route = str(d.get("route") or "?")
    return f"{route} ({d['escalation']})" if d.get("escalation") else route


def _actions(transcript: list[str] | None) -> list[str]:
    """Agent decisions and tool calls only (not the retrieved text)."""
    out = []
    for line in transcript or []:
        if line.startswith("[decide]"):
            out.append(line[:160])
        elif line.startswith("["):
            out.append(line.split("] ->", 1)[0][:160] + "]")
        if len(out) >= 24:
            break
    return out


def format_answer_trace(d: dict, question: str, latency_ms: float,
                        transcript: list[str] | None = None) -> str:
    u = d.get("usage") or {}
    tin, tout = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    c = d.get("cache") or {}
    cited = [s.get("source", "") for s in d.get("sources") or [] if s.get("cited")]
    srcs = cited or [s.get("source", "") for s in (d.get("sources") or [])[:6]]
    standalone = d.get("standalone_query")
    lines = [
        f"Q: {question}",
        f"STANDALONE: {standalone if standalone and standalone != question else '(same)'}",
        f"CACHE: {c.get('note') or '(off)'}",
        f"ROUTE: {d.get('route', '')}  ESCALATION: {d.get('escalation') or '-'}  "
        f"SOURCE_ROUTE: {d.get('source_route') or '-'}  AGENT_STEPS: {d.get('agent_steps', 0)}",
        f"ACTIONS: {_actions(transcript)}",
        f"STAGES: {u.get('by_stage') or {}}",
        f"TOKENS: {tin + tout}  (in {tin} / out {tout})  CALLS: {u.get('llm_calls', 0)}  "
        f"COST: {float(u.get('est_cost_usd') or 0):.6f}  LATENCY_MS: {latency_ms:.0f}",
    ]
    if d.get("route") == "cache":
        lines.append(f"SAVED: ~{c.get('tokens_saved', 0)} tokens / ${c.get('cost_saved', 0):.6f} "
                     f"vs the original answer (route {c.get('original_route', '?')})")
    elif c.get("stored") is not None:
        lines.append(f"STORED: {'yes (entry %s)' % c['stored'] if c['stored'] else 'no — ' + c.get('not_stored', '')}")
    lines.append(f"SOURCES: {', '.join(dict.fromkeys(srcs)) or '-'}")
    lines.append(f"A: {d.get('answer', '')}")
    return "\n".join(lines)
