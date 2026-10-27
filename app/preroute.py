"""Production 1.0 — free heuristic pre-route (plain Python, no LLM call).

Relant keeps its signal routing (fast path first; escalate on weak retrieval signals or
INSUFFICIENT_CONTEXT). This module only catches the questions that are OBVIOUSLY multi-step,
so they go straight to the agent instead of wasting one fast-path LLM call first:

* several questions in one message          "¿Qué presión...? ¿Y qué caudal...?"
* comparisons                               "compara", "diferencia entre", "vs", "better than"
* enumerated multi-part requests            lines starting with 1. 2. / a) b) / - -

Questions with part codes are NOT pre-routed: router.py already looks every code up in the
CSV tables (exact, no tokens) and answers several codes in ONE fast call. ProductionRouter
also skips the pre-route for catalog questions and "page N of document X",
which router.py handles with its own dedicated paths.

It never sends a question to the fast path: when no rule fires it returns None and the
normal router decides. Returns a short reason (shown as the escalation).
"""
from __future__ import annotations

import re

from app.config import settings
from app.textutils import fold

# accent-free, matched on fold()ed text with word boundaries
_COMPARE_RE = re.compile(
    r"\b(compar\w*|diferenci\w*|difference\w*|differ|distinto|distinta|"
    r"versus|vs|frente a|en comparacion|mejor que|peor que|better than|worse than|"
    r"cual es mejor|which is better|ventajas y desventajas|pros and cons|"
    r"contrast\w*|equivalen\w*|alternativ\w*)\b")
_ENUM_RE = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|[a-e][.)]|[-*•])\s+\S")


def heuristic_preroute(question: str) -> str | None:
    """'agent' reason or None (let the normal fast-path / signal routing decide)."""
    if not settings.preroute_enabled:
        return None
    q = str(question or "")
    # a clarification answer is appended to the original question: judge the original
    from app.conversation_context import CLARIFY_JOIN
    original = q.split(CLARIFY_JOIN)[0]
    n_q = max(original.count("?"), original.count("¿"))
    if n_q >= 2:
        return "preroute: several questions"
    f = fold(original)
    m = _COMPARE_RE.search(f)
    if m:
        return f"preroute: comparison ({m.group(1)})"
    if len(_ENUM_RE.findall(original)) >= 2:
        return "preroute: multi-part list"
    return None
