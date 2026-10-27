"""Production— conversation context: follow-up questions become standalone questions.

Why: router.py, the retriever and the answer cache see ONE question at a time. Before 1.0
a follow-up such as "¿y cuál es su caudal?" was searched literally: the retriever found
nothing about "su", the router escalated (low_relevance) and the agent asked a clarifying
question although the product was named one turn earlier.

What: before routing, a follow-up is rewritten from the last CONVERSATION_HISTORY_TURNS
question/answer pairs into a question that stands on its own:

    history:  "¿Cuál es la presión máxima de la VDV-25?"  ->  "... 250 bar ..."
    new:      "¿y su caudal?"
    search:   "¿Cuál es el caudal de la dosificadora VDV-25?"

* CONVERSATION_REWRITE=auto (default): only messages that LOOK like follow-ups (pronouns,
  "y ...", "what about ...", very short) cost the extra call; a standalone question is
  passed through unchanged (no call).
* One small call to the agent model. If it fails, a deterministic fallback carries the part
  codes of the previous question over ("¿y su caudal?" -> "¿y su caudal? (VDV-25)").
* The rewritten question is what the router searches (the `chat` command shows it).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.textutils import fold, raw_codes

log = logging.getLogger("rag.conversation")

CLARIFY_JOIN = "\nAclaración del usuario: "       # clarification reply appended to the question

# folded text (no accents). "esta" (= está) is left out on purpose: too common as a verb.
_START_RE = re.compile(
    r"^\s*(y|e|o|pero|entonces|tambien|ademas|ok|vale|and|also|but|so|what about|how about|"
    r"und|auch|aber|was ist mit)\b")
_ANAPHORA_RE = re.compile(
    r"\b(su|sus|suyo|suya|este|estos|estas|ese|esa|esos|esas|eso|esto|aquel|aquella|"
    r"dicho|dicha|dichos|mismo|misma|mismos|anterior|anteriores|previo|previa|el otro|"
    r"la otra|los otros|las otras|lo mismo|"
    r"it|its|this|that|these|those|they|them|their|same|previous|above|former|latter|"
    r"other one|ihr|ihre|sein|seine|dieser|diese|dieses|derselbe|dasselbe|vorherige)\b")
_CONTENT_RE = re.compile(r"[a-z0-9]{3,}")

REWRITE_SYSTEM = (
    "You rewrite the LAST user message of a conversation with a technical assistant (spare "
    "parts, pumps, valves, dispensing equipment) into ONE standalone question that can be "
    "understood without the conversation.\n"
    "Rules:\n"
    "- Replace pronouns and references (su, este, the other one, it, ...) with the product, "
    "model, part code, document or topic they refer to in the conversation.\n"
    "- Keep every part code, model name and number EXACTLY as written.\n"
    "- Keep the language of the last user message.\n"
    "- If the message is already standalone, return it unchanged.\n"
    "- Do NOT answer the question and do not add information that is not in the conversation.\n"
    "Output ONLY the rewritten question, on one line."
)


def followup_reason(question: str, history: list[dict]) -> str | None:
    """Why the message looks like a follow-up (None = standalone). Free, no LLM."""
    if not history:
        return None
    if CLARIFY_JOIN in question:
        return "clarification"
    f = fold(question)
    if _START_RE.search(f):
        return "starts like a continuation"
    if _ANAPHORA_RE.search(f):
        return "reference to earlier turn"
    if len(_CONTENT_RE.findall(f)) <= 3 and not raw_codes(question):
        return "very short"
    return None


def _history_text(history: list[dict]) -> str:
    lines = []
    for h in history[-2 * max(1, settings.conversation_history_turns):]:
        text = " ".join(str(h.get("content") or "").split())
        if h.get("role") == "assistant":
            lines.append(f"Assistant: {text[:settings.conversation_answer_chars]}")
        else:
            lines.append(f"User: {text[:600]}")
    return "\n".join(lines)


def _carry_codes(question: str, history: list[dict]) -> str:
    """Deterministic fallback: the previous question's part codes, if the new one has none."""
    if raw_codes(question):
        return question
    for h in reversed(history):
        if h.get("role") == "user":
            codes = list(raw_codes(str(h.get("content") or "")).values())
            if codes:
                return f"{question} ({', '.join(codes[:3])})"
            break
    return question


def contextualize(question: str, history: list[dict], llm: Any = None,
                  meter: Any = None) -> tuple[str, str]:
    """(standalone question, method). method: "" (unchanged), "llm", "carry-codes", "off"."""
    mode = (settings.conversation_rewrite or "auto").strip().lower()
    if mode == "off" or not history:
        return question, ""
    reason = followup_reason(question, history)
    if mode == "auto" and reason is None:
        return question, ""
    if llm is None:
        out = _carry_codes(question, history)
        return out, "carry-codes" if out != question else ""
    user = (f"Conversation:\n{_history_text(history)}\n\nLast user message: {question}\n\n"
            "Standalone question:")
    msgs = [SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=user)]
    try:
        raw = meter.invoke(llm, msgs, stage="rewrite") if meter is not None \
            else str(llm.invoke(msgs).content)
    except Exception as e:
        log.warning("follow-up rewrite failed (%s) -> carrying part codes", str(e)[:120])
        out = _carry_codes(question, history)
        return out, "carry-codes" if out != question else ""
    out = " ".join(str(raw or "").strip().strip('"').strip("'").split())
    out = re.sub(r"^(standalone question|pregunta)\s*:\s*", "", out, flags=re.I)
    if not out or len(out) > 600:
        out = _carry_codes(question, history)
        return out, "carry-codes" if out != question else ""
    # never lose a code the user typed
    missing = [c for n, c in raw_codes(question).items() if n not in {k for k in raw_codes(out)}]
    if missing:
        out = f"{out} ({', '.join(missing)})"
    return out, "llm" if out != question else ""
