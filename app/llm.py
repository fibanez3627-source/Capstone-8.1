"""OpenRouter clients (LangChain), key check, preflight and a per-question usage meter."""
from __future__ import annotations

import functools
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from app.config import settings
from app.errors import OpenRouterError, api_error_message, classify_api_error

log = logging.getLogger("rag.llm")
_PLACEHOLDERS = ("sk-or-...", "sk-or-xxx", "your", "tu_clave", "tu-clave", "changeme", "<")


@functools.lru_cache(maxsize=1)
def check_api_key() -> str:
    key = (settings.openrouter_api_key or "").strip().strip('"').strip("'").strip()
    if not key:
        raise OpenRouterError("OPENROUTER_API_KEY_RELANT no está definida (variable de entorno "
                              "o .env).")
    if any(key.lower().startswith(p) for p in _PLACEHOLDERS) or key.endswith("..."):
        raise OpenRouterError("OPENROUTER_API_KEY_RELANT tiene un texto de ejemplo, no una "
                              "clave real. Cópiala desde https://openrouter.ai/keys")
    if any(ch.isspace() for ch in key):
        raise OpenRouterError("OPENROUTER_API_KEY_RELANT contiene espacios o saltos de línea.")
    if not key.startswith("sk-or-"):
        log.warning("la clave no empieza con 'sk-or-'; ¿es realmente de OpenRouter?")
    return key


def _openrouter_get(path: str, key: str, timeout: float = 10) -> dict:
    req = Request(f"{settings.openrouter_base_url}{path}",
                  headers={"Authorization": f"Bearer {key}", "User-Agent": "relant-rag/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def preflight(key: str) -> dict:
    """Ask OpenRouter about the key and the balance BEFORE indexing.
    Raises OpenRouterError on fatal problems; only warns if OpenRouter can't be reached."""
    report: dict = {}
    try:
        info = _openrouter_get("/key", key).get("data", {}) or {}
    except HTTPError as e:
        c = classify_api_error(e)
        if e.code in (401, 403) or (c and c[0] != "rate"):
            raise OpenRouterError(c[1] if c else f"OpenRouter /key -> HTTP {e.code}") from e
        log.warning("preflight: /key respondió HTTP %s; continúo sin verificar.", e.code)
        return report
    except (URLError, TimeoutError, OSError, ValueError) as e:
        log.warning("preflight: no pude contactar OpenRouter (%s); continúo.", e)
        return report
    report.update(label=info.get("label"), usage=info.get("usage"), limit=info.get("limit"),
                  limit_remaining=info.get("limit_remaining"), free_tier=info.get("is_free_tier"))
    lr = report["limit_remaining"]
    if lr is not None and lr <= 0:
        raise OpenRouterError("La clave alcanzó su límite de gasto (OpenRouter → Keys).")
    try:
        cr = _openrouter_get("/credits", key).get("data", {}) or {}
        total, used = float(cr.get("total_credits") or 0), float(cr.get("total_usage") or 0)
        report["balance"] = total - used
        if total - used <= 0:
            raise OpenRouterError(api_error_message(Exception("insufficient credits")))
    except OpenRouterError:
        raise
    except Exception as e:
        log.warning("preflight: no pude leer /credits (%s); continúo.", e)
    return report


@functools.lru_cache(maxsize=8)
def make_llm(model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    """One shared client per model (ChatOpenAI is thread-safe; reuses HTTP connections)."""
    return ChatOpenAI(model=model or settings.llm_model,
                      temperature=settings.temperature if temperature is None else temperature,
                      api_key=SecretStr(check_api_key()), base_url=settings.openrouter_base_url,
                      timeout=settings.llm_timeout, max_retries=settings.max_retries)


def make_embeddings() -> OpenAIEmbeddings:
    # check_embedding_ctx_length=False: OpenRouter does not accept token-id arrays.
    return OpenAIEmbeddings(model=settings.embed_model, api_key=SecretStr(check_api_key()),
                            base_url=settings.openrouter_base_url, timeout=settings.embed_timeout,
                            max_retries=settings.max_retries, check_embedding_ctx_length=False)


@dataclass
class UsageMeter:
    """Counts LLM calls / tokens / time for ONE request (never shared between requests)."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_seconds: float = 0.0
    by_stage: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def invoke(self, llm: Any, messages: list[BaseMessage], stage: str = "llm") -> str:
        t0 = time.perf_counter()
        resp = llm.invoke(messages)
        dt = time.perf_counter() - t0
        usage = getattr(resp, "usage_metadata", None) or {}
        with self._lock:
            self.calls += 1
            self.llm_seconds += dt
            self.input_tokens += int(usage.get("input_tokens", 0) or 0)
            self.output_tokens += int(usage.get("output_tokens", 0) or 0)
            self.by_stage[stage] = self.by_stage.get(stage, 0) + 1
        return str(resp.content)

    def as_dict(self) -> dict:
        cost = (self.input_tokens * settings.price_in_per_mtok
                + self.output_tokens * settings.price_out_per_mtok) / 1e6
        return {"llm_calls": self.calls, "by_stage": dict(self.by_stage),
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "llm_seconds": round(self.llm_seconds, 2), "est_cost_usd": round(cost, 6)}
