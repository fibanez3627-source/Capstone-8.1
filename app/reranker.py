"""Production 1.0 — cross-encoder reranker (multilingual: Spanish, English, German).

The hybrid search returns up to RERANKER_CANDIDATES (40) fused candidates; a cross-encoder
reads (question, chunk) pairs together and re-scores them; the retriever keeps the top_k
(8-10). A cross-encoder is far more precise than the bi-encoder vectors, but too slow to
run over the whole collection — hence only over the fused short list.

Backends (RERANKER_BACKEND in the .env):
    local   sentence-transformers CrossEncoder on this server. Default model
            cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual, ~120 MB, fast on
            CPU). Higher quality: BAAI/bge-reranker-v2-m3 (~2.2 GB, use a GPU).
            The model is downloaded once from Hugging Face:
                python -m app.cli reranker-check
    cohere  Cohere Rerank API (COHERE_API_KEY, model rerank-v3.5, multilingual).
    off     no reranking: the RRF order is used.

FAIL-SAFE: if the model cannot be loaded or a call fails, the reranker reports it once
in the log and the retriever keeps the RRF order. A search never fails because of it.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

from app.config import settings

log = logging.getLogger("rag.reranker")


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class Reranker:
    def __init__(self, backend: str | None = None, model: str | None = None) -> None:
        self.backend = (backend or settings.reranker_backend or "off").strip().lower()
        if self.backend not in ("local", "cohere", "off"):
            log.warning("RERANKER_BACKEND=%s unknown -> off", self.backend)
            self.backend = "off"
        if self.backend == "cohere" and not settings.cohere_api_key:
            log.warning("RERANKER_BACKEND=cohere but COHERE_API_KEY is empty -> off")
            self.backend = "off"
        self.model_name = model or (settings.cohere_rerank_model if self.backend == "cohere"
                                    else settings.reranker_model)
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()   # one forward pass at a time (CPU-bound)
        self._failed = ""                       # reason the backend is disabled
        self.stats = {"calls": 0, "pairs": 0, "errors": 0, "ms": 0.0}

    # ------------------------------------------------------------------ state
    @property
    def enabled(self) -> bool:
        return self.backend != "off" and not self._failed

    def status(self) -> dict:
        calls = self.stats["calls"] or 1
        return {"backend": self.backend, "model": self.model_name, "enabled": self.enabled,
                "loaded": self._model is not None or self.backend == "cohere",
                "disabled_reason": self._failed, "calls": self.stats["calls"],
                "pairs": self.stats["pairs"], "errors": self.stats["errors"],
                "avg_ms": round(self.stats["ms"] / calls, 1)}

    def _disable(self, reason: str) -> None:
        if not self._failed:
            self._failed = reason
            log.error("reranker disabled: %s (searches keep the RRF order)", reason)

    # ------------------------------------------------------------------ local model
    def _load(self) -> Any:
        if self._model is not None or self.backend != "local":
            return self._model
        with self._load_lock:
            if self._model is None and not self._failed:
                t0 = time.time()
                try:
                    from sentence_transformers import CrossEncoder   # type: ignore
                    kw: dict[str, Any] = {"max_length": 512}
                    if settings.reranker_device:
                        kw["device"] = settings.reranker_device
                    self._model = CrossEncoder(self.model_name, **kw)
                    log.info("reranker loaded: %s in %.1f s", self.model_name, time.time() - t0)
                except Exception as e:                # missing package, no network, OOM...
                    self._disable(f"cannot load {self.model_name}: {type(e).__name__}: {e}")
        return self._model

    def warmup(self) -> None:
        """Load the model (and run one tiny pair) so the first question is not slow."""
        if self.backend == "local" and self.enabled:
            self.score("warmup", ["warmup"])

    # ------------------------------------------------------------------ scoring
    def score(self, query: str, texts: list[str]) -> list[float] | None:
        """Relevance in [0, 1] for each text, or None (backend off / failed)."""
        if not self.enabled or not texts or not (query or "").strip():
            return None
        texts = [str(t or "")[:settings.reranker_max_chars] for t in texts]
        t0 = time.perf_counter()
        try:
            out = self._score_cohere(query, texts) if self.backend == "cohere" \
                else self._score_local(query, texts)
        except Exception as e:
            self.stats["errors"] += 1
            if self.stats["errors"] >= 3:
                self._disable(f"{type(e).__name__}: {e}")
            else:
                log.warning("reranker call failed (%s); RRF order used", e)
            return None
        if out is None:
            return None
        self.stats["calls"] += 1
        self.stats["pairs"] += len(texts)
        self.stats["ms"] += (time.perf_counter() - t0) * 1000
        return out

    def _score_local(self, query: str, texts: list[str]) -> list[float] | None:
        model = self._load()
        if model is None:
            return None
        with self._predict_lock:
            raw = model.predict([(query, t) for t in texts], batch_size=settings.reranker_batch,
                                show_progress_bar=False)
        vals = [float(x) for x in (raw.tolist() if hasattr(raw, "tolist") else raw)]
        # some models return logits, others probabilities -> always map to [0, 1]
        if any(v < 0.0 or v > 1.0 for v in vals):
            vals = [_sigmoid(v) for v in vals]
        return vals

    def _score_cohere(self, query: str, texts: list[str]) -> list[float]:
        body = json.dumps({"model": self.model_name, "query": query, "documents": texts,
                           "top_n": len(texts)}).encode("utf-8")
        req = Request("https://api.cohere.com/v2/rerank", data=body, method="POST",
                      headers={"Authorization": f"Bearer {settings.cohere_api_key}",
                               "Content-Type": "application/json"})
        with urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
        out = [0.0] * len(texts)
        for item in data.get("results", []):
            out[int(item["index"])] = float(item["relevance_score"])
        return out


_instance: Reranker | None = None
_instance_lock = threading.Lock()


def get_reranker() -> Reranker:
    """One reranker per process (shared by every knowledge base and every request)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = Reranker()
    return _instance


def set_reranker(r: Reranker | None) -> None:
    """Tests: inject a fake reranker (or None to rebuild from the settings)."""
    global _instance
    _instance = r
