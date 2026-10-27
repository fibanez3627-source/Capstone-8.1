"""API error classification (from 5.7, unchanged in behavior)."""
from __future__ import annotations


class OpenRouterError(RuntimeError):
    """Key/account problem that no retry will fix (no credits, invalid key...)."""


class IndexingAborted(RuntimeError):
    """Indexing stopped; finished files are saved and resume on the next run."""


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return None


def classify_api_error(exc: BaseException) -> tuple[str, str] | None:
    """-> (kind, human message) for known OpenRouter/OpenAI errors, else None.
    kind: credits | auth | forbidden | rate."""
    m, code = str(exc), _status_code(exc)
    low = m.lower()
    if code == 402 or "error code: 402" in low or "insufficient credits" in low:
        return ("credits",
                "OpenRouter respondió 402: SIN CRÉDITOS. Verifica que la clave sea del "
                "workspace con créditos o compra créditos en "
                "https://openrouter.ai/settings/credits")
    if code == 401 or "error code: 401" in low or "unauthorized" in low or "user not found" in low:
        return ("auth", "OpenRouter respondió 401: CLAVE INVÁLIDA o revocada "
                        "(OPENROUTER_API_KEY_RELANT en el .env).")
    if code == 403 or "error code: 403" in low:
        return ("forbidden", "OpenRouter respondió 403: la clave no tiene permiso para este "
                             "modelo o superó su límite de gasto (OpenRouter → Keys → limit).")
    if code == 429 or "error code: 429" in low or "rate limit" in low:
        return ("rate", "OpenRouter respondió 429: demasiadas peticiones. Espera un momento.")
    return None


def is_fatal_api_error(exc: BaseException) -> bool:
    c = classify_api_error(exc)
    return bool(c and c[0] != "rate")


def api_error_message(exc: BaseException, default: str | None = None) -> str:
    c = classify_api_error(exc)
    if c is not None:
        return c[1]
    return default if default is not None else str(exc)
