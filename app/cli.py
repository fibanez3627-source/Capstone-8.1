"""RAG command line :

Preguntar:
    python -m app.cli chat                       conversación interactiva (recuerda el contexto)
    python -m app.cli ask "¿presión máxima de la VDV-25?"   una sola pregunta
        opciones:  --agent  fuerza el agente      --debug  muestra los pasos
                   --json   salida JSON completa  --instrucciones "texto"  tono / formato
                   --no-cache  no usa la caché semántica de respuestas

Caché semántica de respuestas (SQLite, ver CACHE_SEMANTICA.md):
    python -m app.cli cache-stats                hits, tokens y USD ahorrados, entradas más usadas
    python -m app.cli cache-clear                vacía la caché de respuestas (no los embeddings)
    python scripts/cache_compare.py              comparativa SIN caché vs CON caché

Índice:
    python -m app.cli update                     indexa archivos nuevos / cambiados / borrados
    python -m app.cli rebuild                    borra e indexa todo (la caché conserva lo pagado)
    python -m app.cli status                     archivos, fragmentos y vectores del índice
    python -m app.cli docs [--keyword X] [--type manual] [--manufacturer abnox]

Configuración y diagnóstico:
    python -m app.cli paths                      qué .env se leyó y todas las rutas efectivas
    python -m app.cli init-data                  crea las carpetas RAG_DIR_* del .env (vacías)
    python -m app.cli check                      clave, créditos, embeddings y modelo de chat
    python -m app.cli reranker-check             descarga / carga el cross-encoder y lo prueba
    python -m app.cli web-update                 descarga las páginas de los fabricantes

Productos y hojas técnicas:
    python -m app.cli data-report                productos sin foto, duplicados, huérfanos...
    python -m app.cli datasheet 123456 [--pdf] [--out archivo]   una hoja técnica (HTML o PDF)
    python -m app.cli datasheet --all            todas -> RAG_DATASHEET_OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from app.config import settings

log = logging.getLogger("rag.cli")

CHAT_HELP = """Comandos dentro del chat:
  /salir        termina (también Ctrl+C o Ctrl+Z / Ctrl+D)
  /exit         termina
  /nuevo        olvida la conversación y empieza otra
  /agente       activa / desactiva el agente forzado (más pasos, más tokens)
  /debug        activa / desactiva los pasos del agente
  /fuentes      muestra todas las fuentes de la última respuesta
  /cache        activa / desactiva la caché semántica de respuestas
  /ayuda        esta ayuda"""


# =========================================================================================
# entry point
# =========================================================================================
def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    ap = argparse.ArgumentParser(prog="python -m app.cli",
                                 description="Relant RAG por línea de comandos",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("-v", "--verbose", action="store_true", help="muestra el log en la consola")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="comando")

    def answer_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--agent", action="store_true", help="fuerza el agente")
        p.add_argument("--debug", action="store_true", help="muestra los pasos")
        p.add_argument("--instrucciones", default=None,
                       help="instrucciones extra (tono, formato); por defecto RAG_INSTRUCTIONS")
        p.add_argument("--no-cache", action="store_true",
                       help="no usa la caché semántica de respuestas")

    p = sub.add_parser("ask", help="una sola pregunta")
    p.add_argument("question", nargs="+")
    p.add_argument("--json", action="store_true", help="resultado completo en JSON")
    answer_opts(p)
    p = sub.add_parser("chat", help="conversación interactiva")
    answer_opts(p)
    sub.add_parser("update", help="indexa archivos nuevos / cambiados / borrados")
    sub.add_parser("rebuild", help="borra e indexa todo")
    sub.add_parser("status", help="estado del índice")
    p = sub.add_parser("docs", help="lista los documentos indexados")
    p.add_argument("--keyword", default="")
    p.add_argument("--type", default="", help="manual, datasheet, table, drawing, catalogue...")
    p.add_argument("--manufacturer", default="")
    p.add_argument("--limit", type=int, default=60)
    sub.add_parser("paths", help="rutas efectivas del .env")
    p = sub.add_parser("cache-stats", help="estadísticas de la caché semántica de respuestas")
    p.add_argument("--json", action="store_true")
    sub.add_parser("cache-clear", help="vacía la caché semántica de respuestas")
    sub.add_parser("init-data", help="crea las carpetas del .env")
    sub.add_parser("check", help="prueba la clave de OpenRouter y los modelos")
    sub.add_parser("reranker-check", help="prueba el reranker")
    p = sub.add_parser("web-update", help="descarga las páginas de los fabricantes")
    p.add_argument("--refresh", action="store_true", help="aunque la copia sea reciente")
    sub.add_parser("data-report", help="reporte de productos / fotos / hojas")
    p = sub.add_parser("datasheet", help="genera hojas técnicas")
    p.add_argument("part", nargs="?", default="")
    p.add_argument("--all", action="store_true")
    p.add_argument("--out", default="")
    p.add_argument("--no-embed", action="store_true", help="enlaza las imágenes en vez de incrustarlas")
    p.add_argument("--pdf", action="store_true", help="PDF en vez de HTML")
    args = ap.parse_args(argv)
    _setup_logging(args.verbose or args.cmd in ("update", "rebuild", "web-update"))

    from app.errors import IndexingAborted, OpenRouterError
    try:
        if args.cmd == "paths":
            print(json.dumps(settings.paths_report(), indent=2, ensure_ascii=False))
            return 0
        if args.cmd == "init-data":
            return init_data()
        if args.cmd in ("cache-stats", "cache-clear"):
            return cache_command(args)           # no API key needed
        if args.cmd == "reranker-check":
            return reranker_check()
        if args.cmd == "web-update":
            return web_update(args.refresh)
        if args.cmd in ("status", "docs", "data-report", "datasheet"):
            return _local_command(args)          # no API key needed

        from app.llm import check_api_key, make_embeddings, make_llm, preflight
        key = check_api_key()
        if args.cmd == "check":
            print("preflight:", preflight(key))
            print("embedding dims:", len(make_embeddings().embed_query("ping")))
            print("chat:", make_llm().invoke("Reply with OK").content)
            return 0
        if args.cmd in ("update", "rebuild"):
            return index(args.cmd, key)
        router = _make_router()
        if args.cmd == "ask":
            return ask(router, " ".join(args.question), args)
        if args.cmd == "chat":
            return chat(router, args)
        return 1
    except (OpenRouterError, IndexingAborted, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n(cancelado)", file=sys.stderr)
        return 130


# =========================================================================================
# setup
# =========================================================================================
def _utf8_console() -> None:
    """Accents and ñ on the Windows console, also when the output is redirected."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _setup_logging(console: bool) -> None:
    """Console: warnings only (INFO with -v or while indexing). File: RAG_LOG_DIR, INFO."""
    level = os.getenv("LOG_LEVEL", "").upper()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for old in [h for h in root.handlers if getattr(h, "_relant_cli", False)]:
        root.removeHandler(old)                   # main() called twice in one process (tests)
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(level or (logging.INFO if console else logging.WARNING))
    h.setFormatter(fmt)
    h._relant_cli = True
    root.addHandler(h)
    if settings.log_dir:
        from logging.handlers import RotatingFileHandler
        Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(Path(settings.log_dir) / "relant_rag.log",
                                 maxBytes=10_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        fh._relant_cli = True
        root.addHandler(fh)
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _vision_llm():
    from app.llm import make_llm
    return make_llm(settings.effective_vision_model) if settings.describe_images else None


def _load_retriever(vision=None):
    from app.retriever import HybridEvaluationRetriever
    r = HybridEvaluationRetriever(vision_llm=vision)
    r.load()
    return r


def _make_router():
    from app.llm import make_llm
    from app.production_router import ProductionRouter
    retriever = _load_retriever()
    if retriever.snapshot.empty:
        raise RuntimeError("el índice está vacío. Ejecuta primero:  python -m app.cli update")
    llm = make_llm(settings.llm_model)
    agent_llm = make_llm(settings.effective_agent_model) if settings.agent_model else llm
    cache = validator = None
    if settings.semantic_cache:
        from app.semantic_cache import open_semantic_cache
        cache = open_semantic_cache(retriever)
        validator = (make_llm(settings.effective_validator_model)
                     if settings.semantic_cache_validator_model else llm)
    return ProductionRouter(retriever, llm, agent_llm, cache=cache, validator_llm=validator)


# =========================================================================================
# questions
# =========================================================================================
def ask(router, question: str, args) -> int:
    res = router.answer(question, force_agent=args.agent, debug=args.debug,
                        instructions=args.instrucciones, use_cache=not args.no_cache)
    d = res.to_dict()
    if args.json:
        print(json.dumps(d, indent=2, ensure_ascii=False))
    else:
        print_result(d, debug=args.debug)
    return 0


def chat(router, args) -> int:
    print("Relant RAG — escribe tu pregunta (/ayuda para ver los comandos, /salir para terminar)")
    history: list[dict] = []
    pending_clarify = ""            # original question when the assistant asked something back
    force_agent, debug, last = args.agent, args.debug, {}
    use_cache = not args.no_cache
    from app.conversation_context import CLARIFY_JOIN
    while True:
        try:
            text = input("\nTú> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        cmd = text.lower()
        if cmd in ("/salir", "/exit", "/quit", "salir"):
            return 0
        if cmd in ("/ayuda", "/help", "?"):
            print(CHAT_HELP)
            continue
        if cmd == "/nuevo":
            history, pending_clarify, last = [], "", {}
            print("(conversación nueva)")
            continue
        if cmd == "/agente":
            force_agent = not force_agent
            print(f"(agente forzado: {'sí' if force_agent else 'no'})")
            continue
        if cmd == "/debug":
            debug = not debug
            print(f"(debug: {'sí' if debug else 'no'})")
            continue
        if cmd == "/cache":
            use_cache = not use_cache
            print(f"(caché semántica: {'sí' if use_cache else 'no'})")
            continue
        if cmd == "/fuentes":
            _print_sources(last.get("sources") or [], limit=50)
            continue

        query = f"{pending_clarify}{CLARIFY_JOIN}{text}" if pending_clarify else text
        t0 = time.time()
        try:
            res = router.answer(query, force_agent=force_agent, debug=debug,
                                instructions=args.instrucciones, history=history,
                                use_cache=use_cache)
        except KeyboardInterrupt:
            print("\n(pregunta cancelada)")
            continue
        except Exception as e:
            from app.errors import OpenRouterError
            if isinstance(e, OpenRouterError):
                print(f"ERROR: {e}", file=sys.stderr)
                continue
            log.exception("question failed")
            print(f"ERROR: {e}", file=sys.stderr)
            continue
        last = res.to_dict()
        print()
        print_result(last, debug=debug, label="Relant> ", elapsed=time.time() - t0)
        pending_clarify = query.split(CLARIFY_JOIN)[0] if last.get("clarify") else ""
        history.append({"role": "user", "content": query.replace(CLARIFY_JOIN, " — ")})
        history.append({"role": "assistant", "content": last.get("answer") or ""})
        history = history[-2 * max(1, settings.conversation_history_turns) - 2:]


def print_result(d: dict, debug: bool = False, label: str = "", elapsed: float | None = None) -> None:
    if d.get("rewrite"):
        print(f"(entendí: {d.get('standalone_query')})")
    print(f"{label}{(d.get('answer') or '').strip()}")
    if not d.get("clarify"):
        _print_sources(d.get("sources") or [])
        drawings = d.get("drawings") or []
        if drawings:
            print("\nDibujos:")
            for x in drawings:
                print(f"  - {x.get('url') or x.get('source')}")
    u = d.get("usage") or {}
    route = d.get("route", "")
    if d.get("escalation"):
        route += f" ({d['escalation']})"
    if d.get("source_route"):
        route += f" · fuente: {d['source_route']}"
    c = d.get("cache") or {}
    if d.get("route") == "cache":
        kind = "exacta" if c.get("kind") == "exact" else f"semántica, sim {c.get('similarity', 0):.2f}"
        route += f" ({kind}) · ahorro ~{c.get('tokens_saved', 0)} tokens"
    secs = elapsed if elapsed is not None else (d.get("timings_ms") or {}).get("total", 0) / 1000
    print(f"\n[ruta: {route} · {u.get('llm_calls', 0)} llamadas LLM · "
          f"{u.get('input_tokens', 0)}/{u.get('output_tokens', 0)} tokens · "
          f"~${u.get('est_cost_usd', 0):.4f} · {secs:.1f} s]")
    if debug and d.get("transcript"):
        print("\nPasos:")
        for line in d["transcript"]:
            print("  " + (line if len(line) < 600 else line[:600] + " ..."))


def _print_sources(sources: list[dict], limit: int = 6) -> None:
    """Cited sources first; one line per file (several chunks of one file -> one line)."""
    if not sources:
        return
    seen, rows = set(), []
    for s in sources:
        key = s.get("url") or s.get("source")
        if key in seen:
            continue
        seen.add(key)
        rows.append(s)
    cited = [s for s in rows if s.get("cited")]
    shown = (cited or rows)[:limit]
    print("\nFuentes" + (" citadas:" if cited else ":"))
    for s in shown:
        sec = f" [{s['section']}]" if s.get("section") else ""
        print(f"  - {s.get('url') or s.get('source')}{sec}")
    if len(rows) > len(shown):
        print(f"  (+{len(rows) - len(shown)} más: /fuentes en el chat o --json)")


# =========================================================================================
# indexing
# =========================================================================================
def index(mode: str, key: str) -> int:
    from app.jobs import index_lock_for
    from app.llm import preflight
    from app.retriever import HybridEvaluationRetriever
    retriever = HybridEvaluationRetriever(vision_llm=_vision_llm())
    if settings.preflight:
        preflight(key)
    with index_lock_for(retriever):
        sync = retriever.rebuild_index() if mode == "rebuild" else retriever.update_index()
    print(json.dumps(sync, indent=2, ensure_ascii=False, default=str))
    return 0


def _local_command(args) -> int:
    """Commands that only read the index on disk (no API calls)."""
    from app.retriever import HybridEvaluationRetriever

    class _NoEmbeddings:                      # never called: these commands do not search
        def embed_documents(self, texts):
            raise RuntimeError("embeddings are not available in this command")

        def embed_query(self, text):
            raise RuntimeError("embeddings are not available in this command")

    retriever = HybridEvaluationRetriever(embeddings=_NoEmbeddings())
    retriever.load()
    if args.cmd == "status":
        st = retriever.stats()
        st["env_file"] = settings.paths_report()["env_file"]
        st["data_dir"], st["index_dir"] = settings.data_dir, settings.index_dir
        print(json.dumps(st, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.cmd == "docs":
        from app.catalog import format_catalog_entry
        items, total = retriever.list_documents(args.keyword, args.manufacturer, args.type,
                                                max(1, args.limit))
        print(f"{len(items)} de {total} documentos ({retriever.catalog_summary()})")
        for e in items:
            print(format_catalog_entry(e))
        return 0
    if args.cmd == "data-report":
        from app.products import get_product_index
        print(json.dumps(get_product_index(retriever).report(), indent=2, ensure_ascii=False))
        return 0
    return datasheet(retriever, args)


def datasheet(retriever, args) -> int:
    from app.products import build_datasheet_html, get_product_index, write_all_datasheets
    if args.all:
        print(json.dumps(write_all_datasheets(retriever, args.out or None,
                                              embed=not args.no_embed), indent=2))
        return 0
    if not args.part:
        print("Indica el número de parte (o --all).", file=sys.stderr)
        return 1
    idx = get_product_index(retriever)
    one, variants, found = idx.resolve(args.part)
    if one is None:
        if not found:
            print(f"No hay producto '{args.part}' en las tablas de productos.", file=sys.stderr)
            return 1
        if len(found) > 1:
            print("Varias coincidencias, se usa la primera: "
                  + ", ".join(f"{p.part} ({p.name})" for p in found[:8]), file=sys.stderr)
        one, variants = found[0], []
    out = args.out
    settings.datasheet_out_dir.mkdir(parents=True, exist_ok=True)
    if args.pdf:
        from app.products import build_datasheet_pdf, datasheet_pdf_filename
        out = out or str(settings.datasheet_out_dir / datasheet_pdf_filename(one))
        with open(out, "wb") as f:
            f.write(build_datasheet_pdf(one, variants=variants or None))
    else:
        out = out or str(settings.datasheet_out_dir / f"hoja_{one.part or 'producto'}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(build_datasheet_html(one, embed=not args.no_embed,
                                         variants=variants or None))
    print(os.path.abspath(out))
    return 0


# =========================================================================================
# diagnostics
# =========================================================================================
def reranker_check() -> int:
    from app.reranker import get_reranker
    rr = get_reranker()
    t0 = time.time()
    q = "presión máxima de la bomba"
    docs = ["Maximum operating pressure: 250 bar (Max. Betriebsdruck 250 bar).",
            "Instrucciones de limpieza del filtro de aire.",
            "La presión máxima de trabajo de la bomba es de 250 bar."]
    scores = rr.score(q, docs)
    print(json.dumps({"status": rr.status(), "seconds": round(time.time() - t0, 2),
                      "query": q, "scores": dict(zip(docs, scores or []))},
                     indent=2, ensure_ascii=False))
    return 0 if scores else 1


def web_update(refresh: bool) -> int:
    if not settings.web_enabled:
        print("WEB_ENABLED=false en el .env: la búsqueda en los sitios de los fabricantes "
              "está desactivada.")
        return 0
    from app.manufacturer_web import get_web
    web = get_web()
    web.ensure(refresh=refresh, allow_crawl=True)
    print(json.dumps(web.status(), indent=2, ensure_ascii=False, default=str))
    return 0


def cache_command(args) -> int:
    """cache-stats / cache-clear: only the SQLite file (no API key, no index)."""
    from app.semantic_cache import SemanticAnswerCache
    path = settings.semantic_cache_path
    if not os.path.exists(path):
        print(f"(no hay caché todavía: {path})")
        return 0
    cache = SemanticAnswerCache(path, None)
    try:
        if args.cmd == "cache-clear":
            print(f"caché vaciada: {cache.clear()} respuestas borradas ({path})")
            return 0
        st = cache.stats()
        if getattr(args, "json", False):
            print(json.dumps(st, indent=2, ensure_ascii=False))
            return 0
        ev = st["events"]
        print(f"Caché semántica: {path}")
        print(f"  entradas: {st['entries']}   umbral: {st['threshold']}   "
              f"validación LLM: {'sí' if st['validate_semantic_hits'] else 'no'}   "
              f"invalidación: {st['invalidation']}   TTL: {st['ttl_days']} días")
        rate = "-" if st["hit_rate"] is None else f"{100 * st['hit_rate']:.0f}%"
        print(f"  consultas: {st['lookups']}   hits exactos: {ev.get('hit-exact', 0)}   "
              f"hits semánticos: {ev.get('hit-semantic', 0)}   rechazados: {ev.get('rejected', 0)}   "
              f"miss: {ev.get('miss', 0)}   tasa de hit: {rate}")
        print(f"  guardadas: {ev.get('stored', 0)}   no guardadas: {ev.get('not-stored', 0)}")
        print(f"  tokens ahorrados: {st['tokens_saved']}   (gastados en validar: "
              f"{st['tokens_spent_on_validation']})   USD ahorrados (aprox.): ${st['usd_saved_est']:.4f}")
        if st["top_entries"]:
            print("  más usadas:")
            for e in st["top_entries"]:
                print(f"    [{e['id']}] {e['hits']} hits · {e['tokens_saved']} tok · "
                      f"{e['route']} · {e['question'][:70]}")
        return 0
    finally:
        cache.close()


DATA_LAYOUT = {      # sección -> qué va ahí (las CARPETAS vienen del .env, RAG_DIR_*)
    "productos": "un CSV / Excel por familia: bombas.csv, valvulas_atomizado.xlsx ...",
    "precios": "precios_existencias.csv (precio, costo, existencia)",
    "fotos": "foto del producto: <NumParte>.jpg",
    "explosionados": "vista explosionada: <NumParte>_explosionado.png",
    "drawings": "otros dibujos / planos",
    "hojas_tecnicas": "PDF existentes: <familia>/<NumParte>_<Nombre>.pdf",
    "manuales": "<fabricante>/<archivo>.pdf",
    "catalogos": "<fabricante>/<archivo>.pdf",
}


def init_data() -> int:
    """Create the folders declared in the .env (safe: never touches existing files)."""
    from app.config import SECTION_ENV
    root = Path(settings.data_dir)
    for section, what in DATA_LAYOUT.items():
        dirs = settings.section_dirs(section)
        if not dirs:
            print(f"({SECTION_ENV[section]} vacía: sección no usada)")
        for d in dirs:
            (root / d).mkdir(parents=True, exist_ok=True)
            print(f"{root / d}  <- {what}  [{SECTION_ENV[section]}]")
    Path(settings.index_dir).mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
