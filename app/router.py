"""Fast path / agent router.

Default (target ~80% of questions):  1 hybrid retrieval + 1 LLM call.
    * the question contains a model/article code and there are CSV tables
      -> table_lookup first (local, exact, no tokens) and its rows go into the same call.
Escalate to the 5.7 LangGraph agent only when:
    * "which documents / manuals / datasheets do we have ..."   (reason: catalog)
    * nothing found                                              (no_results)
    * the code asked for appears in no table row and no chunk    (code_not_found)
    * weak vector relevance AND few query words in the top chunks (low_relevance)
    * the fast-path LLM answers INSUFFICIENT_CONTEXT              (insufficient_context)
When escalating, the agent is seeded with the fast-path results, so its first step is
not a repeat of the search that was already done.

Source routing (SOURCE_ROUTING in the .env): the question is classified as asking for
technical data (-> datasheets + product CSVs), a procedure (-> manuals) or parts/drawings
(-> drawing folders). The search runs first ONLY in those folders (declared in the .env as
RAG_DIR_*); with "prefer" it widens to everything only when that source has nothing
relevant, with "strict" it never does.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent import HybridGraphAgent, step_signature
from app.config import SECTION_LABEL, SOURCE_GROUPS, settings
from app.prompts import FAST_SYSTEM, INSUFFICIENT, with_instructions
from app.query_lang import expand_query, llm_search_query, looks_english
from app.retriever import HybridEvaluationRetriever, SearchSignals
from app.textutils import (STEM_STOPWORDS, extract_codes, fold, format_docs, norm, raw_codes,
                           simple_tokenize, source_tag)
from app.tools import RequestContext, new_context

log = logging.getLogger("rag.router")

_STOPWORDS = {
    # es
    "que", "qué", "cual", "cuál", "cuales", "como", "cómo", "para", "por", "con", "sin", "los",
    "las", "del", "una", "uno", "unos", "unas", "este", "esta", "ese", "esa", "son", "hay",
    "tiene", "tienen", "dime", "sobre", "entre", "cuanto", "cuánto", "donde", "dónde", "más",
    "mas", "muy", "pero", "cuando", "cuándo", "puedo", "puede", "quiero", "necesito",
    # en
    "the", "and", "for", "with", "what", "which", "how", "does", "are", "is", "can", "this",
    "that", "from", "about", "have", "has", "need", "tell", "give", "show", "much", "many",
    "where", "when", "there", "their", "your", "into",
}
_CAT_NOUNS = (r"(documentos|manuales|archivos|ficheros|fichas(?:\s+t[eé]cnicas)?|datasheets|"
              r"documents|manuals|files|tablas|tables|dibujos|drawings|planos|"
              r"cat[aá]logos|catalogues|catalogs|despieces)")
_CATALOG_RES = [
    re.compile(r"\b(qu[eé]|cu[aá]les|cu[aá]ntos|cu[aá]ntas|which|what|list|lista|listar|"
               r"enumera|muestra|mu[eé]strame|show)\b.{0,40}\b" + _CAT_NOUNS + r"\b", re.I | re.S),
    re.compile(r"\b(tenemos|hay|existen|do we have|are there|available|disponibles)\b.{0,40}\b"
               + _CAT_NOUNS + r"\b", re.I | re.S),
    re.compile(r"\b" + _CAT_NOUNS + r"\b.{0,30}\b(tenemos|hay|disponibles|do we have|available)\b",
               re.I | re.S),
]
# The user NAMES the kind of document -> only that source (wins over topic words)
_EXPLICIT_SOURCE = {
    "manual": re.compile(r"\b(manual(es)?|instructivos?|instrucciones de (uso|operacion))\b"),
    "datasheet": re.compile(r"\b(hojas? tecnicas?|fichas? tecnicas?|data ?sheets?|hojas? de datos)\b"),
    "drawing": re.compile(r"\b(despieces?|explosionad\w*|dibujos?|planos?|exploded|drawings?)\b"),
}


def _words_re(words: tuple[str, ...]) -> re.Pattern:
    alts = sorted({re.escape(fold(w)) for w in words if fold(w)}, key=len, reverse=True)
    # whole words / phrases only ("peso" must not match "pesos" = money)
    return re.compile(r"\b(" + "|".join(alts) + r")\b") if alts else re.compile(r"(?!x)x")


_DATASHEET_RE = re.compile(r"\b(hojas?\s+t[eé]cnicas?|fichas?\s+t[eé]cnicas?|data\s?sheets?|"
                           r"hojas?\s+de\s+datos|especificaciones\s+completas)\b", re.I)
# "genérame un PDF de la AX-4192400", "descargar pdf del AX-..." = a datasheet PDF request
_PDF_REQ_RE = re.compile(r"\b(gen[eé]r\w*|crea\w*|haz\w*|hacer\w*|dame|descarg\w*|export\w*|"
                         r"convi[eé]rt\w*|imprim\w*|generate|create|make|download|export)\b"
                         r".{0,60}\bpdf\b|\bpdf\b.{0,40}\b(datasheet|ficha|hoja)", re.I | re.S)
# the user asks to PRODUCE a sheet ("genérame la ficha técnica de ...", "dame una hoja de
# datos de cualquier válvula"), not a value that happens to be in a datasheet
_SHEET_REQ_RE = re.compile(r"\b(gen[eé]r\w*|crea\w*|haz\w*|hacer\w*|dame|d[aá]me|descarg\w*|"
                           r"export\w*|mand\w*|env[ií]\w*|necesito|quiero|generate|create|make|"
                           r"download|send|give me)\b.{0,50}\b(hojas?\s+t[eé]cnicas?|"
                           r"fichas?\s+t[eé]cnicas?|data\s?sheets?|hojas?\s+de\s+datos|pdf)\b",
                           re.I | re.S)

# "document 0893", "documento 1203-00", "doc-id 0893", "documnet 0893" (typos tolerated)
_DOC_REF_RE = re.compile(r"\b(?:doc\w{0,7}|archivo|file|manual)(?:[\s\-]*id)?\s*(?:no\.?|n[°º]|#|:)?\s*"
                         r"([0-9]{3,6}(?:-[0-9]{2})?)\b", re.I)
# "page 40", "página 40", "pag. 40", "p. 40"
_PAGE_REF_RE = re.compile(r"\b(?:p[aá]g(?:ina|e)?\.?|page|p\.)\s*(\d{1,4})\b", re.I)
# the model sometimes wraps or misspells the escape word ("**INSUFFICIENT_CONTEXT**")
_INSUFFICIENT_RE = re.compile(r"IN\w{0,3}SUFFICIENT[_ ]CONTEXT", re.I)


@dataclass
class QueryResult:
    answer: str
    route: str                      # fast | fast+table | agent | clarify
    escalation: str = ""            # why the agent ran ("" on the fast path)
    clarify: bool = False           # answer is a question back to the user
    sources: list[dict] = field(default_factory=list)
    drawings: list[dict] = field(default_factory=list)
    agent_steps: int = 0
    usage: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    baseline: str | None = None     # only when settings.run_baseline (evaluation)
    source_route: str = ""          # e.g. "manual", "datasheet", "manual->all" (fallback)
    transcript: list[str] | None = None   # only with debug=True

    def to_dict(self) -> dict:
        return asdict(self)


class QueryRouter:
    def __init__(self, retriever: HybridEvaluationRetriever, llm: Any,
                 agent_llm: Any | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.agent = HybridGraphAgent(agent_llm or llm)
        self._stats: Counter = Counter()
        self._stats_lock = threading.Lock()

    # ---------------- classification ----------------
    @staticmethod
    def is_catalog_question(q: str) -> bool:
        return any(r.search(q or "") for r in _CATALOG_RES)

    @staticmethod
    def is_datasheet_question(q: str) -> bool:
        return bool(_DATASHEET_RE.search(q or "") or _PDF_REQ_RE.search(q or ""))

    @staticmethod
    def source_intent(q: str) -> list[str]:
        """Which source the question asks for: ['manual'], ['datasheet'], ['drawing'],
        several, or [] (no preference). Explicit names win over topic words."""
        if settings.source_routing == "off":
            return []
        f = fold(q)
        explicit = [g for g, rx in _EXPLICIT_SOURCE.items() if rx.search(f)]
        if explicit:
            return explicit
        topic = []
        for g, words in (("datasheet", settings.intent_datasheet_words),
                         ("manual", settings.intent_manual_words),
                         ("drawing", settings.intent_drawing_words)):
            if _words_re(tuple(words)).search(f):
                topic.append(g)
        return topic

    def preferred_sections(self, q: str, rtr: Any) -> tuple[set[str] | None, str]:
        """(sections to search first, intent label) — None when there is no preference
        or none of those folders has indexed files in the selected bases."""
        intent = self.source_intent(q)
        if not intent:
            return None, ""
        wanted = {s for g in intent for s in SOURCE_GROUPS[g]}
        try:
            present = rtr.sections_present()
        except AttributeError:
            present = wanted
        sections = wanted & present
        return (sections or None), "+".join(intent)

    # ---------------- document + page references ----------------
    @staticmethod
    def detect_doc_page(q: str) -> tuple[str, int | None]:
        """('0893', 40) for 'table on page 40 of document 0893'; ('', None) if no doc named."""
        m = _DOC_REF_RE.search(q or "")
        p = _PAGE_REF_RE.search(q or "")
        return (m.group(1) if m else ""), (int(p.group(1)) if p else None)

    @staticmethod
    def _resolve_doc_ref(rtr: Any, ref: str) -> str | None:
        """File whose NAME contains the reference, else whose TEXT says 'Doc-ID <ref>'."""
        rel, _ = rtr.resolve_doc(ref)
        if rel:
            return rel
        base = ref.split("-")[0]
        rx = re.compile(r"Doc[\s\-]*ID\s*:?\s*" + re.escape(base) + r"\b", re.I)
        votes: Counter = Counter()
        for c in rtr.snapshot.chunks.values():
            if rx.search(c.get("text", "")):
                votes[c["source"]] += 1
        return votes.most_common(1)[0][0] if votes else None

    @staticmethod
    def _page_chunks(rtr: Any, rel: str, page: int | None, question: str,
                     max_chunks: int = 6) -> list[dict]:
        """Chunks of one file, in order. With a page: the chunks of that page (+ the next
        ones, tables often continue). Works with page-aware chunks ('page' key) and, for
        the current index, with the printed page number ('40' alone on a line)."""
        chunks = [c for c in rtr.snapshot.chunks.values() if c.get("source") == rel]
        if not chunks:
            return []
        if page is not None:
            # users quote the PRINTED page number; the PDF index often differs
            rx = re.compile(rf"(?m)^\s*(?:-\s*)?{page}(?:\s*-)?\s*$|impresa {page}\]")
            if any("page" in c for c in chunks):
                phys = next((c["page"] for c in chunks if c.get("page_label") == page), None)
                if phys is None:
                    phys = next((c["page"] for c in chunks if rx.search(c.get("text", ""))), None)
                if phys is None:
                    phys = page if any(c.get("page") == page for c in chunks) else None
                if phys is not None:
                    sel = [c for c in chunks if c.get("page") in (phys, phys + 1)]
                    sel.sort(key=lambda c: (c["page"], c.get("kind") == "table", c.get("seq", 0)))
                    return sel[:max_chunks + 4]
            else:
                hit = [i for i, c in enumerate(chunks) if rx.search(c.get("text", ""))]
                if hit:
                    first = hit[0]
                    return chunks[max(0, first - 1): first + max_chunks - 1]
        # no page (or not found): the passages of that file most related to the question
        toks = {t for t in simple_tokenize(question) if len(t) >= 3 and t not in _STOPWORDS}
        scored = sorted(range(len(chunks)), reverse=True,
                        key=lambda i: len(toks & set(simple_tokenize(chunks[i]["text"]))))
        keep = sorted(scored[:max_chunks])
        return [chunks[i] for i in keep]

    @staticmethod
    def detect_codes(q: str) -> list[str]:
        """Model/article codes as written by the user ('VDV-25', '123456')."""
        return list(raw_codes(q).values())

    @staticmethod
    def _coverage(question: str, docs: list[dict]) -> float:
        """Share of the question's words found in the top chunks. A word in another language
        counts as found when its glossary translation is there ('Düsen' -> 'nozzle')."""
        terms = {t for t in simple_tokenize(question) if len(t) >= 3 and t not in _STOPWORDS}
        if not terms:
            return 1.0
        seen = set(simple_tokenize(" ".join(d["text"] for d in docs)))

        def found(t: str) -> bool:
            return t in seen or bool(set(simple_tokenize(expand_query(t))) & seen)
        return sum(1 for t in terms if found(t)) / len(terms)

    def _weak_reason(self, question: str, codes: list[str], docs: list[dict],
                     sig: SearchSignals, table_rows: int) -> str:
        if not docs and not table_rows:
            return "no_results"
        if codes and not table_rows:
            wanted = {norm(c) for c in codes}
            present: set[str] = set()
            for d in docs:
                present |= extract_codes(d["text"]) | {norm(d.get("source", ""))}
            if not any(w in p for w in wanted for p in present):
                return "code_not_found"
        if not table_rows:
            low_vec = (not sig.vector_ok) or sig.max_vector < settings.fast_min_vector_score
            if low_vec and self._coverage(question, docs[:5]) < settings.fast_min_term_coverage:
                return "low_relevance"
        return ""

    # ---------------- main entry ----------------
    def answer(self, question: str, force_agent: bool = False, debug: bool = False,
               retriever: Any = None, instructions: str = "") -> QueryResult:
        """retriever: per-question override of the index; instructions: optional extra
        instructions appended to every prompt (RAG_INSTRUCTIONS)."""
        t0 = time.perf_counter()
        question = (question or "").strip()
        rtr = retriever or self.retriever
        rtr.maybe_reload()
        ctx = new_context(rtr, instructions, llm=self.llm)
        if force_agent:
            return self._run_agent(question, ctx, [], "forced", t0, debug)
        # "show tables of valve KA-2 HD" is a content question, not "which tables do we have":
        # a named document or a product/model code means it is never a catalog question
        if self.is_catalog_question(question) and not self.detect_doc_page(question)[0] \
                and not self.detect_codes(question):
            return self._run_agent(question, ctx, [], "catalog", t0, debug)

        # 0) the user names a document (and maybe a page): read THAT document, not a search
        doc_ref, page = self.detect_doc_page(question)
        if doc_ref:
            rel = self._resolve_doc_ref(rtr, doc_ref)
            if rel:
                t_r = time.perf_counter()
                docs = self._page_chunks(rtr, rel, page, question)
                retrieval_ms = (time.perf_counter() - t_r) * 1000
                if docs:
                    ctx.hits.extend(docs)
                    args = {"doc": rel} | ({"page": page} if page else {})
                    seed = [f"{step_signature('read_document', args)} {format_docs(docs, n=len(docs), width=4000)}"]
                    ans = self._fast_answer(ctx, question, docs, "")
                    if settings.fast_escalate_on_insufficient and _INSUFFICIENT_RE.search(ans):
                        return self._run_agent(question, ctx, seed, "insufficient_context", t0,
                                               debug, retrieval_ms)
                    return self._finish(question, ans, "fast+doc", "", ctx, t0, retrieval_ms,
                                        debug=debug, transcript=seed)

        # 1) exact code lookup in the CSV tables (local, no tokens)
        seed: list[str] = []
        codes = self.detect_codes(question)
        table_text, table_rows = "", 0
        has_tables = any(e["type"] == "table" for e in rtr.snapshot.catalog.values())
        if codes and has_tables:
            before = len(ctx.hits)
            code_arg = ", ".join(codes)
            table_text = rtr.table_lookup(code_arg, hits=ctx.hits)
            table_rows = len(ctx.hits) - before
            seed.append(f"{step_signature('table_lookup', {'code': code_arg})} {table_text}")

        # 1b) "hoja técnica del 123456": product master data + datasheet link (local)
        #     every code is tried (the first one may be a series, not the product)
        wants_sheet = self.is_datasheet_question(question)
        sheet_found = False
        for code in (codes if wants_sheet else [])[:3]:
            sheet = ctx.tools["product_datasheet"](part=code)
            if sheet.startswith("[datasheet:"):
                seed.append(f"{step_signature('product_datasheet', {'part': code})} {sheet}")
                table_text = (table_text + "\n\n" if table_rows else "") + sheet
                table_rows += 1
                sheet_found = True
                break
        # 1c) a datasheet asked for by product NAME ("ficha técnica de la válvula de apertura
        #     y cierre", "una hoja de datos de cualquier válvula"): the fast path cannot pick
        #     the product, the agent can (list_documents / product_datasheet by name)
        if wants_sheet and not codes and not sheet_found and _SHEET_REQ_RE.search(question):
            return self._run_agent(question, ctx, seed, "datasheet_by_name", t0, debug)

        # 2) one hybrid retrieval — first in the source the question asks for
        t_r = time.perf_counter()
        search_q = question
        if settings.query_translate == "llm" and not looks_english(question):
            # optional: English keywords from the cheap model (+1 small call)
            en = llm_search_query(self.agent.llm, ctx.meter, question)
            if en:
                search_q = f"{question} {en}"
        sections, intent = self.preferred_sections(search_q, rtr)
        ctx.source_route = intent
        if sections and settings.source_routing == "strict":
            ctx.default_sections = sections          # the agent stays in that source too
        docs, sig = rtr.retrieve_with_signals(
            search_q, top_k=settings.fast_top_k, sections=sections)
        if sections and settings.source_routing == "prefer" and \
                self._weak_reason(question, codes, docs, sig, table_rows):
            log.info("source %s has nothing relevant -> searching every source", intent)
            docs, sig = rtr.retrieve_with_signals(search_q, top_k=settings.fast_top_k)
            ctx.source_route = f"{intent}->all"
            sections = None
        ctx.hits.extend(docs)
        retrieval_ms = (time.perf_counter() - t_r) * 1000
        args = {"query": question}
        if sections:
            args["source"] = intent
        seed.append(f"{step_signature('semantic_search', args)} {format_docs(docs, n=len(docs))}")

        reason = self._weak_reason(question, codes, docs, sig, table_rows)
        if reason:
            return self._run_agent(question, ctx, seed, reason, t0, debug, retrieval_ms)

        # 3) one LLM call
        ans = self._fast_answer(ctx, question, docs, table_text if table_rows else "",
                                intent=intent if sections else "")
        if settings.fast_escalate_on_insufficient and _INSUFFICIENT_RE.search(ans):
            return self._run_agent(question, ctx, seed, "insufficient_context", t0, debug,
                                   retrieval_ms)
        route = "fast+table" if table_rows else "fast"
        return self._finish(question, ans, route, "", ctx, t0, retrieval_ms, debug=debug,
                            transcript=seed)

    # ---------------- paths ----------------
    def _fast_answer(self, ctx: RequestContext, question: str, docs: list[dict],
                     table_text: str, intent: str = "") -> str:
        parts = []
        if intent:
            labels = ", ".join(SECTION_LABEL[s] for g in intent.split("+")
                               for s in SOURCE_GROUPS.get(g, ()))
            parts.append(f"This question asks for: {intent}. The documents below were "
                         f"searched ONLY in: {labels}.")
        focus = list(dict.fromkeys(d["source"] for d in docs if d.get("focus")))
        if focus:
            names = ", ".join(Path(s).stem for s in focus)
            parts.append(f"The question is about the product of: {names}. Answer from that "
                         "document; use fragments of OTHER products only if the user asks to "
                         "compare, and then label them as another product.")
        if table_text:
            parts.append(f"Exact table lookup / product data:\n{table_text}")
        parts.append("Documents:\n" + ("\n\n".join(f"[{d['id']}] {source_tag(d)} {d['text']}"
                                                   for d in docs)
                                       or "(no documents retrieved)"))
        return ctx.meter.invoke(self.llm, [SystemMessage(content=with_instructions(
                                               FAST_SYSTEM, ctx.instructions)),
                                           HumanMessage(content="\n\n".join(parts)
                                                        + f"\n\nQuestion: {question}")],
                                stage="fast")

    def _run_agent(self, question: str, ctx: RequestContext, seed: list[str], reason: str,
                   t0: float, debug: bool, retrieval_ms: float = 0.0) -> QueryResult:
        baseline = None
        if settings.run_baseline:
            docs = self._rtr(ctx).retrieve(question, top_k=settings.fast_top_k)
            baseline = self._fast_answer(ctx, question, docs, "")
        final = self.agent.run(question, ctx, seed_transcript=seed)
        route = "clarify" if final.get("clarify") else "agent"
        return self._finish(question, final["answer"], route, reason, ctx, t0, retrieval_ms,
                            steps=final["iterations"], clarify=bool(final.get("clarify")),
                            baseline=baseline, debug=debug, transcript=final["transcript"])

    # ---------------- result assembly ----------------
    @staticmethod
    def _with_downloads(answer: str, ctx: RequestContext, clarify: bool) -> str:
        """Datasheet PDFs for this answer -> the console command that writes each one.
        Added here (not by the LLM) so the command is always right and always present."""
        downloads = getattr(ctx, "downloads", None) or []
        if clarify or not downloads:
            return answer
        links = [d for d in downloads[:3] if d["cmd"] not in (answer or "")]
        if not links:
            return answer
        block = "\n".join(f"- {d['label']}:  {d['cmd']}" for d in links)
        return f"{(answer or '').rstrip()}\n\nHojas técnicas en PDF (ejecuta en la consola):\n{block}"

    def _rtr(self, ctx: RequestContext) -> Any:
        return ctx.retriever or self.retriever

    def _sources(self, hits: list[dict], answer: str, limit: int = 10,
                 retriever: Any = None) -> list[dict]:
        catalog = (retriever or self.retriever).snapshot.catalog
        seen: dict[str, dict] = {}
        for h in hits:
            if h["id"] in seen:
                continue
            src = h.get("source", "")
            # web page -> its URL; indexed file -> its path on disk
            url = (src if h.get("kind") == "web"
                   else (catalog[src].get("path", "") if src in catalog else ""))
            seen[h["id"]] = {"id": h["id"], "source": src, "kind": h.get("kind", "text"),
                             "section": h.get("section", ""),
                             "score": h.get("score"), "cited": h["id"] in answer, "url": url}
        items = list(seen.values())
        items.sort(key=lambda s: not s["cited"])          # cited first, then retrieval order
        return items[:limit]

    def _drawings(self, question: str, answer: str, retriever: Any = None) -> list[dict]:
        """(a) id/file name quoted, (b) shared codes, (c) >= 2 file-name words (from 5.7)."""
        qa = f"{question}\n{answer}"
        qa_low, qa_codes, qa_words = qa.lower(), extract_codes(qa), set(simple_tokenize(qa))
        scored = []
        for d in (retriever or self.retriever).drawings():
            p = Path(d["path"])
            score = 0.0
            if d["id"].lower() in qa_low or p.name.lower() in qa_low or p.stem.lower() in qa_low:
                score += 10
            overlap = qa_codes & (extract_codes(d["text"]) | {c for c in qa_codes if c in norm(p.stem)})
            score += 3 * len(overlap)
            stem_words = {w for w in simple_tokenize(p.stem) if len(w) >= 4 and w not in STEM_STOPWORDS}
            if len(stem_words & qa_words) >= 2:
                score += len(stem_words & qa_words)
            if score > 0 and p.exists():
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"id": d["id"], "source": d["source"], "url": str(d["path"])}
                for _, d in scored[:settings.max_drawings]]

    def _finish(self, question: str, answer: str, route: str, reason: str,
                ctx: RequestContext, t0: float, retrieval_ms: float, steps: int = 0,
                clarify: bool = False, baseline: str | None = None, debug: bool = False,
                transcript: list[str] | None = None) -> QueryResult:
        total_ms = (time.perf_counter() - t0) * 1000
        answer = self._with_downloads(answer, ctx, clarify)
        res = QueryResult(
            answer=answer, route=route, escalation=reason, clarify=clarify,
            sources=[] if clarify else self._sources(ctx.hits, answer, retriever=self._rtr(ctx)),
            drawings=[] if clarify else self._drawings(question, answer, self._rtr(ctx)),
            agent_steps=steps, usage=ctx.meter.as_dict(),
            timings_ms={"retrieval": round(retrieval_ms, 1), "total": round(total_ms, 1),
                        "llm": round(ctx.meter.llm_seconds * 1000, 1)},
            baseline=baseline, transcript=transcript if debug else None,
            source_route=getattr(ctx, "source_route", ""))
        with self._stats_lock:
            self._stats["queries"] += 1
            self._stats[f"route:{route}"] += 1
            if reason:
                self._stats[f"escalation:{reason}"] += 1
            self._stats["llm_calls"] += res.usage["llm_calls"]
        log.info("route=%s%s calls=%d tokens=%d/%d total=%.0fms q=%r", route,
                 f"({reason})" if reason else "", res.usage["llm_calls"],
                 res.usage["input_tokens"], res.usage["output_tokens"], total_ms, question[:80])
        return res

    def stats(self) -> dict:
        with self._stats_lock:
            s = dict(self._stats)
        q = s.get("queries", 0)
        fast = s.get("route:fast", 0) + s.get("route:fast+table", 0)
        s["fast_share"] = round(fast / q, 3) if q else None
        s["avg_llm_calls"] = round(s.get("llm_calls", 0) / q, 2) if q else None
        return s
