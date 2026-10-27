"""Agent tools, built PER QUESTION around a RequestContext (hits + usage meter).
`clarify` is not executed here: the agent treats it as a terminal step and returns the
question to the user (the `chat` command shows it and waits for the reply)."""
from __future__ import annotations

import ast
import operator as op
from dataclasses import dataclass, field
from typing import Any, Callable

from app.catalog import format_catalog_entry
from app.config import SOURCE_ALIASES, SOURCE_GROUPS, settings
from app.llm import UsageMeter
from app.textutils import fold, format_docs


@dataclass
class RequestContext:
    hits: list[dict] = field(default_factory=list)
    meter: UsageMeter = field(default_factory=UsageMeter)
    tools: dict[str, Callable[..., str]] = field(default_factory=dict)
    instructions: str = ""          # optional extra instructions (RAG_INSTRUCTIONS)
    retriever: Any = None           # the index searched by this question
    llm: Any = None
    source_route: str = ""          # source the router chose for this question (router.py)
    default_sections: set | None = None   # SOURCE_ROUTING=strict: searches stay in these
    downloads: list[dict] = field(default_factory=list)   # datasheet PDFs offered with the
                                    # answer; router._finish appends the CLI command


def _txt(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v or "")


def _calculator(expression: str = "", **_: Any) -> str:
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
           ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos, ast.Mod: op.mod,
           ast.FloorDiv: op.floordiv}

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    try:
        return str(ev(ast.parse(str(expression).replace(",", ""), mode="eval").body))
    except Exception as e:
        return f"calc error: {e}"


def source_sections(source: Any) -> set[str] | None:
    """'manual' -> {'manuales'}; 'datasheet' -> {'productos', 'hojas_tecnicas'} ...
    '' / 'all' / unknown -> None (no filter)."""
    names = [fold(x) for x in str(_txt(source)).replace(";", ",").split(",") if x.strip()]
    out: set[str] = set()
    for n in names:
        g = SOURCE_ALIASES.get(n) or SOURCE_ALIASES.get(n.rstrip("s"))
        if g in SOURCE_GROUPS:
            out |= set(SOURCE_GROUPS[g])
    return out or None


def search_queries(query: Any = "", **kw: Any) -> list[str]:
    """Production 1.0: a search tool call may carry up to MAX_QUERIES_PER_SEARCH (2)
    queries — a list in `query` / `queries`, or 'a || b'. Duplicates and empty strings
    are dropped; each query is cut to MAX_QUERY_CHARS (least privilege)."""
    raw: list[Any] = []
    for v in (query, kw.get("queries"), kw.get("q")):
        if isinstance(v, (list, tuple)):
            raw.extend(v)
        elif v:
            raw.extend(str(v).split("||"))
    out: list[str] = []
    for q in raw:
        q = " ".join(str(q or "").split())[:settings.max_query_chars]
        if q and q.lower() not in {x.lower() for x in out}:
            out.append(q)
    return out[:max(1, settings.max_queries_per_search)]


def make_tools(retriever: Any, hits: list[dict],
               ctx: "RequestContext | None" = None) -> dict[str, Callable[..., str]]:
    def _search(query: Any, sections: set[str] | None, label: str, top_k: int | None = None,
                n: int = 14, extra: Any = None) -> str:   # room for table pieces
        queries = search_queries(query, queries=extra)
        if not queries:
            return "This search tool needs query = a text (or a list of up to 2 texts)."
        if len(queries) == 1:
            docs = retriever.retrieve(queries[0], top_k=top_k, hits=hits, sections=sections)
            if not docs and sections:
                return (f"No {label} matches '{queries[0]}' (folders declared in the .env for "
                        "that source are empty or do not contain it). Try another source.")
            return format_docs(docs, n=n)
        # two topics in one call: one block per query, a chunk is shown only once
        seen: set[str] = set()
        blocks = []
        per = max(3, (n * 3) // 4)
        for q in queries:
            docs = retriever.retrieve(q, top_k=top_k, sections=sections)
            new = [d for d in docs if d["id"] not in seen]
            seen.update(d["id"] for d in new)
            hits.extend(new)
            again = len(docs) - len(new)
            body = format_docs(new, n=per) if new else (
                f"No {label} matches." if not docs else "(only chunks already shown above)")
            blocks.append(f"== results for '{q}'"
                          + (f" ({again} already shown above)" if again and new else "")
                          + f" ==\n{body}")
        return "\n\n".join(blocks)

    def semantic_search(query: Any = "", source: Any = "", **kw: Any) -> str:
        source = source or kw.get("doc_type") or kw.get("type") or kw.get("sources")
        sections = source_sections(source)
        if sections is None and ctx is not None and ctx.default_sections:
            sections = set(ctx.default_sections)     # SOURCE_ROUTING=strict
        return _search(query, sections, f"{_txt(source) or 'preferred-source'} document",
                       extra=kw.get("queries") or kw.get("q"))

    def datasheet_search(query: Any = "", **kw: Any) -> str:
        """Technical data ONLY from the datasheets folder and the product CSVs."""
        return _search(query, set(SOURCE_GROUPS["datasheet"]), "datasheet / product data",
                       extra=kw.get("queries") or kw.get("q"))

    def manual_search(query: Any = "", **kw: Any) -> str:
        """Operation / installation / maintenance ONLY from the manuals folder."""
        return _search(query, set(SOURCE_GROUPS["manual"]), "manual", extra=kw.get("queries") or kw.get("q"))

    def drawing_search(query: Any = "", **kw: Any) -> str:
        """Drawings ONLY: exploded views, drawings and product photos folders."""
        return _search(query, set(SOURCE_GROUPS["drawing"]), "drawing", top_k=3, n=3,
                       extra=kw.get("queries") or kw.get("q"))

    def manufacturer_web_search(query: str = "", site: str = "all", k: int = 4, **_: Any) -> str:
        if not settings.web_enabled:
            return "manufacturer_web_search is disabled. Answer from the local index."
        from app.manufacturer_web import get_web
        web = get_web()
        try:
            found = web.search(str(query), site=str(site or "all"), k=int(k))
        except Exception as e:
            return f"manufacturer_web_search failed: {e}. Answer from the local index."
        notes = "".join(f"(note: {n})\n" for n in web.notes.values())
        if not found:
            return notes + "No matching page on the manufacturers' websites."
        hits.extend({"id": h["url"], "source": h["url"], "text": h["text"], "kind": "web",
                     "path": ""} for h in found)
        return notes + "\n".join(
            f"[WEB {web.sites[h['site']]['name']}] {h['title']} — {h['url']}\n{h['text'][:600]}"
            for h in found)

    def list_documents(keyword: str = "", manufacturer: str = "", doc_type: str = "",
                       limit: int = 0, **kw: Any) -> str:
        keyword = _txt(keyword or kw.get("query") or kw.get("topic") or kw.get("code"))
        doc_type = _txt(doc_type or kw.get("type") or kw.get("kind"))
        manufacturer = _txt(manufacturer or kw.get("brand") or kw.get("maker"))
        try:
            limit = max(1, min(int(limit or settings.list_docs_max), 60))
        except (TypeError, ValueError):
            limit = settings.list_docs_max
        items, total = retriever.list_documents(keyword, manufacturer, doc_type, limit)
        head = (f"list_documents(keyword='{keyword}', manufacturer='{manufacturer}', "
                f"doc_type='{doc_type}') -> showing {len(items)} of {total} matching "
                f"(collection: {retriever.catalog_summary()})"
                + (" [ranked by relevance to keyword]" if keyword else ""))
        return head + "\n" + ("\n".join(format_catalog_entry(e) for e in items)
                               or "(no document matched — try fewer filters)")

    def read_document(doc: Any = "", part: Any = 1, query: str = "", **kw: Any) -> str:
        doc = doc or kw.get("document") or kw.get("file") or kw.get("doc_id") \
            or kw.get("doc_ids") or kw.get("name") or kw.get("id") or ""
        docs = [str(d) for d in doc] if isinstance(doc, (list, tuple)) else [str(doc)]
        docs = [d for d in docs if d.strip()][:2]
        if not docs:
            return "read_document needs doc = a file name (see list_documents)."
        try:
            part = int(part or kw.get("page") or 1)
        except (TypeError, ValueError):
            part = 1
        budget = settings.read_doc_max_chars // len(docs)
        return "\n\n".join(retriever.read_document(d, part=part, query=_txt(query),
                                                   max_chars=budget, hits=hits) for d in docs)

    def table_lookup(code: Any = "", columns: Any = "", table: Any = "", **kw: Any) -> str:
        code = _txt(code or kw.get("codes") or kw.get("article") or kw.get("part_number")
                    or kw.get("model") or kw.get("query"))
        columns = _txt(columns or kw.get("column") or kw.get("fields"))
        table = _txt(table or kw.get("file") or kw.get("doc"))
        return retriever.table_lookup(code, columns=columns, table=table, hits=hits)

    def product_datasheet(part: Any = "", **kw: Any) -> str:
        """Product master data + the command that generates its PDF datasheet."""
        from app.products import datasheet_cli_command, get_product_index
        q = _txt(part or kw.get("code") or kw.get("part_number") or kw.get("name")
                 or kw.get("product") or kw.get("query"))
        if not q.strip():
            return "product_datasheet needs part = a part number or product name."
        idx = get_product_index(retriever)
        one, variants, candidates = idx.resolve(q)
        # the same part number in several rows / bases is ONE datasheet (variants are
        # columns of it); only different products are listed separately
        found = [one] if one is not None else candidates
        if not found:
            return (f"No product '{q}' in the product tables (CSV / Excel of RAG_DIR_PRODUCTOS / "
                    "RAG_DIR_HOJAS_TECNICAS). Use table_lookup or semantic_search.")
        out = []
        for p in found[:3]:
            ident = p.part or p.name
            sid = f"datasheet:{ident}"
            specs = "; ".join(f"{h}: {v}" for h, v in p.spec_rows()[:25])
            pdfs = ", ".join(p.pdfs) or "none"
            hits.append({"id": sid, "source": sid, "kind": "datasheet",
                         "text": f"{p.name} {p.part} {specs}", "path": ""})
            for pdf in p.pdfs[:2]:
                hits.append({"id": pdf, "source": pdf, "kind": "document", "text": "", "path": ""})
            cmd = datasheet_cli_command(p)
            if ctx is not None and all(d["cmd"] != cmd for d in ctx.downloads):
                ctx.downloads.append({"kind": "datasheet_pdf", "cmd": cmd,
                                      "label": f"Hoja técnica PDF · {ident}"})
            if p is one and variants:
                specs += (f" || {len(variants)} more row(s) with the same part number; the "
                          "datasheet shows them as variant columns")
            out.append(f"[{sid}] {p.name} | part {p.part} | family {p.family} | "
                       f"data [{p.source_id}]\n  specs: {specs or '(none)'}\n"
                       f"  datasheet: [{sid}] | PDF datasheet: the command that generates it is "
                       f"added to the answer automatically | existing PDF datasheet: {pdfs}"
                       + (f" | photo: {p.image['rel']}" if p.image else " | no photo"))
        more = f"\n(+{len(found) - 3} more matches; be more specific)" if len(found) > 3 else ""
        return "\n".join(out) + more

    return {"semantic_search": semantic_search,
            "datasheet_search": datasheet_search, "manual_search": manual_search,
            "drawing_search": drawing_search,
            "list_documents": list_documents, "read_document": read_document,
            "table_lookup": table_lookup,
            "product_datasheet": product_datasheet,
            "manufacturer_web_search": manufacturer_web_search,
            "web_search": manufacturer_web_search,          # alias the LLM sometimes uses
            "calculator": _calculator}


def new_context(retriever: Any, instructions: str = "", llm: Any = None) -> RequestContext:
    ctx = RequestContext(instructions=instructions, retriever=retriever, llm=llm)
    ctx.tools = make_tools(retriever, ctx.hits, ctx)
    return ctx
