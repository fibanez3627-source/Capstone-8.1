"""Document catalog (one line per file, built from the index, no API calls)"""
from __future__ import annotations

import re
from pathlib import Path

from app.config import (TABLE_EXTS, IMAGE_EXTS, MANUFACTURERS, OTHER_SECTION, SECTION_DOC_TYPE,
                        SECTION_LABEL, TEXT_EXTS, settings)
from app.textutils import fold, raw_codes

DOC_TYPE_LABEL = {"manual": "manual", "datasheet": "datasheet", "table": "table (CSV/Excel)",
                  "drawing": "drawing", "catalogue": "catalogue",
                  "spare_parts": "spare-parts list", "web": "web page",
                  "document": "document"}
_TYPE_NAME_WORDS = (
    ("datasheet", ("datasheet", "data sheet", "datenblatt", "ficha tecnica", "hoja de datos",
                   "hoja tecnica", "hojas tecnicas", "fichas tecnicas", "product data", "tds")),
    ("manual", ("manual", "instructions", "anleitung", "instructivo", "handbuch", "bda")),
    ("spare_parts", ("spare parts", "spare part", "ersatzteil", "refacciones", "repuestos",
                     "parts list")),
    ("catalogue", ("catalog", "catalogue", "katalog", "catalogo", "brochure", "prospekt",
                   "folleto", "price list", "preisliste", "lista de precios")),
)
_TYPE_TEXT_WORDS = (
    ("manual", ("operating instructions", "instruction manual", "instructions for use",
                "user manual", "operating manual", "original instructions",
                "betriebsanleitung", "bedienungsanleitung", "gebrauchsanweisung",
                "manual de usuario", "manual de operacion", "manual de instrucciones")),
    ("datasheet", ("datasheet", "data sheet", "datenblatt", "ficha tecnica", "hoja de datos")),
    ("catalogue", ("catalogue", "catalog", "katalog", "catalogo", "price list", "preisliste")),
    ("spare_parts", ("spare parts list", "ersatzteilliste", "lista de refacciones",
                     "parts list")),
)
_TYPE_ALIASES = {
    "manual": "manual", "manuals": "manual", "instructions": "manual", "instructivo": "manual",
    "datasheet": "datasheet", "datasheets": "datasheet", "data sheet": "datasheet",
    "ficha": "datasheet", "ficha tecnica": "datasheet", "fichas tecnicas": "datasheet",
    "hoja de datos": "datasheet", "tds": "datasheet",
    "table": "table", "tables": "table", "csv": "table", "excel": "table", "xlsx": "table", "tabla": "table",
    "spreadsheet": "table", "stock": "table",
    "drawing": "drawing", "drawings": "drawing", "dibujo": "drawing", "image": "drawing",
    "exploded view": "drawing", "despiece": "drawing", "plano": "drawing",
    "catalogue": "catalogue", "catalog": "catalogue", "catalogo": "catalogue",
    "brochure": "catalogue", "price list": "catalogue",
    "spare parts": "spare_parts", "spare_parts": "spare_parts", "parts list": "spare_parts",
    "refacciones": "spare_parts", "repuestos": "spare_parts",
    "web": "web", "web page": "web", "document": "document", "pdf": "document",
}


def canon_type(s: str) -> str:
    f = fold(s).replace("_", " ")
    if not f:
        return ""
    if f in _TYPE_ALIASES:
        return _TYPE_ALIASES[f]
    for k, v in _TYPE_ALIASES.items():
        if k in f:
            return v
    return f


def _guess_manufacturer(rel: str, head: str) -> str:
    name = " " + fold(rel).replace("ue", "u") + " "
    text = " " + fold(head[:2500]).replace("ue", "u") + " "
    for where in (name, text):
        for disp, keys in MANUFACTURERS:
            for k in keys:
                if (f" {k} " in where) if len(k) < 5 else (k in where):
                    return disp
    # Fallback: the folder name. Only manuales/<fabricante>/, catalogos/<fabricante>/ (the
    # folders declared in the .env as RAG_DIR_MANUALES / RAG_DIR_CATALOGOS) or a folder
    # outside the declared structure name a manufacturer; family folders do not.
    section, inner = settings.section_root(rel)
    if section not in ("manuales", "catalogos", OTHER_SECTION):
        return ""
    folders = inner.split("/")[:-1]
    return folders[0] if folders else ""


def _guess_doc_type(ext: str, rel: str, head: str) -> str:
    """The folder declared in the .env decides the type (a PDF in RAG_DIR_MANUALES is a
    manual even if its name says 'ficha técnica'). Names / first page are only used for
    files outside every declared folder."""
    if ext in IMAGE_EXTS:
        return "drawing"
    if ext in TABLE_EXTS:
        return "table"
    section = settings.section_of(rel)
    if section in SECTION_DOC_TYPE:
        return SECTION_DOC_TYPE[section]
    name = " " + fold(rel) + " "
    for t, words in _TYPE_NAME_WORDS:
        if t == "datasheet":          # only RAG_DIR_HOJAS_TECNICAS holds datasheets
            continue
        if any((f" {w} " in name) if len(w) <= 3 else (w in name) for w in words):
            return t
    if ext in TEXT_EXTS and re.search(r"https?://", head[:400]):
        return "web"
    text = fold(head[:3000])
    for t, words in _TYPE_TEXT_WORDS:
        if t == "datasheet":
            continue
        if any(w in text for w in words):
            return t
    return "document"


def _guess_title(kind: str, rel: str, chunks: list[dict]) -> str:
    stem = Path(rel).stem
    if not chunks:
        return stem
    first = chunks[0]["text"]
    if kind == "drawing":
        m = re.search(r"(?im)^\W*(?:1\.\s*)?product[^:\n]*:\s*(.+)$", first)
        prod = m.group(1).strip(" *") if m else ""
        return f"{stem} — {prod[:90]}" if prod else stem
    if kind == "table":
        m = re.search(r"(?m)^Title: (.+)$", first)
        return m.group(1).strip()[:110] if m else stem
    for line in first.splitlines():
        t = " ".join(line.split())
        if (len(t) >= 8 and sum(ch.isalpha() for ch in t) >= 5
                and not re.match(r"(?i)^(page|seite|pagina|página)\s*\d", t)):
            return t[:110]
    return stem


def build_catalog_entry(path: str, rel: str, chunks: list[dict]) -> dict:
    ext = Path(path).suffix.lower()
    head = "\n".join(c["text"] for c in chunks[:3])
    kind = _guess_doc_type(ext, rel, head)
    codes = raw_codes("\n".join(c["text"] for c in chunks) + " " + Path(rel).stem)
    section = settings.section_of(rel)
    entry = {"rel": rel, "path": path, "ext": ext, "type": kind, "section": section,
             "manufacturer": _guess_manufacturer(rel, head),
             "title": _guess_title(kind, rel, chunks), "codes": set(codes),
             "codes_show": list(codes.values())[:10], "n_codes": len(codes),
             "n_chunks": len(chunks), "snippet": "", "columns": "", "n_rows": 0}
    if kind == "table" and chunks:
        m = re.search(r"(?m)^Columns: (.+)$", chunks[0]["text"])
        entry["columns"] = m.group(1).strip() if m else ""
        m = re.search(r"(?m)^Number of rows: (\d+)", chunks[0]["text"])
        entry["n_rows"] = int(m.group(1)) if m else max(0, len(chunks) - 1)
    elif kind != "drawing" and chunks:
        entry["snippet"] = " ".join(chunks[0]["text"].split())[:180]
    return entry


def format_catalog_entry(e: dict) -> str:
    line = (f"- {e['rel']} | {DOC_TYPE_LABEL.get(e['type'], e['type'])} | "
            f"source: {SECTION_LABEL.get(e.get('section', OTHER_SECTION), '?')} | "
            f"{e['manufacturer'] or '?'} | {e['title'][:100]}")
    if e["type"] == "table":
        line += f" | {e['n_rows']} rows | columns: {e['columns'][:160]}"
    if e["codes_show"]:
        more = f" (+{e['n_codes'] - len(e['codes_show'])})" if e["n_codes"] > len(e["codes_show"]) else ""
        line += f"\n    codes: {', '.join(e['codes_show'])}{more}"
    if e["snippet"]:
        line += f"\n    {e['snippet'][:160]}"
    return line


def catalog_entry_public(e: dict) -> dict:
    """JSON-safe view of a catalog entry for the API (sets -> lists, no server paths)."""
    rel = e["rel"]
    kb = rel[1:].split("/", 1)[0] if rel.startswith("@") else ""
    return {"rel": rel, "kb": kb, "type": e["type"], "section": e.get("section", OTHER_SECTION),
            "type_label": DOC_TYPE_LABEL.get(e["type"], e["type"]),
            "manufacturer": e["manufacturer"], "title": e["title"],
            "codes": e["codes_show"], "n_codes": e["n_codes"], "n_chunks": e["n_chunks"],
            "n_rows": e["n_rows"], "columns": e["columns"], "snippet": e["snippet"]}
