"""Product master data and technical datasheets (HTML).

Where the data comes from (all of it from the index catalog, so "indexed" = "exists").
Each folder is the one DECLARED IN THE .env (nothing else is read for products):

    RAG_DIR_PRODUCTOS/<familia>.csv          one CSV per family: Bombas, Válvulas ...
    RAG_DIR_PRECIOS/*.csv                    price + stock list (never printed on a datasheet)
    RAG_DIR_FOTOS/<NumParte>.jpg             product photo
    RAG_DIR_EXPLOSIONADOS/<NumParte>_explosionado.png   exploded view (optional)
    RAG_DIR_HOJAS_TECNICAS/<familia>/<NumParte>_<Nombre>.pdf   datasheets that already exist

Everything is joined by PART NUMBER (preferred) or PRODUCT NAME, the two columns every
CSV shares. Column names are recognised with settings.part_col_aliases / name_col_aliases.

    idx = get_product_index(retriever)
    prods = idx.find("123456")              # or a product name
    html = build_datasheet_html(prods[0])   # self-contained page (images embedded)

No LLM call anywhere in this file: a datasheet must contain exactly what the CSV says.
"""
from __future__ import annotations

import base64
import functools
import html
import io
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from app.config import IMAGE_SECTIONS, settings
from app.textutils import fold, norm

log = logging.getLogger("rag.products")
_SEP = re.compile(r"[\s_\-.()]+")


# =========================================================================================
# Column detection
# =========================================================================================
def _ckey(s: str) -> str:
    return fold(s).replace(" ", "")


def find_col(headers: list[str], aliases: tuple[str, ...], exclude: set[int] = frozenset()) -> int | None:
    """Exact header match first (any alias), then 'header contains alias' in alias order."""
    keys = [_ckey(h) for h in headers]
    for a in aliases:
        ak = _ckey(a)
        for i, k in enumerate(keys):
            if i not in exclude and k == ak:
                return i
    for a in aliases:
        ak = _ckey(a)
        for i, k in enumerate(keys):
            if i not in exclude and ak and ak in k:
                return i
    return None


@functools.lru_cache(maxsize=1)
def _hidden_keys(words: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_ckey(w) for w in words)


@functools.lru_cache(maxsize=4096)
def _is_hidden(header: str) -> bool:
    """Cached: the same few headers repeat on every row of every family CSV."""
    k = _ckey(header)
    return any(w in k for w in _hidden_keys(tuple(settings.datasheet_hide_cols)))


def _family_label(rel: str) -> str:
    stem = Path(rel).stem
    return re.sub(r"[_\-]+", " ", stem).strip().capitalize()


# =========================================================================================
# Data model
# =========================================================================================
@dataclass
class Product:
    part: str
    name: str
    family: str
    rel: str                                   # family CSV (relative path)
    row_number: int                            # 1-based, as in the CSV (after the header)
    fields: list[tuple[str, str]]              # (header, value) in CSV column order
    source_id: str = ""                        # chunk id of that row, for citations
    image: dict | None = None                  # catalog entry of the product photo
    exploded: list[dict] = field(default_factory=list)
    pdfs: list[str] = field(default_factory=list)       # existing datasheet PDFs (rel)
    stock: list[tuple[str, str]] = field(default_factory=list)   # row of the price/stock CSV

    @property
    def key(self) -> str:
        return norm(self.part) or norm(self.name)

    def spec_rows(self) -> list[tuple[str, str]]:
        """The rows printed on the datasheet: everything except identity, price, stock,
        image and internal columns, and empty cells."""
        skip = {h for h in (self.part_header, self.name_header, self.family_header) if h}
        return [(h, v) for h, v in self.fields
                if v and h not in skip and not _is_hidden(h)]

    part_header: str = ""
    name_header: str = ""
    family_header: str = ""

    def summary(self) -> dict:
        return {"part": self.part, "name": self.name, "family": self.family,
                "source": self.rel, "row": self.row_number, "source_id": self.source_id,
                "specs": dict(self.spec_rows()),
                "image": self.image["rel"] if self.image else None,
                "exploded": [e["rel"] for e in self.exploded], "pdfs": list(self.pdfs),
                "datasheet_cmd": datasheet_cli_command(self)}


# =========================================================================================
# Image matching
# =========================================================================================
def plain_rel(rel: str) -> str:
    """'@base/productos/x.csv' -> 'productos/x.csv': a multi-RAG namespace (base slug, e.g.
    'precios-imp') must never trigger the folder/name keyword rules below."""
    return rel.split("/", 1)[1] if rel.startswith("@") and "/" in rel else rel


def _is_exploded(rel: str) -> bool:
    section = settings.section_of(rel)
    if section == "explosionados":
        return True
    if section == "fotos":
        return False
    f = fold(plain_rel(rel))
    return any(fold(w) in f for w in settings.exploded_words)


def _image_keys(rel: str) -> tuple[set[str], str]:
    """(part-number keys, name key) of an image file.
    'drawings/fotos/BP-200_explosionado.jpg' -> ({'bp200', 'bp', '200'}, 'bp200')"""
    stem = Path(rel).stem
    clean = stem
    for w in settings.exploded_words:
        clean = re.sub(rf"{re.escape(w)}\w*", " ", clean, flags=re.I)
    clean = re.sub(r"\b(vista|view|foto|photo|img|imagen|image)\b", " ", clean, flags=re.I)
    clean = re.sub(r"[\s_\-.]+$", "", re.sub(r"^[\s_\-.]+", "", clean))
    tokens = [norm(t) for t in _SEP.split(clean) if t]
    keys = {norm(clean)} | {t for t in tokens if len(t) >= 3}
    # also the first two tokens glued ("BP-200_bomba" -> 'bp200')
    if len(tokens) >= 2:
        keys.add(tokens[0] + tokens[1])
    return {k for k in keys if k}, norm(clean)


# =========================================================================================
# Product index
# =========================================================================================
class ProductIndex:
    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever
        self._lock = threading.Lock()
        self._built_for: tuple[int, float] = (0, 0.0)
        self._t = 0.0
        self.products: list[Product] = []
        self.by_part: dict[str, list[Product]] = {}
        self.by_name: dict[str, list[Product]] = {}
        self.problems: list[str] = []
        self.unmatched_images: list[str] = []
        self.stock_only: list[str] = []

    # ------------------------------------------------------------------ build
    def _fresh(self) -> bool:
        snap = self.retriever.snapshot
        return (self._built_for == (id(snap), getattr(snap, "built_at", 0.0))
                and time.time() - self._t < settings.product_cache_secs)

    def ensure(self) -> "ProductIndex":
        if not self._fresh():
            with self._lock:
                if not self._fresh():
                    self._build()
        return self

    def _table(self, e: dict) -> tuple[list[str], list[list[str]]] | None:
        try:
            headers, rows, _ = self.retriever._table_rows(e["path"])
            return headers, rows
        except Exception as ex:
            self.problems.append(f"{e['rel']}: no se pudo leer ({ex})")
            return None

    @staticmethod
    def _is_price_table(rel: str) -> bool:
        return settings.section_of(rel) == "precios"

    @staticmethod
    def _is_family_table(rel: str) -> bool:
        sections = tuple(settings.product_table_sections) or ("productos",)
        return settings.section_of(rel) in sections

    def _build(self) -> None:
        snap = self.retriever.snapshot
        catalog = snap.catalog
        self.problems, products = [], []
        stock_rows: dict[str, list[tuple[str, str]]] = {}
        stock_names: dict[str, str] = {}
        step = max(1, int(settings.csv_rows_per_chunk))

        for e in sorted(catalog.values(), key=lambda x: x["rel"]):
            # only the CSVs of RAG_DIR_PRODUCTOS and RAG_DIR_PRECIOS: an import / invoice /
            # stock CSV elsewhere must never become a product family
            if e["type"] != "table" or not (self._is_family_table(e["rel"])
                                            or self._is_price_table(e["rel"])):
                continue
            got = self._table(e)
            if not got:
                continue
            headers, rows = got
            pcol = find_col(headers, settings.part_col_aliases)
            ncol = find_col(headers, settings.name_col_aliases, exclude={pcol} if pcol is not None else set())
            if pcol is None and ncol is None:
                self.problems.append(f"{e['rel']}: sin columna de número de parte ni de nombre "
                                     f"(columnas: {', '.join(headers)[:150]})")
                continue
            if self._is_price_table(e["rel"]):
                for r in rows:
                    k = norm(r[pcol]) if pcol is not None else ""
                    k = k or (norm(r[ncol]) if ncol is not None else "")
                    if k:
                        stock_rows[k] = [(h, v) for h, v in zip(headers, r) if v]
                        stock_names[k] = (r[pcol] if pcol is not None else "") or r[ncol]
                continue
            fcol = find_col(headers, settings.family_col_aliases,
                            exclude={c for c in (pcol, ncol) if c is not None})
            if pcol is None:
                self.problems.append(f"{e['rel']}: sin columna de número de parte; se une sólo por nombre")
            for i, r in enumerate(rows):
                part = r[pcol].strip() if pcol is not None else ""
                name = r[ncol].strip() if ncol is not None else ""
                if not (part or name):
                    continue
                products.append(Product(
                    part=part, name=name,
                    family=(r[fcol].strip() if fcol is not None and r[fcol].strip() else _family_label(e["rel"])),
                    rel=e["rel"], row_number=i + 1,
                    fields=list(zip(headers, r)),
                    source_id=f"{e['rel']}_row{(i // step) * step + 1}",
                    part_header=headers[pcol] if pcol is not None else "",
                    name_header=headers[ncol] if ncol is not None else "",
                    family_header=headers[fcol] if fcol is not None else ""))

        by_part: dict[str, list[Product]] = {}
        by_name: dict[str, list[Product]] = {}
        for p in products:
            if norm(p.part):
                by_part.setdefault(norm(p.part), []).append(p)
            if norm(p.name):
                by_name.setdefault(norm(p.name), []).append(p)
        for k, ps in by_part.items():
            if len(ps) > 1:
                self.problems.append(f"número de parte {ps[0].part} repetido en: "
                                     + ", ".join(f"{p.rel} fila {p.row_number}" for p in ps))

        # images: product photos and exploded views
        used: set[str] = set()
        images = [e for e in catalog.values() if e["type"] == "drawing"
                  and settings.section_of(e["rel"]) in IMAGE_SECTIONS]
        for e in images:
            pkeys, nkey = _image_keys(e["rel"])
            targets = [p for k in pkeys for p in by_part.get(k, [])] or by_name.get(nkey, [])
            for prod in {id(t): t for t in targets}.values():
                if _is_exploded(e["rel"]):
                    prod.exploded.append(e)
                elif prod.image is None:
                    prod.image = e
                else:
                    self.problems.append(f"{prod.part or prod.name}: dos fotos "
                                         f"({prod.image['rel']}, {e['rel']}); se usa la primera")
                used.add(e["rel"])
        self.unmatched_images = sorted(e["rel"] for e in images if e["rel"] not in used)

        # existing PDF datasheets
        for e in catalog.values():
            if e.get("ext", Path(e["rel"]).suffix.lower()) != ".pdf":
                continue
            # only RAG_DIR_HOJAS_TECNICAS: a "ficha técnica" inside an import folder or a
            # manual is NOT the product's datasheet
            if settings.section_of(e["rel"]) != "hojas_tecnicas":
                continue
            stem = Path(e["rel"]).stem
            keys = set(e.get("codes") or ()) | {norm(t) for t in _SEP.split(stem) if t} | {norm(stem)}
            for k in keys:
                for p in by_part.get(k, []):
                    if e["rel"] not in p.pdfs:
                        p.pdfs.append(e["rel"])

        for p in products:
            p.stock = stock_rows.get(norm(p.part)) or stock_rows.get(norm(p.name)) or []
        known = set(by_part) | set(by_name)
        self.stock_only = sorted(stock_names[k] for k in stock_rows if k not in known)

        self.products, self.by_part, self.by_name = products, by_part, by_name
        self._built_for = (id(snap), getattr(snap, "built_at", 0.0))
        self._t = time.time()
        log.info("products: %d in %d family table(s), %d with photo, %d with PDF",
                 len(products), len({p.rel for p in products}),
                 sum(p.image is not None for p in products), sum(bool(p.pdfs) for p in products))

    # ------------------------------------------------------------------ lookup
    def find(self, query: str, limit: int = 10) -> list[Product]:
        """Exact part number > exact name > name/part containing the query."""
        self.ensure()
        q = norm(query)
        if not q:
            return []
        if q in self.by_part:
            return self.by_part[q][:limit]
        if q in self.by_name:
            return self.by_name[q][:limit]
        if len(q) < 3:
            return []
        hits = [p for p in self.products if q in norm(p.part) or q in norm(p.name)]
        hits.sort(key=lambda p: (len(norm(p.name)), p.part))
        return hits[:limit]

    def resolve(self, query: str) -> tuple[Product | None, list[Product], list[Product]]:
        """The product of ONE datasheet: (product, variants, candidates).

        * A part number found in several rows is still ONE datasheet: rows with the same
          data (the same CSV / Excel in two knowledge bases, a .csv next to its .xlsx, a
          repeated row) collapse into one; rows with different data become variants
          (one column each in the datasheet), as the original pdfdatagen did.
        * Only a query that matches DIFFERENT products (a name shared by several part
          numbers, a partial code) is ambiguous: product None + the candidates."""
        q = norm(query)
        found = self.find(query, limit=50)
        if not found or not q:
            return None, [], found
        by_part = [p for p in found if norm(p.part) == q]
        by_name = [p for p in found if norm(p.name) == q]
        if by_part:
            group = by_part
        elif by_name and len({norm(p.part) for p in by_name}) <= 1:
            group = by_name                       # one product, only known by its name
        elif len(found) == 1:
            group = found
        else:
            return None, [], _unique(found)
        group = _unique(group)
        return group[0], group[1:], []

    def report(self) -> dict:
        """Data-quality report: run it after copying new files into data/."""
        self.ensure()
        ps = self.products
        return {
            "products": len(ps),
            "families": sorted({f"{p.family} ({p.rel})" for p in ps}),
            "with_photo": sum(p.image is not None for p in ps),
            "with_exploded": sum(bool(p.exploded) for p in ps),
            "with_pdf": sum(bool(p.pdfs) for p in ps),
            "with_stock_row": sum(bool(p.stock) for p in ps),
            "missing_photo": [p.part or p.name for p in ps if p.image is None][:200],
            "missing_stock_row": [p.part or p.name for p in ps if not p.stock][:200],
            "stock_rows_without_product": self.stock_only[:200],
            "images_without_product": self.unmatched_images[:200],
            "problems": self.problems[:200],
        }


def _data_key(p: Product) -> tuple:
    return (norm(p.part), norm(p.name), tuple((h, v) for h, v in p.fields if v))


def _unique(products: list[Product]) -> list[Product]:
    """Drop rows that repeat the same product data; the kept row borrows the photo,
    exploded views, PDFs and stock row of its copies. The default base (rel without '@')
    and rows with a photo go first."""
    order = sorted(products, key=lambda p: (p.rel.startswith("@"), p.image is None, p.rel,
                                            p.row_number))
    kept: dict[tuple, Product] = {}
    for p in order:
        k = _data_key(p)
        first = kept.get(k)
        if first is None:
            kept[k] = p
            continue
        if first.image is None and p.image is not None:
            first.image = p.image
        for x in p.exploded:
            if all(x["rel"] != y["rel"] for y in first.exploded):
                first.exploded.append(x)
        first.pdfs += [x for x in p.pdfs if x not in first.pdfs]
        if not first.stock and p.stock:
            first.stock = p.stock
    return list(kept.values())


def variant_title(p: Product, all_same_part: bool) -> str:
    """Column title of a variant: the part number, plus where it comes from when several
    rows share the same part number."""
    ident = p.part or p.name
    return f"{ident} ({Path(plain_rel(p.rel)).stem}, fila {p.row_number})" if all_same_part else ident


def get_product_index(retriever: Any) -> ProductIndex:
    """One ProductIndex per retriever (rebuilds itself when the index snapshot changes)."""
    idx = getattr(retriever, "_product_index", None)
    if idx is None:
        idx = ProductIndex(retriever)
        try:
            retriever._product_index = idx
        except AttributeError:
            pass
    return idx


# =========================================================================================
# HTML datasheet
# =========================================================================================
@functools.lru_cache(maxsize=256)
def _data_uri(path: str, max_side: int, mtime: float) -> str:
    """Image file -> data: URI (resized; JPEG for photos, PNG if it has transparency)."""
    from PIL import Image  # type: ignore
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        im.draft("RGB", (max_side, max_side))
        alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
        im = im.convert("RGBA" if alpha else "RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        if alpha:
            im.save(buf, format="PNG", optimize=True)
            mime = "image/png"
        else:
            im.save(buf, format="JPEG", quality=85, optimize=True)
            mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _img_src(entry_or_path: dict | str | None, embed: bool) -> str:
    if not entry_or_path:
        return ""
    path = entry_or_path if isinstance(entry_or_path, str) else entry_or_path["path"]
    if embed or isinstance(entry_or_path, str):
        try:
            return _data_uri(str(path), settings.datasheet_image_max_side, os.path.getmtime(path))
        except Exception as e:
            log.warning("datasheet: cannot read image %s: %s", path, e)
            return ""
    try:                                   # --no-embed: link the local file
        return Path(path).resolve().as_uri()
    except ValueError:
        return str(path)


_SHEET_CSS = """
@page { size: letter; margin: 14mm 16mm; }
* { box-sizing: border-box; }
body { margin: 0; background: #e9ecef; color: #1d1d1b; font: 10.5pt/1.45 Calibri, Carlito, "Segoe UI", Arial, sans-serif; }
.page { width: 215.9mm; min-height: 279.4mm; margin: 12px auto; background: #fff; padding: 14mm 16mm 18mm; position: relative; box-shadow: 0 2px 12px rgba(0,0,0,.15); }
.top { display: flex; justify-content: flex-end; }
.top img { height: 11mm; }
h1 { text-align: center; font-size: 15pt; margin: 6mm 0 2mm; }
.pn { text-align: center; font-weight: 700; font-size: 11.5pt; margin: 0 0 9mm; }
.pn span { font-weight: 400; color: #555; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) 62mm; gap: 10mm; align-items: start; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; }
th, td { border: 1px solid #1d1d1b; padding: 1.2mm 2mm; text-align: left; vertical-align: top; }
th { background: #f2f2f2; }
td:first-child { width: 45%; }
.imgs { display: grid; gap: 8mm; }
.imgbox { border: 1px solid #c8ccd0; border-radius: 4mm; height: 52mm; display: grid; place-items: center; overflow: hidden; background: #fff; }
.imgbox img { max-width: 100%; max-height: 100%; object-fit: contain; }
.imgbox.none { border-style: dashed; color: #8a9097; font-size: 9pt; }
.cap { text-align: center; font-size: 8.5pt; color: #555; margin-top: 1.5mm; }
.foot { position: absolute; left: 16mm; right: 16mm; bottom: 8mm; border-top: 1px solid #c8ccd0; padding-top: 2mm; font-size: 7.5pt; color: #6b7178; display: flex; justify-content: space-between; gap: 6mm; }
.foot .red { color: #e30613; font-weight: 700; }
@media print { body { background: #fff; } .page { margin: 0; box-shadow: none; width: auto; min-height: auto; padding: 0; } .foot { position: fixed; left: 0; right: 0; bottom: 0; } }
@media (max-width: 230mm) { .page { width: auto; min-height: 0; margin: 0; padding: 16px 16px 60px; } .grid { grid-template-columns: 1fr; } }
"""


def build_datasheet_html(p: Product, embed: bool = True, show_sources: bool = True,
                         variants: list[Product] | None = None) -> str:
    """The datasheet as one HTML page laid out like 'Hoja técnica ejemplo.docx' (Letter):
    logo top-right, name + part number centred, characteristics table on the left, product
    image on the right, exploded view below it only if it exists.
    embed=True puts every image inside the file (it can be e-mailed or saved alone)."""
    e = html.escape
    logo = (_img_src(settings.datasheet_logo, True)
            if settings.datasheet_logo and Path(settings.datasheet_logo).exists() else "")
    variants = [v for v in (variants or []) if v is not p]
    if variants:
        allp = [p] + variants
        same = len({norm(x.part or x.name) for x in allp}) == 1
        cols = [dict(x.spec_rows()) for x in allp]
        keys: list[str] = []
        for c in cols:
            keys += [k for k in c if k not in keys]
        table = ("<table><thead><tr><th>Característica</th>"
                 + "".join(f"<th>{e(variant_title(x, same))}</th>" for x in allp)
                 + "</tr></thead><tbody>"
                 + "".join(f"<tr><td>{e(k)}</td>" + "".join(f"<td>{e(c.get(k) or '-')}</td>"
                                                           for c in cols) + "</tr>"
                           for k in keys)
                 + "</tbody></table>") if keys else "<p>Sin características registradas.</p>"
    else:
        rows = p.spec_rows()
        table = ("<table><thead><tr><th>Característica</th><th>Valor</th></tr></thead><tbody>"
                 + "".join(f"<tr><td>{e(h)}</td><td>{e(v)}</td></tr>" for h, v in rows)
                 + "</tbody></table>") if rows else "<p>Sin características registradas.</p>"

    def box(entry: dict | None, label: str) -> str:
        src = _img_src(entry, embed)
        if not src:
            return f'<div><div class="imgbox none">{e(label)} no disponible</div></div>'
        return (f'<div><div class="imgbox"><img src="{src}" alt="{e(label)} {e(p.name)}"></div>'
                f'<div class="cap">{e(label)}</div></div>')

    imgs = box(p.image, "Imagen del producto")
    for x in p.exploded[:2]:                 # "imagen explosionada, si existe"
        imgs += box(x, "Vista explosionada")
    src_note = (f"Datos: {e(p.rel)}, fila {p.row_number}"
                + (f" · Foto: {e(p.image['rel'])}" if p.image else "")) if show_sources else ""
    title = f"{p.name} · {p.part}" if p.name and p.part else (p.name or p.part)
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hoja técnica {e(title)}</title>
<meta name="part-number" content="{e(p.part)}">
<style>{_SHEET_CSS}</style></head>
<body><div class="page">
  <div class="top">{f'<img src="{logo}" alt="Relant">' if logo else '<b>Relant</b>'}</div>
  <h1>{e(p.name or p.part)}</h1>
  <p class="pn">{e(p.part) if p.part else ''}{f' <span>· {e(p.family)}</span>' if p.family else ''}</p>
  <div class="grid">
    <div>{table}</div>
    <div class="imgs">{imgs}</div>
  </div>
  <div class="foot"><span><span class="red">Relant</span> · Hoja técnica generada el {date.today():%d/%m/%Y}</span><span>{src_note}</span></div>
</div></body></html>"""


# =========================================================================================
# PDF datasheet (app/pdfdatagen.py, xhtml2pdf)
# =========================================================================================
_pdf_gen_lock = threading.Lock()
_pdf_gen: tuple[str, float, Any] | None = None       # (config path, mtime, generator)


def _pdf_generator() -> Any:
    """One DatasheetGenerator, rebuilt when RAG_DATASHEET_PDF_CONFIG changes on disk.
    Without that variable: built-in template + default colours."""
    global _pdf_gen
    from app.pdfdatagen import DatasheetGenerator
    cfg = settings.datasheet_pdf_config or ""
    mtime = os.path.getmtime(cfg) if cfg and os.path.isfile(cfg) else 0.0
    with _pdf_gen_lock:
        if _pdf_gen is None or _pdf_gen[:2] != (cfg, mtime):
            if cfg and mtime:
                gen = DatasheetGenerator.desde_archivo_json(
                    cfg, crear_carpetas=False, base_dir=str(Path(cfg).parent))
            else:
                if cfg:
                    log.warning("RAG_DATASHEET_PDF_CONFIG no existe: %s (se usa la plantilla "
                                "por defecto)", cfg)
                gen = DatasheetGenerator(crear_carpetas=False)
            _pdf_gen = (cfg, mtime, gen)
        return _pdf_gen[2]


def datasheet_pdf_filename(p: Product) -> str:
    ident = re.sub(r"[^\w\-.]+", "_", p.part or p.name).strip("_") or f"fila{p.row_number}"
    fam = re.sub(r"[^\w\-]+", "_", fold(p.family).upper()).strip("_")
    return f"Ficha_Tecnica_{fam + '_' if fam else ''}{ident}.pdf"


def build_datasheet_pdf(p: Product, variants: list[Product] | None = None) -> bytes:
    """The datasheet as a PDF (xhtml2pdf via app/pdfdatagen.py): the same rows as the HTML
    sheet (spec_rows: no price, stock or internal columns), photo + first exploded view
    and the logo of RAG_DATASHEET_LOGO. variants: other rows of the same part number, one
    column each. No LLM call."""
    gen = _pdf_generator()
    logo = settings.datasheet_logo if settings.datasheet_logo and Path(settings.datasheet_logo).exists() else None
    allp = [p] + [v for v in (variants or []) if v is not p]
    same = len({norm(x.part or x.name) for x in allp}) == 1
    titles = [variant_title(x, same) for x in allp] if len(allp) > 1 else None
    return gen.generar_pdf_bytes(
        modelo=p.part or p.name, nombre=p.name or p.part, categoria=p.family or "",
        variantes=[x.spec_rows() for x in allp], titulos=titles,
        ruta_foto=p.image["path"] if p.image else None,
        ruta_vista_explosionada=p.exploded[0]["path"] if p.exploded else None,
        ruta_logo=logo)


def datasheet_cli_command(p: Product) -> str:
    """The console command that writes this product's PDF datasheet."""
    ident = p.part or p.name
    arg = f'"{ident}"' if re.search(r"[\s&|<>^]", ident) else ident
    return f"python -m app.cli datasheet {arg} --pdf"


def write_all_datasheets(retriever: Any, out_dir: str | Path | None = None,
                         embed: bool = True) -> dict:
    """Batch: one HTML per product into index/hojas_generadas/<familia>/<NumParte>.html."""
    idx = get_product_index(retriever).ensure()
    out = Path(out_dir or settings.datasheet_out_dir)
    n = 0
    for p in idx.products:
        name = re.sub(r"[^\w\-.]+", "_", p.part or p.name).strip("_") or f"row{p.row_number}"
        d = out / re.sub(r"[^\w\-]+", "_", p.family).strip("_")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.html").write_text(build_datasheet_html(p, embed=embed), encoding="utf-8")
        n += 1
    return {"written": n, "out_dir": str(out)}
