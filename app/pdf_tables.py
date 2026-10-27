"""Table-aware PDF chunking for both RAG pipelines (checkpoint 5.1 and app/retriever.py).

Why: pypdf.extract_text() and pymupdf get_text() flatten tables into loose lines, and the
character splitter then cuts a table in the middle, so only the first chunk keeps the
column headers. Retrieval returns one fragment and the LLM answers with part of the table.

What this does, per PDF page:
  1. pymupdf page.find_tables() detects the tables (ruled and, with a fallback, unruled).
  2. Every table becomes its own chunk(s) in Markdown, kind="table". Long tables are split
     every ROWS_PER_TABLE_CHUNK rows and EVERY piece repeats the header row, the document
     title and the page, so any piece found by BM25/vector search is self-explanatory.
     A table that continues on the next page without a header reuses the previous header.
  3. The text OUTSIDE the tables goes through the normal splitter, kind="text", with the
     page number in the chunk.
Every chunk carries "page" and "table_id", so a retriever can pull ALL the pieces of a
table once one piece matches (see expand_table_hits).

pip install pymupdf
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

ROWS_PER_TABLE_CHUNK = 25        # rows per table chunk (header repeated in each)
MAX_TABLE_CHUNK_CHARS = 3500     # safety cap for very wide tables


def _clean(cell) -> str:
    return " ".join(str(cell or "").split()).replace("|", "/")


def _md(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _looks_like_header(row: list[str]) -> bool:
    filled = [c for c in row if c]
    if not filled:
        return False
    numeric = sum(1 for c in filled if re.fullmatch(r"[\d.,/\-–x×± ]+(mm|bar|kg|l|ml|°c)?", c.lower()))
    return numeric <= len(filled) // 3


_SECTION_RE = re.compile(r"^\d+(\.\d+)+\.?\s+\S")


def _page_lines(page) -> list[tuple[float, float, str]]:
    """(y0, y1, text) of every text line on the page."""
    lines: dict[tuple, list] = {}
    for w in page.get_text("words"):
        lines.setdefault((w[5], w[6]), []).append(w)
    out = []
    for ws in lines.values():
        ws = sorted(ws, key=lambda x: x[0])
        out.append((min(x[1] for x in ws), max(x[3] for x in ws), " ".join(x[4] for x in ws)))
    return sorted(out)


def _caption_above(lines, bbox, reach: float = 160, carry: str = "") -> list[str]:
    """The section heading ('8.4.2 Adhesive version') and the line right above a table,
    so a table chunk can be found by its heading and told apart from similar tables.
    No heading on the page above the table -> the last heading of the previous pages
    (`carry`): a list that continues on the next page keeps its section number."""
    top = bbox[1]
    sec = [l for l in lines if l[1] <= top + 1 and _SECTION_RE.match(l[2])]
    out = [sec[-1][2]] if sec else ([carry] if carry else [])
    above = [l for l in lines if l[1] <= top + 1 and top - l[1] < reach]
    if not above:
        return out
    nearest = above[-1][2]
    if nearest not in out and top - above[-1][1] < 40:
        out.append(nearest)
    return out


_PAGE_WORD_RE = re.compile(r"^(?:page|p[aá]gina|seite|p\.|pag\.)\s*(\d{1,4})(?:\s*(?:of|de|von|/)\s*\d+)?$", re.I)


def _printed_page(page) -> int | None:
    """Page number PRINTED on the page (header/footer), which is what users quote.
    It often differs from the PDF page index (covers, tables of contents...).
    Accepts a number alone on a line, 'Page 40 / Seite 40', or a number inside a footer
    line such as '2019. All rights reserved. 40 Doc-ID 0893' (years and Doc-IDs skipped)."""
    h = page.rect.height or 1
    lines: dict[tuple, list] = {}
    for w in page.get_text("words"):
        lines.setdefault((w[5], w[6]), []).append(w)
    alone, inline = [], []
    for ws in lines.values():
        ws = sorted(ws, key=lambda x: x[0])
        y = min(x[1] for x in ws)
        if not (y < 0.12 * h or y > 0.85 * h):           # only header / footer area
            continue
        edge = min(y, h - y)
        txt = " ".join(x[4] for x in ws).strip()
        m = re.fullmatch(r"-?\s*(\d{1,4})\s*-?", txt) or _PAGE_WORD_RE.match(txt)
        if m:
            alone.append((edge, int(m.group(1))))
            continue
        toks = [x[4] for x in ws]
        for i, tok in enumerate(toks):
            if not re.fullmatch(r"\d{1,3}", tok):
                continue
            prev = toks[i - 1].lower() if i else ""
            if prev.startswith(("id", "doc", "rev", "art", "no", "nr", "©")) or prev.endswith(":"):
                continue
            inline.append((edge, int(tok)))
    best = min(alone) if alone else (min(inline) if inline else None)
    return best[1] if best else None


_CODE_CELL_RE = re.compile(r"^(?:[A-Z]{1,4}[-.]?)?\d{4,8}(?:[-./]\d{1,4})?$")


def _visual_rows(words, gap: float = 12.0, tol: float = 3.0) -> list[dict]:
    """Words -> visual rows (same baseline), each split into cells at wide x gaps.
    Keeps a code and its description on ONE line even when the PDF stores them as
    separate text blocks (the usual layout of parts lists)."""
    rows: list[dict] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        yc = (w[1] + w[3]) / 2
        if rows and abs(rows[-1]["yc"] - yc) <= tol:
            rows[-1]["words"].append(w)
        else:
            rows.append({"yc": yc, "words": [w]})
    for r in rows:
        ws = sorted(r["words"], key=lambda w: w[0])
        cells, cur, last_x1 = [], [], None
        for w in ws:
            if last_x1 is not None and w[0] - last_x1 > gap:
                cells.append(" ".join(cur)); cur = []
            cur.append(w[4]); last_x1 = w[2]
        cells.append(" ".join(cur))
        r.update(cells=cells, y0=min(w[1] for w in ws), y1=max(w[3] for w in ws))
    return rows


def _parts_runs(rows: list[dict], min_rows: int = 3) -> list[tuple[int, int]]:
    """(start, end) index ranges of rows that form an unruled parts list: consecutive
    rows whose first cell is an item code and that have a description cell."""
    runs, i = [], 0
    while i < len(rows):
        j = i
        while j < len(rows) and len(rows[j]["cells"]) >= 2 and _CODE_CELL_RE.match(rows[j]["cells"][0]):
            j += 1
        if j - i >= min_rows:
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


class TableWriter:
    """Turns grids (list of rows) into table chunks: header detection, empty columns and
    title rows removed, caption ('Sección: ...'), header repeated in every piece, pieces of
    one table share a table_id. Shared by the PDF, Word and PowerPoint extractors."""

    def __init__(self, rel: str, path: str, title: str, chunks: list[dict]) -> None:
        self.rel, self.path, self.title, self.chunks = rel, path, title, chunks
        self.prev_header: list[str] | None = None
        self.t_no = 0

    @staticmethod
    def _drop_empty_cols(grid: list[list[str]]) -> list[list[str]]:
        keep = [j for j in range(max(map(len, grid), default=0))
                if any(j < len(r) and r[j] for r in grid)]
        return [[r[j] if j < len(r) else "" for j in keep] for r in grid]

    def add(self, grid: list[list[str]], caption: list[str], page, label, where: str,
            continuation_ok: bool = True) -> None:
        grid = [[_clean(c) for c in r] for r in grid]
        grid = [r for r in grid if any(r)]
        if not grid:
            return
        grid = self._drop_empty_cols(grid)
        caption = list(caption)
        while grid and sum(1 for c in grid[0] if c) <= 1:              # title rows
            caption.append(next(c for c in grid[0] if c))
            grid = grid[1:]
        grid = self._drop_empty_cols(grid)
        if len(grid) < 2:
            return
        prev = self.prev_header
        if _looks_like_header(grid[0]):
            header, rows = grid[0], grid[1:]
            self.t_no += 1
        elif continuation_ok and prev and len(grid[0]) == len(prev) and not caption:
            header, rows = prev, grid                       # continues from previous page
        else:
            header, rows = [f"Col {j + 1}" for j in range(len(grid[0]))], grid
            self.t_no += 1
        header = [h or f"Col {j + 1}" for j, h in enumerate(header)]
        self.prev_header = header
        t_no = self.t_no
        cap = " — ".join(dict.fromkeys(caption))
        table_id = f"{self.rel}_table{t_no}"
        head = (f"Sección: {cap}\n" if cap else "") + f"Columnas: {', '.join(header)}\n"
        pg = f"_p{page}" if page is not None else ""
        s0 = 0
        while s0 < len(rows):
            part = rows[s0:s0 + ROWS_PER_TABLE_CHUNK]
            body = _md(header, part)
            while len(body) > MAX_TABLE_CHUNK_CHARS and len(part) > 1:
                part = part[: max(1, len(part) // 2)]
                body = _md(header, part)
            self.chunks.append({
                "id": f"{table_id}{pg}_r{s0 + 1}", "source": self.rel,
                "text": (f"[TABLA] {self.title} — {where} — tabla {t_no} "
                         f"(filas {s0 + 1}-{s0 + len(part)})\n{head}{body}"),
                "kind": "table", "path": self.path, "page": page, "page_label": label,
                "table_id": table_id, "seq": len(self.chunks)})
            s0 += len(part)


def text_chunks(chunks: list[dict], rel: str, path: str, title: str, text: str, splitter,
                page, label, where: str, id_part: str) -> None:
    if text.strip():
        for i, t in enumerate(splitter.split_text(text)):
            chunks.append({"id": f"{rel}_{id_part}_chunk{i}", "source": rel,
                           "text": f"[{title} — {where}]\n{t}", "kind": "text",
                           "path": path, "page": page, "page_label": label,
                           "table_id": "", "seq": len(chunks)})


def extract_pdf_chunks(path: str, rel: str, splitter) -> list[dict]:
    """Returns chunk dicts compatible with both retrievers:
    {id, source, text, kind, path, page, page_label, table_id, seq}."""
    title = Path(path).stem
    chunks: list[dict] = []
    tw = TableWriter(rel, path, title, chunks)
    carry = ""                                      # last section heading seen so far

    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc, start=1):
            label = _printed_page(page)
            where = f"página {pno}" + (f", impresa {label}" if label and label != pno else "")
            try:                                            # ruled tables only: the
                tables = [t for t in page.find_tables().tables   # "text" strategy splits
                          if t.row_count >= 2 and t.col_count >= 2]  # words on real manuals
            except Exception:
                tables = []
            tables.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
            all_lines = _page_lines(page)
            rects = [pymupdf.Rect(t.bbox) for t in tables]
            words = [w for w in page.get_text("words")
                     if not any(pymupdf.Rect(w[:4]).intersects(r) for r in rects)]
            vrows = _visual_rows(words)

            # unruled parts lists (code | description) -> tables; the row above is the header
            used: set[int] = set()
            found: list[tuple[float, list[list[str]], list[str]]] = []
            for a, b in _parts_runs(vrows):
                start = a
                if a > 0 and len(vrows[a - 1]["cells"]) >= 2 and _looks_like_header(vrows[a - 1]["cells"]) \
                        and not _CODE_CELL_RE.match(vrows[a - 1]["cells"][0]):
                    start = a - 1
                grid = [r["cells"] for r in vrows[start:b]]
                if start == a:
                    grid = [["Item no.", "Description"] + [""] * (max(map(len, grid)) - 2)] + grid
                bbox = (0, vrows[start]["y0"], 0, 0)
                found.append((vrows[start]["y0"], grid, _caption_above(all_lines, bbox, carry=carry)))
                used.update(range(start, b))

            # text outside every table, one visual row per line, cells joined with ' | '
            text = "\n".join(" | ".join(r["cells"]) for i, r in enumerate(vrows) if i not in used)
            text_chunks(chunks, rel, path, title, text, splitter, pno, label, where, f"p{pno}")

            items = [(t.bbox[1], t.extract(),
                      _caption_above(all_lines, t.bbox, carry=carry)) for t in tables] + found
            for _, grid, caption in sorted(items, key=lambda x: x[0]):
                tw.add(grid, caption, pno, label, where)
            secs = [l[2] for l in all_lines if _SECTION_RE.match(l[2])]
            if secs:
                carry = secs[-1]
    return chunks


_GROUP_CACHE: dict = {}


def _groups(all_chunks) -> tuple[dict, dict]:
    """(table_id -> pieces, (source, page) -> table pieces), cached per index snapshot."""
    key = (id(all_chunks), len(all_chunks))
    got = _GROUP_CACHE.get(key)
    if got is None:
        by_table: dict[str, list[dict]] = {}
        by_page: dict[tuple, list[dict]] = {}
        by_source: dict[str, list[dict]] = {}
        for c in all_chunks.values():
            if c.get("table_id"):
                by_table.setdefault(c["table_id"], []).append(c)
                by_page.setdefault((c.get("source"), c.get("page")), []).append(c)
                by_source.setdefault(c.get("source"), []).append(c)
        for v in list(by_table.values()) + list(by_page.values()) + list(by_source.values()):
            v.sort(key=lambda c: c.get("seq", 0))
        _GROUP_CACHE.clear()
        got = _GROUP_CACHE[key] = (by_table, by_page, by_source)
    return got


def expand_table_hits(results: list[dict], all_chunks, max_extra: int = 8) -> list[dict]:
    """After retrieve(), in rank order:
    * a table piece matched  -> the other pieces of the SAME table follow it;
    * a page's text matched (e.g. its headings '8.4.1 Standard version') -> the tables of
      THAT page follow it, because their rows live in separate table chunks.
    So the LLM sees whole tables next to the text that found them."""
    by_table, by_page, _ = _groups(all_chunks)
    seen = {r["id"] for r in results}
    out, extra = [], 0
    for r in results:
        out.append(r)
        if r.get("table_id"):
            extra_c = by_table.get(r["table_id"], [])
        elif r.get("page") is not None:
            extra_c = by_page.get((r.get("source"), r.get("page")), [])
        else:
            extra_c = []
        for c in extra_c:
            if extra >= max_extra:
                break
            if c["id"] not in seen:
                out.append(dict(c, score=r.get("score"), expanded=True))
                seen.add(c["id"])
                extra += 1
    return out


# words that never decide which table answers ("show me the available ... please")
_FOCUS_STOP = set("""
the and for with show list available all give tell what which how please about from table
tables version versions type types complete me my of to in on is are can you want need
el la los las de del para con que cual cuales muestra muestrame dame lista todas todos
disponibles tabla tablas por favor una uno quiero necesito sobre en y o mi me
die der das und fuer mit zeige mir bitte alle welche verfuegbar liste tabelle von im ist sind
now also just only answer reply respond english spanish german language
responde contesta espanol ingles aleman idioma ahora tambien solo
antworte antwort deutsch englisch spanisch sprache jetzt auch nur auf
""".split())


def _stem(w: str) -> str:
    """Light plural stemmer, same result for singular and plural:
    nozzle/nozzles -> nozzl, needle/needles -> needl, duese/duesen -> dues,
    boquilla/boquillas -> boquilla, version/versiones -> version."""
    if len(w) > 5 and w.endswith("en"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("s"):
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def focus_doc_tables(query: str, results: list[dict], all_chunks,
                     max_chunks: int = 14, max_chars: int = 16000) -> list[dict]:
    """List questions ('available nozzles for KA-2 HD') are answered by SEVERAL tables of
    one manual (8.3.1 ... 8.3.6), more than a top-k search returns. Here:
      1. target manual = the file whose NAME contains the model named in the question
         (KA-2 HD -> '2. KA-2 HD - 0893-01 - Dep.pdf'), else the file of the best hit;
      2. every table chunk of that file is scored by how many of the question's words
         (already expanded to English by query_lang) it contains;
      3. the best-scoring tables are appended in document order (whole list, no gaps)."""
    from app.textutils import fold, mkey
    _, _, by_source = _groups(all_chunks)
    if not by_source:
        return results
    f = fold(query)
    toks = f.split()
    # candidate model names: 1-3 consecutive words with letters AND digits ('ka 2 hd')
    spans = [(i, i + n) for n in (3, 2, 1) for i in range(len(toks) - n + 1)]
    grams = [("".join(toks[i:j]), i, j) for i, j in spans]
    grams = [g for g in grams if len(g[0]) >= 4 and re.search(r"\d", g[0]) and re.search(r"[a-z]", g[0])]
    targets: list[str] = []
    model_toks: set[str] = set()
    if grams:
        stems = {src: mkey(Path(str(src)).stem) for src in by_source}
        best_len, best_span, hits_src = 0, None, []
        for g, i, j in grams:
            srcs = [src for src, k in stems.items() if g in k]
            if srcs and len(g) > best_len:
                best_len, best_span, hits_src = len(g), (i, j), srcs
        if best_span:
            targets = hits_src[:2]
            model_toks = set(toks[best_span[0]:best_span[1]])
    if not targets:
        for r in results:
            if r.get("source") in by_source:
                targets = [r["source"]]
                break
    if not targets:
        return results
    model_toks |= {t for t in toks if re.search(r"\d", t)}
    terms = {_stem(t) for t in toks if len(t) >= 3 and t not in _FOCUS_STOP and t not in model_toks}
    if not terms:
        return results
    cands = []
    for src in targets:
        for c in by_source[src]:
            lines = c["text"].splitlines()
            cap = next((l for l in lines if l.startswith("Sección: ")), lines[0] if lines else "")
            words = {_stem(w) for w in fold(c["text"]).split()}
            cap_words = {_stem(w) for w in fold(cap).split()}
            sc = (len(terms & cap_words), len(terms & words))   # section title first
            if sc[1]:
                cands.append((sc, c))
    if not cands:
        return results
    best_cap = max(sc[0] for sc, _ in cands)
    if best_cap:           # tables whose TITLE names what was asked ('Nozzle, ...')
        chosen = [c for sc, c in cands if sc[0] == best_cap]
    else:                  # no title matches: tables whose body matches best
        best_txt = max(sc[1] for sc, _ in cands)
        chosen = [c for sc, c in cands if sc[1] == best_txt]
    chosen.sort(key=lambda c: c.get("seq", 0))
    seen = {r["id"] for r in results}
    base = min((r.get("score") or 0 for r in results), default=0) if results else 0
    out, used, n = list(results), 0, 0
    for c in chosen:
        if c["id"] in seen:
            continue
        if n >= max_chunks or used + len(c["text"]) > max_chars:
            break
        out.append(dict(c, score=base, expanded=True, focus=True))
        seen.add(c["id"])
        used += len(c["text"])
        n += 1
    return out
