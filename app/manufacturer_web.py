"""Manufacturer websites (abnox.com, schuetze-gmbh.de): crawled once into a local cache,
searched with BM25 (no API key, no tokens). From 5.7, with one web change: a query never
waits for a crawl — if the copy is missing/stale it searches what exists and refreshes in
the background."""
from __future__ import annotations

import html
import json
import logging
import os
import re
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from app.config import WEB_SITES, WEB_SKIP, settings
from app.retriever import BM25Okapi
from app.textutils import simple_tokenize

log = logging.getLogger("rag.web")


class _PageParser(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form",
            "iframe", "template", "button", "select"}
    BLOCK = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr",
             "td", "th", "section", "article", "dd", "dt", "table", "main"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.in_title = False
        self.title = ""
        self.parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag == "title":
            self.in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif tag == "title":
            self.in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines, seen = [], set()
        for ln in "".join(self.parts).splitlines():
            ln = " ".join(ln.split())
            if len(ln) > 1 and ln not in seen:
                seen.add(ln)
                lines.append(ln)
        return "\n".join(lines)


def _http_get(url: str) -> tuple[str, str, str]:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; RelantRAG/1.0)",
                                "Accept-Language": "en"})
    with urlopen(req, timeout=settings.web_timeout) as r:
        ctype = r.headers.get("Content-Type", "")
        raw = r.read(4_000_000)
        charset = r.headers.get_content_charset() or "utf-8"
        return r.geturl(), ctype, raw.decode(charset, errors="ignore")


class ManufacturerWeb:
    def __init__(self, sites: dict | None = None, cache_dir: Path | None = None,
                 fetch: Callable[[str], tuple[str, str, str]] | None = None) -> None:
        self.sites = sites or WEB_SITES
        self.cache_dir = Path(cache_dir or settings.web_cache_dir)
        self._fetch = fetch or _http_get
        self._lock = threading.RLock()
        self._crawl_lock = threading.Lock()
        self._pages: dict[str, dict] = {}
        self._passages: list[dict] = []
        self._bm25: Any = None
        self.notes: dict[str, str] = {}

    def _cache_file(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict:
        try:
            with open(self._cache_file(key), encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) and d.get("pages") else {}
        except (OSError, ValueError):
            return {}

    def _fresh(self, d: dict) -> bool:
        try:
            age = datetime.now() - datetime.fromisoformat(d["crawled_at"])
            return age.total_seconds() < settings.web_cache_days * 86400
        except (KeyError, ValueError, TypeError):
            return False

    def _url_ok(self, url: str, site: dict) -> str | None:
        u = urlparse(url)
        base = urlparse(site["start"])
        if u.scheme not in ("http", "https"):
            return None
        if u.netloc.lower().removeprefix("www.") != base.netloc.lower().removeprefix("www."):
            return None
        path = u.path.rstrip("/") or "/"
        if not path.startswith(site.get("prefix", "/")):
            return None
        low = path.lower()
        if any(s in low for s in site.get("skip", WEB_SKIP)):
            return None
        if re.search(r"\.(pdf|jpe?g|png|gif|svg|webp|zip|docx?|xlsx?|mp4|mov|dwg|stp|step)$", low):
            return None
        return f"{base.scheme}://{base.netloc}{path}"

    def _sitemap_urls(self, site: dict) -> list[str]:
        out: list[str] = []
        todo, seen = [site["sitemap"]] if site.get("sitemap") else [], set()
        while todo and len(seen) < 20:
            sm = todo.pop(0)
            if sm in seen:
                continue
            seen.add(sm)
            try:
                _, _, xml = self._fetch(sm)
            except Exception:
                continue
            locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)
            if "<sitemapindex" in xml:
                todo.extend(html.unescape(x) for x in locs)
            else:
                out.extend(html.unescape(x) for x in locs)
        return out

    def crawl(self, key: str) -> dict:
        site = self.sites[key]
        log.info("crawling %s (max %d pages)", site["start"], settings.web_max_pages)
        start = self._url_ok(site["start"], site) or site["start"]
        frontier = [start] + [u for u in (self._url_ok(x, site) for x in self._sitemap_urls(site)) if u]
        queued = set(frontier)
        pages: dict[str, dict] = {}
        errors = 0
        with ThreadPoolExecutor(max_workers=settings.web_workers) as ex:
            while frontier and len(pages) < settings.web_max_pages:
                batch = frontier[:settings.web_workers * 2]
                del frontier[:len(batch)]
                futs = {ex.submit(self._fetch, u): u for u in batch}
                for fut in as_completed(futs):
                    url = futs[fut]
                    try:
                        final, ctype, body = fut.result()
                    except Exception:
                        errors += 1
                        continue
                    if "html" not in ctype.lower():
                        continue
                    p = _PageParser()
                    try:
                        p.feed(body)
                    except Exception:
                        continue
                    text = p.text()
                    final = self._url_ok(final, site) or url
                    if len(text) >= 150 and len(pages) < settings.web_max_pages:
                        pages[final] = {"title": " ".join(p.title.split()) or final,
                                        "text": text[:60000]}
                    for href in p.links:
                        n = self._url_ok(urljoin(final, href.split("#")[0]), site)
                        if n and n not in queued:
                            queued.add(n)
                            frontier.append(n)
        if not pages:
            raise RuntimeError(f"could not read {site['start']} ({errors} network errors)")
        d = {"site": key, "crawled_at": datetime.now().isoformat(timespec="seconds"), "pages": pages}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_file(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._cache_file(key))
        log.info("%s: %d pages cached", site["name"], len(pages))
        return d

    def ensure(self, keys: list[str] | None = None, refresh: bool = False,
               allow_crawl: bool = True) -> None:
        """Load cached copies; crawl stale/missing ones only if allow_crawl."""
        stale: list[str] = []
        with self._lock:
            changed = False
            for key in keys or list(self.sites):
                if key in self._pages and not refresh and (
                        self._fresh(self._pages[key]) or key in self.notes):
                    continue
                d = {} if refresh else self._read_cache(key)
                if not d or not self._fresh(d):
                    if not allow_crawl:
                        stale.append(key)
                        if not d:
                            continue
                    else:
                        try:
                            d = self.crawl(key)
                            self.notes.pop(key, None)
                        except Exception as e:
                            old = d or self._read_cache(key)
                            if old:
                                self.notes[key] = (f"offline: using the copy from "
                                                   f"{old.get('crawled_at', '?')[:10]}")
                                d = old
                            else:
                                self.notes[key] = f"{self.sites[key]['name']} unavailable: {e}"
                                continue
                self._pages[key] = d
                changed = True
            if changed or self._bm25 is None:
                self._build()
        if stale:
            self.refresh_in_background(stale)

    def refresh_in_background(self, keys: list[str] | None = None, refresh: bool = False) -> bool:
        if not self._crawl_lock.acquire(blocking=False):
            return False

        def _bg() -> None:
            try:
                self.ensure(keys, refresh=refresh, allow_crawl=True)
            except Exception as e:
                log.warning("web refresh failed: %s", e)
            finally:
                self._crawl_lock.release()

        threading.Thread(target=_bg, name="web-crawl", daemon=True).start()
        return True

    def _build(self) -> None:
        passages = []
        for key, d in self._pages.items():
            for url, pg in d.get("pages", {}).items():
                buf = ""
                for ln in pg["text"].splitlines():
                    if len(buf) + len(ln) > 900 and buf:
                        passages.append({"site": key, "url": url, "title": pg["title"], "text": buf})
                        buf = buf[-150:]
                    buf += ln + "\n"
                if buf.strip():
                    passages.append({"site": key, "url": url, "title": pg["title"], "text": buf})
        corpus = [simple_tokenize(f"{p['title']} {p['title']} {p['text']}") for p in passages]
        self._passages, self._bm25 = passages, (BM25Okapi(corpus) if corpus else None)

    def search(self, query: str, site: str = "all", k: int = 4) -> list[dict]:
        key = (site or "all").lower().strip()
        key = next((s for s in self.sites if s in key or key in s), "all") if key != "all" else key
        keys = list(self.sites) if key == "all" else [key]
        self.ensure(keys, allow_crawl=False)          # never crawl inside a request
        with self._lock:
            if self._bm25 is None or not query.strip():
                return []
            scores = self._bm25.get_scores(simple_tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            out, per_page = [], Counter()
            for i in order:
                p = self._passages[i]
                if scores[i] <= 0 or p["site"] not in keys or per_page[p["url"]] >= 2:
                    continue
                per_page[p["url"]] += 1
                out.append(dict(p, score=round(float(scores[i]), 2)))
                if len(out) >= k:
                    break
            return out

    def status(self) -> dict:
        out = {}
        for key, site in self.sites.items():
            d = self._pages.get(key) or self._read_cache(key)
            out[key] = {"name": site["name"], "pages": len(d.get("pages", {})) if d else 0,
                        "crawled_at": (d.get("crawled_at") if d else None),
                        "note": self.notes.get(key, "")}
        return out


_WEB: ManufacturerWeb | None = None
_WEB_LOCK = threading.Lock()


def get_web() -> ManufacturerWeb:
    global _WEB
    with _WEB_LOCK:
        if _WEB is None:
            _WEB = ManufacturerWeb()
        return _WEB
