"""Ambil halaman publik ftmm.unair.ac.id lewat WP REST API secara konservatif.

Setiap request dicatat ke log JSONL. Setiap dokumen menyimpan URL sumber, berkas cache
mentah, waktu ambil, dan hash teks agar hasil bisa diaudit.
"""

import hashlib
import html as html_lib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

from lib.kb import infer_prodi
from lib.web_crawl import classify, extract_images, extract_links, needs_render

BASE = "https://ftmm.unair.ac.id"
DEFAULT_UA = (
    "FTMM-Compass-Eval/0.1 "
    "(+https://github.com/Primarizki21/FTMM_Compass_chatbot_eval; riset akademik)"
)
DEFAULT_DELAY = 4.0
PER_PAGE = 100
FIELDS = "id,slug,link,date,modified,title,content,excerpt"
JENIS = ("pages", "posts")
TIPE_BY_BASE = {"posts": "post", "ajde_events": "event"}
TYPES_DENY = {
    "attachment",
    "nav_menu_item",
    "wp_block",
    "wp_template",
    "wp_template_part",
    "wp_navigation",
    "wp_font_family",
    "wp_font_face",
    "elementor_library",
    "custom_css",
    "customize_changeset",
    "oembed_cache",
    "user_request",
}
BLOCK_MARKER = "requested url was rejected"
SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "iframe", "template"})
BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "blockquote", "td"}
)


class WafBlocked(RuntimeError):
    """Situs menolak request. Berhenti, jangan retry cepat."""


class HttpStatusError(RuntimeError):
    def __init__(self, status: int, url: str, body: bytes = b""):
        super().__init__(f"HTTP {status} {url}")
        self.status = status
        self.url = url
        self.body = body


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def header(headers: dict[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return ""


def is_blocked(body: bytes, content_type: str, expect: str = "json") -> bool:
    """WAF membalas 200 dengan halaman HTML pendek, bukan JSON dan bukan 403."""
    if "html" not in content_type.lower():
        return False
    text = body[:2000].decode("utf-8", "ignore").lower()
    return any(marker in text for marker in ("requested url was rejected", "request rejected")) or (
        expect == "json" and len(body) <= 300
    )


def robots_allows(robots_text: str, path: str, user_agent: str = "FTMM-Compass-Eval") -> bool:
    parser = RobotFileParser()
    parser.parse(robots_text.splitlines())
    return parser.can_fetch(user_agent, BASE + path)


@dataclass
class Fetcher:
    delay: float = DEFAULT_DELAY
    timeout: float = 30.0
    user_agent: str = DEFAULT_UA
    max_requests: int = 400
    log_path: Path | None = None
    sleeper: Callable[[float], None] = time.sleep
    opener: Callable[[str, dict[str, str], float], tuple[int, dict[str, str], bytes]] | None = None
    requests: int = 0
    last_headers: dict[str, str] = field(default_factory=dict)

    def _open(self, url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict[str, str], bytes]:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()

    def _log(
        self,
        path: str,
        status: int,
        size: int,
        elapsed: float,
        hasil: str,
        pesan: str = "",
        extra: dict | None = None,
    ) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "waktu": now(),
            "path": path,
            "status_http": status,
            "byte": size,
            "detik": elapsed,
            "hasil": hasil,
        }
        if pesan:
            entry["pesan"] = pesan
        if extra:
            entry.update({k: v for k, v in extra.items() if v})
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get(self, path: str, *, expect: str = "json") -> bytes:
        if self.requests >= self.max_requests:
            raise WafBlocked(f"Batas --max-requests {self.max_requests} tercapai sebelum {path}")
        if self.requests:
            self.sleeper(self.delay)
        self.requests += 1

        url = BASE + path
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "identity"}
        headers["Accept"] = "application/json" if expect == "json" else "text/plain"
        open_url = self.opener or self._open

        for backoff in (30.0, 90.0, None):
            started = time.monotonic()
            try:
                status, response_headers, body = open_url(url, headers, self.timeout)
            except HTTPError as exc:
                body = exc.read()
                status, response_headers = exc.code, dict(exc.headers.items())
            except (URLError, TimeoutError, OSError) as exc:
                elapsed = round(time.monotonic() - started, 3)
                self._log(path, 0, 0, elapsed, "error", str(exc))
                if backoff is None:
                    raise
                self.sleeper(backoff)
                continue

            elapsed = round(time.monotonic() - started, 3)
            self.last_headers = response_headers
            if status >= 500:
                self._log(path, status, len(body), elapsed, "error", f"HTTP {status}")
                if backoff is None:
                    raise HttpStatusError(status, url, body)
                self.sleeper(backoff)
                continue
            if status >= 400:
                self._log(path, status, len(body), elapsed, "error", f"HTTP {status}")
                raise HttpStatusError(status, url, body)
            if is_blocked(body, header(response_headers, "Content-Type"), expect):
                self._log(path, status, len(body), elapsed, "blokir", "WAF menolak request")
                raise WafBlocked(f"WAF menolak {path}. Berhenti, ulangi setelah jeda.")

            cache = header(response_headers, "x-litespeed-cache")
            age = header(response_headers, "age")
            self._log(path, status, len(body), elapsed, "ok", extra={"cache": cache, "umur_cache": age})
            return body

        raise AssertionError("tidak mungkin")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in SKIP_TAGS:
            self.skipping += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self.skipping = max(0, self.skipping - 1)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skipping:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    text = html_lib.unescape("".join(extractor.parts))
    text = re.sub(r"[ \t\f\v\xa0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_item(item: dict, tipe: str) -> dict:
    judul = html_lib.unescape(str(item.get("title", {}).get("rendered", ""))).strip()
    slug = item.get("slug", "")
    content = item.get("content", {}).get("rendered", "") or item.get("excerpt", {}).get("rendered", "")
    teks = html_to_text(str(content))
    url = item.get("link", "")
    return {
        "id": item.get("id"),
        "tipe": TIPE_BY_BASE.get(tipe, "page"),
        "url": url,
        "slug": slug,
        "judul": judul,
        "tanggal": item.get("date", ""),
        "modified": item.get("modified", ""),
        "prodi": infer_prodi(judul, slug),
        "teks": teks,
        "jumlah_html": len(content),
        "jumlah_karakter": len(teks),
        "hash_teks": hashlib.sha256(teks.encode("utf-8")).hexdigest(),
        "diambil_pada": item.get("_diambil", ""),
        "sumber_raw": item.get("_sumber_raw", ""),
        "links": extract_links(str(content), url),
        "gambar": extract_images(str(content), url),
        "status": "ok" if teks else "kosong",
        "perlu_render": needs_render(teks, str(content)),
    }


def parse_sitemap(body: bytes) -> tuple[list[str], list[str]]:
    """Kembalikan (url, sub_sitemap) dari urlset atau sitemapindex."""
    root = ElementTree.fromstring(body)
    urls = [(element.findtext("{*}loc") or "").strip() for element in root.findall("{*}url")]
    sitemaps = [(element.findtext("{*}loc") or "").strip() for element in root.findall("{*}sitemap")]
    return [url for url in urls if url], [url for url in sitemaps if url]


def collect_sitemap_urls(fetcher: Fetcher, cache_dir: Path, max_sitemaps: int = 30) -> dict:
    """Baca sitemap_index.xml Yoast dan semua sub-sitemap; simpan body mentah di cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    body = fetcher.get("/sitemap_index.xml", expect="text")
    (cache_dir / "sitemap-index.xml").write_bytes(body)
    _, children = parse_sitemap(body)

    urls: list[str] = []
    terbaca: list[str] = []
    for child in children[:max_sitemaps]:
        name = child.rsplit("/", 1)[-1] or "sitemap.xml"
        cache = cache_dir / name
        if cache.exists():
            payload = cache.read_bytes()
        else:
            payload = fetcher.get("/" + name, expect="text")
            cache.write_bytes(payload)
        page_urls, _ = parse_sitemap(payload)
        urls.extend(page_urls)
        terbaca.append(name)

    unik = sorted({url for url in urls if classify(url) == "page"})
    return {"urls": unik, "sitemaps": terbaca, "jumlah_url": len(unik), "jumlah_mentah": len(urls)}


def public_types(fetcher: Fetcher) -> list[str]:
    """Rest base dari tipe konten publik; menyertakan CPT seperti EventOn bila terekspos."""
    data = json.loads(fetcher.get("/wp-json/wp/v2/types"))
    bases = []
    for slug, info in data.items():
        base = info.get("rest_base") or slug
        if slug not in TYPES_DENY:
            bases.append(base)
    return bases


def enumerate_items(
    fetcher: Fetcher,
    tipe: str,
    raw_dir: Path,
    since: str = "",
    limit: int = 0,
    refresh: bool = False,
) -> tuple[list[dict], dict]:
    """Ambil item per halaman REST, cache per halaman di raw/, lanjutkan dari cache."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    pages = 0
    total_situs = None
    page = 1
    while True:
        cache = raw_dir / f"{tipe}-{page:04d}.json"
        if cache.exists() and not refresh:
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            query = urlencode(
                {"per_page": PER_PAGE, "page": page, "orderby": "id", "order": "asc", "_fields": FIELDS}
            )
            path = f"/wp-json/wp/v2/{tipe}?{query}"
            try:
                body = fetcher.get(path)
            except HttpStatusError as exc:
                if exc.status == 400 and page > 1:
                    break
                raise
            total = header(fetcher.last_headers, "X-WP-Total")
            data = {
                "url": BASE + path,
                "status": 200,
                "byte": len(body),
                "diambil_pada": now(),
                "total_situs": int(total) if total.isdigit() else None,
                "items": json.loads(body),
            }
            cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        batch = data.get("items", [])
        if not batch:
            break
        pages += 1
        if total_situs is None and data.get("total_situs"):
            total_situs = data["total_situs"]
        if since:
            batch = [item for item in batch if (item.get("modified") or item.get("date") or "") >= since]
        for item in batch:
            item["_sumber_raw"] = cache.name
            item["_diambil"] = data.get("diambil_pada", "")
        items.extend(batch)
        if limit and len(items) >= limit:
            items = items[:limit]
            break
        if len(data.get("items", [])) < PER_PAGE:
            break
        page += 1
        if page > 200:
            raise RuntimeError("Pagination melebihi 200 halaman; periksa endpoint.")

    meta = {"halaman_rest": pages, "jumlah_item": len(items), "since": since}
    if total_situs is not None:
        meta["total_situs"] = total_situs
    return items, meta
