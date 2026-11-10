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

from lib.kb import infer_prodi

BASE = "https://ftmm.unair.ac.id"
DEFAULT_UA = (
    "FTMM-Compass-Eval/0.1 "
    "(+https://github.com/Primarizki21/FTMM_Compass_chatbot_eval; riset akademik)"
)
DEFAULT_DELAY = 4.0
PER_PAGE = 100
FIELDS = "id,slug,link,date,modified,title,content,excerpt"
JENIS = ("pages", "posts")
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
    return BLOCK_MARKER in text or (expect == "json" and len(body) <= 300)


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

    def _log(self, path: str, status: int, size: int, elapsed: float, hasil: str, pesan: str = "") -> None:
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

            self._log(path, status, len(body), elapsed, "ok")
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
    return {
        "id": item.get("id"),
        "tipe": "post" if tipe == "posts" else "page",
        "url": item.get("link", ""),
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
        "status": "ok" if teks else "kosong",
    }


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
