"""Utilitas crawl web FTMM: klasifikasi URL, ekstraksi tautan, deteksi halaman tipis, merge dump browser."""

import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

from lib.kb import infer_prodi

HOST = "ftmm.unair.ac.id"
SKIP_PREFIXES = ("/wp-admin", "/wp-login", "/wp-json", "/feed", "/tag/", "/category/", "/author/", "/search/")
LANG_PREFIXES = ("/en/", "/id/", "/zh/", "/ar/", "/ms/")
MEDIA_EXT = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".svg": "image",
    ".doc": "doc",
    ".docx": "doc",
    ".xls": "doc",
    ".xlsx": "doc",
    ".ppt": "doc",
    ".pptx": "doc",
    ".zip": "archive",
}
RENDER_MARKERS = ("datatable", "admin-ajax.php", "handlebars", "ajax_url", "wp-json/wp/v2")
RENDER_CHAR_MIN = 400


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict] = []
        self.href: str | None = None
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        self.href = attributes.get("href")
        self.text = []

    def handle_data(self, data: str) -> None:
        if self.href is not None:
            self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self.href is None:
            return
        href = self.href.strip()
        if href and not href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
            self.links.append({"url": href, "teks": " ".join("".join(self.text).split())})
        self.href = None
        self.text = []


class _ImgParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.images: list[dict] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in ("img", "source"):
            return
        attributes = dict(attrs)
        src = attributes.get("src") or attributes.get("data-src") or ""
        srcset = attributes.get("srcset") or attributes.get("data-srcset") or ""
        if not src and srcset:
            src = srcset.split(",")[0].strip().split(" ")[0]
        if src:
            self.images.append({"url": src, "alt": attributes.get("alt", "")})


def classify(url: str) -> str:
    """Jenis tautan: page, pdf, image, doc, archive, language, external, skip."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "skip"
    if parsed.netloc and parsed.netloc.lower() != HOST:
        return "external"
    path = parsed.path or "/"
    suffix = Path(path).suffix.lower()
    if suffix in MEDIA_EXT:
        return MEDIA_EXT[suffix]
    if path.startswith(SKIP_PREFIXES) or parsed.query.startswith(("replytocom=", "share=")):
        return "skip"
    if path.startswith(LANG_PREFIXES):
        return "language"
    return "page"


def extract_links(html: str, base: str) -> list[dict]:
    parser = _LinkParser()
    parser.feed(html)
    parser.close()
    seen: set[str] = set()
    links = []
    for link in parser.links:
        url = urljoin(base, link["url"]).split("#", 1)[0]
        if url in seen:
            continue
        seen.add(url)
        links.append({"url": url, "teks": link["teks"], "jenis": classify(url)})
    return links


def extract_images(html: str, base: str) -> list[dict]:
    parser = _ImgParser()
    parser.feed(html)
    parser.close()
    seen: set[str] = set()
    images = []
    for image in parser.images:
        url = urljoin(base, image["url"])
        if url in seen:
            continue
        seen.add(url)
        images.append({"url": url, "alt": image["alt"]})
    return images


def needs_render(teks: str, html: str = "") -> bool:
    """Halaman dianggap perlu render browser bila teks pendek atau memuat penanda tabel/AJAX."""
    if len(teks) < RENDER_CHAR_MIN:
        return True
    haystack = html.lower()
    return any(marker in haystack for marker in RENDER_MARKERS)


def _hash_teks(teks: str) -> str:
    return hashlib.sha256(teks.encode("utf-8")).hexdigest()


def _record_from_dump(dump: dict, lama: dict | None = None) -> dict:
    teks = str(dump.get("teks", "")).strip()
    html = str(dump.get("html", ""))
    url = dump.get("url", "")
    slug = dump.get("slug") or Path(urlparse(url).path.rstrip("/")).name
    record = {
        "id": dump.get("id"),
        "tipe": dump.get("tipe", "page"),
        "url": url,
        "slug": slug,
        "judul": dump.get("judul", ""),
        "tanggal": dump.get("tanggal", ""),
        "modified": dump.get("modified", ""),
        "prodi": dump.get("prodi") or infer_prodi(dump.get("judul", ""), slug),
        "teks": teks,
        "jumlah_html": len(html),
        "jumlah_karakter": len(teks),
        "hash_teks": _hash_teks(teks),
        "diambil_pada": dump.get("diambil_pada", ""),
        "sumber_raw": dump.get("sumber_raw", ""),
        "links": [
            {"url": link["url"], "teks": link.get("teks", ""), "jenis": link.get("jenis") or classify(link["url"])}
            for link in dump.get("links") or []
            if link.get("url")
        ],
        "gambar": dump.get("gambar") or [],
        "status": "ok" if teks else "kosong",
        "perlu_render": False,
        "mesin": dump.get("mesin", "browser"),
    }
    if lama:
        for kunci in ("id", "tipe", "tanggal", "modified"):
            if lama.get(kunci):
                record[kunci] = lama[kunci]
        record["prodi"] = lama.get("prodi") or record["prodi"]
    return record


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _kunci(row: dict) -> tuple:
    if row.get("id"):
        return (row.get("tipe"), row.get("id"))
    return ("url", row.get("url"))


def propagate_prodi(rows: list[dict]) -> int:
    """Warisi prodi dari halaman induk (mis. halaman staf -> profil dosen)."""
    masuk: dict[str, str] = {}
    for row in rows:
        if not row.get("prodi"):
            continue
        for link in row.get("links") or []:
            masuk.setdefault(link["url"], row["prodi"])
    jumlah = 0
    for row in rows:
        if row.get("prodi"):
            continue
        prodi = masuk.get(row["url"])
        if prodi:
            row["prodi"] = prodi
            jumlah += 1
    return jumlah


def merge_dump(dump_dir: Path, out_dir: Path) -> dict:
    """Gabungkan dump browser ke pages.jsonl; teks yang lebih panjang menang."""
    pages_path = out_dir / "pages.jsonl"
    records = {}
    if pages_path.exists():
        for line in pages_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            records[_kunci(row)] = row

    diganti = 0
    ditambah = 0
    for path in sorted(dump_dir.glob("*.json")):
        dump = json.loads(path.read_text(encoding="utf-8"))
        url = dump.get("url", "")
        lama = next((row for row in records.values() if row["url"] == url), None)
        record = _record_from_dump(dump, lama)
        if record["status"] != "ok":
            continue
        if lama is None:
            ditambah += 1
        elif record["jumlah_karakter"] > lama.get("jumlah_karakter", 0):
            diganti += 1
        else:
            continue
        record["sumber_raw"] = path.name
        if lama is not None:
            records.pop(_kunci(lama), None)
        records[_kunci(record)] = record

    rows = sorted(records.values(), key=lambda row: ((row.get("tipe") or ""), (row.get("id") or 0), row["url"]))
    diwarisi = propagate_prodi(rows)
    _write_jsonl(pages_path, rows)
    stats = build_graph(rows, out_dir)
    return {"jumlah_dokumen": len(rows), "ditambah": ditambah, "diganti": diganti, "prodi_diwarisi": diwarisi, **stats}


def build_graph(rows: list[dict], out_dir: Path) -> dict:
    """Tulis graf.jsonl (tautan) dan media.jsonl (PDF/gambar) dari pages.jsonl."""
    edges = []
    media = []
    seen_media: set[tuple[str, str]] = set()
    for row in rows:
        for link in row.get("links") or []:
            edges.append(
                {
                    "dari": row["url"],
                    "ke": link["url"],
                    "anchor": link.get("teks", ""),
                    "jenis": link.get("jenis", "page"),
                }
            )
            if link.get("jenis") in ("pdf", "doc", "archive", "image"):
                key = (link["url"], row["url"])
                if key not in seen_media:
                    seen_media.add(key)
                    media.append({"url": link["url"], "jenis": link["jenis"], "halaman_sumber": row["url"]})
        for image in row.get("gambar") or []:
            key = (image["url"], row["url"])
            if key not in seen_media:
                seen_media.add(key)
                media.append(
                    {"url": image["url"], "jenis": "image", "halaman_sumber": row["url"], "teks": image.get("alt", "")}
                )
    _write_jsonl(out_dir / "graf.jsonl", edges)
    _write_jsonl(out_dir / "media.jsonl", media)
    return {"jumlah_edge": len(edges), "jumlah_media": len(media)}
