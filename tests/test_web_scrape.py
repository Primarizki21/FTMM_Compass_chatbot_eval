import json

import pytest

from lib import web_scrape as ws
from lib.kb import PRODI_ALIASES, PRODI_BY_ACRONYM, format_web_chunk, infer_prodi, load_web_categorized

HTML = (
    '<div class="elementor"><script>alert("x")</script><h2>Visi</h2>'
    "<p>Menjadi fakultas unggul &amp; berdaya saing.</p><style>.x{color:red}</style></div>"
)

ITEM = {
    "id": 7,
    "slug": "profil-pimpinan",
    "link": "https://ftmm.unair.ac.id/profil-pimpinan/",
    "date": "2024-01-02T03:04:05",
    "modified": "2026-11-10T09:09:19",
    "title": {"rendered": "Profil Pimpinan"},
    "content": {"rendered": HTML},
    "_sumber_raw": "pages-0001.json",
    "_diambil": "2026-11-11T01:00:00+00:00",
}


def opener(body: bytes, content_type: str, status: int = 200, counter: list[int] | None = None):
    def call(url, headers, timeout):
        if counter is not None:
            counter.append(1)
        return status, {"content-type": content_type}, body

    return call


def test_html_to_text_drops_script_and_style():
    text = ws.html_to_text(HTML)
    assert "Visi" in text
    assert "berdaya saing" in text
    assert "\xa0" not in text
    assert "alert" not in text
    assert "color:red" not in text


def test_waf_block_detected_without_retry(tmp_path):
    counter = []
    fetcher = ws.Fetcher(
        log_path=tmp_path / "log.jsonl",
        opener=opener(b"<html>The requested URL was rejected.</html>", "text/html; charset=UTF-8", counter=counter),
    )
    with pytest.raises(ws.WafBlocked):
        fetcher.get("/wp-json/wp/v2/pages?per_page=1")
    assert len(counter) == 1
    assert json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])["hasil"] == "blokir"


def test_fetch_writes_trace_and_returns_body(tmp_path):
    fetcher = ws.Fetcher(
        log_path=tmp_path / "log.jsonl",
        opener=opener(b'[{"id": 1}]', "application/json"),
    )
    assert fetcher.get("/wp-json/wp/v2/pages") == b'[{"id": 1}]'
    entry = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert entry["hasil"] == "ok"
    assert entry["status_http"] == 200
    assert entry["path"] == "/wp-json/wp/v2/pages"


def test_delay_between_requests_only(tmp_path):
    slept = []
    fetcher = ws.Fetcher(
        delay=6.0,
        sleeper=slept.append,
        opener=opener(b"[]", "application/json"),
    )
    fetcher.get("/a")
    fetcher.get("/b")
    assert slept == [6.0]


def test_robots_disallow_is_honoured():
    robots = "User-agent: *\nDisallow: /wp-json/\nSitemap: https://ftmm.unair.ac.id/sitemap_index.xml\n"
    assert not ws.robots_allows(robots, "/wp-json/wp/v2/pages")
    assert ws.robots_allows(robots, "/profil-pimpinan/")


def test_parse_item_keeps_provenance_and_marks_empty():
    record = ws.parse_item(ITEM, "pages")
    assert record["tipe"] == "page"
    assert record["sumber_raw"] == "pages-0001.json"
    assert record["diambil_pada"] == "2026-11-11T01:00:00+00:00"
    assert record["jumlah_karakter"] == len(record["teks"])
    assert record["status"] == "ok"

    kosong = ws.parse_item({**ITEM, "content": {"rendered": ""}}, "pages")
    assert kosong["status"] == "kosong"


def test_enumerate_uses_cache_on_second_run(tmp_path):
    raw_dir = tmp_path / "raw"
    payload = json.dumps([{"id": 1, "title": {"rendered": "A"}, "content": {"rendered": "<p>A</p>"}}]).encode()
    first = ws.Fetcher(delay=0, opener=opener(payload, "application/json"))
    items, meta = ws.enumerate_items(first, "pages", raw_dir)
    assert items[0]["_sumber_raw"] == "pages-0001.json"
    assert meta["jumlah_item"] == 1

    second = ws.Fetcher(delay=0, opener=lambda *a: (_ for _ in ()).throw(AssertionError("cache tidak dipakai")))
    items, _ = ws.enumerate_items(second, "pages", raw_dir)
    assert items[0]["id"] == 1


def test_infer_prodi_and_chunk():
    assert infer_prodi("Profil Staff Teknik Industri", "profil-staff-teknik-industri") == "Teknik Industri"
    assert infer_prodi("Berita Fakultas", "berita-fakultas") == ""
    assert set(PRODI_ALIASES) <= set(PRODI_BY_ACRONYM.values())

    chunk = format_web_chunk(ws.parse_item(ITEM, "pages"))
    assert chunk.startswith("[INFORMASI WEB FTMM]")
    assert "Fakultas" in chunk
    assert "https://ftmm.unair.ac.id/profil-pimpinan/" in chunk


def test_load_web_categorized_skips_dynamic(tmp_path):
    dasar = {
        "tipe": "page",
        "kategori": "akademik",
        "judul": "Alur KTM",
        "url": "https://ftmm.unair.ac.id/alur-ktm/",
        "prodi": "",
        "halaman": None,
        "teks": "Isi halaman " * 20,
    }
    rows = [
        {**dasar, "chunk_id": "WEB-page-7-001", "dipakai_kb": True},
        {**dasar, "chunk_id": "WEB-post-8-001", "tipe": "post", "dipakai_kb": False},
    ]
    (tmp_path / "chunks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    documents = load_web_categorized(tmp_path)
    assert [doc.kode for doc in documents] == ["WEB-page-7-001"]
    assert documents[0].sumber == "https://ftmm.unair.ac.id/alur-ktm/"
    assert len(load_web_categorized(tmp_path, semua=True)) == 2
