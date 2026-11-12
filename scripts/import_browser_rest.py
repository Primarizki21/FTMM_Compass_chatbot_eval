"""Susun web_kb/ dari dump REST hasil browser (raw_browser_rest/) tanpa jaringan."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import run_manifest, web_crawl, web_scrape  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def muat_pages(raw: Path, mesin: str) -> tuple[list[dict], dict]:
    records = []
    per_jenis: dict[str, dict] = {}
    for path in sorted(raw.glob("*.json")):
        base = path.stem.rsplit("-", 1)[0]
        if base not in web_scrape.TIPE_BY_BASE and base != "pages":
            continue
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        items = wrapper.get("items") or []
        per_jenis.setdefault(base, {"halaman": 0, "jumlah_item": 0, "total_situs": wrapper.get("total_situs")})
        per_jenis[base]["halaman"] += 1
        per_jenis[base]["jumlah_item"] += len(items)
        for item in items:
            record = web_scrape.parse_item(item, base)
            record["mesin"] = mesin
            record["sumber_raw"] = path.name
            record["diambil_pada"] = wrapper.get("diambil_pada", record["diambil_pada"])
            records.append(record)
    records.sort(key=lambda row: (row["tipe"], row["id"] or 0, row["url"]))
    web_crawl.propagate_prodi(records)
    return records, per_jenis


def muat_media_lokal(out: Path) -> list[dict]:
    """Dokumen hasil OCR/PDF menjadi record tipe dokumen agar ikut korpus retrieval."""
    peta_path = out / "media_lokal.json"
    if not peta_path.exists():
        return []
    peta = json.loads(peta_path.read_text(encoding="utf-8"))
    rows = []
    for nama, info in peta.items():
        teks_path = ROOT / info["teks"]
        if not teks_path.exists():
            continue
        teks = teks_path.read_text(encoding="utf-8").strip()
        if not teks:
            continue
        rows.append(
            {
                "id": None,
                "tipe": "dokumen",
                "url": info["url"],
                "slug": Path(nama).stem,
                "judul": info.get("judul", nama),
                "tanggal": "",
                "modified": "",
                "prodi": "",
                "teks": teks,
                "jumlah_html": 0,
                "jumlah_karakter": len(teks),
                "hash_teks": hashlib.sha256(teks.encode("utf-8")).hexdigest(),
                "diambil_pada": web_scrape.now(),
                "sumber_raw": str(teks_path.relative_to(ROOT)),
                "links": [],
                "gambar": [],
                "status": "ok",
                "perlu_render": False,
                "mesin": "b",
            }
        )
    return rows


def tulis_log_browser(raw: Path, out: Path) -> int:
    """Jejak per respons browser: wrapper REST, sitemap, dan dump render."""
    entries = []
    for path in sorted((raw).glob("*.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        if "items" not in wrapper:
            continue
        entries.append(
            {
                "waktu": wrapper.get("diambil_pada", ""),
                "path": wrapper.get("url", ""),
                "status_http": wrapper.get("status", 0),
                "byte": wrapper.get("byte", 0),
                "dokumen": len(wrapper.get("items") or []),
                "hasil": "ok",
                "mesin": "b",
                "sumber_raw": path.name,
            }
        )
    for path in sorted((raw).glob("*.xml")):
        entries.append(
            {
                "waktu": "",
                "path": path.name,
                "status_http": 200,
                "byte": path.stat().st_size,
                "dokumen": 0,
                "hasil": "ok",
                "mesin": "b",
                "sumber_raw": path.name,
            }
        )
    for path in sorted((out / "raw_browser").glob("*.json")):
        dump = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "waktu": dump.get("diambil_pada", ""),
                "path": dump.get("url", ""),
                "status_http": 200,
                "byte": len(dump.get("html", "")),
                "dokumen": 1,
                "hasil": "render",
                "mesin": "b",
                "sumber_raw": path.name,
            }
        )
    entries.sort(key=lambda entry: (entry["waktu"], entry["path"]))
    write_jsonl(out / "log_browser.jsonl", entries)
    return len(entries)


def muat_sitemap(raw: Path) -> dict:
    urls: list[str] = []
    terbaca = []
    gagal = []
    for path in sorted(raw.glob("sitemap-*.xml")):
        if path.name == "sitemap-index.xml":
            continue
        try:
            page_urls, _ = web_scrape.parse_sitemap(path.read_bytes())
        except Exception:
            gagal.append(path.name)
            continue
        urls.extend(page_urls)
        terbaca.append(path.name)
    unik = sorted({url for url in urls if web_crawl.classify(url) == "page"})
    return {
        "urls": unik,
        "sitemaps": terbaca,
        "gagal": gagal,
        "jumlah_url": len(unik),
        "jumlah_mentah": len(urls),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("web_kb/raw_browser_rest"))
    parser.add_argument("--out", type=Path, default=Path("web_kb"))
    parser.add_argument("--mesin", default="b")
    parser.add_argument("--user-agent", default="")
    args = parser.parse_args()

    raw: Path = args.raw
    out: Path = args.out
    if not raw.exists():
        print(f"[FAIL] Dump REST browser tidak ada di {raw}")
        raise SystemExit(1)

    mulai = web_scrape.now()
    records, per_jenis = muat_pages(raw, args.mesin)
    media_lokal = muat_media_lokal(out)
    records.extend(media_lokal)
    records.sort(key=lambda row: (row["tipe"], row["id"] or 0, row["url"]))
    if not records:
        print(f"[FAIL] Tidak ada item di {raw}")
        raise SystemExit(1)

    write_jsonl(out / "pages.jsonl", records)
    graph = web_crawl.build_graph(records, out)

    sitemap = muat_sitemap(raw)
    jumlah_log = tulis_log_browser(raw, out)
    write_jsonl(
        out / "urls.jsonl",
        [{"url": url, "jenis": web_crawl.classify(url), "sumber": "sitemap"} for url in sitemap["urls"]],
    )

    render = [
        {"url": row["url"], "jumlah_karakter": row["jumlah_karakter"], "judul": row["judul"]}
        for row in records
        if row.get("perlu_render")
    ]
    write_jsonl(out / "render_queue.jsonl", render)

    sitemap_urls = set(sitemap["urls"])
    terambil = {row["url"] for row in records}
    tidak_terambil = sorted(sitemap_urls - terambil)

    manifest = run_manifest.write_manifest(
        out / "manifest.json",
        {
            "sumber": web_scrape.BASE,
            "mesin": args.mesin,
            "basis": [f"wp-json/wp/v2/{base}" for base in per_jenis],
            "user_agent": args.user_agent,
            "per_jenis": per_jenis,
            "jumlah_dokumen": len(records),
            "jumlah_kosong": sum(1 for row in records if row["status"] != "ok"),
            "jumlah_dokumen_lokal": len(media_lokal),
            "trace_browser": jumlah_log,
            "jumlah_perlu_render": len(render),
            "sitemap": {
                "sitemaps": sitemap["sitemaps"],
                "jumlah_url": sitemap["jumlah_url"],
                "sitemaps_gagal": sitemap["gagal"],
                "jumlah_tidak_terambil": len(tidak_terambil),
                "contoh_tidak_terambil": tidak_terambil[:20],
            },
            "graf": graph,
            "hash_konten": hashlib.sha256((out / "pages.jsonl").read_bytes()).hexdigest(),
            "mulai": mulai,
            "selesai": web_scrape.now(),
        },
    )
    print(
        f"[OK] {manifest['jumlah_dokumen']} dokumen "
        f"(kosong={manifest['jumlah_kosong']}, render={len(render)}, "
        f"sitemap={sitemap['jumlah_url']}, tidak_terambil={len(tidak_terambil)}, "
        f"edge={graph['jumlah_edge']}, media={graph['jumlah_media']}) → {out / 'pages.jsonl'}"
    )


if __name__ == "__main__":
    main()
