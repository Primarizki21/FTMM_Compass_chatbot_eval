"""Ambil informasi publik FTMM UNAIR ke web_kb/ tanpa menyentuh knowledge_base/.

Isolasi: data masuk ke web_kb/ (gitignored). Jejak: log.jsonl per request, raw/ per halaman
REST, sitemap/, pages.jsonl, graf.jsonl, media.jsonl, dan manifest.json.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import run_manifest, web_crawl, web_scrape  # noqa: E402

MIN_DELAY = 4.0


def log_stats(log_path: Path, since: str) -> dict:
    counts = {"ok": 0, "blokir": 0, "error": 0}
    if not log_path.exists():
        return counts
    for line in log_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("waktu", "") >= since and entry.get("hasil") in counts:
            counts[entry["hasil"]] += 1
    return counts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def resolve_jenis(fetcher: web_scrape.Fetcher, diminta: str) -> tuple[list[str], list[str]]:
    """auto = semua rest base publik dari /wp-json/wp/v2/types; selain itu pakai daftar eksplisit."""
    tersedia = web_scrape.public_types(fetcher)
    if diminta == "auto":
        return tersedia, []
    diminta_list = [bagian.strip() for bagian in diminta.split(",") if bagian.strip()]
    dipakai = [base for base in diminta_list if base in tersedia]
    dilewati = [base for base in diminta_list if base not in tersedia]
    return dipakai, dilewati


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("web_kb"))
    parser.add_argument("--jenis", default="auto", help="auto atau daftar rest base: pages,posts,ajde_events")
    parser.add_argument("--no-sitemap", action="store_true", help="Lewati enumerasi sitemap Yoast")
    parser.add_argument("--delay", type=float, default=web_scrape.DEFAULT_DELAY)
    parser.add_argument("--since", default="", help="Batas modified minimal, format YYYY-MM-DD; kosong = semua")
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah dokumen per jenis; 0 = semua")
    parser.add_argument("--max-requests", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--refresh", action="store_true", help="Abaikan cache raw/")
    parser.add_argument("--dry-run", action="store_true", help="Hanya cek akses dan jumlah dokumen")
    parser.add_argument("--user-agent", default=web_scrape.DEFAULT_UA)
    args = parser.parse_args()

    if args.delay < MIN_DELAY:
        parser.error(f"--delay minimal {MIN_DELAY} detik agar tidak memicu WAF")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    mulai = web_scrape.now()
    fetcher = web_scrape.Fetcher(
        delay=args.delay,
        timeout=args.timeout,
        user_agent=args.user_agent,
        max_requests=args.max_requests,
        log_path=out / "log.jsonl",
    )

    try:
        robots = fetcher.get("/robots.txt", expect="text").decode("utf-8", "ignore")
        jenis, dilewati = resolve_jenis(fetcher, args.jenis)
        jenis = [base for base in jenis if web_scrape.robots_allows(robots, f"/wp-json/wp/v2/{base}")]
        if not jenis:
            print("[FAIL] Tidak ada rest base yang boleh diambil menurut robots.txt/types")
            raise SystemExit(3)

        if args.dry_run:
            print(f"[OK] rest base: {', '.join(jenis)}")
            for rest_base in jenis:
                fetcher.get(f"/wp-json/wp/v2/{rest_base}?per_page=1&_fields=id")
                total = web_scrape.header(fetcher.last_headers, "X-WP-Total") or "?"
                print(f"[OK] {rest_base}: {total} dokumen")
            return

        sitemap = {"urls": [], "sitemaps": [], "jumlah_url": 0}
        if not args.no_sitemap:
            sitemap = web_scrape.collect_sitemap_urls(fetcher, out / "raw")
            write_jsonl(
                out / "urls.jsonl",
                [{"url": url, "jenis": web_crawl.classify(url), "sumber": "sitemap"} for url in sitemap["urls"]],
            )

        records: list[dict] = []
        per_jenis: dict[str, dict] = {}
        for rest_base in jenis:
            try:
                items, meta = web_scrape.enumerate_items(
                    fetcher, rest_base, out / "raw", since=args.since, limit=args.limit, refresh=args.refresh
                )
            except web_scrape.HttpStatusError as exc:
                per_jenis[rest_base] = {"status": exc.status, "jumlah_item": 0}
                dilewati.append(f"{rest_base}:HTTP{exc.status}")
                continue
            per_jenis[rest_base] = meta
            records.extend(web_scrape.parse_item(item, rest_base) for item in items)

        records.sort(key=lambda row: (row["tipe"], row["id"] or 0, row["url"]))
        write_jsonl(out / "pages.jsonl", records)
        graph = web_crawl.build_graph(records, out)

        render = [
            {"url": row["url"], "jumlah_karakter": row["jumlah_karakter"], "judul": row["judul"]}
            for row in records
            if row.get("perlu_render")
        ]
        write_jsonl(out / "render_queue.jsonl", render)

        sitemap_urls = set(sitemap["urls"])
        terambil = {row["url"] for row in records}
        tidak_terambil = sorted(sitemap_urls - terambil)

        stats = log_stats(out / "log.jsonl", mulai)
        manifest = run_manifest.write_manifest(
            out / "manifest.json",
            {
                "sumber": web_scrape.BASE,
                "basis": [f"wp-json/wp/v2/{rest_base}" for rest_base in jenis],
                "user_agent": args.user_agent,
                "delay_detik": args.delay,
                "since": args.since,
                "limit": args.limit,
                "per_jenis": per_jenis,
                "jumlah_dokumen": len(records),
                "jumlah_kosong": sum(1 for row in records if row["status"] != "ok"),
                "jumlah_perlu_render": len(render),
                "jumlah_request": stats["ok"] + stats["blokir"] + stats["error"],
                "request": stats,
                "sitemap": {
                    "sitemaps": sitemap["sitemaps"],
                    "jumlah_url": sitemap["jumlah_url"],
                    "jumlah_tidak_terambil": len(tidak_terambil),
                    "contoh_tidak_terambil": tidak_terambil[:20],
                },
                "jenis_dilewati": dilewati,
                "graf": graph,
                "hash_konten": hashlib.sha256((out / "pages.jsonl").read_bytes()).hexdigest(),
                "mulai": mulai,
                "selesai": web_scrape.now(),
            },
        )
        print(
            f"[OK] {manifest['jumlah_dokumen']} dokumen "
            f"(kosong={manifest['jumlah_kosong']}, render={len(render)}, "
            f"request={manifest['jumlah_request']}, blokir={stats['blokir']}, "
            f"edge={graph['jumlah_edge']}) → {out / 'pages.jsonl'}"
        )
    except web_scrape.WafBlocked as exc:
        print(f"[FAIL] {exc}")
        print(
            f"Blokir WAF dapat bertahan >20 menit. Ulangi ≥30 menit dengan "
            f"--delay {max(args.delay * 2, 8):.0f}; cache raw/ dipakai ulang."
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
