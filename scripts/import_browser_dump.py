"""Gabungkan hasil render browser ke web_kb/ tanpa jaringan."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import run_manifest, web_crawl  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path("web_kb/raw_browser"))
    parser.add_argument("--out", type=Path, default=Path("web_kb"))
    args = parser.parse_args()

    if not args.dump.exists():
        print(f"[FAIL] Dump browser tidak ada di {args.dump}")
        raise SystemExit(1)

    stats = web_crawl.merge_dump(args.dump, args.out)

    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        lama = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (args.out / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
        run_manifest.write_manifest(
            manifest_path,
            {
                **lama,
                "jumlah_dokumen": len(rows),
                "jumlah_kosong": sum(1 for row in rows if row["status"] != "ok"),
                "jumlah_perlu_render": sum(1 for row in rows if row.get("perlu_render")),
                "graf": {"jumlah_edge": stats["jumlah_edge"], "jumlah_media": stats["jumlah_media"]},
                "hash_konten": hashlib.sha256((args.out / "pages.jsonl").read_bytes()).hexdigest(),
                "selesai": web_crawl.datetime.now(web_crawl.timezone.utc).isoformat(timespec="seconds"),
            },
        )

    print(
        f"[OK] {stats['jumlah_dokumen']} dokumen "
        f"(ditambah={stats['ditambah']}, diganti={stats['diganti']}, prodi_diwarisi={stats['prodi_diwarisi']}, "
        f"edge={stats['jumlah_edge']}, media={stats['jumlah_media']})"
    )


if __name__ == "__main__":
    main()
