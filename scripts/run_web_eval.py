"""Evaluasi retrieval web_kb_categorized memakai dataset query -> URL.

Metrik dihitung pada URL unik: beberapa chunk dari halaman yang sama tidak
menghabiskan slot Top-5. Tidak memanggil LLM.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib import rag, run_manifest  # noqa: E402
from lib.kb import WEB_CAT_DIR, load_documents, load_web_categorized  # noqa: E402
from lib.metrics import score_ranking  # noqa: E402
METRIK = ["hit@1", "hit@3", "hit@5", "mrr@5", "recall@5", "ndcg@5"]


def baca_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Dataset tidak ada: {path}. Jalankan: uv run python scripts/build_web_eval.py")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def agregat(kelompok: str, records: list[dict]) -> dict:
    row = {"kelompok": kelompok, "n": len(records)}
    if not records:
        return row | {nama: None for nama in METRIK} | {"latensi_p50_ms": None, "latensi_p95_ms": None}
    for nama in METRIK:
        row[nama] = float(np.mean([record["skor"][nama] for record in records]))
    latensi = [record["latency_ms"] for record in records]
    row["latensi_p50_ms"] = float(np.percentile(latensi, 50))
    row["latensi_p95_ms"] = float(np.percentile(latensi, 95))
    return row


def sel(nilai) -> str:
    if nilai is None:
        return "-"
    return f"{nilai:.4f}" if isinstance(nilai, float) else str(nilai)


def tabel(rows: list[dict]) -> str:
    headers = ["kelompok", "n", *METRIK, "latensi_p50_ms", "latensi_p95_ms"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(sel(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def slug(teks: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", teks.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=WEB_CAT_DIR / "web_eval.jsonl")
    metode_tersedia = [metode for metode in rag.METHODS if metode != rag.HYBRID_REFORMULATED]
    parser.add_argument("--metode", choices=metode_tersedia, default=rag.BM25)
    parser.add_argument("--top-k", type=int, default=50, help="Kandidat yang diambil sebelum deduplikasi URL")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "web_eval")
    parser.add_argument("--gabung", action="store_true", help="Sertakan kurikulum dan regulasi sebagai distraktor")
    args = parser.parse_args()

    courses = load_documents(web_dir=WEB_CAT_DIR) if args.gabung else load_web_categorized()
    queries = baca_dataset(args.dataset)
    retriever = rag.CourseRetriever(courses)

    records = []
    luar_cakupan = []
    for query in queries:
        try:
            hasil = retriever.retrieve(args.metode, query["query"], top_k=args.top_k)
        except rag.DenseIndexError as error:
            raise SystemExit(f"{error}\nIndex web: uv run python build_index.py --web") from error
        kandidat = [
            {
                "rank": rank,
                "kode": hit.course.kode,
                "judul": hit.course.nama,
                "url": hit.course.sumber,
                "skor": round(hit.score, 4),
            }
            for rank, hit in enumerate(hasil.hits, start=1)
        ]
        if not query.get("target_url"):
            luar_cakupan.append(
                {
                    "id": query["id"],
                    "kategori": query["kategori"],
                    "query": query["query"],
                    "jenis_kueri": query.get("jenis_kueri", ""),
                    "latency_ms": hasil.latency_ms,
                    "skor_teratas": hasil.hits[0].score if hasil.hits else 0.0,
                    "kandidat": kandidat,
                }
            )
            continue
        skor = score_ranking((hit.course.sumber for hit in hasil.hits), {query["target_url"]})
        records.append(
            {
                "id": query["id"],
                "kategori": query["kategori"],
                "query": query["query"],
                "jenis_kueri": query.get("jenis_kueri", ""),
                "target_url": query["target_url"],
                "target_chunk": query["target_chunk"],
                "latency_ms": hasil.latency_ms,
                "skor": skor,
                "kandidat": kandidat,
            }
        )

    if not records:
        raise SystemExit("Dataset tidak punya kueri dengan target URL.")
    rows = [agregat("semua", records)]
    for kategori in sorted({record["kategori"] for record in records}):
        rows.append(agregat(kategori, [record for record in records if record["kategori"] == kategori]))

    nama_run = slug(args.metode) + ("-gabung" if args.gabung else "")
    if "gold" in args.dataset.name:
        nama_run += "-gold"
    out = args.output / nama_run
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for record in records + luar_cakupan:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = [
        f"# Evaluasi Retrieval Web: {args.metode}",
        "",
        f"- Korpus: {len(courses)} dokumen "
        + ("(kurikulum + regulasi + web)" if args.gabung else "(web statis saja)"),
        f"- Dataset: {args.dataset.name}, {len(records)} kueri dinilai, {len(luar_cakupan)} di luar cakupan",
        f"- Kandidat: {args.top_k}; URL dideduplikasi sebelum penilaian Top-5",
        "- Metrik: Hit@1/3/5, MRR@5, Recall@5, NDCG@5",
        "",
        tabel(rows),
        "",
        "## Kueri di luar cakupan",
        "",
        f"- Jumlah: {len(luar_cakupan)}",
        f"- Mengembalikan hasil: {sum(bool(record['kandidat']) for record in luar_cakupan)}",
        f"- Rata-rata skor teratas: {np.mean([record['skor_teratas'] for record in luar_cakupan]):.4f}"
        if luar_cakupan
        else "- Tidak ada kueri luar cakupan pada dataset ini.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    run_manifest.write_manifest(
        out / "run.json",
        {
            "jenis": "retrieval-web",
            "metode": args.metode,
            "top_k": args.top_k,
            "n": len(records),
            "n_luar_cakupan": len(luar_cakupan),
            "corpus_sha256": rag.corpus_hash(courses),
            "dataset": str(args.dataset),
            "dataset_sha256": run_manifest.file_sha256(args.dataset),
            "gabung": args.gabung,
        },
    )
    print(f"[OK] {len(records)} kueri -> {out}")
    print(tabel(rows))


if __name__ == "__main__":
    main()
