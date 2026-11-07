"""Menyusun ledger agregat dari results/ tanpa menyentuh data institusi.

Hanya kolom whitelist yang dibaca: metrik, n, latensi, dan metadata run.
Kolom pertanyaan, jawaban, jawaban_acuan, konteks, serta traces.jsonl tidak pernah dibuka.
"""

import argparse
import csv
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "ledger"
ARCHIVES = OUT / "archives.csv"

RETRIEVAL_COLUMNS = [
    "dataset",
    "penilaian",
    "kelompok",
    "n",
    "hit@1",
    "hit@3",
    "hit@5",
    "mrr@5",
    "recall@5",
    "ndcg@5",
    "latensi_p50_ms",
    "latensi_p95_ms",
]
EVAL_META_COLUMNS = ["penyedia_llm", "model_llm", "metode", "top_k", "model_embedding"]
SUMMARY_COLUMNS = [
    "kelompok",
    "n",
    "n_retrieval",
    "hit@1",
    "hit@k",
    "mrr",
    "bert_precision",
    "bert_recall",
    "bert_f1",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
    "ttft_p50_ms",
    "total_p50_ms",
    "total_p95_ms",
]
RUN_FIELDS = [
    "run",
    "jenis",
    "tanggal",
    "commit",
    "provider",
    "model",
    "metode",
    "top_k",
    "embedding",
    "dataset",
    "penilaian",
    "kelompok",
    "n",
    "hit_at_1",
    "hit_at_k",
    "mrr",
    "ndcg",
    "bert_f1",
    "rouge_l_f1",
    "latensi_p50_ms",
    "latensi_p95_ms",
    "ttft_p50_ms",
    "total_p95_ms",
    "dataset_hash",
    "arsip_drive",
]


def read_rows(path: Path, columns: list[str]) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{column: row.get(column, "") for column in columns} for row in csv.DictReader(f)]


def first_row(path: Path, columns: list[str]) -> dict:
    rows = read_rows(path, columns)
    return rows[0] if rows else {}


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dataset_hash() -> str:
    path = ROOT / "ground_truth" / "GT_MASTER.csv"
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def last_modified(paths: list[Path]) -> str:
    stamps = [p.stat().st_mtime for p in paths if p.exists()]
    return datetime.fromtimestamp(max(stamps)).strftime("%Y-%m-%d %H:%M") if stamps else ""


def summary_title(path: Path) -> str:
    if not path.exists():
        return ""
    first = path.read_text(encoding="utf-8").splitlines()[0]
    return first.removeprefix("# Hasil Retrieval:").strip()


def load_archives() -> dict[str, str]:
    if not ARCHIVES.exists():
        return {}
    return {row["run"]: row["drive_url"] for row in read_rows(ARCHIVES, ["run", "drive_url"])}


def collect(results_dir: Path) -> list[dict]:
    commit = git_commit()
    gt_hash = dataset_hash()
    archives = load_archives()
    records: list[dict] = []

    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        run = run_dir.name
        metrics = run_dir / "metrics.csv"
        if metrics.exists():
            for row in read_rows(metrics, RETRIEVAL_COLUMNS):
                records.append(
                    {
                        "run": run,
                        "jenis": "retrieval",
                        "tanggal": last_modified([metrics, run_dir / "summary.md"]),
                        "commit": commit,
                        "provider": "lokal",
                        "model": "-",
                        "metode": summary_title(run_dir / "summary.md") or run,
                        "top_k": "5",
                        "embedding": "-",
                        "dataset": row["dataset"],
                        "penilaian": row["penilaian"],
                        "kelompok": row["kelompok"],
                        "n": row["n"],
                        "hit_at_1": row["hit@1"],
                        "hit_at_k": row["hit@5"],
                        "mrr": row["mrr@5"],
                        "ndcg": row["ndcg@5"],
                        "latensi_p50_ms": row["latensi_p50_ms"],
                        "latensi_p95_ms": row["latensi_p95_ms"],
                        "dataset_hash": gt_hash,
                        "arsip_drive": archives.get(run, ""),
                    }
                )

        for summary in sorted(run_dir.glob("ringkasan_*.csv")):
            scores = run_dir / f"skor_{summary.name.removeprefix('ringkasan_')}"
            meta = first_row(scores, EVAL_META_COLUMNS)
            for row in read_rows(summary, SUMMARY_COLUMNS):
                records.append(
                    {
                        "run": summary.stem.removeprefix("ringkasan_"),
                        "jenis": "ground_truth",
                        "tanggal": last_modified([summary, scores]),
                        "commit": commit,
                        "provider": meta.get("penyedia_llm", ""),
                        "model": meta.get("model_llm", ""),
                        "metode": meta.get("metode", ""),
                        "top_k": meta.get("top_k", ""),
                        "embedding": meta.get("model_embedding", ""),
                        "dataset": "GT_MASTER",
                        "penilaian": "",
                        "kelompok": row["kelompok"],
                        "n": row["n"],
                        "hit_at_1": row["hit@1"],
                        "hit_at_k": row["hit@k"],
                        "mrr": row["mrr"],
                        "bert_f1": row["bert_f1"],
                        "rouge_l_f1": row["rougeL_f1"],
                        "ttft_p50_ms": row["ttft_p50_ms"],
                        "total_p95_ms": row["total_p95_ms"],
                        "dataset_hash": gt_hash,
                        "arsip_drive": archives.get(summary.stem.removeprefix("ringkasan_"), ""),
                    }
                )
    return records


def overall(records: list[dict]) -> list[dict]:
    """Satu baris ringkas per run: ambil kelompok agregat bila ada, selain itu baris pertama."""
    seen: dict[tuple[str, str, str], dict] = {}
    for record in records:
        if record["kelompok"].lower() not in {"semua", "semua (dalam cakupan)"}:
            continue
        seen[(record["run"], record["dataset"], record["penilaian"])] = record
    return [seen[key] for key in sorted(seen)]


def format_number(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return value or "-"


def render_markdown(records: list[dict]) -> str:
    lines = [
        "# Ledger Evaluasi",
        "",
        "Hanya agregat. Tidak ada teks pertanyaan, jawaban, jawaban acuan, konteks, atau log chat di berkas ini.",
        "Dibuat ulang dengan `uv run python scripts/ledger.py`; detail angka per kelompok ada di `runs.csv`.",
        "",
        "| Run | Data | Jenis | Tanggal | Commit | Provider / Model | Metode | k | n | Metrik utama | Hash GT | Arsip |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in overall(records):
        if record["jenis"] == "retrieval":
            metrics = (
                f"Hit@1 {format_number(record['hit_at_1'])} · Hit@5 {format_number(record['hit_at_k'])} · "
                f"MRR {format_number(record['mrr'])} · NDCG {format_number(record['ndcg'])}"
            )
        else:
            metrics = (
                f"BERT F1 {format_number(record.get('bert_f1', ''))} · "
                f"ROUGE-L {format_number(record.get('rouge_l_f1', ''))} · "
                f"Hit@1 {format_number(record['hit_at_1'])} · TTFT {record['ttft_p50_ms'] or '-'} ms"
            )
        arsip = f"[Drive]({record['arsip_drive']})" if record["arsip_drive"] else "-"
        provider = " / ".join(v for v in (record["provider"], record["model"]) if v) or "-"
        data = record["dataset"] + (f" / {record['penilaian']}" if record["penilaian"] else "")
        lines.append(
            f"| {record['run']} | {data} | {record['jenis']} | {record['tanggal']} | `{record['commit']}` | "
            f"{provider} | {record['metode']} | {record['top_k']} | {record['n']} | {metrics} | "
            f"{record['dataset_hash'] or '-'} | {arsip} |"
        )
    return "\n".join(lines) + "\n"


def write_ledger(records: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (out / "RUNS.md").write_text(render_markdown(records), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    if not args.results.exists():
        print(f"[WARN] {args.results} belum ada; jalankan evaluasi lebih dulu")
        return

    records = collect(args.results)
    if not records:
        print(f"[WARN] tidak ada metrik di {args.results}")
        return

    write_ledger(records, args.out)
    print(f"[OK] {len(overall(records))} run, {len(records)} baris -> {args.out}")


if __name__ == "__main__":
    main()
