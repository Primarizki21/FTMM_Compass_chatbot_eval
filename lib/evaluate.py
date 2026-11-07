"""Menjalankan satu metode retrieval pada kedua ground truth dan menulis laporan."""

import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lib import run_manifest
from lib.kb import PRODI_BY_ACRONYM, ROOT, Course
from lib.metrics import score_ranking
from lib.rag import corpus_hash

GT_DIR = ROOT / "ground_truth"
METRIC_NAMES = ["hit@1", "hit@3", "hit@5", "mrr@5", "recall@5", "ndcg@5"]
TOP_TRACE = 5
TRACE_DEPTH = 20

SearchFn = Callable[[str], list[tuple[int, float]]]


def load_queries() -> tuple[list[dict], list[dict]]:
    ir_eval = json.loads((GT_DIR / "ir_eval_dataset.json").read_text(encoding="utf-8"))
    complexity = json.loads((GT_DIR / "retrieval_complexity_benchmark.json").read_text(encoding="utf-8"))
    return ir_eval, complexity


def run_search(search: SearchFn, query: str) -> tuple[list[tuple[int, float]], float]:
    start = time.perf_counter()
    hits = search(query)
    return hits, (time.perf_counter() - start) * 1000.0


def split_hit(hit: tuple) -> tuple[int, float, dict]:
    """Terima (idx, skor) atau (idx, skor, komponen) agar pemanggil lama tetap jalan."""
    idx, score = hit[0], hit[1]
    komponen = hit[2] if len(hit) > 2 else {}
    return int(idx), float(score), komponen


def to_trace(hits: list[tuple], courses: list[Course], depth: int = TRACE_DEPTH) -> list[dict]:
    return [
        {
            "rank": rank,
            "kode": courses[i].kode,
            "nama": courses[i].nama,
            "prodi": courses[i].prodi,
            "skor": round(score, 4),
            "komponen": komponen,
        }
        for rank, hit in enumerate(hits[:depth], start=1)
        for i, score, komponen in [split_hit(hit)]
    ]


def eval_ir(search: SearchFn, courses: list[Course], queries: list[dict]) -> list[dict]:
    records = []
    for q in queries:
        targets = {k.strip() for k in q["target_kode_mk"]}
        hits, latency = run_search(search, q["question"])
        records.append(
            {
                "dataset": "ir_eval",
                "id": q["id"],
                "query": q["question"],
                "register": q["register"],
                "jumlah_target": "tunggal" if len(targets) == 1 else "ganda",
                "target": sorted(targets),
                "latency_ms": latency,
                "top5": to_trace(hits, courses, TOP_TRACE),
                "kandidat": to_trace(hits, courses),
                "skor_kode": score_ranking((courses[i].kode for i, *_ in hits), targets),
            }
        )
    return records


def eval_complexity(search: SearchFn, courses: list[Course], queries: list[dict]) -> list[dict]:
    records = []
    for q in queries:
        hits, latency = run_search(search, q["query"])
        record = {
            "dataset": "complexity",
            "id": q["id"],
            "query": q["query"],
            "tier": q["complexity_tier"],
            "latency_ms": latency,
            "top5": to_trace(hits, courses, TOP_TRACE),
            "kandidat": to_trace(hits, courses),
            "skor_teratas": hits[0][1] if hits else 0.0,
        }
        if not q["is_out_of_scope"]:
            codes = {k.strip() for k in q["target_kode_mk"]}
            prodi = PRODI_BY_ACRONYM[q["program_studi"]]
            record["target"] = sorted(codes)
            record["prodi"] = prodi
            record["skor_kode"] = score_ranking((courses[i].kode for i, *_ in hits), codes)
            record["skor_kode_prodi"] = score_ranking(
                ((courses[i].kode, courses[i].prodi) for i, *_ in hits),
                {(code, prodi) for code in codes},
            )
        records.append(record)
    return records


def count_unreachable(courses: list[Course], queries: list[dict]) -> int:
    """Target yang kodenya ada, tetapi tidak terdaftar di prodi yang disebut kueri."""
    available = {(c.kode, c.prodi) for c in courses}
    return sum(
        (code.strip(), PRODI_BY_ACRONYM[q["program_studi"]]) not in available
        for q in queries
        if not q["is_out_of_scope"]
        for code in q["target_kode_mk"]
    )


def aggregate(dataset: str, penilaian: str, kelompok: str, records: list[dict], score_key: str) -> dict:
    row = {"dataset": dataset, "penilaian": penilaian, "kelompok": kelompok, "n": len(records)}
    if not records:
        return row | {name: None for name in METRIC_NAMES} | {"latensi_p50_ms": None, "latensi_p95_ms": None}
    for name in METRIC_NAMES:
        row[name] = float(np.mean([r[score_key][name] for r in records]))
    latencies = [r["latency_ms"] for r in records]
    row["latensi_p50_ms"] = float(np.percentile(latencies, 50))
    row["latensi_p95_ms"] = float(np.percentile(latencies, 95))
    return row


def build_rows(ir_records: list[dict], complexity_records: list[dict]) -> list[dict]:
    rows = [aggregate("ir_eval", "kode", "semua", ir_records, "skor_kode")]
    for register in ("formal", "informal", "bilingual"):
        subset = [r for r in ir_records if r["register"] == register]
        rows.append(aggregate("ir_eval", "kode", f"register: {register}", subset, "skor_kode"))
    for jumlah in ("tunggal", "ganda"):
        subset = [r for r in ir_records if r["jumlah_target"] == jumlah]
        rows.append(aggregate("ir_eval", "kode", f"target {jumlah}", subset, "skor_kode"))

    in_scope = [r for r in complexity_records if "skor_kode" in r]
    tiers = sorted({r["tier"] for r in in_scope})
    for penilaian, key in (("kode", "skor_kode"), ("kode+prodi", "skor_kode_prodi")):
        rows.append(aggregate("complexity", penilaian, "semua (dalam cakupan)", in_scope, key))
        for tier in tiers:
            subset = [r for r in in_scope if r["tier"] == tier]
            rows.append(aggregate("complexity", penilaian, tier, subset, key))
    return rows


def out_of_scope_stats(complexity_records: list[dict]) -> dict:
    out = [r for r in complexity_records if "skor_kode" not in r]
    inside = [r for r in complexity_records if "skor_kode" in r]
    mean_top = lambda records: float(np.mean([r["skor_teratas"] for r in records])) if records else 0.0
    return {
        "n": len(out),
        "tetap_mengembalikan_hasil": sum(bool(r["top5"]) for r in out),
        "rata_skor_teratas_luar_cakupan": mean_top(out),
        "rata_skor_teratas_dalam_cakupan": mean_top(inside),
    }


def format_cell(value) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def markdown_table(rows: list[dict]) -> str:
    headers = ["penilaian", "kelompok", "n", *METRIC_NAMES, "latensi_p50_ms", "latensi_p95_ms"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def write_report(
    method: str,
    output_dir: Path,
    courses: list[Course],
    ir_records: list[dict],
    complexity_records: list[dict],
    unreachable: int,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(ir_records, complexity_records)
    oos = out_of_scope_stats(complexity_records)

    with open(output_dir / "traces.jsonl", "w", encoding="utf-8") as f:
        for record in ir_records + complexity_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = [
        f"# Hasil Retrieval: {method}",
        "",
        f"- Korpus: {len(courses)} baris mata kuliah",
        "- Peringkat dideduplikasi sebelum dinilai; semua metrik dihitung pada 5 hasil teratas",
        "- recall@5 = target yang ditemukan / min(jumlah target, 5)",
        f"- Target `kode+prodi` yang tidak terdaftar di prodi kuerinya: {unreachable}",
        "",
        "## ir_eval_dataset.json",
        "",
        markdown_table([r for r in rows if r["dataset"] == "ir_eval"]),
        "",
        "## retrieval_complexity_benchmark.json",
        "",
        markdown_table([r for r in rows if r["dataset"] == "complexity"]),
        "",
        "## Kueri di luar cakupan (tidak dinilai dengan metrik di atas)",
        "",
        f"- Jumlah kueri: {oos['n']}",
        f"- Tetap mengembalikan hasil: {oos['tetap_mengembalikan_hasil']}",
        f"- Rata-rata skor teratas, luar cakupan: {oos['rata_skor_teratas_luar_cakupan']:.4f}",
        f"- Rata-rata skor teratas, dalam cakupan: {oos['rata_skor_teratas_dalam_cakupan']:.4f}",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    run_manifest.write_manifest(
        output_dir / "run.json",
        {
            "jenis": "retrieval",
            "metode": method,
            "provider": "lokal",
            "model": "-",
            "top_k": TOP_TRACE,
            "embedding": "-",
            "corpus_sha256": corpus_hash(courses),
            "dataset": "ir_eval+complexity",
            "dataset_sha256": run_manifest.file_sha256(GT_DIR / "ir_eval_dataset.json"),
            "n_unit": len(ir_records) + len(complexity_records),
        },
    )
    return rows


def run(method: str, search: SearchFn, courses: list[Course], output_dir: Path, limit: int | None = None) -> None:
    ir_queries, complexity_queries = load_queries()
    if limit is not None:
        ir_queries = ir_queries[:limit]
        complexity_queries = complexity_queries[:limit] + complexity_queries[-limit:]

    ir_records = eval_ir(search, courses, ir_queries)
    complexity_records = eval_complexity(search, courses, complexity_queries)
    unreachable = count_unreachable(courses, complexity_queries)
    rows = write_report(method, output_dir, courses, ir_records, complexity_records, unreachable)

    print(f"Metode: {method} | korpus: {len(courses)} | target tak terjangkau kode+prodi: {unreachable}")
    for row in rows:
        if row["n"]:
            print(
                f"{row['dataset']:<11}{row['penilaian']:<11}{row['kelompok']:<36}n={row['n']:<4}"
                f"Hit@1={row['hit@1']:.3f} Hit@3={row['hit@3']:.3f} Hit@5={row['hit@5']:.3f} "
                f"MRR={row['mrr@5']:.3f} NDCG={row['ndcg@5']:.3f}"
            )
    print(f"Laporan: {output_dir}")
