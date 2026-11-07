import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("ledger", ROOT / "scripts" / "ledger.py")
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)

SECRET = "RAHASIA-SILABUS-JANGAN-BOCOR"

RETRIEVAL_ROW = {
    "dataset": "ir_eval",
    "penilaian": "kode",
    "kelompok": "semua",
    "n": "10",
    "hit@1": "0.9000",
    "hit@3": "1.0000",
    "hit@5": "1.0000",
    "mrr@5": "0.9400",
    "recall@5": "0.9700",
    "ndcg@5": "0.9500",
    "latensi_p50_ms": "0.6000",
    "latensi_p95_ms": "0.7000",
}

SUMMARY_ROW = {
    "kelompok": "Semua",
    "n": "20",
    "n_retrieval": "20",
    "hit@1": "0.8000",
    "hit@k": "0.9500",
    "mrr": "0.8800",
    "bert_precision": "0.7000",
    "bert_recall": "0.7100",
    "bert_f1": "0.7050",
    "rouge1_f1": "0.4000",
    "rouge2_f1": "0.2000",
    "rougeL_f1": "0.3900",
    "ttft_p50_ms": "812.0",
    "total_p50_ms": "1400.0",
    "total_p95_ms": "2600.0",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_results(tmp_path: Path) -> Path:
    results = tmp_path / "results"

    write_csv(results / "bm25" / "metrics.csv", [RETRIEVAL_ROW])
    (results / "bm25" / "summary.md").write_text("# Hasil Retrieval: BM25 (k1=1.5, b=0.75)\n", encoding="utf-8")

    write_csv(
        results / "evaluasi" / "skor_lokal-uji.csv",
        [
            {
                "pertanyaan": SECRET,
                "jawaban": SECRET,
                "jawaban_acuan": SECRET,
                "konteks": SECRET,
                "penyedia_llm": "9Router (OpenAI-compatible)",
                "model_llm": "ag/gemini-x",
                "metode": "BM25 (Sparse)",
                "top_k": "3",
                "model_embedding": "",
            }
        ],
    )
    write_csv(results / "evaluasi" / "ringkasan_lokal-uji.csv", [SUMMARY_ROW])
    (results / "evaluasi" / "traces.jsonl").write_text(SECRET + "\n", encoding="utf-8")
    return results


def test_ledger_keeps_only_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ARCHIVES", tmp_path / "archives.csv")
    records = ledger.collect(make_results(tmp_path))
    out = tmp_path / "ledger"
    ledger.write_ledger(records, out)

    published = (out / "runs.csv").read_text(encoding="utf-8") + (out / "RUNS.md").read_text(encoding="utf-8")
    assert SECRET not in published

    header = next(csv.reader(open(out / "runs.csv", encoding="utf-8")))
    assert not {"pertanyaan", "jawaban", "jawaban_acuan", "konteks"} & set(header)

    rows = list(csv.DictReader(open(out / "runs.csv", encoding="utf-8")))
    retrieval = next(row for row in rows if row["jenis"] == "retrieval")
    assert retrieval["metode"] == "BM25 (k1=1.5, b=0.75)"
    assert retrieval["hit_at_k"] == "1.0000"

    ground_truth = next(row for row in rows if row["jenis"] == "ground_truth")
    assert ground_truth["provider"] == "9Router (OpenAI-compatible)"
    assert ground_truth["model"] == "ag/gemini-x"
    assert ground_truth["bert_f1"] == "0.7050"
    assert ground_truth["run"] == "lokal-uji"


def test_ledger_summary_prefers_aggregate_group(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ARCHIVES", tmp_path / "archives.csv")
    records = ledger.collect(make_results(tmp_path))
    rows = ledger.overall(records)

    assert len(rows) == 2
    assert {row["run"] for row in rows} == {"bm25", "lokal-uji"}


def test_ledger_merges_drive_archive_link(tmp_path, monkeypatch):
    archives = tmp_path / "archives.csv"
    write_csv(archives, [{"run": "bm25", "drive_url": "https://drive.google.com/drive/folders/uji"}])
    monkeypatch.setattr(ledger, "ARCHIVES", archives)

    rows = ledger.collect(make_results(tmp_path))
    links = {row["arsip_drive"] for row in rows if row["run"] == "bm25"}

    assert links == {"https://drive.google.com/drive/folders/uji"}
    assert "drive.google.com" in ledger.render_markdown(rows)


def test_ledger_uses_manifest_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "ARCHIVES", tmp_path / "archives.csv")
    results = make_results(tmp_path)
    (results / "bm25" / "run.json").write_text(
        json.dumps(
            {
                "commit": "deadbee",
                "ditulis": "2026-01-02T03:04:05",
                "metode": "BM25 (k1=1.5, b=0.75)",
                "provider": "lokal",
                "model": "-",
                "top_k": 5,
                "embedding": "-",
                "corpus_sha256": "korpus123",
                "dataset_sha256": "dataset456",
            }
        ),
        encoding="utf-8",
    )
    (results / "evaluasi" / "run_lokal-uji.json").write_text(
        json.dumps(
            {
                "commit": "cafe123",
                "provider": "9Router (OpenAI-compatible)",
                "model": "ag/gemini-x",
                "metode": "BM25 (Sparse)",
                "top_k": 3,
                "embedding": "",
                "corpus_sha256": "korpus123",
                "gt_sha256": "gt789",
            }
        ),
        encoding="utf-8",
    )

    rows = ledger.collect(results)
    retrieval = next(row for row in rows if row["jenis"] == "retrieval")
    ground_truth = next(row for row in rows if row["jenis"] == "ground_truth")

    assert retrieval["corpus_hash"] == "korpus123"
    assert retrieval["dataset_hash"] == "dataset456"
    assert retrieval["commit"] == "deadbee"
    assert ground_truth["corpus_hash"] == "korpus123"
    assert ground_truth["dataset_hash"] == "gt789"
    assert ground_truth["commit"] == "cafe123"
