"""Evaluasi jawaban terhadap ground truth: menjawab soal satu per satu, lalu menghitung skor secara terpisah."""

import csv
import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from lib import llm, rag
from lib.chat_log import append_row
from lib.kb import ROOT

GT_PATH = ROOT / "ground_truth" / "GT_MASTER.csv"
EVAL_DIR = ROOT / "results" / "evaluasi"
# Model BERTScore berukuran ratusan MB; drive C sering penuh, jadi disimpan di folder project.
HF_CACHE_DIR = ROOT / ".hf_cache"

ANSWER_FIELDS = [
    "id",
    "kategori",
    "source_id",
    "penyedia_llm",
    "model_llm",
    "metode",
    "top_k",
    "model_embedding",
    "pertanyaan",
    "jawaban_acuan",
    "jawaban",
    "kueri_reformulasi",
    "kode_diambil",
    "file_sumber",
    "skor_retrieval",
    "konteks",
    "peringkat_sumber",
    "latensi_retrieval_ms",
    "ttft_ms",
    "latensi_generasi_ms",
    "token_prompt",
    "token_jawaban",
    "waktu",
]
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]
SCORE_FIELDS = ["bert_precision", "bert_recall", "bert_f1", *(f"{name}_f1" for name in ROUGE_METRICS)]
SUMMARY_FIELDS = [
    "kelompok",
    "n",
    "n_retrieval",
    "hit@1",
    "hit@k",
    "mrr",
    *SCORE_FIELDS,
    "ttft_p50_ms",
    "total_p50_ms",
    "total_p95_ms",
]


def load_ground_truth(path: Path = GT_PATH) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sample_evenly(rows: list[dict], count: int) -> list[dict]:
    """Sampel tetap yang tersebar merata, agar uji kecil tidak hanya berisi soal silabus di awal file."""
    if count >= len(rows):
        return rows
    positions = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[i] for i in positions]


def run_name(config: llm.LLMConfig, method: str, top_k: int) -> str:
    raw = f"{config.provider.split()[0]}_{config.model}_{method.split()[0]}_{method.split()[-1]}_k{top_k}".lower()
    return re.sub(r"[^a-z0-9.-]+", "-", raw)


def answers_path(name: str, eval_dir: Path | None = None) -> Path:
    return (eval_dir or EVAL_DIR) / f"jawaban_{name}.csv"


def scores_path(name: str, eval_dir: Path | None = None) -> Path:
    return (eval_dir or EVAL_DIR) / f"skor_{name}.csv"


def summary_path(name: str, eval_dir: Path | None = None) -> Path:
    return (eval_dir or EVAL_DIR) / f"ringkasan_{name}.csv"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def source_rank(item: dict, retrieved_codes: list[str]) -> str:
    """Peringkat sumber hanya bisa dihitung untuk silabus; source_id regulasi tidak menunjuk ke satu butir yang pasti."""
    if not item["source_type"].startswith("Silabus"):
        return ""
    unique_codes = list(dict.fromkeys(retrieved_codes))
    source = item["source_id"].strip()
    return str(unique_codes.index(source) + 1) if source in unique_codes else "0"


def answer_question(
    retriever: rag.CourseRetriever, item: dict, method: str, top_k: int, config: llm.LLMConfig
) -> dict:
    """Tiap soal berdiri sendiri: tanpa riwayat chat, dan jawaban acuan tidak pernah dikirim ke LLM."""
    question = item["question"]
    result = retriever.retrieve(method, question, top_k, reformulate=lambda q: llm.reformulate_query(q, config))
    stats: dict = {}
    messages = rag.build_messages(question, result.hits, history=[])
    answer = "".join(llm.stream_chat(messages, config, rag.keep_alive_for(method), stats))
    codes = [hit.course.kode for hit in result.hits]
    return {
        "id": item["id"],
        "kategori": item["source_type"],
        "source_id": item["source_id"],
        "penyedia_llm": config.provider,
        "model_llm": config.model,
        "metode": method,
        "top_k": top_k,
        "model_embedding": llm.EMBED_MODEL if method in rag.METHODS_NEEDING_EMBEDDING else "",
        "pertanyaan": question,
        "jawaban_acuan": item["answer"],
        "jawaban": answer,
        "kueri_reformulasi": result.reformulated_query,
        "kode_diambil": "; ".join(codes),
        "file_sumber": "; ".join(hit.course.sumber for hit in result.hits),
        "skor_retrieval": "; ".join(f"{hit.score:.4f}" for hit in result.hits),
        "konteks": "\n\n---\n\n".join(hit.course.chunk for hit in result.hits),
        "peringkat_sumber": source_rank(item, codes),
        "latensi_retrieval_ms": round(result.latency_ms, 1),
        "ttft_ms": round(stats.get("ttft_ms", 0.0), 1),
        "latensi_generasi_ms": round(stats.get("generation_ms", 0.0), 1),
        "token_prompt": stats.get("prompt_tokens", 0),
        "token_jawaban": stats.get("answer_tokens", 0),
        "waktu": datetime.now().isoformat(timespec="seconds"),
    }


def save_answer(name: str, row: dict, eval_dir: Path | None = None) -> None:
    append_row(answers_path(name, eval_dir), ANSWER_FIELDS, row)


def compute_bert(candidates: list[str], references: list[str]) -> list[tuple[float, float, float]]:
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    from bert_score import score

    precision, recall, f1 = score(candidates, references, lang="id", device="cpu", batch_size=16, verbose=False)
    return list(zip(precision.tolist(), recall.tolist(), f1.tolist()))


def compute_rouge(candidates: list[str], references: list[str]) -> list[dict[str, float]]:
    from rouge_score import rouge_scorer

    # Stemmer bawaan hanya untuk bahasa Inggris, jadi dimatikan untuk teks Indonesia.
    scorer = rouge_scorer.RougeScorer(ROUGE_METRICS, use_stemmer=False)
    results = []
    for candidate, reference in zip(candidates, references):
        scores = scorer.score(reference, candidate)
        results.append({f"{name}_f1": scores[name].fmeasure for name in ROUGE_METRICS})
    return results


def score_answers(
    rows: list[dict],
    bert: Callable[[list[str], list[str]], list[tuple[float, float, float]]] | None = None,
    rouge: Callable[[list[str], list[str]], list[dict[str, float]]] | None = None,
) -> list[dict]:
    candidates = [row["jawaban"] for row in rows]
    references = [row["jawaban_acuan"] for row in rows]
    bert_scores = (bert or compute_bert)(candidates, references)
    rouge_results = (rouge or compute_rouge)(candidates, references)
    scored = []
    for row, (precision, recall, f1), rouge_scores in zip(rows, bert_scores, rouge_results):
        scored.append({**row, "bert_precision": precision, "bert_recall": recall, "bert_f1": f1, **rouge_scores})
    return scored


def summarize_group(label: str, rows: list[dict]) -> dict:
    ranks = [int(row["peringkat_sumber"]) for row in rows if row["peringkat_sumber"] != ""]
    totals = [float(row["latensi_retrieval_ms"]) + float(row["latensi_generasi_ms"]) for row in rows]
    summary = {
        "kelompok": label,
        "n": len(rows),
        "n_retrieval": len(ranks),
        "hit@1": np.mean([rank == 1 for rank in ranks]) if ranks else "",
        "hit@k": np.mean([rank > 0 for rank in ranks]) if ranks else "",
        "mrr": np.mean([1.0 / rank if rank else 0.0 for rank in ranks]) if ranks else "",
        "ttft_p50_ms": np.percentile([float(row["ttft_ms"]) for row in rows], 50),
        "total_p50_ms": np.percentile(totals, 50),
        "total_p95_ms": np.percentile(totals, 95),
    }
    for field in SCORE_FIELDS:
        summary[field] = np.mean([float(row[field]) for row in rows])
    return {key: round(float(value), 4) if isinstance(value, (float, np.floating)) else value for key, value in summary.items()}


def summarize(scored: list[dict]) -> list[dict]:
    categories = sorted({row["kategori"] for row in scored})
    groups = [("Semua", scored)] + [(c, [row for row in scored if row["kategori"] == c]) for c in categories]
    return [summarize_group(label, rows) for label, rows in groups if rows]


def write_scores(name: str, scored: list[dict], summary: list[dict], eval_dir: Path | None = None) -> None:
    target = scores_path(name, eval_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*ANSWER_FIELDS, *SCORE_FIELDS], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(scored)
    with open(summary_path(name, eval_dir), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
