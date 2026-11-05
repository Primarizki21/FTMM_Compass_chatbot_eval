"""Menyimpan setiap giliran chat ke CSV agar bisa dievaluasi. Kolom jawaban_acuan diisi manual saat menilai."""

import csv
from pathlib import Path

from lib.kb import ROOT

LOG_PATH = ROOT / "chat_logs" / "chat_history.csv"

FIELDS = [
    "waktu",
    "sesi_id",
    "giliran",
    "metode",
    "top_k",
    "filter_prodi",
    "penyedia_llm",
    "model_llm",
    "model_embedding",
    "pertanyaan",
    "kueri_reformulasi",
    "jawaban",
    "kode_mk_diambil",
    "file_sumber",
    "skor_retrieval",
    "konteks",
    "latensi_retrieval_ms",
    "ttft_ms",
    "latensi_generasi_ms",
    "token_prompt",
    "token_jawaban",
    "jawaban_acuan",
]


def migrate_columns(path: Path, fields: list[str]) -> None:
    """Menyesuaikan file lama saat kolom bertambah, agar baris baru tidak bergeser dari judul kolomnya."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == fields:
            return
        rows = list(reader)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, fields: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    if not is_new:
        migrate_columns(path, fields)
    # BOM hanya di awal file supaya Excel membaca huruf dengan benar.
    with open(path, "a", encoding="utf-8-sig" if is_new else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        if is_new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def append_turn(row: dict, path: Path | None = None) -> None:
    append_row(path or LOG_PATH, FIELDS, row)
