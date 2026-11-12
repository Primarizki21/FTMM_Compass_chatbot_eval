# Ledger Evaluasi

Hanya agregat. Tidak ada teks pertanyaan, jawaban, jawaban acuan, konteks, atau log chat di berkas ini.
Dibuat ulang dengan `uv run python scripts/ledger.py`; detail angka per kelompok ada di `runs.csv`.

| Run | Data | Jenis | Tanggal | Commit | Provider / Model | Metode | k | n | Metrik utama | Hash GT | Arsip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | complexity / kode | retrieval | 2026-11-07T09:53:16 | `f52f8c6` | lokal / - | BM25 (k1=1.5, b=0.75) | 5 | 90 | Hit@1 0.5778 · Hit@5 0.8778 · MRR 0.6993 · NDCG 0.7090 | 6beddcdd8492 | [Drive](https://drive.google.com/drive/folders/1rJIeOMZW_FV9cEOcIsjLgX79MlP8loj9) |
| bm25 | complexity / kode+prodi | retrieval | 2026-11-07T09:53:16 | `f52f8c6` | lokal / - | BM25 (k1=1.5, b=0.75) | 5 | 90 | Hit@1 0.5333 · Hit@5 0.8444 · MRR 0.6585 · NDCG 0.6674 | 6beddcdd8492 | [Drive](https://drive.google.com/drive/folders/1rJIeOMZW_FV9cEOcIsjLgX79MlP8loj9) |
| bm25 | ir_eval / kode | retrieval | 2026-11-07T09:53:16 | `f52f8c6` | lokal / - | BM25 (k1=1.5, b=0.75) | 5 | 149 | Hit@1 0.5705 · Hit@5 0.7383 · MRR 0.6295 · NDCG 0.5963 | 6beddcdd8492 | [Drive](https://drive.google.com/drive/folders/1rJIeOMZW_FV9cEOcIsjLgX79MlP8loj9) |
| lokal-llama3.2-3b-dense-embedding-k3 | GT_MASTER | ground_truth | 2026-11-12 16:55 | `87c8ce2` | Lokal (Ollama) / llama3.2:3b | Dense (Embedding) | 3 | 449 | BERT F1 0.7776 · ROUGE-L 0.3986 · Hit@1 0.9512 · TTFT 8918.2 ms | 034daa4881e5 | - |
