# Ledger Evaluasi

Hanya agregat. Tidak ada teks pertanyaan, jawaban, jawaban acuan, konteks, atau log chat di berkas ini.
Dibuat ulang dengan `uv run python scripts/ledger.py`; detail angka per kelompok ada di `runs.csv`.

| Run | Data | Jenis | Tanggal | Commit | Provider / Model | Metode | k | n | Metrik utama | Hash GT | Arsip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | complexity / kode | retrieval | 2026-11-07T09:53:16 | `f52f8c6` | lokal / - | BM25 (k1=1.5, b=0.75) | 5 | 90 | Hit@1 0.5778 · Hit@5 0.8778 · MRR 0.6993 · NDCG 0.7090 | 6beddcdd8492 | [Drive](https://drive.google.com/drive/folders/1rJIeOMZW_FV9cEOcIsjLgX79MlP8loj9) |
| bm25 | complexity / kode+prodi | retrieval | 2026-11-07T09:53:16 | `f52f8c6` | lokal / - | BM25 (k1=1.5, b=0.75) | 5 | 90 | Hit@1 0.5333 · Hit@5 0.8444 · MRR 0.6585 · NDCG 0.6674 | 6beddcdd8492 | [Drive](https://drive.google.com/drive/folders/1rJIeOMZW_FV9cEOcIsjLgX79MlP8loj9) |
| bm25 | ir_eval / kode | retrieval | 2026-11-07T09:53:16 | `f52f8c6` | lokal / - | BM25 (k1=1.5, b=0.75) | 5 | 149 | Hit@1 0.5705 · Hit@5 0.7383 · MRR 0.6295 · NDCG 0.5963 | 6beddcdd8492 | [Drive](https://drive.google.com/drive/folders/1rJIeOMZW_FV9cEOcIsjLgX79MlP8loj9) |
| bm25-uid | complexity / kode+prodi | retrieval | 2026-11-15T15:53:56 | `42e3698` | - / - | BM25 (Sparse) | 5 | 90 | Hit@1 0.5444 · Hit@5 0.8222 · MRR 0.6576 · NDCG 0.6625 | dfa2a1c94da9 | - |
| bm25-uid | complexity / uid | retrieval | 2026-11-15T15:53:56 | `42e3698` | - / - | BM25 (Sparse) | 5 | 90 | Hit@1 0.5444 · Hit@5 0.8222 · MRR 0.6576 · NDCG 0.6625 | dfa2a1c94da9 | - |
| bm25-uid | ir_eval / uid | retrieval | 2026-11-15T15:53:56 | `42e3698` | - / - | BM25 (Sparse) | 5 | 149 | Hit@1 0.5034 · Hit@5 0.7047 · MRR 0.5865 · NDCG 0.5471 | dfa2a1c94da9 | - |
| dense-uid | complexity / kode+prodi | retrieval | 2026-11-15T15:52:24 | `42e3698` | - / - | Dense (Embedding) | 5 | 90 | Hit@1 0.7889 · Hit@5 0.9222 · MRR 0.8491 · NDCG 0.8490 | dfa2a1c94da9 | - |
| dense-uid | complexity / uid | retrieval | 2026-11-15T15:52:24 | `42e3698` | - / - | Dense (Embedding) | 5 | 90 | Hit@1 0.7889 · Hit@5 0.9222 · MRR 0.8491 · NDCG 0.8490 | dfa2a1c94da9 | - |
| dense-uid | ir_eval / uid | retrieval | 2026-11-15T15:52:24 | `42e3698` | - / - | Dense (Embedding) | 5 | 149 | Hit@1 0.7114 · Hit@5 0.8389 · MRR 0.7642 · NDCG 0.7216 | dfa2a1c94da9 | - |
| hybrid-reformulasi-uid | complexity / kode+prodi | retrieval | 2026-11-15T19:49:51 | `77709db` | OpenAI API / gpt-6-luna | Hybrid + Query Reformulation | 5 | 90 | Hit@1 0.7889 · Hit@5 0.9444 · MRR 0.8500 · NDCG 0.8446 | dfa2a1c94da9 | - |
| hybrid-reformulasi-uid | complexity / uid | retrieval | 2026-11-15T19:49:51 | `77709db` | OpenAI API / gpt-6-luna | Hybrid + Query Reformulation | 5 | 90 | Hit@1 0.7889 · Hit@5 0.9444 · MRR 0.8500 · NDCG 0.8446 | dfa2a1c94da9 | - |
| hybrid-reformulasi-uid | ir_eval / uid | retrieval | 2026-11-15T19:49:51 | `77709db` | OpenAI API / gpt-6-luna | Hybrid + Query Reformulation | 5 | 149 | Hit@1 0.7315 · Hit@5 0.8993 · MRR 0.7993 · NDCG 0.7618 | dfa2a1c94da9 | - |
| hybrid-rrf-uid | complexity / kode+prodi | retrieval | 2026-11-15T15:53:45 | `42e3698` | - / - | Hybrid RRF (Sparse + Dense) | 5 | 90 | Hit@1 0.7556 · Hit@5 0.9111 · MRR 0.8161 · NDCG 0.8093 | dfa2a1c94da9 | - |
| hybrid-rrf-uid | complexity / uid | retrieval | 2026-11-15T15:53:45 | `42e3698` | - / - | Hybrid RRF (Sparse + Dense) | 5 | 90 | Hit@1 0.7556 · Hit@5 0.9111 · MRR 0.8161 · NDCG 0.8093 | dfa2a1c94da9 | - |
| hybrid-rrf-uid | ir_eval / uid | retrieval | 2026-11-15T15:53:45 | `42e3698` | - / - | Hybrid RRF (Sparse + Dense) | 5 | 149 | Hit@1 0.6510 · Hit@5 0.8054 · MRR 0.7142 · NDCG 0.6822 | dfa2a1c94da9 | - |
| lokal-llama3.2-3b-dense-embedding-k3 | GT_MASTER | ground_truth | 2026-11-12 16:55 | `77709db` | Lokal (Ollama) / llama3.2:3b | Dense (Embedding) | 3 | 449 | BERT F1 0.7776 · ROUGE-L 0.3986 · Hit@1 0.9512 · TTFT 8918.2 ms | 034daa4881e5 | - |
