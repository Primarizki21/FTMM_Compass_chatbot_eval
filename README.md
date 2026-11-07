# RAG

UI chatbot dan evaluasi metode RAG FTMM COMPASS. LLM bisa memakai 9Router (OpenAI-compatible, lewat tunnel tim), model lokal (Ollama), atau Gemini API.
Knowledge base berasal dari tim (`Dataset Matkul Prodi` di Drive: 5 JSON kurikulum + panduan regulasi); ground truth diturunkan dari `gt_master.json` repo chatbot.

- **Mode Chat Bebas**: bertanya apa saja, riwayat tersimpan ke `chat_logs/chat_history.csv`.
- **Mode Evaluasi Ground Truth**: soal `GT_MASTER.csv` dijawab otomatis, lalu dinilai dengan BERTScore dan ROUGE.
- **Metode retrieval**: BM25 (Sparse), Dense (Embedding), Hybrid RRF, dan Hybrid + Query Reformulation.

## Instalasi

Butuh [uv](https://docs.astral.sh/uv/) (Astral). `uv` yang mengelola versi Python >= 3.12 dan seluruh dependensi, jadi
tidak perlu venv manual. Berjalan di Linux, macOS, WSL2, maupun Windows; perintah di dokumen ini relatif terhadap root repo
dan tidak bergantung pada path absolut atau lokasi cache tertentu.

```
uv sync
```

### Data yang tidak ikut repo

`knowledge_base/` dan `ground_truth/` sengaja gitignored, jadi tidak ada di clone baru. Minta ke tim, lalu letakkan di root repo:

- `knowledge_base/` — JSON kurikulum per prodi dan subfolder `regulasi/`, apa adanya (format chunk memengaruhi hasil evaluasi).
- `ground_truth/` — `GT_MASTER.csv`, `ir_eval_dataset.json`, dan `retrieval_complexity_benchmark.json`.

Tanpa keduanya aplikasi gagal memuat dokumen dan sebagian test gagal karena data tidak ada, bukan karena bug kode.
Ground truth diturunkan dari repo chatbot: `data/ground_truth/gt_master.json` dengan kolom `id, question, answer, source_id, source_type`.

Setelah data lengkap, periksa hasilnya; semua test harus lolos:

```
uv run pytest
```

Bila cache `uv` perlu dipindah (disk kecil atau kebijakan mesin), arahkan `UV_CACHE_DIR` pada shell masing-masing;
jangan mengubah perintah proyek agar bergantung pada lokasi cache.

### Perintah singkat (Makefile)

Opsional. Butuh `make` (tersedia di Linux, macOS, WSL2). Di Windows tanpa `make`, jalankan perintah `uv run ...` yang sama langsung.

| Perintah | Fungsi |
|---|---|
| `make sync` | Pasang dependensi |
| `make test` | Jalankan seluruh test |
| `make index` | Buat atau timpa index Dense untuk `EMBED_MODEL_NAME` aktif |
| `make run` atau `make streamlit` | Jalankan UI |
| `make bm25` | Benchmark retrieval BM25 tanpa LLM |
| `make ledger` | Susun ledger agregat dari `results/` |
| `make clean-index` | Hapus index Dense agar dibuat ulang |

Model embedding mengikuti `EMBED_MODEL_NAME`. Ganti model = jalankan `make index` sekali; berkas model lama tetap ada di `index/` dan bisa dihapus manual.

### Memakai 9Router (jalur tim)

Akses LLM tim memakai tunnel, bukan `localhost`: kalau tunnel mati, kondisi yang terlihat sama untuk semua orang.

1. Salin `.env.example` menjadi `.env`.
2. Isi `NINE_ROUTER_API_KEY=` dengan key dari dashboard 9router. Kunci Google Gemini ditolak.
3. `NINE_ROUTER_HOST` dan `NINE_ROUTER_MODEL` sudah terisi default tim; model wajib memakai prefix provider (`ag/...`).
4. Pilih `9Router (OpenAI-compatible)` di sidebar.

Pertanyaan dan potongan knowledge base dikirim ke 9Router. Embedding tetap memakai model lokal, jadi metode selain BM25 tetap membutuhkan Ollama.

### Memakai Gemini (opsional)

1. Isi `GEMINI_API_KEY=` di `.env`. Jangan membagikan file `.env`.
2. Pilih `Gemini API` di sidebar.

Pertanyaan dan potongan knowledge base akan dikirim ke server Google. Pastikan ini diizinkan untuk data internal FTMM.

### Memakai LLM lokal

`ollama list` harus menampilkan `llama3.2:3b` (atau nilai `LLM_MODEL_NAME`). Kalau daemon tidak terbaca, cek `OLLAMA_HOST`
dan pastikan layanan Ollama aktif, lalu muat ulang halaman.

## Menjalankan

1. Buat index Dense. Cukup sekali per kombinasi isi `knowledge_base/` dan model embedding; tidak perlu diulang setiap
   kali menjalankan UI. Jalankan ulang hanya bila knowledge base berubah atau `EMBED_MODEL_NAME` diganti. Metode BM25
   tidak membutuhkannya. Butuh `nomic-embed-text-v2-moe` di Ollama.
   ```
   uv run python build_index.py
   ```
   Index dimuat saat kueri Dense/Hybrid pertama dan divalidasi terhadap hash korpus. Bila index belum ada atau basi,
   UI menampilkan pesan untuk menjalankan skrip ini. Setelah rebuild, restart Streamlit agar matriks lama tidak tertahan
   di cache.
2. Jalankan UI, lalu buka alamat lokal yang dicetak Streamlit (default `http://127.0.0.1:8501`)
   ```
   uv run streamlit run app.py
   ```
3. Untuk membandingkan retrieval tanpa LLM/GPU (opsional):
   ```
   uv run python run_bm25.py --limit 20
   ```
   Tanpa `--limit` seluruh dataset dijalankan; hasil ditulis ke `results/bm25/`.

## Evaluasi Ground Truth

1. Pilih metode, sumber LLM, dan top-k di sidebar.
2. Isi jumlah soal, misalnya 30 untuk uji coba. Sampel tersebar merata ke semua kategori.
3. Klik **Mulai / Lanjutkan**. Kalau berhenti atau error, klik lagi; soal yang selesai tidak diulang.
4. Klik **Hitung skor** setelah selesai. BERTScore berjalan di CPU. Saat pertama kali, model
   `bert-base-multilingual-cased` diunduh ke `.hf_cache/`.

Hasil disimpan per kombinasi LLM, metode, dan top-k di `results/evaluasi/`:

| File | Isi |
|---|---|
| `jawaban_*.csv` | Pertanyaan, jawaban acuan, jawaban chatbot, konteks, dan latensi per soal |
| `skor_*.csv` | Isi file jawaban ditambah skor BERTScore dan ROUGE per soal |
| `ringkasan_*.csv` | Rata-rata per kategori soal |
| `jejak_*.jsonl` | Satu baris per soal: kandidat retrieval (`rank`, kode, nama, prodi, skor, `komponen` dense/sparse/rrf, `peringkat` tiap tahap), konteks, jawaban, dan `error` bila gagal |
| `nilai_*.jsonl` | Metrik per soal (BERTScore, ROUGE) setelah scoring |
| `run_*.json` | Manifest run: konfigurasi, hash korpus, hash GT, commit |

Kalau file CSV terbuka di Excel saat evaluasi berjalan, penyimpanan bisa gagal. Tutup dulu, atau gunakan tombol unduh.

### Metrik

| Metrik | Mengukur | Cakupan |
|---|---|---|
| Hit@1, Hit@k, MRR | Dokumen sumber ditemukan dan seberapa tinggi peringkatnya | Soal silabus saja |
| BERTScore (P, R, F1) | Kemiripan makna jawaban dengan jawaban acuan | Semua soal |
| ROUGE-1, ROUGE-2, ROUGE-L (F1) | Kemiripan kata jawaban dengan jawaban acuan | Semua soal |
| TTFT, latensi total P50 dan P95 | Kecepatan | Semua soal |

Retrieval soal regulasi tidak dinilai karena `source_id`-nya tidak menunjuk ke satu butir yang pasti:
angka yang sama dipakai di beberapa file panduan, dan banyak soal merujuk beberapa sumber sekaligus.

Saat menafsirkan hasil:

- Semua soal silabus menyebut kode MK, sehingga BM25 diuntungkan. Untuk membandingkan metode retrieval,
  gunakan juga `run_bm25.py` dengan `ir_eval_dataset.json` dan `retrieval_complexity_benchmark.json`.
- Jawaban acuan silabus dibuat dari template katalog, dan sebagian baris GT memuat pertanyaan duplikat. Hitung ulang dari `GT_MASTER.csv` bila butuh angka; sampel GT tidak sepenuhnya independen.
- Jawaban yang benar tapi memakai kata lain mendapat ROUGE rendah. BERTScore lebih tepat untuk parafrasa.
- BERTScore dan ROUGE tidak mendeteksi jawaban yang terdengar benar tapi tidak didukung konteks.

## Ledger

`scripts/ledger.py` merangkum `results/` menjadi `docs/ledger/RUNS.md` dan `docs/ledger/runs.csv`. Isinya agregat per run:
metode, provider/model, top-k, n, metrik, commit, hash ground truth, dan tautan arsip Drive.

```
uv run python scripts/ledger.py
```

Teks pertanyaan, jawaban, jawaban acuan, konteks, dan `traces.jsonl` tidak ikut disalin. Hasil mentah tetap di `results/`
dan dibagikan lewat Drive, bukan git. Aturan lengkap ada di `docs/ledger/README.md`.

## Pengaturan

Bisa diisi di `.env`:

| Variabel | Fungsi |
|---|---|
| `NINE_ROUTER_HOST`, `NINE_ROUTER_MODEL`, `NINE_ROUTER_API_KEY` | Provider utama tim lewat tunnel |
| `GEMINI_API_KEY`, `GEMINI_MODEL_NAME` | Provider pembanding langsung ke Google |
| `LLM_MODEL_NAME`, `EMBED_MODEL_NAME`, `OLLAMA_HOST` | LLM lokal dan embedding |

Model LLM juga bisa diganti di sidebar. Index harus dibuat ulang setelah mengganti model embedding.

## Isi folder

| Lokasi | Isi |
|---|---|
| `app.py` | UI Streamlit |
| `build_index.py` | Membuat index Dense |
| `run_bm25.py` | Evaluasi retrieval BM25 (`--output`, `--limit`) |
| `scripts/ledger.py` | Menyusun ledger agregat dari `results/` |
| `lib/` | Knowledge base, retriever, LLM, log CSV, evaluasi, metrik |
| `tests/` | Test pytest (retrieval, evaluasi, provider, ledger) |
| `docs/ledger/` | Ledger agregat; satu-satunya artefak evaluasi yang di-commit |
| `knowledge_base/` | JSON kurikulum per prodi dan `regulasi/` dari tim |
| `ground_truth/` | `GT_MASTER.csv` dan dataset retrieval; turunan `gt_master.json` |
| `index/`, `chat_logs/`, `results/`, `.hf_cache/` | Dibuat otomatis saat program berjalan |

Yang disamakan dengan aplikasi: format chunk kurikulum dan regulasi, opsi LLM lokal `PureRAGEngine`, tokenizer dan
parameter BM25, RRF k=60, serta prompt reformulasi.

Yang berbeda: kurikulum dan regulasi dicari dalam satu index (di aplikasi terpisah), prompt sistem menyebut regulasi,
pencarian Dense memakai NumPy (hasil setara `IndexFlatIP`), batas keluaran 9Router 512 token mengikuti `PureRAGEngine`,
batas Gemini 2.048 token karena token thinking ikut terhitung, dan ada filter prodi opsional.

Batasan: kueri lanjutan seperti "prasyaratnya apa?" dicari apa adanya, sama seperti `PureRAGEngine`.

## Kerahasiaan

`.env`, `knowledge_base/`, `ground_truth/`, `index/`, `chat_logs/`, dan `results/` sudah masuk `.gitignore`.
Jangan diunggah ke repositori publik.

Repo ini publik. Satu-satunya artefak evaluasi yang boleh di-commit adalah `docs/ledger/` (agregat, tanpa teks soal atau
jawaban). Hasil mentah dibagikan lewat Google Drive tim.
