# RAG

UI chatbot dan evaluasi metode RAG FTMM COMPASS. LLM bisa memakai model lokal (Ollama) atau Gemini API.
Knowledge base disalin dari aplikasi (`E:\IRIS_longterm\1`).

- **Mode Chat Bebas**: bertanya apa saja, riwayat tersimpan ke `chat_logs/chat_history.csv`.
- **Mode Evaluasi Ground Truth**: soal `GT_MASTER.csv` (449) dijawab otomatis, lalu dinilai dengan BERTScore dan ROUGE.
- **Metode retrieval**: BM25 (Sparse), Dense (Embedding), Hybrid RRF, dan Hybrid + Query Reformulation.

## Instalasi

Jalankan dari `E:\IRIS_longterm\RAG`. Cache diarahkan ke drive E karena drive C hampir penuh.

```
set UV_CACHE_DIR=E:\IRIS_longterm\.uv-cache
python -m uv sync
```

Di jendela CMD baru, `set UV_CACHE_DIR=...` perlu dijalankan lagi kalau daftar paket berubah.

Periksa hasilnya; semua test harus lolos:

```
python -m uv run pytest
```

### Memakai Gemini (opsional)

1. Salin `.env.example` menjadi `.env`.
2. Isi `GEMINI_API_KEY=` dengan API key. Jangan membagikan file `.env`.

Pertanyaan dan potongan knowledge base akan dikirim ke server Google. Pastikan ini diizinkan untuk data internal FTMM.
Embedding tetap memakai model lokal, jadi metode selain BM25 tetap membutuhkan Ollama.

### Memakai LLM lokal

`ollama list` harus menampilkan `llama3.2:3b`. Kalau modelnya tidak terbaca, tutup Ollama lalu buka lagi
agar membaca `E:\Ollama_models`.

## Menjalankan

1. Buat index Dense. Wajib untuk metode selain BM25, dan diulang setiap kali knowledge base berubah.
   Butuh `nomic-embed-text-v2-moe` di Ollama.
   ```
   python -m uv run python build_index.py
   ```
2. Jalankan UI, lalu buka http://localhost:8501
   ```
   python -m uv run streamlit run app.py
   ```

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

Kalau file CSV terbuka di Excel saat evaluasi berjalan, penyimpanan bisa gagal. Tutup dulu, atau gunakan tombol unduh.

### Metrik

| Metrik | Mengukur | Cakupan |
|---|---|---|
| Hit@1, Hit@k, MRR | Dokumen sumber ditemukan dan seberapa tinggi peringkatnya | 369 soal silabus |
| BERTScore (P, R, F1) | Kemiripan makna jawaban dengan jawaban acuan | Semua soal |
| ROUGE-1, ROUGE-2, ROUGE-L (F1) | Kemiripan kata jawaban dengan jawaban acuan | Semua soal |
| TTFT, latensi total P50 dan P95 | Kecepatan | Semua soal |

Retrieval soal regulasi tidak dinilai karena `source_id`-nya tidak menunjuk ke satu butir yang pasti:
angka yang sama dipakai di beberapa file panduan, dan banyak soal merujuk beberapa sumber sekaligus.

Saat menafsirkan hasil:

- Semua soal silabus menyebut kode MK, sehingga BM25 diuntungkan. Untuk membandingkan metode retrieval,
  gunakan juga `run_bm25.py` dengan `ir_eval_dataset.json` dan `retrieval_complexity_benchmark.json`.
- Jawaban acuan silabus dibuat dari template katalog, dan 7 pertanyaan muncul dua kali di GT.
- Jawaban yang benar tapi memakai kata lain mendapat ROUGE rendah. BERTScore lebih tepat untuk parafrasa.
- BERTScore dan ROUGE tidak mendeteksi jawaban yang terdengar benar tapi tidak didukung konteks.

## Pengaturan

Bisa diisi di `.env`: `GEMINI_API_KEY`, `GEMINI_MODEL_NAME`, `LLM_MODEL_NAME`, `EMBED_MODEL_NAME`.
Model LLM juga bisa diganti di sidebar. Index harus dibuat ulang setelah mengganti model embedding.

## Isi folder

| Lokasi | Isi |
|---|---|
| `app.py` | UI Streamlit |
| `build_index.py` | Membuat index Dense |
| `run_bm25.py` | Evaluasi retrieval BM25 |
| `lib/` | Knowledge base, retriever, LLM, log CSV, evaluasi, metrik |
| `knowledge_base/` | 5 file kurikulum (369 MK) dan `regulasi/` (262 butir), salinan dari folder `1` |
| `ground_truth/` | Kunci jawaban evaluasi, bukan knowledge base |
| `index/`, `chat_logs/`, `results/`, `.hf_cache/` | Dibuat otomatis saat program berjalan |

Yang disamakan dengan aplikasi: format chunk kurikulum dan regulasi, opsi LLM lokal `PureRAGEngine`, tokenizer dan
parameter BM25, RRF k=60, serta prompt reformulasi.

Yang berbeda: kurikulum dan regulasi dicari dalam satu index (di aplikasi terpisah), prompt sistem menyebut regulasi,
pencarian Dense memakai NumPy (hasil setara `IndexFlatIP`), batas keluaran Gemini 2.048 token karena token thinking ikut
terhitung (lokal 512), dan ada filter prodi opsional.

Batasan: kueri lanjutan seperti "prasyaratnya apa?" dicari apa adanya, sama seperti `PureRAGEngine`.

## Kerahasiaan

`.env`, `knowledge_base/`, `ground_truth/`, `index/`, `chat_logs/`, dan `results/` sudah masuk `.gitignore`.
Jangan diunggah ke repositori publik.
