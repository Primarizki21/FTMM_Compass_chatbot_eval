# Ledger Evaluasi

Berkas di folder ini adalah satu-satunya artefak evaluasi yang boleh masuk repo publik. Isinya agregat, bukan data institusi.

## Isi

| Berkas | Isi | Dibaca oleh |
|---|---|---|
| `RUNS.md` | satu baris per run per dataset: metode, provider/model, k, n, metrik agregat, commit, hash GT, tautan arsip | manusia, PR review |
| `runs.csv` | versi panjang: satu baris per (run, dataset, penilaian, kelompok) dengan angka mentah | spreadsheet, analisis |
| `archives.csv` | peta `run` → URL folder Drive hasil mentah | pengisi `RUNS.md` |

`runs.csv` dan `RUNS.md` dihasilkan otomatis; jangan diedit manual.

```bash
uv run python scripts/ledger.py
```

Jalankan setelah evaluasi selesai, lalu commit bersama perubahan kode eksperimennya.

## Batas kerahasiaan

Generator hanya membaca `metrics.csv`, `ringkasan_*.csv`, dan lima kolom metadata dari `skor_*.csv` (`penyedia_llm`, `model_llm`, `metode`, `top_k`, `model_embedding`). Kolom dan berkas berikut tidak pernah dibuka maupun disalin:

- `pertanyaan`, `jawaban`, `jawaban_acuan`, `konteks`
- `jawaban_*.csv`, `skor_*.csv` (isi teks), `traces.jsonl`, `chat_logs/`
- `knowledge_base/`, `ground_truth/`

Repo ini publik. Jangan menambahkan teks soal, jawaban acuan, atau potongan silabus ke berkas apa pun di folder ini, termasuk pada nama run dan pesan commit.

## Arsip hasil mentah di Google Drive

Hasil mentah (CSV jawaban/skor, traces, log chat) tetap lokal karena memuat isi kurikulum dan pertanyaan mahasiswa. Untuk dibagikan ke tim, unggah sebagai satu folder per run ke Drive proyek:

- Folder induk: `Project FTMM COMPASS` — https://drive.google.com/drive/folders/127DPRBrV9Ng7uiIwRGCPF_7OBeaVVnsz
- Subfolder tujuan: `Hasil Evaluasi RAG` (belum dibuat; root folder saat ini hanya berisi `Dataset Matkul Prodi`, shortcut `Ground Truth NLP`, dan `Paper Referensi`)
- Nama folder arsip: `<YYYY-MM-DD>_<metode>_<provider-model>_k<k>`, contoh `2026-11-07_bm25_lokal-k3`

Setelah folder terunggah, catat tautannya:

```csv
run,drive_url
bm25,https://drive.google.com/drive/folders/<id>
```

lalu jalankan ulang `scripts/ledger.py` agar kolom Arsip di `RUNS.md` terisi.

Status akses: Drive MCP pada mesin ini menolak permintaan (`authorization was revoked`). Upload dan pembuatan subfolder menunggu otorisasi ulang akun Drive.
