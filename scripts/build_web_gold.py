"""Susun gold set evaluasi retrieval web dari unit KB yang sudah dikurasi.

Kueri ditulis manual dalam bahasa mahasiswa, bukan disalin dari judul. Target
diselesaikan ke URL + chunk dari `chunks.jsonl`; bila ada target yang tidak
ditemukan, skrip berhenti agar dataset tidak diam-diam salah.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPESIFIKASI = [
    # akademik
    {"query": "Bagaimana cara mengajukan cuti akademik?", "kategori": "akademik", "judul": "Alur Pengajuan Cuti Akademik"},
    {"query": "Apa yang harus dilakukan kalau KTM saya hilang?", "kategori": "akademik", "judul": "Alur Permohonan Pengantar KTM Hilang"},
    {"query": "Bagaimana cara mengajukan surat keterangan mahasiswa aktif?", "kategori": "akademik", "judul": "Alur Pengajuan Surat Keterangan Mahasiswa Aktif"},
    {"query": "Saya terlambat membayar UKT, apa prosedurnya?", "kategori": "akademik", "judul": "Alur Permohonan Terlambat Bayar UKT"},
    {"query": "Bagaimana cara mengundurkan diri dari kampus?", "kategori": "akademik", "judul": "Alur Pengajuan Undur Diri"},
    {"query": "Bagaimana cara legalisir transkrip nilai?", "kategori": "akademik", "judul": "Alur Pengajuan Legalisir & Transkrip Nilai"},
    {"query": "Akun email OneDrive saya terkunci, bagaimana resetnya?", "kategori": "akademik", "judul": "Cara Reset Akun Email Onedrive via ULT"},
    {"query": "Apa saja syarat mengikuti yudisium?", "kategori": "akademik", "judul": "Informasi Yudisium"},
    {"query": "Bagaimana alur pengajuan skripsi?", "kategori": "akademik", "judul": "Informasi Skripsi"},
    {"query": "Di mana saya bisa melihat informasi PKL?", "kategori": "akademik", "judul": "Informasi PKL"},
    {"query": "Kapan kalender akademik FTMM berlaku?", "kategori": "akademik", "judul": "Kalender Akademik"},
    {"query": "Bagaimana cara mendaftar sebagai mahasiswa baru FTMM?", "kategori": "akademik", "judul": "Pendaftaran Mahasiswa"},
    {"query": "Bagaimana cara mengajukan surat rekomendasi?", "kategori": "akademik", "judul": "Alur Pengajuan Surat Rekomendasi"},
    {"query": "Apa yang bisa saya lakukan di portal informasi mahasiswa?", "kategori": "akademik", "judul": "Portal Informasi Mahasiswa"},
    # program studi
    {"query": "Apa yang dipelajari di program studi Teknik Elektro?", "kategori": "program_studi", "judul": "Apa yang dipelajari di program studi Teknik Elektro?"},
    {"query": "Mengapa memilih jurusan Teknik Industri?", "kategori": "program_studi", "judul": "Mengapa belajar keilmuan Teknik Industri?"},
    {"query": "Apa kompetensi lulusan Teknik Robotika dan Kecerdasan Buatan?", "kategori": "program_studi", "judul": "Kompetensi Lulusan Teknik Robotika dan Kecerdasan Buatan"},
    {"query": "Apa saja bidang peminatan di Teknologi Sains Data?", "kategori": "program_studi", "judul": "Rincian Bidang Peminatan Teknologi Sains Data Universitas Airlangga"},
    {"query": "Apa prospek kerja lulusan Rekayasa Nanoteknologi?", "kategori": "program_studi", "judul": "Apa sajakah kesempatan karir lulusan Rekayasa Nanoteknologi?"},
    {"query": "Bagaimana profil program studi Teknik Industri?", "kategori": "program_studi", "judul": "Teknik Industri – Program Studi"},
    {"query": "Apa yang dipelajari di Sains Data Technology?", "kategori": "program_studi", "judul": "Sains Data Technology – Study Program"},
    {"query": "Apa visi program studi Teknologi Sains Data?", "kategori": "program_studi", "judul": "Teknologi Sains Data – Program Studi", "frasa": "Visi"},
    # profil fakultas
    {"query": "Siapa dekan FTMM saat ini?", "kategori": "profil_fakultas", "judul": "Pimpinan Fakultas"},
    {"query": "Apa visi dan misi FTMM?", "kategori": "profil_fakultas", "judul": "Visi, Misi, dan Tujuan"},
    {"query": "Bagaimana sejarah berdirinya FTMM?", "kategori": "profil_fakultas", "judul": "Sejarah Fakultas"},
    {"query": "Bagaimana struktur organisasi FTMM?", "kategori": "profil_fakultas", "judul": "Struktur Organisasi FTMM (OCR)"},
    {"query": "Apa itu Fakultas Teknologi Maju dan Multidisiplin?", "kategori": "profil_fakultas", "judul": "Tentang Fakultas"},
    {"query": "Siapa saja wakil dekan FTMM?", "kategori": "profil_fakultas", "judul": "Profil Pimpinan"},
    {"query": "Siapa Prof. Dr. Ir. Retna Apsari?", "kategori": "profil_staf", "judul": "Retna Apsari"},
    {"query": "Apa profil Ika Qutsiati Utami?", "kategori": "profil_staf", "judul": "Ika Qutsiati Utami, S.Kom., M.Sc. Ph.D"},
    {"query": "Siapa Fadjar Mulya?", "kategori": "profil_staf", "judul": "Fadjar Mulya, Ph.D"},
    {"query": "Apa profil Dr. Fadli Ama?", "kategori": "profil_staf", "judul": "Dr. Fadli Ama"},
    {"query": "Departemen apa saja yang ada di FTMM?", "kategori": "profil_fakultas", "judul": "Profil Departemen"},
    # fasilitas
    {"query": "Fasilitas apa saja yang tersedia untuk mahasiswa FTMM?", "kategori": "fasilitas", "judul": "Fasilitas FTMM"},
    {"query": "Apakah ada fasilitas tempat ibadah di FTMM?", "kategori": "fasilitas", "judul": "Fasilitas Tempat Ibadah"},
    {"query": "Fasilitas olahraga apa yang ada di FTMM?", "kategori": "fasilitas", "judul": "Fasilitas Olahraga"},
    {"query": "Fasilitas penelitian apa saja yang dimiliki FTMM?", "kategori": "fasilitas", "judul": "Fasilitas Penelitian"},
    {"query": "Fasilitas penunjang apa yang dimiliki FTMM?", "kategori": "fasilitas", "judul": "Fasilitas Penunjang FTMM"},
    {"query": "Fasilitas akademik apa saja yang tersedia?", "kategori": "fasilitas", "judul": "Fasilitas Akademik"},
    {"query": "Fasilitas kemahasiswaan apa saja yang ada?", "kategori": "fasilitas", "judul": "Fasilitas Kemahasiswaan"},
    # kemahasiswaan
    {"query": "Apa itu BEM FTMM?", "kategori": "kemahasiswaan", "judul": "BEM FTMM UNAIR"},
    {"query": "Apa tugas BLM FTMM?", "kategori": "kemahasiswaan", "judul": "BLM FTMM UNAIR"},
    {"query": "Komunitas mahasiswa apa saja yang ada di FTMM?", "kategori": "kemahasiswaan", "judul": "Komunitas"},
    {"query": "Apa itu MBKM di FTMM?", "kategori": "kemahasiswaan", "judul": "MBKM FTMM"},
    {"query": "Apa saja layanan alumni FTMM?", "kategori": "kemahasiswaan", "judul": "Alumni FTMM"},
    # penelitian, pengabdian, kerjasama, sdgs
    {"query": "Apa saja research group di FTMM?", "kategori": "penelitian", "judul": "Research Groups"},
    {"query": "Apa itu SEMNAS FTMM?", "kategori": "penelitian", "judul": "SEMNAS FTMM"},
    {"query": "Apa fokus penelitian di FTMM?", "kategori": "penelitian", "judul": "Penelitian"},
    {"query": "Apa saja kegiatan pengabdian masyarakat FTMM?", "kategori": "pengabdian", "judul": "Pengabdian"},
    {"query": "Di mana informasi kegiatan pengmas FTMM?", "kategori": "pengabdian", "judul": "Berita Pengmas FTMM"},
    {"query": "Siapa saja mitra kerja sama FTMM?", "kategori": "kerjasama", "judul": "Mitra Kolaborasi"},
    {"query": "Siapa mitra industri FTMM?", "kategori": "kerjasama", "judul": "Mitra Kolaborasi", "frasa": "industri"},
    {"query": "Apa program pembangunan berkelanjutan FTMM?", "kategori": "sdgs", "judul": "Program Berkelanjutan"},
    {"query": "SDGs apa saja yang didukung FTMM?", "kategori": "sdgs", "judul": "Program Pembangunan Berkelanjutan Fakultas"},
    {"query": "Apa program SDGs 6 di FTMM?", "kategori": "sdgs", "judul": "SDGs 6 – Clean Water and Sanitation"},
    {"query": "Apa kegiatan pengabdian masyarakat FTMM?", "kategori": "pengabdian", "judul": "Pengabdian"},
    # regulasi dan mutu
    {"query": "Di mana saya bisa membaca SOP layanan FTMM?", "kategori": "regulasi", "judul": "SOP FTMM"},
    {"query": "Apa aturan perilaku bagi mahasiswa FTMM?", "kategori": "regulasi", "judul": "Buku Aturan Berperilaku"},
    {"query": "Apa itu PPID di FTMM?", "kategori": "regulasi", "judul": "PPID FTMM"},
    {"query": "Apa rencana strategis FTMM?", "kategori": "regulasi", "judul": "Rencana Strategis FTMM"},
    {"query": "Apa itu Sistem Kredit Prestasi?", "kategori": "regulasi", "judul": "Sistem Kredit Prestasi"},
    {"query": "Bagaimana mengakses Daftar Informasi Publik FTMM?", "kategori": "regulasi", "judul": "Daftar Informasi Publik (DIP) Fakultas"},
    {"query": "Bagaimana hasil tracer study lulusan FTMM?", "kategori": "mutu", "judul": "Hasil Tracer Study Fakultas Teknologi Maju dan Multidisiplin"},
    {"query": "Bagaimana survei kepuasan mahasiswa FTMM?", "kategori": "mutu", "judul": "Survei Kepuasan"},
    # dokumen resmi (PDF)
    {"query": "Apa ketentuan cuti akademik di FTMM?", "kategori": "dokumen_resmi", "judul": "Panduan Pendidikan FTMM UNAIR 2024/2025", "frasa": "5.5 Cuti Akademik"},
    {"query": "Bagaimana aturan yudisium dan wisuda?", "kategori": "dokumen_resmi", "judul": "Buku Pedoman Pendidikan 2024/2025", "frasa": "BAB XVIII"},
    {"query": "Apa syarat kelulusan program sarjana?", "kategori": "dokumen_resmi", "judul": "Buku Pedoman Pendidikan 2024/2025", "frasa": "Telah menyelesaikan tugas akhir/skripsi"},
    {"query": "Bagaimana ketentuan beasiswa di FTMM?", "kategori": "dokumen_resmi", "judul": "Panduan Pendidikan FTMM UNAIR 2024/2025", "frasa": "7.1 Beasiswa"},
    {"query": "Berapa minimal kehadiran untuk bisa mengikuti ujian?", "kategori": "dokumen_resmi", "judul": "Panduan Pendidikan FTMM UNAIR 2024/2025", "frasa": "minimal 80%"},
    {"query": "Apa sanksi jika melanggar aturan akademik?", "kategori": "dokumen_resmi", "judul": "Buku Pedoman Pendidikan 2024/2025", "frasa": "BAB XVII"},
    {"query": "Bagaimana ketentuan pengisian KRS?", "kategori": "dokumen_resmi", "judul": "Buku Pedoman Pendidikan 2024/2025", "frasa": "BAB XII"},
    {"query": "Apa aturan semester antara?", "kategori": "dokumen_resmi", "judul": "Buku Pedoman Pendidikan 2024/2025", "frasa": "Semester Antara"},
    {"query": "Apa syarat cuti akademik menurut Buku Pedoman?", "kategori": "dokumen_resmi", "judul": "Buku Pedoman Pendidikan 2024/2025", "frasa": "BAB XVI"},
    # FAQ
    {"query": "Di mana saja letak ruang layanan FTMM?", "kategori": "faq", "judul": "Frequently Asked Question (FAQ)", "frasa": "ruang layanan"},
    {"query": "Apa saja yang harus saya persiapkan untuk yudisium?", "kategori": "faq", "judul": "Frequently Asked Question (FAQ)", "frasa": "Yudisium"},
    {"query": "Bagaimana cara mendaftar Praktik Kerja Lapangan?", "kategori": "faq", "judul": "Frequently Asked Question (FAQ)", "frasa": "Praktik Kerja Lapangan"},
    {"query": "Di mana saya bisa melihat prosedur skripsi?", "kategori": "faq", "judul": "Frequently Asked Question (FAQ)", "frasa": "prosedur Skripsi"},
    {"query": "Di mana saya bisa mengajukan surat keterangan mahasiswa aktif?", "kategori": "faq", "judul": "Frequently Asked Question (FAQ)", "frasa": "Surat Keterangan Mahasiswa Aktif"},
    {"query": "Bagaimana saya mendapatkan pengumuman perkuliahan?", "kategori": "faq", "judul": "Frequently Asked Question (FAQ)", "frasa": "pengumuman dan informasi perkuliahan"},
    # luar cakupan
    {"query": "Siapa dosen tetap Program Studi Teknik Sipil?", "kategori": "luar_cakupan", "luar_cakupan": True},
    {"query": "Berapa biaya kuliah Fakultas Kedokteran UNAIR?", "kategori": "luar_cakupan", "luar_cakupan": True},
    {"query": "Bagaimana cara mendaftar beasiswa LPDP?", "kategori": "luar_cakupan", "luar_cakupan": True},
    {"query": "Apa jadwal KRS semester gasal 2027/2028?", "kategori": "luar_cakupan", "luar_cakupan": True},
    {"query": "Siapa rektor Universitas Airlangga saat ini?", "kategori": "luar_cakupan", "luar_cakupan": True},
    {"query": "Apa itu kurikulum Merdeka Belajar di SMA?", "kategori": "luar_cakupan", "luar_cakupan": True},
]


def normal(teks: str) -> str:
    return re.sub(r"\s+", " ", (teks or "").strip().lower())


def susun(chunks: list[dict], spesifikasi: list[dict] | None = None) -> list[dict]:
    per_judul: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        if chunk.get("dipakai_kb"):
            per_judul[normal(chunk["judul"])].append(chunk)

    rows = []
    gagal = []
    for nomor, spec in enumerate(spesifikasi or SPESIFIKASI, start=1):
        if spec.get("luar_cakupan"):
            rows.append(
                {
                    "id": f"GOLD-{nomor:03d}",
                    "kategori": spec["kategori"],
                    "query": spec["query"],
                    "jenis_kueri": "luar-cakupan",
                    "target_url": "",
                    "target_chunk": "",
                    "target_judul": "",
                    "target_halaman": None,
                    "sumber": "manual",
                }
            )
            continue

        kandidat = sorted(per_judul.get(normal(spec["judul"]), []), key=lambda chunk: chunk["chunk_id"])
        if not kandidat:
            gagal.append(f"{spec['judul']!r} tidak ditemukan")
            continue
        frasa = normal(spec.get("frasa", ""))
        chunk = next((item for item in kandidat if frasa and frasa in normal(item["teks"])), None)
        if frasa and chunk is None:
            gagal.append(f"{spec['judul']!r} tidak memuat frasa {spec['frasa']!r}")
            continue
        chunk = chunk or kandidat[0]
        rows.append(
            {
                "id": f"GOLD-{nomor:03d}",
                "kategori": spec["kategori"],
                "query": spec["query"],
                "jenis_kueri": "gold",
                "target_url": chunk["url"],
                "target_chunk": chunk["chunk_id"],
                "target_judul": chunk["judul"],
                "target_halaman": chunk.get("halaman"),
                "sumber": "manual",
            }
        )

    if gagal:
        raise SystemExit("Target gold tidak lengkap:\n- " + "\n- ".join(gagal))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumber", type=Path, default=ROOT / "web_kb_categorized" / "chunks.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "web_kb_categorized" / "web_eval_gold.jsonl")
    args = parser.parse_args()

    if not args.sumber.exists():
        raise SystemExit(f"chunks.jsonl tidak ada di {args.sumber}. Jalankan: make kategori")
    chunks = [json.loads(line) for line in args.sumber.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = susun(chunks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    dinilai = sum(1 for row in rows if row["target_url"])
    print(f"[OK] {len(rows)} kueri gold ({dinilai} dinilai, {len(rows) - dinilai} luar cakupan) -> {args.out}")


if __name__ == "__main__":
    main()
