from scripts.categorize_web_kb import (
    buang_duplikat_kb,
    buat_chunk,
    daftar_isi,
    klasifikasi,
    layak_chunk,
    layak_kb,
    pecah_faq,
    potong_teks,
    prodi_bersih,
    rapikan_judul,
    rapikan_teks,
    tandai_arsip,
)


def rec(**kwargs):
    data = {
        "tipe": "page",
        "url": "https://ftmm.unair.ac.id/contoh/",
        "judul": "",
        "teks": "",
        "prodi": "",
        "status": "ok",
    }
    data.update(kwargs)
    return data


def test_profil_staf_dari_nik_dan_gelar():
    assert klasifikasi(rec(judul="Retna Apsari", teks="NIK196806261993032003"))[0] == "profil_staf"
    assert klasifikasi(rec(judul="Dr. Fadli Ama"))[0] == "profil_staf"


def test_berita_bergelar_bukan_profil_staf():
    assert klasifikasi(rec(tipe="post", judul="Reasoning AI Mengalahkan Ph.D: Apa yang Sebenarnya Terjadi?"))[0] == "berita"


def test_kata_pendek_tidak_cocok_di_tengah_kata():
    assert klasifikasi(rec(tipe="post", judul="Manfaatkan Lahan Non-Produktif, Mahasiswa Raih Medali"))[0] == "prestasi"


def test_struktur_organisasi_ocr_masuk_profil_fakultas():
    hasil = klasifikasi(rec(tipe="dokumen", judul="Struktur Organisasi FTMM (OCR)"))
    assert hasil == ("profil_fakultas", "struktur organisasi")


def test_kepuasan_kinerja_masuk_mutu():
    assert klasifikasi(rec(tipe="post", judul="Kepuasan terhadap Kinerja Sub Bagian Fakultas Genap 2024/2025"))[0] == "mutu"


def test_berita_bertajuk_regulasi_tidak_masuk_regulasi():
    assert klasifikasi(rec(tipe="post", judul="Optimalisasi Teknologi dan Regulasi: Strategi Industri"))[0] == "berita"


def test_dokumen_resmi_dari_judul_post():
    assert klasifikasi(rec(tipe="post", judul="Daftar Informasi Publik (DIP) Fakultas"))[0] == "regulasi"
    assert klasifikasi(rec(tipe="post", judul="Laporan Penyerapan Anggaran TW 4 2025"))[0] == "regulasi"


def test_pelantikan_ormawa_bukan_prestasi():
    assert klasifikasi(rec(tipe="post", judul="Lantik Pengurus BEM dan BLM, Dekan Berpesan untuk Berprestasi"))[0] == "kemahasiswaan"


def test_halaman_navigasi_dibuang():
    assert klasifikasi(rec(judul="Beranda", url="https://ftmm.unair.ac.id/"))[0] == "sampah"


def test_prodi_fakultas_tidak_diwarisi():
    assert prodi_bersih("Tentang Fakultas", "tentang-fakultas", "Teknik Elektro", "profil_fakultas") == ""
    assert prodi_bersih("Kurikulum Teknik Elektro", "kurikulum", "Teknik Elektro", "akademik") == "Teknik Elektro"
    assert prodi_bersih("Retna Apsari", "retna-apsari", "Teknik Elektro", "profil_staf") == "Teknik Elektro"


def test_rapikan_judul_dan_teks():
    assert rapikan_judul("Judul - Fakultas Teknologi Maju dan Multidisiplin | Universitas Airlangga") == "Judul"
    assert rapikan_teks("a\x03b\x0cc\n\n\n d ") == "ab\n\f\nc\n\nd"


def test_potong_teks_tidak_melebihi_batas():
    teks = "\n\n".join("Kalimat panjang nomor %d. " % i + "kata " * 60 for i in range(8))
    potongan = potong_teks(teks, maks=500)
    assert len(potongan) > 1
    assert all(len(bagian) <= 500 for bagian in potongan)


def test_pdf_dipecah_per_halaman():
    doc = {
        "id": None,
        "slug": "buku-pedoman",
        "tipe": "dokumen",
        "kategori": "dokumen_resmi",
        "dasar_kategori": "tipe=dokumen",
        "judul": "Buku Pedoman",
        "url": "https://contoh/buku.pdf",
        "prodi": "",
        "tanggal": "",
        "modified": "",
        "dipakai_kb": True,
        "teks": "Halaman pertama berisi penjelasan yang cukup panjang untuk dinilai. " * 2
        + "\n\f"
        + "Halaman kedua berisi penjelasan yang cukup panjang untuk dinilai. " * 2
        + "\n\f"
        + "Halaman ketiga berisi penjelasan yang cukup panjang untuk dinilai. " * 2,
    }
    chunks = buat_chunk(doc, maks=1200)
    assert [chunk["halaman"] for chunk in chunks] == [1, 2, 3]
    assert chunks[0]["chunk_id"] == "WEB-dokumen-buku-pedoman-001"


def test_faq_dipecah_per_pertanyaan():
    teks = (
        "Pertanyaan yang Sering Diajukan\n\nSemua\n\nLokasiDi mana ruang layanan FTMM berada?\n\n"
        "Ruang layanan berada di lantai 10, lantai 7, dan lantai 1 gedung FTMM.\n\n"
        "AkademikBagaimana cara mengajukan cuti akademik?\n\n"
        "Ajukan permohonan cuti akademik ke PAA program studi masing-masing.\n\n"
        "Pertanyaan tidak ditemukan"
    )
    unit = pecah_faq(teks)
    assert len(unit) == 2
    assert unit[0].startswith("Di mana ruang layanan FTMM berada?")
    assert unit[1].startswith("Bagaimana cara mengajukan cuti akademik?")


def test_edisi_lama_ditandai_arsip():
    dokumen = [
        {"judul": "Buku Pedoman Pendidikan 2022/2023", "kategori": "dokumen_resmi", "arsip": False},
        {"judul": "Buku Pedoman Pendidikan 2024/2025", "kategori": "dokumen_resmi", "arsip": False},
    ]
    tandai_arsip(dokumen)
    assert dokumen[0]["arsip"] is True
    assert dokumen[1]["arsip"] is False


def test_layak_kb_menolak_administratif():
    dasar = {"arsip": False, "sifat": "statis", "kategori": "regulasi", "judul": "Laporan Penyerapan Anggaran TW 4"}
    assert layak_kb(dasar) is False
    assert layak_kb({**dasar, "judul": "SOP FTMM"}) is True
    assert layak_kb({**dasar, "kategori": "berita", "sifat": "dinamis", "judul": "Berita"}) is False


def test_duplikat_kb_disimpan_satu_teks_terpanjang():
    dokumen = [
        {"kategori": "akademik", "judul": "Informasi Yudisium", "dipakai_kb": True, "jumlah_karakter": 900},
        {"kategori": "akademik", "judul": "Informasi Yudisium", "dipakai_kb": True, "jumlah_karakter": 500},
    ]
    assert buang_duplikat_kb(dokumen) == 1
    assert [doc["dipakai_kb"] for doc in dokumen] == [True, False]
    assert dokumen[1]["alasan_tidak_kb"] == "duplikat judul"


def test_daftar_isi_dan_chunk_tidak_layak():
    assert daftar_isi("BAB I ........ 1\nBAB II ....... 5\nBAB III ...... 9")
    assert not daftar_isi("Kalimat biasa tanpa titik penuntun.\nParagraf kedua.")
    assert layak_chunk("Panduan ini menjelaskan prosedur pengajuan cuti akademik dengan lengkap.")
    assert not layak_chunk("1 2 3 45 6 7 89 10 11 12 13 14 15 16 17 18")
