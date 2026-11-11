from scripts.categorize_web_kb import klasifikasi, rapikan_judul, rapikan_teks


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


def test_rapikan_judul_dan_teks():
    assert rapikan_judul("Judul - Fakultas Teknologi Maju dan Multidisiplin | Universitas Airlangga") == "Judul"
    assert rapikan_teks("a\x03b\x0cc\n\n\n d ") == "ab\n\nc\n\nd"
