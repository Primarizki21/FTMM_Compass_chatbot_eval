import json

from scripts.build_web_eval import kueri, susun


def chunk(**kwargs):
    data = {
        "chunk_id": "WEB-page-1-001",
        "induk_id": "WEB-page-1",
        "tipe": "page",
        "kategori": "akademik",
        "judul": "Alur Permohonan Pengantar KTM Hilang",
        "url": "https://ftmm.unair.ac.id/alur-ktm/",
        "prodi": "",
        "halaman": None,
        "jenis_unit": "paragraf",
        "dipakai_kb": True,
        "teks": "Isi halaman yang cukup panjang untuk dinilai. " * 5,
    }
    data.update(kwargs)
    return data


def test_kueri_faq_tanpa_label():
    row = chunk(kategori="faq", jenis_unit="faq", teks="Di mana ruang layanan FTMM?\n\nLantai 10.")
    assert kueri(row) == ("Di mana ruang layanan FTMM?", "pertanyaan-faq")


def test_kueri_dokumen_memakai_judul_bagian():
    row = chunk(
        kategori="dokumen_resmi",
        jenis_unit="halaman",
        halaman=216,
        teks="BAB XI\nDEWAN ETIKA\n\nPasal 47",
    )
    kueri_teks, jenis = kueri(row)
    assert "BAB XI" in kueri_teks
    assert jenis == "bagian-dokumen"


def test_susun_menyebar_bagian_dan_membatasi_kategori():
    chunks = []
    for induk in ("WEB-page-1", "WEB-page-2"):
        for urutan in range(1, 6):
            chunks.append(
                chunk(
                    chunk_id=f"{induk}-{urutan:03d}",
                    induk_id=induk,
                    judul=f"Judul {induk} {urutan}",
                    teks=f"Isi {induk} bagian {urutan}. " * 10,
                )
            )
    rows = susun(chunks, per_kategori=3, per_dokumen=2, seed=42)
    assert len(rows) == 3
    assert len({row["target_url"] for row in rows}) <= 2
    assert rows == susun(chunks, per_kategori=3, per_dokumen=2, seed=42)


def test_susun_mengabaikan_chunk_dinamis():
    rows = susun([chunk(dipakai_kb=False)], per_kategori=8, per_dokumen=3, seed=42)
    assert rows == []
    json.dumps(rows)
