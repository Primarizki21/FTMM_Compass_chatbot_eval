import pytest

from scripts.build_web_gold import susun


def chunk(**kwargs):
    data = {
        "chunk_id": "WEB-page-1-001",
        "induk_id": "WEB-page-1",
        "judul": "Halaman Uji",
        "url": "https://contoh/halaman/",
        "halaman": None,
        "dipakai_kb": True,
        "teks": "Isi awal halaman.",
    }
    data.update(kwargs)
    return data


def test_gold_memilih_chunk_berfrasa():
    chunks = [
        chunk(chunk_id="WEB-page-1-001", teks="Isi awal halaman."),
        chunk(chunk_id="WEB-page-1-002", teks="Bagian cuti akademik dan prosedurnya."),
    ]
    rows = susun(chunks, [{"query": "Bagaimana cuti?", "kategori": "akademik", "judul": "Halaman Uji", "frasa": "cuti akademik"}])
    assert rows[0]["target_chunk"] == "WEB-page-1-002"
    assert rows[0]["target_url"] == "https://contoh/halaman/"


def test_gold_luar_cakupan_tanpa_target():
    rows = susun([], [{"query": "Siapa dosen Teknik Sipil?", "kategori": "luar_cakupan", "luar_cakupan": True}])
    assert rows[0]["target_url"] == ""
    assert rows[0]["jenis_kueri"] == "luar-cakupan"


def test_gold_gagal_bila_target_hilang():
    with pytest.raises(SystemExit):
        susun([], [{"query": "Apa isi?", "kategori": "akademik", "judul": "Tidak Ada"}])
