"""Rapikan web_kb/pages.jsonl menjadi JSON per kategori dan unit retrieval.

Sumber tetap `web_kb/pages.jsonl` (mentah). Skrip ini hanya menurunkan artefak:
satu berkas JSON per kategori (dokumen utuh, untuk dibaca manusia) dan
`chunks.jsonl` (unit yang dipakai retriever: PDF per halaman, teks panjang
dipotong, FAQ per pertanyaan). Tidak ada akses jaringan.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent

KATEGORI = {
    "akademik": "Layanan dan aturan akademik: kalender, KRS, ujian, UKT, cuti, skripsi, PKL, yudisium, beasiswa, pendaftaran.",
    "program_studi": "Profil program studi dan informasi tentang pilihan prodi.",
    "profil_staf": "Profil dosen dan tenaga kependidikan.",
    "profil_fakultas": "Identitas fakultas: sejarah, visi, misi, pimpinan, dan struktur organisasi.",
    "kemahasiswaan": "Organisasi mahasiswa, kegiatan, MBKM, alumni, dan komunitas.",
    "prestasi": "Prestasi mahasiswa dan dosen.",
    "penelitian": "Penelitian, publikasi, konferensi, seminar, dan research group.",
    "pengabdian": "Pengabdian kepada masyarakat.",
    "kerjasama": "Kerja sama, mitra, kunjungan, dan kegiatan internasional.",
    "fasilitas": "Fasilitas dan sarana prasarana.",
    "regulasi": "Dokumen resmi: SOP, pedoman, panduan, PPID, zona integritas, renstra, laporan kinerja.",
    "mutu": "Penjaminan mutu: audit mutu, akreditasi, survei kepuasan, dan tracer study.",
    "agenda": "Agenda dan acara fakultas.",
    "sdgs": "Program dan berita pembangunan berkelanjutan/SDGs.",
    "dokumen_resmi": "Dokumen PDF hasil scraping: buku pedoman dan panduan pendidikan.",
    "faq": "Pertanyaan yang sering diajukan.",
    "berita": "Berita umum yang tidak masuk kategori lain.",
    "lainnya": "Halaman umum yang tidak masuk kategori lain.",
}

# Kategori ini tetap statis walau bertipe `post` (profil dosen, info prodi,
# dokumen resmi). `berita` dan `agenda` selalu dinamis, sekalipun bertipe `page`.
KATEGORI_STATIS = {"profil_staf", "program_studi", "regulasi", "dokumen_resmi"}
KATEGORI_DINAMIS = {"berita", "agenda"}

# Kata penanda dokumen resmi. "regulasi" generik hanya berlaku untuk non-post
# agar berita bertajuk "regulasi" tidak ikut masuk.
PENANDA_DOKUMEN = (
    "sop",
    "standar operasional",
    "aturan berperilaku",
    "buku aturan",
    "pedoman",
    "panduan",
    "kontrak kinerja",
    "laporan kinerja",
    "laporan penyerapan anggaran",
    "laporan kemajuan",
    "daftar informasi publik",
    "daftar informasi dikecualikan",
    "maklumat pelayanan",
    "inventarisasi",
    "sistem kredit prestasi",
    "ppid",
    "rencana strategis",
    "renstra",
)

# Dokumen resmi yang tetap disimpan, tetapi tidak layak masuk KB chatbot:
# laporan anggaran/inventaris dan kontrak kerja internal.
ADMIN_TIDAK_KB = (
    "laporan penyerapan anggaran",
    "laporan kemajuan",
    "daftar inventarisasi",
    "kontrak kinerja",
    "maklumat pelayanan",
)

# Penanda artikel prodi yang evergreen, bukan berita kegiatan.
INFO_PRODI = (
    "mengapa belajar",
    "kompetensi lulusan",
    "kesempatan karir",
    "bidang peminatan",
    "prospek",
    "yang dipelajari",
    "profil lulusan",
)

SUFIKS_JUDUL = re.compile(r"\s*[-–|]\s*Fakultas Teknologi Maju dan Multidisiplin.*$", re.IGNORECASE)
GELAR = re.compile(
    r"\b(?:prof\.|dr\.|ir\.|dr\.eng|s\.t|s\.si|s\.kom|s\.sos|s\.e\b|s\.m|s\.ip|s\.farm|"
    r"s\.pd|s\.st|s\.ikom|m\.t|m\.sc|m\.kom|m\.si|m\.stat|m\.eng|m\.tr|m\.farm|m\.ba|"
    r"m\.m\b|m\.imun|ph\.d|a\.md|sp\.)"
)
TAHUN = re.compile(r"\b(20\d{2})(?:\s*[/–-]\s*20\d{2})?\b")

# `prodi` di pages.jsonl diwarisi dari halaman induk, sehingga halaman fakultas
# bisa berlabel prodi tertentu. Label dipertahankan hanya bila nama prodi muncul
# di judul/slug; profil staf dan info prodi dikecualikan karena warisannya benar.
ALIAS_PRODI = {
    "Teknologi Sains Data": ("teknologi sains data", "data science"),
    "Teknik Robotika dan Kecerdasan Buatan": ("teknik robotika", "robotika dan kecerdasan buatan", "robotics"),
    "S1 Rekayasa Nanoteknologi": ("rekayasa nanoteknologi", "nanoteknologi", "nanotechnology"),
    "Teknik Industri": ("teknik industri", "industrial engineering"),
    "Teknik Elektro": ("teknik elektro", "electrical engineering"),
}
PRODI_DIWARISI = {"profil_staf", "program_studi"}

MAKS_CHUNK = 1200
MIN_CHUNK = 80


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ada(teks: str, *frasa: str) -> bool:
    """Cocokkan kata/frasa utuh; 'ukt' tidak boleh cocok di 'produktif'."""
    return any(re.search(r"(?<![a-z0-9])" + re.escape(frasa_) + r"(?![a-z0-9])", teks) for frasa_ in frasa)


def rapikan_teks(teks: str) -> str:
    """Normalisasi teks; form feed dipertahankan sebagai batas halaman PDF."""
    teks = (teks or "").replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    teks = teks.replace("\f", "\n\f\n")
    teks = "".join(ch for ch in teks if ch in "\n\t\f" or ord(ch) >= 32)
    baris = [re.sub(r"[ \t]+", " ", baris).strip(" \t") for baris in teks.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(baris)).strip()


def rapikan_judul(judul: str) -> str:
    judul = re.sub(r"\s+", " ", (judul or "").replace("\xa0", " ")).strip()
    judul = SUFIKS_JUDUL.sub("", judul)
    return judul.strip(" -–|")


def klasifikasi(rec: dict) -> tuple[str, str]:
    """Kembalikan (kategori, dasar). Urutan aturan = prioritas."""
    tipe = rec["tipe"]
    path = urlparse(rec["url"].lower()).path
    judul = re.sub(r"\s+", " ", (rec.get("judul") or "").lower()).strip()
    kepala = (rec.get("teks") or "")[:1200].lower()

    if tipe in ("page", "dokumen") and ada(judul, "struktur organisasi"):
        return "profil_fakultas", "struktur organisasi"
    if tipe == "dokumen":
        return "dokumen_resmi", "tipe=dokumen"
    if tipe == "event":
        return "agenda", "tipe=event"
    if "page not found" in judul or "halaman tidak ditemukan" in judul:
        return "sampah", "page not found"
    if tipe == "page" and path in ("/", "/beranda-baru/", "/download/", "/berita/"):
        return "sampah", "halaman navigasi"
    if tipe == "page" and ada(judul, "events", "agenda", "event directory", "rundown"):
        return "agenda", "halaman agenda"
    if "/faq" in path or "frequently asked" in judul:
        return "faq", "faq"
    if "sdgs" in path or ada(judul, "sdg", "sdgs", "sustainable development", "program berkelanjutan", "pembangunan berkelanjutan"):
        return "sdgs", "sdgs"
    if (
        "/tenaga-kependidikan" in path
        or "/staf-dosen" in path
        or "/profil-staff" in path
        or "profil staff" in judul
        or "staff profiles" in judul
        or judul.startswith("peneliti ")
    ):
        return "profil_staf", "halaman staf"
    if (
        re.search(r"(?<![a-z])(?:nik|nip)\s*[:.]?\s*\d{4,}", kepala)
        or ("jabatan" in kepala and "unit kerja" in kepala)
        or ("program studi" in kepala and "pendidikan" in kepala and "jabatan" in kepala)
    ):
        return "profil_staf", "marker profil di teks"
    if GELAR.search(judul) and (tipe == "page" or re.search(r"(?<![a-z])(?:nik|nip)\s*[:.]?\s*\d{4,}", kepala)):
        return "profil_staf", "gelar di judul"
    if tipe == "page" and (ada(judul, "program studi", "study program") or "/program-studi" in path):
        return "program_studi", "program studi"
    if tipe == "page" and ada(
        judul,
        "tentang fakultas",
        "sejarah fakultas",
        "visi",
        "misi",
        "selayang pandang",
        "struktur organisasi",
        "pimpinan fakultas",
        "profil pimpinan",
        "profil departemen",
        "leadership profile",
    ):
        return "profil_fakultas", "halaman profil fakultas"
    if ada(
        judul,
        "audit mutu",
        "survei",
        "survey",
        "tracer study",
        "akreditasi",
        "kepuasan",
        "asesmen lapangan",
        "asesmen prodi",
        "ban-pt",
    ):
        return "mutu", "mutu"
    if tipe == "page" and ada(judul, "zona integritas"):
        return "regulasi", "zona integritas"
    if ada(judul, *PENANDA_DOKUMEN) or (tipe != "post" and ada(judul, "regulasi")):
        return "regulasi", "regulasi"
    if ada(judul, "fasilitas", "gedung", "laboratorium", "perpustakaan", "sarana", "prasarana"):
        return "fasilitas", "fasilitas"
    if ada(
        judul,
        "portal informasi",
        "surat rekomendasi",
        "surat keterangan",
        "skma",
        "undur diri",
        "ukt",
        "ktm",
        "transkrip",
        "legalisir",
        "yudisium",
        "skripsi",
        "pkl",
        "kalender akademik",
        "pendaftaran",
        "calon mahasiswa",
        "beasiswa",
        "wisuda",
        "ujian",
        "semester",
        "kurikulum",
        "mata kuliah",
        "krs",
        "ipk",
        "sks",
        "cuti akademik",
        "reset akun",
        "onedrive",
    ):
        return "akademik", "akademik"
    if ada(judul, "juara", "raih", "sabet", "menang", "medali", "award", "penghargaan", "kompetisi", "lomba", "finalis", "winner"):
        return "prestasi", "prestasi"
    if ada(judul, "pengabdian", "pengmas", "abdimas", "kkn"):
        return "pengabdian", "pengabdian"
    if ada(judul, "penelitian", "research", "riset", "publikasi", "jurnal", "konferensi", "seminar", "icatam", "segta", "semnas", "research group"):
        return "penelitian", "penelitian"
    if ada(judul, "kerja sama", "kerjasama", "kolaborasi", "mitra", "mou", "kunjungan", "guest lecture", "internasional", "universiti", "inbound", "outbound", "exchange"):
        return "kerjasama", "kerjasama"
    if ada(judul, "kemahasiswaan", "bem", "blm", "hima", "mahasiswa", "pkkmb", "lkmm", "pemira", "komunitas", "aktivitas", "mbkm", "alumni", "organisasi"):
        return "kemahasiswaan", "kemahasiswaan"
    if tipe == "post" and rec.get("prodi") and ada(judul, *INFO_PRODI):
        return "program_studi", "informasi prodi"
    if tipe == "page" and ada(judul, "tentang fakultas", "sejarah", "visi", "misi", "profil", "pimpinan", "struktur organisasi", "selayang pandang", "program berkelanjutan", "program pembangunan"):
        return "profil_fakultas", "halaman profil fakultas"
    return ("berita", "default berita") if tipe == "post" else ("lainnya", "default halaman")


def muat(sumber: Path) -> tuple[list[dict], dict]:
    """Baca JSONL, buang kosong, dan pilih satu baris terbaik per URL."""
    baris = []
    with sumber.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                baris.append(json.loads(line))

    terbaik: dict[str, dict] = {}
    kosong = 0
    for rec in baris:
        if rec.get("status") != "ok" or not (rec.get("teks") or "").strip():
            kosong += 1
            continue
        url = rec.get("url") or ""
        lama = terbaik.get(url)
        if lama is None or len(rec.get("teks") or "") > len(lama.get("teks") or ""):
            terbaik[url] = rec

    stat = {
        "baris_sumber": len(baris),
        "kosong_atau_gagal": kosong,
        "duplikat_url": len(baris) - kosong - len(terbaik),
        "duplikat_teks": 0,
    }
    return list(terbaik.values()), stat


def prodi_bersih(judul: str, slug: str, prodi: str, kategori: str) -> str:
    """Buang label prodi warisan pada halaman yang tidak menyebut prodi itu."""
    if not prodi or kategori in PRODI_DIWARISI:
        return prodi
    teks = f"{judul} {slug}".lower().replace("-", " ")
    return prodi if any(alias in teks for alias in ALIAS_PRODI.get(prodi, ())) else ""


def tahun_edisi(judul: str) -> int | None:
    angka = [int(tahun) for tahun in TAHUN.findall(judul)]
    return max(angka) if angka else None


def kunci_edisi(judul: str) -> str:
    teks = TAHUN.sub(" ", judul.lower())
    teks = re.sub(r"\bocr\b", " ", teks)
    return re.sub(r"[^a-z0-9]+", " ", teks).strip()


def tandai_arsip(dokumen: list[dict]) -> None:
    """Edisi lama buku/panduan tetap disimpan, tetapi tidak dipakai sebagai KB."""
    grup: dict[str, list[dict]] = defaultdict(list)
    for doc in dokumen:
        if doc["kategori"] == "dokumen_resmi":
            grup[kunci_edisi(doc["judul"])].append(doc)

    for items in grup.values():
        bertahun = [doc for doc in items if tahun_edisi(doc["judul"]) is not None]
        if len(bertahun) < 2:
            continue
        terbaru = max(tahun_edisi(doc["judul"]) for doc in bertahun)
        for doc in bertahun:
            doc["arsip"] = tahun_edisi(doc["judul"]) < terbaru


def sifat(doc: dict) -> str:
    if doc["kategori"] in KATEGORI_DINAMIS:
        return "dinamis"
    if doc["tipe"] in ("page", "dokumen") or doc["kategori"] in KATEGORI_STATIS:
        return "statis"
    return "dinamis"


def layak_kb(doc: dict) -> bool:
    """Statis saja tidak cukup: laporan administratif tetap di luar KB."""
    if doc["arsip"] or doc["sifat"] != "statis":
        return False
    if doc["kategori"] == "regulasi" and any(penanda in doc["judul"].lower() for penanda in ADMIN_TIDAK_KB):
        return False
    return True


def kunci_judul(doc: dict) -> str:
    """Kunci untuk membuang dokumen KB yang judulnya sama atau nama stafnya kembar."""
    judul = doc["judul"].lower()
    if doc["kategori"] == "profil_staf":
        judul = GELAR.sub("", judul)
        judul = re.sub(r"\(cand\)|,|\bprof\b|\bdr\b|\bir\b", " ", judul)
    return re.sub(r"[^a-z0-9]+", " ", judul).strip()


def buang_duplikat_kb(dokumen: list[dict]) -> int:
    """Simpan satu dokumen per (kategori, judul) dan tandai sisanya tidak dipakai KB."""
    grup: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for doc in dokumen:
        if doc["dipakai_kb"]:
            grup[(doc["kategori"], kunci_judul(doc))].append(doc)

    dibuang = 0
    for items in grup.values():
        if len(items) < 2:
            continue
        items.sort(key=lambda doc: doc["jumlah_karakter"], reverse=True)
        for doc in items[1:]:
            doc["dipakai_kb"] = False
            doc["alasan_tidak_kb"] = "duplikat judul"
            dibuang += 1
    return dibuang


def daftar_isi(teks: str) -> bool:
    """Halaman daftar isi: mayoritas baris punya titik-titik penuntun."""
    baris = [baris for baris in teks.splitlines() if baris.strip()]
    if not baris:
        return True
    bertitik = sum(1 for baris in baris if re.search(r"\.{4,}", baris))
    return bertitik >= max(3, len(baris) * 0.3)


def bersihkan_halaman(teks: str) -> str:
    """Buang header/footer berulang yang ikut terbaca dari PDF."""
    pola = re.compile(
        r"^(?:PEDOMAN|PANDUAN|BUKU PEDOMAN) PENDIDIKAN(?: UNIVERSITAS AIRLANGGA| FTMM UNAIR)?"
        r"\s*\d{4}\s*/\s*\d{4}$",
        re.IGNORECASE,
    )
    return "\n".join(baris for baris in teks.splitlines() if not pola.match(baris.strip()))


def layak_chunk(teks: str) -> bool:
    huruf = sum(karakter.isalpha() for karakter in teks)
    return huruf >= 60 and huruf / max(len(teks), 1) >= 0.5


def potong_kalimat(teks: str, maks: int) -> list[str]:
    potongan = []
    for kalimat in re.split(r"(?<=[.!?])\s+|\n+", teks):
        kalimat = kalimat.strip()
        if not kalimat:
            continue
        while len(kalimat) > maks:
            potongan.append(kalimat[:maks])
            kalimat = kalimat[maks:]
        if kalimat:
            potongan.append(kalimat)
    return potongan


def potong_teks(teks: str, maks: int = MAKS_CHUNK) -> list[str]:
    """Bungkus paragraf/kalimat menjadi potongan <= maks karakter."""
    teks = teks.strip()
    if not teks:
        return []
    if len(teks) <= maks:
        return [teks]

    unit = []
    for paragraf in re.split(r"\n\s*\n", teks):
        paragraf = paragraf.strip()
        if not paragraf:
            continue
        unit.extend([paragraf] if len(paragraf) <= maks else potong_kalimat(paragraf, maks))

    hasil = []
    buf = ""
    for bagian in unit:
        if buf and len(buf) + 2 + len(bagian) > maks:
            hasil.append(buf)
            buf = bagian
        else:
            buf = f"{buf}\n\n{bagian}" if buf else bagian
    if buf:
        hasil.append(buf)
    return hasil


def pecah_faq(teks: str) -> list[str]:
    """Pisahkan halaman FAQ menjadi satu unit per pertanyaan."""
    unit = []
    buf: list[str] = []
    mulai = False
    for blok in (bagian.strip() for bagian in re.split(r"\n\s*\n", teks) if bagian.strip()):
        if blok.startswith("Pertanyaan tidak ditemukan"):
            break
        if "?" in blok:
            if mulai and buf:
                unit.append(bersihkan_label_faq("\n\n".join(buf)))
            buf = [blok]
            mulai = True
        elif mulai:
            buf.append(blok)
    if mulai and buf:
        unit.append(bersihkan_label_faq("\n\n".join(buf)))
    return [bagian for bagian in unit if len(bagian) >= MIN_CHUNK]


def bersihkan_label_faq(teks: str) -> str:
    """Label filter FAQ menempel di awal pertanyaan, mis. 'LokasiDi mana saja...'."""
    for label in ("Lokasi", "Akun", "Layanan", "Akademik", "Kemahasiswaan", "Bantuan", "Informasi"):
        if teks.startswith(label) and len(teks) > len(label) and teks[len(label)] not in " \n":
            return teks[len(label) :].lstrip()
    return teks


def kunci_dokumen(doc: dict) -> str:
    """ID dokumen mentah; PDF hasil impor tidak punya `id`, jadi pakai slug."""
    if doc.get("id") not in (None, "", 0):
        return str(doc["id"])
    return doc.get("slug") or hashlib.sha1(doc["url"].encode("utf-8")).hexdigest()[:8]


def buat_chunk(doc: dict, maks: int = MAKS_CHUNK, stat: dict | None = None) -> list[dict]:
    induk = f"WEB-{doc['tipe']}-{kunci_dokumen(doc)}"
    unit: list[tuple[int | None, str, str]] = []
    if doc["kategori"] == "faq":
        unit = [(None, "faq", teks) for teks in pecah_faq(doc["teks"])]
    elif doc["tipe"] == "dokumen":
        for halaman, teks in enumerate(doc["teks"].split("\f"), start=1):
            teks = bersihkan_halaman(teks)
            if daftar_isi(teks):
                if stat is not None:
                    stat["dibuang_daftar_isi"] = stat.get("dibuang_daftar_isi", 0) + 1
                continue
            for bagian in potong_teks(teks, maks):
                unit.append((halaman, "halaman", bagian))
    else:
        unit = [(None, "paragraf", teks) for teks in potong_teks(doc["teks"], maks)]

    hasil = []
    for urutan, (halaman, jenis_unit, teks) in enumerate(unit, start=1):
        teks = teks.strip()
        if len(teks) < MIN_CHUNK or not layak_chunk(teks):
            if stat is not None:
                stat["dibuang_chunk_tidak_layak"] = stat.get("dibuang_chunk_tidak_layak", 0) + 1
            continue
        hasil.append(
            {
                "chunk_id": f"{induk}-{urutan:03d}",
                "induk_id": induk,
                "tipe": doc["tipe"],
                "kategori": doc["kategori"],
                "dasar_kategori": doc["dasar_kategori"],
                "judul": doc["judul"],
                "url": doc["url"],
                "prodi": doc["prodi"],
                "tanggal": doc["tanggal"],
                "modified": doc["modified"],
                "halaman": halaman,
                "jenis_unit": jenis_unit,
                "jumlah_karakter": len(teks),
                "dipakai_kb": doc["dipakai_kb"],
                "teks": teks,
            }
        )
    return hasil


def tulis(out: Path, dokumen: list[dict], stat: dict, maks_chunk: int = MAKS_CHUNK, chunk_semua: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    grup: dict[str, list[dict]] = {nama: [] for nama in KATEGORI}
    dibuang = 0
    for rec in dokumen:
        kategori, dasar = klasifikasi(rec)
        if kategori == "sampah":
            dibuang += 1
            continue
        teks = rapikan_teks(rec.get("teks") or "")
        if not teks:
            stat["kosong_atau_gagal"] += 1
            continue
        judul = rapikan_judul(rec.get("judul") or "")
        slug = rec.get("slug") or ""
        grup[kategori].append(
            {
                "id": rec.get("id"),
                "tipe": rec.get("tipe"),
                "kategori": kategori,
                "dasar_kategori": dasar,
                "judul": judul,
                "url": rec.get("url") or "",
                "slug": slug,
                "tanggal": rec.get("tanggal") or "",
                "modified": rec.get("modified") or "",
                "prodi": prodi_bersih(judul, slug, rec.get("prodi") or "", kategori),
                "jumlah_karakter": len(teks),
                "hash_teks": hashlib.sha256(teks.encode("utf-8")).hexdigest(),
                "arsip": False,
                "teks": teks,
            }
        )

    unik: list[dict] = []
    terlihat: set[str] = set()
    for doc in (doc for items in grup.values() for doc in items):
        if doc["hash_teks"] in terlihat:
            stat["duplikat_teks"] += 1
            continue
        terlihat.add(doc["hash_teks"])
        unik.append(doc)

    tandai_arsip(unik)
    stat.setdefault("dibuang_daftar_isi", 0)
    stat.setdefault("dibuang_chunk_tidak_layak", 0)
    for doc in unik:
        doc["sifat"] = sifat(doc)
        doc["dipakai_kb"] = layak_kb(doc)
        doc["alasan_tidak_kb"] = (
            ""
            if doc["dipakai_kb"]
            else "arsip"
            if doc["arsip"]
            else "dinamis"
            if doc["sifat"] != "statis"
            else "administratif"
        )
    stat["duplikat_kb"] = buang_duplikat_kb(unik)

    grup = {nama: [doc for doc in unik if doc["kategori"] == nama] for nama in KATEGORI}
    chunks: list[dict] = []
    for doc in unik:
        if doc["dipakai_kb"] or chunk_semua:
            chunks.extend(buat_chunk(doc, maks_chunk, stat))

    manifest = {
        "sumber": "web_kb/pages.jsonl",
        "skrip": "scripts/categorize_web_kb.py",
        "dibuat": now(),
        "baris_sumber": stat["baris_sumber"],
        "kosong_atau_gagal": stat["kosong_atau_gagal"],
        "duplikat_url": stat["duplikat_url"],
        "duplikat_teks": stat["duplikat_teks"],
        "dibuang": dibuang,
        "jumlah_dokumen": sum(len(items) for items in grup.values()),
        "jumlah_chunk": len(chunks),
        "jumlah_kb": sum(doc["dipakai_kb"] for doc in unik),
        "jumlah_statis": sum(doc["sifat"] == "statis" for doc in unik),
        "jumlah_dinamis": sum(doc["sifat"] != "statis" for doc in unik),
        "jumlah_arsip": sum(doc["arsip"] for doc in unik),
        "duplikat_kb": stat["duplikat_kb"],
        "dibuang_daftar_isi": stat["dibuang_daftar_isi"],
        "dibuang_chunk_tidak_layak": stat["dibuang_chunk_tidak_layak"],
        "maks_chunk": maks_chunk,
        "kategori": {},
        "catatan": [
            "Teks dinormalisasi: spasi/newline dirapikan, karakter kontrol dibuang, form feed dipertahankan sebagai batas halaman PDF.",
            "Satu URL hanya muncul sekali; baris terpilih adalah yang teksnya paling panjang.",
            "Dokumen dengan hash teks sama hanya disimpan sekali.",
            "Judul dibersihkan dari sufiks ' - Fakultas Teknologi Maju dan Multidisiplin ...'.",
            "KB hanya memakai dokumen statis: halaman, dokumen resmi, profil staf, info prodi, dan regulasi; berita/agenda tetap tersimpan tetapi tidak dipakai.",
            "Edisi lama buku pedoman ditandai arsip dan tidak dipakai KB.",
            "Dokumen administratif (laporan anggaran, inventaris, kontrak kinerja, maklumat) tidak dipakai KB.",
            "Dokumen KB dengan judul atau nama staf sama hanya disimpan satu; sisanya diberi alasan_tidak_kb.",
            "Halaman daftar isi dan potongan dengan rasio huruf rendah dibuang dari chunks.jsonl.",
            "chunks.jsonl adalah unit retrieval: PDF per halaman, teks panjang dipotong, FAQ per pertanyaan.",
            "Field mentah yang tidak disalin: links, gambar, jumlah_html, diambil_pada, sumber_raw, status, perlu_render, mesin.",
            "Data mentah tetap di web_kb/pages.jsonl; berkas ini turunan dan gitignored.",
        ],
    }

    for nama, items in grup.items():
        if not items:
            continue
        items.sort(key=lambda d: (d["tanggal"] or d["modified"] or "", d["judul"]), reverse=True)
        path = out / f"{nama}.json"
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["kategori"][nama] = {
            "file": path.name,
            "jumlah": len(items),
            "statis": sum(doc["sifat"] == "statis" for doc in items),
            "kb": sum(doc["dipakai_kb"] for doc in items),
            "jumlah_chunk": sum(doc["kategori"] == nama for doc in chunks),
            "deskripsi": KATEGORI[nama],
        }

    with (out / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumber", type=Path, default=ROOT / "web_kb" / "pages.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "web_kb_categorized")
    parser.add_argument("--maks-chunk", type=int, default=MAKS_CHUNK)
    parser.add_argument("--chunk-semua", action="store_true", help="Buat chunk untuk berita/agenda juga")
    args = parser.parse_args()

    if not args.sumber.exists():
        raise SystemExit(f"Sumber tidak ada: {args.sumber}. Jalankan scrape lebih dulu.")
    dokumen, stat = muat(args.sumber)
    manifest = tulis(args.out, dokumen, stat, args.maks_chunk, args.chunk_semua)
    print(f"[OK] {manifest['jumlah_dokumen']} dokumen, {manifest['jumlah_chunk']} chunk -> {args.out}")
    print(
        f"     KB: {manifest['jumlah_kb']} dokumen, statis={manifest['jumlah_statis']}, "
        f"dinamis={manifest['jumlah_dinamis']}, arsip={manifest['jumlah_arsip']}, "
        f"duplikat_kb={manifest['duplikat_kb']}"
    )
    for nama, info in manifest["kategori"].items():
        print(f"  {nama:<16} {info['jumlah']:>5} dok  {info['jumlah_chunk']:>5} chunk  {info['file']}")


if __name__ == "__main__":
    main()
