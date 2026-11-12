"""Rapikan web_kb/pages.jsonl menjadi JSON per kategori informasi.

Sumber tetap `web_kb/pages.jsonl` (mentah). Skrip ini hanya menurunkan artefak:
satu berkas JSON per kategori, teks dinormalisasi, URL duplikat dibuang, dan
manifest berisi aturan klasifikasi. Tidak ada akses jaringan.
"""

import argparse
import hashlib
import json
import re
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

SUFIKS_JUDUL = re.compile(r"\s*[-–|]\s*Fakultas Teknologi Maju dan Multidisiplin.*$", re.IGNORECASE)
GELAR = re.compile(
    r"\b(?:prof\.|dr\.|ir\.|dr\.eng|s\.t|s\.si|s\.kom|s\.sos|s\.e\b|s\.m|s\.ip|s\.farm|"
    r"s\.pd|s\.st|s\.ikom|m\.t|m\.sc|m\.kom|m\.si|m\.stat|m\.eng|m\.tr|m\.farm|m\.ba|"
    r"m\.m\b|m\.imun|ph\.d|a\.md|sp\.)"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ada(teks: str, *frasa: str) -> bool:
    """Cocokkan kata/frasa utuh; 'ukt' tidak boleh cocok di 'produktif'."""
    return any(re.search(r"(?<![a-z0-9])" + re.escape(frasa_) + r"(?![a-z0-9])", teks) for frasa_ in frasa)


def rapikan_teks(teks: str) -> str:
    teks = (teks or "").replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    teks = teks.replace("\f", "\n\n")
    teks = "".join(ch for ch in teks if ch in "\n\t" or ord(ch) >= 32)
    baris = [re.sub(r"[ \t]+", " ", baris).strip() for baris in teks.split("\n")]
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
    if ada(judul, "program studi", "study program") or "/program-studi" in path:
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
    if ada(judul, "audit mutu", "survei kepuasan", "tracer study", "akreditasi", "kepuasan mahasiswa", "kepuasan dosen", "ban-pt", "asesmen lapangan"):
        return "mutu", "mutu"
    if ada(judul, "sop", "standar operasional", "aturan berperilaku", "buku aturan", "pedoman", "panduan", "regulasi", "kontrak kinerja", "laporan kinerja", "ppid", "zona integritas", "rencana strategis", "renstra"):
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
    if ada(judul, "juara", "raih", "sabet", "menang", "medali", "award", "penghargaan", "kompetisi", "lomba", "finalis", "winner", "berprestasi"):
        return "prestasi", "prestasi"
    if ada(judul, "pengabdian", "pengmas", "abdimas", "kkn"):
        return "pengabdian", "pengabdian"
    if ada(judul, "penelitian", "research", "riset", "publikasi", "jurnal", "konferensi", "seminar", "icatam", "segta", "semnas", "research group"):
        return "penelitian", "penelitian"
    if ada(judul, "kerja sama", "kerjasama", "kolaborasi", "mitra", "mou", "kunjungan", "guest lecture", "internasional", "universiti", "inbound", "outbound", "exchange"):
        return "kerjasama", "kerjasama"
    if ada(judul, "kemahasiswaan", "bem", "blm", "hima", "mahasiswa", "pkkmb", "lkmm", "pemira", "komunitas", "aktivitas", "mbkm", "alumni", "organisasi"):
        return "kemahasiswaan", "kemahasiswaan"
    if tipe == "post" and rec.get("prodi") and ada(
        judul, "belajar", "kompetensi lulusan", "kesempatan karir", "bidang peminatan", "prospek", "yang dipelajari", "penerapan"
    ):
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
    }
    return list(terbaik.values()), stat


def tulis(out: Path, dokumen: list[dict], stat: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    grup: dict[str, list[dict]] = {nama: [] for nama in KATEGORI}
    for rec in dokumen:
        kategori, dasar = klasifikasi(rec)
        if kategori == "sampah":
            stat["dibuang"] = stat.get("dibuang", 0) + 1
            continue
        teks = rapikan_teks(rec.get("teks") or "")
        if not teks:
            stat["kosong_atau_gagal"] += 1
            continue
        grup[kategori].append(
            {
                "id": rec.get("id"),
                "tipe": rec.get("tipe"),
                "kategori": kategori,
                "dasar_kategori": dasar,
                "judul": rapikan_judul(rec.get("judul") or ""),
                "url": rec.get("url") or "",
                "slug": rec.get("slug") or "",
                "tanggal": rec.get("tanggal") or "",
                "modified": rec.get("modified") or "",
                "prodi": rec.get("prodi") or "",
                "jumlah_karakter": len(teks),
                "hash_teks": hashlib.sha256(teks.encode("utf-8")).hexdigest(),
                "teks": teks,
            }
        )

    manifest = {
        "sumber": "web_kb/pages.jsonl",
        "skrip": "scripts/categorize_web_kb.py",
        "dibuat": now(),
        "baris_sumber": stat["baris_sumber"],
        "kosong_atau_gagal": stat["kosong_atau_gagal"],
        "duplikat_url": stat["duplikat_url"],
        "dibuang": stat.get("dibuang", 0),
        "jumlah_dokumen": sum(len(items) for items in grup.values()),
        "kategori": {},
        "catatan": [
            "Teks dinormalisasi: spasi/newline dirapikan, karakter kontrol dibuang, form feed PDF menjadi baris kosong.",
            "Satu URL hanya muncul sekali; baris terpilih adalah yang teksnya paling panjang.",
            "Judul dibersihkan dari sufiks ' - Fakultas Teknologi Maju dan Multidisiplin ...'.",
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
            "deskripsi": KATEGORI[nama],
        }

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumber", type=Path, default=ROOT / "web_kb" / "pages.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "web_kb_categorized")
    args = parser.parse_args()

    if not args.sumber.exists():
        raise SystemExit(f"Sumber tidak ada: {args.sumber}. Jalankan scrape lebih dulu.")
    dokumen, stat = muat(args.sumber)
    manifest = tulis(args.out, dokumen, stat)
    print(f"[OK] {manifest['jumlah_dokumen']} dokumen -> {args.out}")
    for nama, info in manifest["kategori"].items():
        print(f"  {nama:<16} {info['jumlah']:>5}  {info['file']}")


if __name__ == "__main__":
    main()
