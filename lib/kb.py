"""Memuat kurikulum dari knowledge_base/ dengan format chunk yang sama seperti aplikasi."""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"

PRODI_BY_ACRONYM = {
    "TSD": "Teknologi Sains Data",
    "TRKB": "Teknik Robotika dan Kecerdasan Buatan",
    "RN": "S1 Rekayasa Nanoteknologi",
    "TI": "Teknik Industri",
    "TE": "Teknik Elektro",
}


WEB_DIR = ROOT / "web_kb"

# Halaman web tidak selalu menyebut nama prodi dalam Bahasa Indonesia; alias ini hanya untuk inferensi.
PRODI_ALIASES = {
    "Teknologi Sains Data": ("teknologi sains data", "data science"),
    "Teknik Robotika dan Kecerdasan Buatan": (
        "teknik robotika",
        "robotika dan kecerdasan buatan",
        "robotics and artificial intelligence",
    ),
    "S1 Rekayasa Nanoteknologi": ("rekayasa nanoteknologi", "nanoteknologi", "nanotechnology"),
    "Teknik Industri": ("teknik industri", "industrial engineering"),
    "Teknik Elektro": ("teknik elektro", "electrical engineering"),
}


REGULATION_SOURCES = [
    ("panduanskripsi.json", "skripsi", "Panduan Skripsi FTMM UNAIR"),
    ("panduanpkl.json", "pkl", "Panduan PKL FTMM UNAIR"),
    ("panduanyudisium.json", "yudisium", "Panduan Yudisium FTMM UNAIR"),
    ("panduanakademik.json", "akademik", "Panduan Akademik FTMM UNAIR"),
    ("google_drive_sop.json", "sop", "SOP Administrasi Akademik FTMM UNAIR"),
]


@dataclass(frozen=True)
class Course:
    kode: str
    nama: str
    prodi: str
    chunk: str
    sumber: str = ""


def format_course_chunk(mk: dict) -> str:
    """Salinan format_course_chunk di 1/scripts/ingest_faiss.py, jangan diubah agar setara."""
    prodi = mk.get("program_studi", "FTMM")
    kode = mk.get("kode_mk", "").strip()
    nama = mk.get("nama_mk", "").strip()
    sks = mk.get("sks", 0)
    sem = mk.get("semester", 0)
    jenis = mk.get("jenis", "wajib")

    prasyarat_list = mk.get("prasyarat", [])
    if isinstance(prasyarat_list, list) and prasyarat_list:
        prasyarat = ", ".join(str(p).strip() for p in prasyarat_list)
    else:
        prasyarat = "Tidak Ada"

    deskripsi = mk.get("deskripsi", "").strip()

    cp_list = mk.get("capaian_pembelajaran", [])
    if isinstance(cp_list, list):
        cp = " ".join(str(c).strip() for c in cp_list)
    else:
        cp = str(cp_list).strip()

    dosen_list = mk.get("dosen_pengampu", [])
    if isinstance(dosen_list, list) and dosen_list:
        dosen = ", ".join(str(d).strip() for d in dosen_list)
    else:
        dosen = "Dosen Pengampu FTMM"

    return (
        f"[MATA KULIAH]\n"
        f"Program Studi: {prodi} | Jenis: {jenis}\n"
        f"Kode: {kode} | Nama: {nama} | SKS: {sks} | Semester: {sem}\n"
        f"Prasyarat: {prasyarat}\n"
        f"Deskripsi: {deskripsi}\n"
        f"Capaian Pembelajaran (CPMK): {cp}\n"
        f"Dosen: {dosen}"
    ).strip()


def format_regulation_chunk(item: dict, default_source: str, topic: str) -> str:
    """Salinan format_regulation_chunk di 1/scripts/ingest_regulations_faiss.py, jangan diubah agar setara."""
    source = item.get("source_name") or default_source
    chapter = item.get("chapter") or item.get("category") or ""
    subchapter = item.get("subchapter", "")
    title = item.get("title", "").strip()
    desc = item.get("description", "").strip()
    steps = item.get("step_by_step", "").strip()

    parts = [f"[REGULASI: {source.upper()}]", f"Kategori/Topik: {topic.upper()}"]
    if chapter:
        parts.append(f"Bab: {chapter}")
    if subchapter:
        parts.append(f"Pasal/Sub-Bab: {subchapter}")
    if title:
        parts.append(f"Judul: {title}")
    if desc:
        parts.append(f"Ketentuan/Isi: {desc}")
    if steps:
        parts.append(f"Prosedur/Alur: {steps}")
    return "\n".join(parts)


def load_regulations(kb_dir: Path = KB_DIR) -> list[Course]:
    """Butir regulasi memakai bentuk Course dengan prodi kosong, karena berlaku untuk semua prodi."""
    documents = []
    for filename, topic, default_source in REGULATION_SOURCES:
        items = json.loads((kb_dir / "regulasi" / filename).read_text(encoding="utf-8"))
        for position, item in enumerate(items, start=1):
            documents.append(
                Course(
                    kode=f"REG-{topic}-{item.get('id') or position}",
                    nama=item.get("title", "").strip(),
                    prodi="",
                    chunk=format_regulation_chunk(item, default_source, topic),
                    sumber=f"regulasi/{filename}",
                )
            )
    return documents


def infer_prodi(judul: str, slug: str) -> str:
    """Tebak prodi dari judul/slug halaman web; kosong berarti berlaku untuk fakultas."""
    haystack = f"{judul} {slug}".lower().replace("-", " ")
    for prodi, aliases in PRODI_ALIASES.items():
        if any(alias in haystack for alias in aliases):
            return prodi
    return ""


def format_web_chunk(page: dict) -> str:
    prodi = page.get("prodi") or "Fakultas"
    jenis = "Berita" if page.get("tipe") == "post" else "Halaman"
    tanggal = (page.get("modified") or page.get("tanggal") or "")[:10]
    return (
        f"[INFORMASI WEB FTMM]\n"
        f"Jenis: {jenis} | Program Studi: {prodi}\n"
        f"Judul: {page.get('judul', '')}\n"
        f"Sumber: {page.get('url', '')}\n"
        f"Diperbarui: {tanggal}\n"
        f"Isi: {page.get('teks', '')}"
    ).strip()


def load_web_pages(web_dir: Path = WEB_DIR) -> list[Course]:
    """Muat hasil scraping web_kb/pages.jsonl sebagai korpus terpisah dari knowledge_base/."""
    path = web_dir / "pages.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Hasil scraping web belum ada di {path}. Jalankan: uv run python scripts/scrape_ftmm.py")
    documents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        page = json.loads(line)
        if page.get("status") != "ok":
            continue
        documents.append(
            Course(
                kode=f"WEB-{page.get('tipe', 'page')}-{page.get('id') or page.get('slug', '')}",
                nama=page.get("judul", ""),
                prodi=page.get("prodi", ""),
                chunk=format_web_chunk(page),
                sumber=page.get("url", ""),
            )
        )
    return documents


def load_documents(kb_dir: Path = KB_DIR, web_dir: Path | None = None) -> list[Course]:
    documents = load_courses(kb_dir) + load_regulations(kb_dir)
    if web_dir is not None:
        documents += load_web_pages(web_dir)
    return documents


def load_courses(kb_dir: Path = KB_DIR) -> list[Course]:
    files = sorted(kb_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file JSON kurikulum di {kb_dir}")

    courses = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        default_prodi = data.get("metadata", {}).get("program_studi", path.stem.replace("FIX ", ""))
        for mk in data.get("mata_kuliah", []):
            mk["program_studi"] = mk.get("program_studi") or default_prodi
            courses.append(
                Course(
                    kode=mk.get("kode_mk", "").strip(),
                    nama=mk.get("nama_mk", "").strip(),
                    prodi=mk["program_studi"].strip(),
                    chunk=format_course_chunk(mk),
                    sumber=path.name,
                )
            )
    return courses
