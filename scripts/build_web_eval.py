"""Susun kandidat dataset evaluasi retrieval dari web_kb_categorized/chunks.jsonl.

Kueri dibuat dari unit yang dipakai KB (halaman statis, profil, regulasi, PDF,
FAQ). Hasilnya "silver": otomatis, deterministik, dan siap dikurasi manual.
Target penilaian berupa URL sumber; chunk target ikut disimpan untuk evaluasi
tingkat halaman.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEMPLATE = {
    "profil_staf": "Siapa {judul}?",
    "profil_fakultas": "Apa isi halaman {judul} FTMM UNAIR?",
    "program_studi": "Apa yang perlu diketahui tentang {judul}?",
    "akademik": "Apa ketentuan {judul}?",
    "fasilitas": "Fasilitas apa yang ada di {judul}?",
    "regulasi": "Apa isi {judul}?",
    "mutu": "Bagaimana {judul}?",
    "kemahasiswaan": "Apa informasi tentang {judul}?",
    "penelitian": "Apa informasi tentang {judul}?",
    "pengabdian": "Apa informasi tentang {judul}?",
    "kerjasama": "Apa informasi tentang {judul}?",
    "sdgs": "Apa informasi tentang {judul}?",
    "lainnya": "Apa informasi tentang {judul}?",
}

ALUR = ("alur", "panduan", "prosedur", "permohonan", "pengajuan")


def muat_chunk(web_dir: Path) -> list[dict]:
    path = web_dir / "chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"chunks.jsonl tidak ada di {web_dir}. Jalankan: make kategori")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def judul_bagian(teks: str) -> str:
    """Ambil baris pertama yang layak jadi judul bagian PDF."""
    for baris in teks.splitlines():
        baris = baris.strip(" .-–—")
        if not (4 <= len(baris) <= 60):
            continue
        if "..." in baris or baris.isdigit() or not any(c.isalpha() for c in baris):
            continue
        if baris[0].islower() or baris.endswith((",", ";", ":")):
            continue
        rendah = f" {baris.lower()} "
        if any(penanda in rendah for penanda in (" yang ", " adalah ", " merupakan ", " terdiri atas ")):
            continue
        if len(baris.split()) > 9:
            continue
        return baris
    return ""


def kueri(chunk: dict) -> tuple[str, str]:
    """Kembalikan (kueri, jenis_kueri)."""
    judul = (chunk.get("judul") or "").strip()
    kategori = chunk.get("kategori") or "lainnya"

    if chunk.get("jenis_unit") == "faq":
        pertama = chunk["teks"].split("?", 1)[0].strip()
        return f"{pertama}?", "pertanyaan-faq"
    if kategori == "dokumen_resmi":
        bagian = judul_bagian(chunk["teks"])
        if bagian and bagian.lower() not in judul.lower():
            return f"Apa isi bagian {bagian} pada {judul}?", "bagian-dokumen"
        return f"Apa isi halaman {chunk.get('halaman')} pada {judul}?", "halaman-dokumen"
    if kategori == "akademik" and judul.lower().startswith(ALUR):
        return f"Bagaimana {judul}?", "judul"

    template = TEMPLATE.get(kategori, "Apa informasi tentang {judul}?")
    return template.format(judul=judul), "judul"


def pilih_bagian(chunks: list[dict], batas: int) -> list[dict]:
    """Ambil unit yang tersebar di seluruh dokumen, bukan hanya awal."""
    if len(chunks) <= batas:
        return chunks
    posisi = sorted({round(i * (len(chunks) - 1) / (batas - 1)) for i in range(batas)})
    return [chunks[i] for i in posisi]


def susun(chunks: list[dict], per_kategori: int, per_dokumen: int, seed: int) -> list[dict]:
    per_induk: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        if not chunk.get("dipakai_kb") or len(chunk.get("teks") or "") < 120:
            continue
        per_induk[chunk["induk_id"]].append(chunk)

    kandidat: dict[str, list[dict]] = defaultdict(list)
    for induk in sorted(per_induk):
        dokumen = sorted(per_induk[induk], key=lambda c: c["chunk_id"])
        bagian = dokumen if dokumen[0]["kategori"] == "faq" else pilih_bagian(dokumen, per_dokumen)
        kandidat[dokumen[0]["kategori"]].extend(bagian)

    rng = random.Random(seed)
    baris = []
    terlihat: set[str] = set()
    for kategori in sorted(kandidat):
        pilihan = kandidat[kategori][:]
        rng.shuffle(pilihan)
        dipakai = 0
        for chunk in pilihan:
            if dipakai >= per_kategori:
                break
            teks_kueri, jenis_kueri = kueri(chunk)
            if len(teks_kueri) < 12 or teks_kueri in terlihat:
                continue
            terlihat.add(teks_kueri)
            dipakai += 1
            baris.append(
                {
                    "id": "",
                    "kategori": kategori,
                    "query": teks_kueri,
                    "jenis_kueri": jenis_kueri,
                    "target_url": chunk["url"],
                    "target_chunk": chunk["chunk_id"],
                    "target_judul": chunk.get("judul", ""),
                    "target_halaman": chunk.get("halaman"),
                    "sumber": "otomatis",
                }
            )

    baris.sort(key=lambda row: (row["kategori"], row["target_chunk"]))
    for nomor, row in enumerate(baris, start=1):
        row["id"] = f"WEBEVAL-{nomor:03d}"
    return baris


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumber", type=Path, default=ROOT / "web_kb_categorized")
    parser.add_argument("--out", type=Path, default=ROOT / "web_kb_categorized" / "web_eval.jsonl")
    parser.add_argument("--per-kategori", type=int, default=8)
    parser.add_argument("--per-dokumen", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunks = muat_chunk(args.sumber)
    baris = susun(chunks, args.per_kategori, args.per_dokumen, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in baris), encoding="utf-8")
    print(f"[OK] {len(baris)} kueri -> {args.out}")
    print(f"     sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()[:12]}")
    for kategori in sorted({row["kategori"] for row in baris}):
        jumlah = sum(1 for row in baris if row["kategori"] == kategori)
        print(f"  {kategori:<16} {jumlah:>3}")


if __name__ == "__main__":
    main()
