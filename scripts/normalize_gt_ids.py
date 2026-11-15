"""Tambahkan UID global ke dataset ground truth dan laporkan yang belum bisa diselesaikan.

UID berasal dari `lib/kb.py`: `MK:<akronim>:<id>`, `REG:<topik>:<id>`, `chunk_id` web.
Dataset JSON ditulis ulang di tempat (gitignored); GT_MASTER tidak diubah, pemetaannya
ditulis ke `ground_truth/gt_source_uids.csv`.
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.kb import PRODI_BY_ACRONYM, load_documents, load_web_categorized  # noqa: E402
from lib.runlog import catat_event, catat_run  # noqa: E402

GT_DIR = ROOT / "ground_truth"
IR_PATH = GT_DIR / "ir_eval_dataset.json"
CX_PATH = GT_DIR / "retrieval_complexity_benchmark.json"
GT_PATH = GT_DIR / "GT_MASTER.csv"
WEB_GOLD_PATH = ROOT / "web_kb_categorized" / "web_eval_gold.jsonl"
UID_MAP_PATH = GT_DIR / "gt_source_uids.csv"
AUDIT_DIR = ROOT / "results" / "gt_audit"


def muat_korpus() -> tuple[dict[str, object], dict[str, list]]:
    dokumen = load_documents()
    by_uid = {d.uid: d for d in dokumen}
    by_kode: dict[str, list] = defaultdict(list)
    for d in dokumen:
        by_kode[d.kode].append(d)
    return by_uid, by_kode


def resolve_ir(item: dict, by_kode: dict[str, list]) -> tuple[list[str], str]:
    """Target dinilai prodi-agnostik: semua UID dengan kode+nama yang sama."""
    uids: list[str] = []
    gagal = multi = False
    for kode, nama in zip(item["target_kode_mk"], item["target_nama_mk"]):
        kandidat = [d for d in by_kode.get(kode.strip(), []) if d.nama == nama.strip()]
        if not kandidat:
            gagal = True
            continue
        if len(kandidat) > 1:
            multi = True
        uids.extend(d.uid for d in kandidat)
    status = "gagal" if gagal else ("multi-prodi" if multi else "tunggal")
    return sorted(set(uids)), status


def resolve_complexity(item: dict, by_kode: dict[str, list]) -> tuple[list[str], str]:
    """Target punya prodi eksplisit, jadi kode kembar lintas prodi tidak ambigu."""
    if item.get("is_out_of_scope"):
        return [], "luar-cakupan"
    prodi = PRODI_BY_ACRONYM.get(item["program_studi"], item["program_studi"])
    uids = sorted(
        {
            d.uid
            for kode in item["target_kode_mk"]
            for d in by_kode.get(kode.strip(), [])
            if d.prodi == prodi
        }
    )
    jumlah_target = len({k.strip() for k in item["target_kode_mk"]})
    return uids, "ok" if len(uids) >= jumlah_target else "gagal"


def tulis_json(path: Path, data: list[dict]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def proses_ir(by_kode: dict[str, list]) -> Counter:
    data = json.loads(IR_PATH.read_text(encoding="utf-8"))
    status = Counter()
    for item in data:
        item["target_uids"], item["_uid_status"] = resolve_ir(item, by_kode)
        status[item["_uid_status"]] += 1
    tulis_json(IR_PATH, data)
    return status


def proses_complexity(by_kode: dict[str, list]) -> Counter:
    data = json.loads(CX_PATH.read_text(encoding="utf-8"))
    status = Counter()
    for item in data:
        item["target_uids"], item["_uid_status"] = resolve_complexity(item, by_kode)
        status[item["_uid_status"]] += 1
    tulis_json(CX_PATH, data)
    return status


def proses_web_gold() -> Counter:
    if not WEB_GOLD_PATH.exists():
        return Counter({"tidak-ada": 1})
    baris = [json.loads(line) for line in WEB_GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    uid_valid = {d.uid for d in load_web_categorized()}
    status = Counter()
    for item in baris:
        target = item.get("target_chunk")
        calon = [target] if isinstance(target, str) else list(target or [])
        item["target_uids"] = [uid for uid in calon if uid in uid_valid]
        item["_uid_status"] = "luar-cakupan" if item.get("jenis_kueri") == "luar-cakupan" else (
            "ok" if item["target_uids"] else "gagal"
        )
        status[item["_uid_status"]] += 1
    WEB_GOLD_PATH.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in baris) + "\n", encoding="utf-8"
    )
    return status


def prodi_dari_jawaban(teks: str, prodi_names: list[str]) -> str:
    """GT silabus memuat baris `Program Studi: <prodi>` karena jawaban acuan dibuat dari template katalog."""
    ditemukan = [prodi for prodi in prodi_names if prodi.lower() in teks.lower()]
    if len(ditemukan) != 1:
        return ""
    marker = re.search(r"(?i)program\s*studi\s*[:\-]?\s*" + re.escape(ditemukan[0]), teks)
    return ditemukan[0] if marker else ""


def uid_dari_kandidat(teks: str, kandidat: list) -> tuple[str, str]:
    """Urutan: prodi eksplisit -> nama MK di jawaban -> prodi-agnostik bila semua kandidat satu MK."""
    prodi_names = sorted({d.prodi for d in kandidat if d.prodi})
    prodi = prodi_dari_jawaban(teks, prodi_names)
    cocok = [d for d in kandidat if d.prodi == prodi] if prodi else []
    if len(cocok) == 1:
        return cocok[0].uid, "prodi-dari-jawaban"

    teks_lower = teks.lower()
    nama = [d for d in kandidat if d.nama.lower() in teks_lower]
    if len(nama) == 1:
        return nama[0].uid, "nama-dari-jawaban"
    if nama and len({d.nama for d in nama}) == 1:
        return ";".join(sorted(d.uid for d in nama)), "multi-prodi"
    if len({d.nama for d in kandidat}) == 1:
        return ";".join(sorted(d.uid for d in kandidat)), "multi-prodi"
    return "", "kode-ambigu"


def proses_gt_master(by_kode: dict[str, list]) -> tuple[Counter, list[dict]]:
    """GT_MASTER tidak diubah; pemetaan source_id -> UID ditulis sebagai berkas terpisah."""
    baris = list(csv.DictReader(GT_PATH.open(encoding="utf-8-sig")))
    status = Counter()
    keluaran = []
    for row in baris:
        kode = row["source_id"].strip()
        kandidat = by_kode.get(kode, []) if row["source_type"].startswith("Silabus") else []
        if len(kandidat) == 1:
            uid, label = kandidat[0].uid, "ok"
        elif kandidat:
            uid, label = uid_dari_kandidat(row["answer"], kandidat)
        else:
            uid, label = "", "butuh-dokumen-asal"
        status[label] += 1
        keluaran.append({"id": row["id"], "source_id": kode, "source_uid": uid, "status": label})
    UID_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UID_MAP_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "source_id", "source_uid", "status"])
        writer.writeheader()
        writer.writerows(keluaran)
    return status, keluaran


def tulis_laporan(ir: Counter, cx: Counter, web: Counter, gt: Counter, keluaran: list[dict]) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ambigu = [row for row in keluaran if row["status"] == "kode-ambigu"]
    tanpa_dokumen = [row for row in keluaran if row["status"] == "butuh-dokumen-asal"]
    lines = [
        "# Audit UID Ground Truth",
        "",
        "Dibuat `scripts/normalize_gt_ids.py`. Hanya ID dan hitungan; tanpa teks soal/jawaban.",
        "",
        "## Dataset retrieval",
        "",
        "| Dataset | Status | n |",
        "|---|---|---|",
        *[f"| ir_eval | {k} | {v} |" for k, v in sorted(ir.items())],
        *[f"| complexity | {k} | {v} |" for k, v in sorted(cx.items())],
        *[f"| web_gold | {k} | {v} |" for k, v in sorted(web.items())],
        "",
        "## GT_MASTER (pemetaan `source_id` -> `source_uid`)",
        "",
        "| Status | n |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(gt.items())],
        "",
        f"- `kode-ambigu`: {len(ambigu)} baris. Butuh kolom prodi atau nama di sumber.",
        f"- `butuh-dokumen-asal`: {len(tanpa_dokumen)} baris. `source_id` berupa nomor butir tanpa nama dokumen.",
        "",
        "## Baris yang belum punya UID",
        "",
        "| id GT | source_id | status |",
        "|---|---|---|",
        *[f"| {row['id']} | {row['source_id']} | {row['status']} |" for row in ambigu + tanpa_dokumen],
    ]
    path = AUDIT_DIR / "uid_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with catat_run(AUDIT_DIR, sys.argv, nama_log="normalize.log"):
        try:
            _, by_kode = muat_korpus()
            catat_event(AUDIT_DIR, tahap="muat_korpus", kode_unik=len(by_kode))
            ir = proses_ir(by_kode)
            cx = proses_complexity(by_kode)
            web = proses_web_gold()
            gt, keluaran = proses_gt_master(by_kode)
            path = tulis_laporan(ir, cx, web, gt, keluaran)
            catat_event(AUDIT_DIR, tahap="selesai", ir=dict(ir), complexity=dict(cx), web_gold=dict(web), gt_master=dict(gt))
            print(f"ir_eval: {dict(ir)}")
            print(f"complexity: {dict(cx)}")
            print(f"web_gold: {dict(web)}")
            print(f"GT_MASTER: {dict(gt)}")
            print(f"Pemetaan: {UID_MAP_PATH}")
            print(f"Laporan: {path}")
        except BaseException as error:
            catat_event(AUDIT_DIR, tahap="gagal", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
