"""Manifest per run: konfigurasi dan provenance yang ditulis bersama hasil evaluasi."""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def write_manifest(path: Path, fields: dict) -> dict:
    data = {"commit": commit(), "ditulis": datetime.now().isoformat(timespec="seconds"), **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def differences(saved: dict, current: dict, keys: tuple[str, ...]) -> list[str]:
    """Field yang berbeda antara manifest tersimpan dan konfigurasi aktif."""
    beda = []
    for key in keys:
        lama, baru = saved.get(key), current.get(key)
        if str(lama) != str(baru):
            beda.append(f"{key}: {lama!r} -> {baru!r}")
    return beda
