"""Log eksekusi harus tersimpan sendiri, termasuk saat run gagal."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.runlog import catat_event, catat_run


def test_catat_run_menyimpan_terminal_dan_traceback(tmp_path):
    with pytest.raises(RuntimeError):
        with catat_run(tmp_path, ["demo.py", "--limit", "5"]):
            print("baris terminal")
            raise RuntimeError("kegagalan contoh")

    log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "baris terminal" in log
    assert "demo.py --limit 5" in log
    assert "RuntimeError: kegagalan contoh" in (tmp_path / "errors.log").read_text(encoding="utf-8")


def test_catat_event_append_satu_baris_json(tmp_path):
    catat_event(tmp_path, tahap="mulai", dokumen=3)
    catat_event(tmp_path, tahap="selesai")
    baris = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["tahap"] for line in baris] == ["mulai", "selesai"]
    assert json.loads(baris[0])["dokumen"] == 3
