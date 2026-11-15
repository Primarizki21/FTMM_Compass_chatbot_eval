"""UID adalah kunci penilaian retrieval; kode MK tidak unik lintas prodi."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.evaluate import uid_targets
from lib.kb import load_courses, load_regulations

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "ground_truth"


def test_uid_unik_dan_tidak_kosong():
    for docs in (load_courses(), load_regulations()):
        uids = [d.uid for d in docs]
        assert all(uids)
        assert len(set(uids)) == len(docs)


def test_uid_membedakan_kode_kembar_prodi_sama():
    docs = [d for d in load_courses() if d.kode == "MAA103"]
    assert len(docs) == 2
    assert len({d.uid for d in docs}) == 2


@pytest.mark.parametrize("nama", ["ir_eval_dataset.json", "retrieval_complexity_benchmark.json"])
def test_target_uid_resolve_ke_korpus(nama):
    path = GT_DIR / nama
    if not path.exists():
        pytest.skip("ground truth belum ada di clone ini")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data and "target_uids" not in data[0]:
        pytest.skip("dataset belum dinormalisasi")
    korpus = {d.uid for d in load_courses()} | {d.uid for d in load_regulations()}
    for item in data:
        if item.get("is_out_of_scope"):
            assert item["target_uids"] == []
            continue
        assert uid_targets(item) <= korpus, item["id"]
