import json

from lib import gt_eval, run_manifest
from lib.evaluate import TRACE_DEPTH, to_trace
from lib.kb import Course


def courses() -> list[Course]:
    return [Course(kode=f"K{i}", nama=f"Matkul {i}", prodi="Teknik Industri", chunk="x") for i in range(30)]


def test_to_trace_keeps_rank_and_depth():
    hits = [(i, 1.0 - i / 100) for i in range(25)]
    trace = to_trace(hits, courses())

    assert len(trace) == TRACE_DEPTH
    assert [item["rank"] for item in trace] == list(range(1, TRACE_DEPTH + 1))
    assert trace[0]["kode"] == "K0"
    assert to_trace(hits, courses(), depth=3)[-1]["rank"] == 3


def test_manifest_roundtrip_and_differences(tmp_path):
    path = tmp_path / "run.json"
    saved = run_manifest.write_manifest(path, {"metode": "BM25", "corpus_sha256": "abc"})

    assert run_manifest.read_manifest(path) == saved
    assert run_manifest.differences(saved, {"metode": "BM25", "corpus_sha256": "abc"}, ("metode", "corpus_sha256")) == []
    assert run_manifest.differences(saved, {"metode": "Dense", "corpus_sha256": "abc"}, ("metode",)) == ["metode: 'BM25' -> 'Dense'"]


def test_gt_trace_collects_per_question_rows(tmp_path):
    gt_eval.append_trace("uji", {"id": "1", "kandidat": [{"rank": 1}], "error": None}, tmp_path)
    gt_eval.append_trace("uji", {"id": "2", "error": "LLMError: habis"}, tmp_path)

    rows = gt_eval.read_traces("uji", tmp_path)
    assert [row["id"] for row in rows] == ["1", "2"]
    assert rows[1]["error"].startswith("LLMError")
    assert gt_eval.trace_path("uji", tmp_path).name == "jejak_uji.jsonl"


def test_score_trace_written_per_question(tmp_path):
    answer = {field: "x" for field in gt_eval.ANSWER_FIELDS}
    answer.update(
        id="1",
        kategori="Silabus Wajib",
        peringkat_sumber="2",
        latensi_retrieval_ms="1",
        latensi_generasi_ms="2",
        ttft_ms="1",
        jawaban="jawab",
        jawaban_acuan="acuan",
    )
    scored = gt_eval.score_answers(
        [answer],
        bert=lambda c, r: [(0.1, 0.2, 0.3)],
        rouge=lambda c, r: [{"rouge1_f1": 0.4, "rouge2_f1": 0.5, "rougeL_f1": 0.6}],
    )
    gt_eval.write_scores("uji", scored, gt_eval.summarize(scored), tmp_path)

    rows = [json.loads(line) for line in gt_eval.score_trace_path("uji", tmp_path).read_text(encoding="utf-8").splitlines()]
    assert rows[0]["id"] == "1"
    assert rows[0]["bert_f1"] == 0.3
    assert rows[0]["rougeL_f1"] == 0.6
    assert "pertanyaan" not in rows[0]


def test_to_trace_includes_component_scores():
    hits = [(0, 0.9, {"dense": 0.7, "sparse": 0.9, "rrf": 0.03}), (1, 0.5, {})]
    trace = to_trace(hits, courses(), depth=2)

    assert trace[0]["komponen"]["rrf"] == 0.03
    assert trace[1]["komponen"] == {}
