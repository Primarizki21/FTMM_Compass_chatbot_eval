import sys
import types

import pytest
from streamlit.testing.v1 import AppTest

from lib import gt_eval, llm, rag
from lib.kb import ROOT, Course, load_documents

CONFIG = llm.LLMConfig(llm.LOCAL, "uji")


def gt_item(id_="1", source_type="Silabus Wajib", source_id="MAS120", answer="acuan"):
    return {"id": id_, "question": "Apa isi MAS120?", "answer": answer, "source_id": source_id, "source_type": source_type}


def fake_stream(messages, config, keep_alive, stats):
    stats.update(ttft_ms=5.0, generation_ms=9.0, prompt_tokens=100, answer_tokens=2)
    yield "Jawaban "
    yield "uji"


def scored_row(kategori, rank, bert=0.5):
    return {
        "kategori": kategori,
        "peringkat_sumber": rank,
        "latensi_retrieval_ms": "10",
        "latensi_generasi_ms": "90",
        "ttft_ms": "20",
        "bert_precision": bert,
        "bert_recall": bert,
        "bert_f1": bert,
        "rouge1_f1": 0.1,
        "rouge2_f1": 0.2,
        "rougeL_f1": 0.3,
    }


def test_ground_truth_has_all_rows_and_sources():
    rows = gt_eval.load_ground_truth()
    assert len(rows) == 449
    assert all(row["question"] and row["answer"] for row in rows)
    codes = {doc.kode for doc in load_documents()}
    silabus = [row for row in rows if row["source_type"].startswith("Silabus")]
    assert len(silabus) == 369
    assert all(row["source_id"].strip() in codes for row in silabus)


def test_sample_evenly_spreads_across_categories():
    rows = gt_eval.load_ground_truth()
    sample = gt_eval.sample_evenly(rows, 30)
    assert len(sample) == len({row["id"] for row in sample}) == 30
    assert {row["source_type"] for row in sample} == {"Silabus Wajib", "Silabus Pilihan", "Regulasi Akademik", "SOP Layanan"}
    assert gt_eval.sample_evenly(rows, 449) == rows


def test_run_names_differ_per_combination():
    names = {gt_eval.run_name(CONFIG, method, top_k) for method in rag.METHODS for top_k in (3, 5)}
    assert len(names) == 8
    assert gt_eval.run_name(llm.LLMConfig(llm.GEMINI, "gemini-x"), rag.BM25, 3) != gt_eval.run_name(CONFIG, rag.BM25, 3)
    assert all(ch.isalnum() or ch in ".-" for ch in gt_eval.run_name(llm.LLMConfig(llm.LOCAL, "llama3.2:3b"), rag.BM25, 3))


def test_source_rank_dedupes_twins_and_skips_regulation():
    assert gt_eval.source_rank(gt_item(), ["X", "X", "MAS120"]) == "2"
    assert gt_eval.source_rank(gt_item(), ["X"]) == "0"
    assert gt_eval.source_rank(gt_item(source_type="Regulasi Akademik", source_id="4"), ["X"]) == ""


def test_answer_question_has_no_history_and_hides_reference(monkeypatch):
    sent = {}

    def capture(messages, config, keep_alive, stats):
        sent["messages"], sent["keep_alive"] = messages, keep_alive
        yield "ok"

    monkeypatch.setattr(llm, "stream_chat", capture)
    retriever = rag.CourseRetriever([Course("MAS120", "Statistika", "Teknologi Sains Data", "MAS120 statistika")])
    row = gt_eval.answer_question(retriever, gt_item(answer="RAHASIA ACUAN"), rag.BM25, 3, CONFIG)

    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert "RAHASIA ACUAN" not in str(sent["messages"])
    assert sent["keep_alive"] == "10m"
    assert row["jawaban"] == "ok"
    assert row["jawaban_acuan"] == "RAHASIA ACUAN"
    assert row["peringkat_sumber"] == "1"
    kandidat = row["_jejak"]["kandidat"][0]
    assert set(kandidat["komponen"]) == {"sparse"}
    assert kandidat["peringkat"]["final"] == 1


def test_rouge_scores_real_library():
    pytest.importorskip("rouge_score")
    same, different = gt_eval.compute_rouge(
        ["mata kuliah statistika", "kucing hitam"], ["mata kuliah statistika", "mata kuliah statistika"]
    )
    assert same == {"rouge1_f1": 1.0, "rouge2_f1": 1.0, "rougeL_f1": 1.0}
    assert different["rouge1_f1"] == 0.0


def test_summary_counts_retrieval_only_for_silabus():
    rows = [
        scored_row("Silabus Wajib", "1", bert=1.0),
        scored_row("Silabus Wajib", "2", bert=0.5),
        scored_row("Silabus Pilihan", "0", bert=0.0),
        scored_row("Regulasi Akademik", "", bert=0.25),
    ]
    summary = {group["kelompok"]: group for group in gt_eval.summarize(rows)}
    everything = summary["Semua"]
    assert (everything["n"], everything["n_retrieval"]) == (4, 3)
    assert everything["hit@1"] == pytest.approx(1 / 3, abs=1e-3)
    assert everything["hit@k"] == pytest.approx(2 / 3, abs=1e-3)
    assert everything["mrr"] == pytest.approx(0.5)
    assert everything["bert_f1"] == pytest.approx(0.4375)
    assert summary["Regulasi Akademik"]["hit@1"] == ""
    assert everything["total_p50_ms"] == 100.0


def test_write_scores_produces_readable_files(tmp_path):
    answer = {field: "x" for field in gt_eval.ANSWER_FIELDS}
    answer.update(
        kategori="SOP Layanan",
        peringkat_sumber="",
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
    assert gt_eval.read_rows(gt_eval.scores_path("uji", tmp_path))[0]["bert_f1"] == "0.3"
    assert gt_eval.read_rows(gt_eval.summary_path("uji", tmp_path))[0]["kelompok"] == "Semua"


@pytest.fixture
def eval_app(tmp_path, monkeypatch):
    monkeypatch.setattr(gt_eval, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(llm, "installed_models", lambda: {llm.LOCAL_MODEL, llm.EMBED_MODEL})
    monkeypatch.setattr(llm, "stream_chat", fake_stream)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Evaluasi Ground Truth").run()
    assert not at.exception
    at.number_input[0].set_value(3).run()
    return at, tmp_path


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def test_eval_runs_all_selected_questions_then_scores(eval_app, monkeypatch):
    at, eval_dir = eval_app
    click(at, "Mulai / Lanjutkan")
    assert not at.exception
    assert not at.session_state.eval_running

    answers = list(eval_dir.glob("jawaban_*.csv"))
    assert len(answers) == 1
    rows = gt_eval.read_rows(answers[0])
    assert len({row["id"] for row in rows}) == len(rows) == 3
    assert rows[0]["jawaban"] == "Jawaban uji"
    assert rows[0]["jawaban_acuan"]

    monkeypatch.setattr(gt_eval, "compute_bert", lambda c, r: [(0.5, 0.5, 0.5)] * len(c))
    click(at, "Hitung skor")
    assert not at.exception
    assert list(eval_dir.glob("ringkasan_*.csv"))


def test_eval_stops_on_error_and_resumes_without_repeating(eval_app, monkeypatch):
    at, eval_dir = eval_app
    calls = {"n": 0}

    def flaky(messages, config, keep_alive, stats):
        calls["n"] += 1
        if calls["n"] == 2:
            raise llm.LLMError("kuota habis")
        yield "ok"

    monkeypatch.setattr(llm, "stream_chat", flaky)
    click(at, "Mulai / Lanjutkan")
    assert not at.exception
    assert any("kuota habis" in error.value for error in at.error)
    assert not at.session_state.eval_running
    path = next(eval_dir.glob("jawaban_*.csv"))
    assert len(gt_eval.read_rows(path)) == 1

    click(at, "Mulai / Lanjutkan")
    rows = gt_eval.read_rows(path)
    assert len(rows) == len({row["id"] for row in rows}) == 3


def test_gemini_warns_when_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "installed_models", lambda: {llm.EMBED_MODEL})
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.selectbox[1].set_value(llm.GEMINI).run()
    assert not at.exception
    assert any("GEMINI_API_KEY" in warning.value for warning in at.sidebar.warning)
    assert any("server Google" in info.value for info in at.sidebar.info)
    assert at.sidebar.text_input[0].value == llm.GEMINI_MODEL


def test_gemini_stream_converts_roles_and_collects_usage(monkeypatch):
    """Memakai tiruan paket google.genai karena paket aslinya belum terpasang."""
    captured = {}

    class Obj:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class APIError(Exception):
        code = 429
        message = "kuota habis"

    state = {"fail": False}

    class Models:
        def generate_content_stream(self, model, contents, config):
            captured.update(model=model, contents=contents, config=config)
            if state["fail"]:
                raise APIError()
            yield Obj(text="Halo ", usage_metadata=None)
            yield Obj(text="dunia", usage_metadata=Obj(prompt_token_count=11, candidates_token_count=2))

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.types = types.SimpleNamespace(Content=Obj, Part=Obj, GenerateContentConfig=Obj)
    genai_module.errors = types.SimpleNamespace(APIError=APIError)
    genai_module.Client = lambda api_key: Obj(models=Models())
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setenv("GEMINI_API_KEY", "uji-tiruan")

    stats = {}
    messages = [
        {"role": "system", "content": "aturan"},
        {"role": "user", "content": "halo"},
        {"role": "assistant", "content": "hai"},
        {"role": "user", "content": "lagi"},
    ]
    answer = "".join(llm.stream_chat(messages, llm.LLMConfig(llm.GEMINI, "gemini-x"), 0, stats))

    assert answer == "Halo dunia"
    assert captured["model"] == "gemini-x"
    assert captured["config"].system_instruction == "aturan"
    assert [c.role for c in captured["contents"]] == ["user", "model", "user"]
    assert stats["prompt_tokens"] == 11
    assert stats["answer_tokens"] == 2
    assert stats["generation_ms"] >= stats["ttft_ms"]

    state["fail"] = True
    with pytest.raises(llm.LLMError, match="429"):
        list(llm.stream_chat(messages, llm.LLMConfig(llm.GEMINI, "gemini-x"), 0, {}))


def test_env_file_does_not_override_existing_variable(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# komentar\nUJI_A="nilai a"\nUJI_B=baru\n', encoding="utf-8")
    monkeypatch.delenv("UJI_A", raising=False)
    monkeypatch.setenv("UJI_B", "lama")
    llm.load_env_file(env_file)
    assert llm.os.environ["UJI_A"] == "nilai a"
    assert llm.os.environ["UJI_B"] == "lama"
    monkeypatch.delenv("UJI_A")


def test_list_models_keeps_only_chat_models_without_prefix():
    from list_gemini_models import chat_model_rows

    def model(name, actions):
        return types.SimpleNamespace(
            name=name, display_name="Tampilan", supported_actions=actions, input_token_limit=100, output_token_limit=10
        )

    rows = chat_model_rows(
        [
            model("models/zeta", ["generateContent"]),
            model("models/embed", ["embedContent"]),
            model("models/alfa", ["countTokens", "generateContent"]),
            model("models/kosong", None),
        ]
    )
    assert rows == [("alfa", "Tampilan", 100, 10), ("zeta", "Tampilan", 100, 10)]
