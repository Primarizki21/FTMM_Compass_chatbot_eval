import csv

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from lib import chat_log, llm, rag
from lib.kb import ROOT, Course, load_courses, load_documents, load_regulations


def course(kode: str, text: str, prodi: str = "Teknologi Sains Data") -> Course:
    return Course(kode=kode, nama=kode, prodi=prodi, chunk=text)


COURSES = [
    course("AAA", "statistika dasar peluang"),
    course("BBB", "pemrograman python dasar"),
    course("CCC", "statistika lanjut", prodi="Teknik Elektro"),
]
# Vektor dokumen dibuat agar Dense menyukai BBB, sedangkan BM25 menyukai dokumen statistika.
MATRIX = np.array([[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def make_retriever() -> rag.CourseRetriever:
    return rag.CourseRetriever(
        COURSES,
        embed_query=lambda q: np.array([1.0, 0.0], dtype=np.float32),
        load_matrix=lambda: MATRIX,
    )


def codes(result: rag.RetrievalResult) -> list[str]:
    return [hit.course.kode for hit in result.hits]


def test_bm25_returns_only_matching_documents():
    assert set(codes(make_retriever().retrieve(rag.BM25, "statistika", top_k=3))) == {"AAA", "CCC"}


def test_dense_orders_by_cosine():
    assert codes(make_retriever().retrieve(rag.DENSE, "apa saja", top_k=3)) == ["BBB", "AAA", "CCC"]


def test_prodi_filter_applies_before_top_k():
    result = make_retriever().retrieve(rag.DENSE, "apa saja", top_k=1, prodi="Teknik Elektro")
    assert codes(result) == ["CCC"]


def test_hybrid_rrf_rewards_appearing_in_both_lists():
    result = make_retriever().retrieve(rag.HYBRID, "peluang", top_k=3)
    # AAA: dense #2 + sparse #1, BBB: dense #1 saja, CCC: dense #3 saja.
    assert codes(result) == ["AAA", "BBB", "CCC"]
    assert result.hits[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_reformulation_adds_keywords_to_original_query():
    seen = []

    def reformulate(query: str) -> str:
        seen.append(query)
        return "pemrograman"

    result = make_retriever().retrieve(rag.HYBRID_REFORMULATED, "koding", top_k=1, reformulate=reformulate)
    assert seen == ["koding"]
    assert result.reformulated_query == "pemrograman"
    assert codes(result) == ["BBB"]


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        make_retriever().retrieve("Lainnya", "x")


def test_missing_index_explains_how_to_build(tmp_path):
    with pytest.raises(rag.DenseIndexError, match="build_index"):
        rag.load_dense_index(COURSES, "model-x", tmp_path)


def test_stale_index_is_rejected(tmp_path):
    matrix_path, meta_path = rag.index_paths("model-x", tmp_path)
    np.save(matrix_path, MATRIX)
    meta_path.write_text('{"corpus_sha256": "lama"}', encoding="utf-8")
    with pytest.raises(rag.DenseIndexError, match="berubah"):
        rag.load_dense_index(COURSES, "model-x", tmp_path)


def test_build_messages_limits_history_and_handles_no_context():
    history = [{"role": "user", "content": str(i)} for i in range(6)]
    messages = rag.build_messages("tanya", [], history)
    assert "Tidak ada dokumen relevan" in messages[0]["content"]
    assert [m["content"] for m in messages[1:]] == ["2", "3", "4", "5", "tanya"]


def test_csv_roundtrip_with_newlines_and_single_bom(tmp_path):
    path = tmp_path / "log" / "chat.csv"
    chat_log.append_turn({"pertanyaan": 'apa "itu"?', "jawaban": "baris 1\nbaris 2"}, path)
    chat_log.append_turn({"pertanyaan": "kedua"}, path)

    assert path.read_bytes().count(b"\xef\xbb\xbf") == 1
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["pertanyaan"] == 'apa "itu"?'
    assert rows[0]["jawaban"] == "baris 1\nbaris 2"
    assert rows[1]["pertanyaan"] == "kedua"
    assert list(rows[0]) == chat_log.FIELDS


def test_documents_record_their_source_file():
    documents = load_documents()
    assert all(doc.sumber for doc in documents)
    nano = next(d for d in documents if d.kode == "AGI101" and "Nano" in d.prodi)
    assert nano.sumber == "FIX REKAYASA NANOTEKNOLOGI.json"
    assert {d.sumber for d in load_regulations()} == {
        "regulasi/panduanskripsi.json",
        "regulasi/panduanpkl.json",
        "regulasi/panduanyudisium.json",
        "regulasi/panduanakademik.json",
        "regulasi/google_drive_sop.json",
    }


def test_old_csv_keeps_rows_aligned_after_new_column(tmp_path):
    path = tmp_path / "chat.csv"
    chat_log.append_row(path, ["a", "c"], {"a": "1", "c": "3"})
    chat_log.append_row(path, ["a", "b", "c"], {"a": "4", "b": "5", "c": "6"})

    assert path.read_bytes().count(b"\xef\xbb\xbf") == 1
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"a": "1", "b": "", "c": "3"}, {"a": "4", "b": "5", "c": "6"}]


def test_kb_loads_courses_and_regulations():
    assert len(load_courses()) == 369
    regulations = load_regulations()
    assert len(regulations) == 262
    assert all(doc.prodi == "" for doc in regulations)
    assert len(load_documents()) == 631


def test_regulations_survive_prodi_filter():
    retriever = rag.CourseRetriever(
        [course("MK1", "skripsi", prodi="Teknik Elektro"), Course("REG-1", "Skripsi", "", "skripsi aturan")]
    )
    result = retriever.retrieve(rag.BM25, "skripsi", top_k=5, prodi="Teknologi Sains Data")
    assert codes(result) == ["REG-1"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    log_path = tmp_path / "chat.csv"
    monkeypatch.setattr(chat_log, "LOG_PATH", log_path)
    monkeypatch.setattr(llm, "installed_models", lambda: {llm.LOCAL_MODEL, llm.EMBED_MODEL})
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60), log_path


def read_log(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_app_answers_and_logs_each_turn(app, monkeypatch):
    at, log_path = app
    prompts = []

    def fake_stream(messages, config, keep_alive, stats):
        prompts.append((messages, config, keep_alive))
        stats.update(ttft_ms=5.0, generation_ms=9.0, prompt_tokens=100, answer_tokens=4)
        yield "Jawaban "
        yield "uji"

    monkeypatch.setattr(llm, "stream_chat", fake_stream)
    at.run()
    assert not at.exception

    at.chat_input[0].set_value("Apa prasyarat Pengantar Statistika MAS120?").run()
    assert not at.exception
    at.chat_input[0].set_value("lalu berapa SKS?").run()
    assert not at.exception

    rows = read_log(log_path)
    assert [r["giliran"] for r in rows] == ["1", "2"]
    assert rows[0]["jawaban"] == "Jawaban uji"
    assert rows[0]["metode"] == rag.BM25
    assert rows[0]["sesi_id"] == rows[1]["sesi_id"]
    assert rows[0]["kode_mk_diambil"].split("; ")[0] == "MAS120"
    assert "Kode: MAS120" in rows[0]["konteks"]
    assert rows[0]["file_sumber"].split("; ")[0].endswith(".json")
    assert rows[0]["token_prompt"] == "100"
    assert rows[0]["model_embedding"] == ""
    assert rows[0]["penyedia_llm"] == llm.LOCAL

    second_prompt, config, keep_alive = prompts[1]
    assert keep_alive == "10m"
    assert config == llm.LLMConfig(llm.LOCAL, llm.LOCAL_MODEL)
    assert second_prompt[1]["content"] == "Apa prasyarat Pengantar Statistika MAS120?"
    assert second_prompt[2]["content"] == "Jawaban uji"


def test_app_shows_error_and_logs_nothing_when_model_fails(app, monkeypatch):
    at, log_path = app

    def failing_stream(messages, config, keep_alive, stats):
        raise llm.ollama.ResponseError("model not found", 404)
        yield

    monkeypatch.setattr(llm, "stream_chat", failing_stream)
    at.run()
    at.chat_input[0].set_value("halo").run()

    assert not at.exception
    assert any("model not found" in error.value for error in at.error)
    assert not log_path.exists()


def test_app_explains_missing_dense_index(app, monkeypatch, tmp_path):
    at, log_path = app
    monkeypatch.setattr(rag, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(rag, "load_dense_index", lambda *a, **k: (_ for _ in ()).throw(rag.DenseIndexError("Jalankan build_index.py")))
    monkeypatch.setattr(llm, "embed_query", lambda q: np.zeros(2))
    at.run()
    at.sidebar.selectbox[0].set_value(rag.DENSE).run()
    at.chat_input[0].set_value("halo").run()

    assert not at.exception
    assert any("build_index" in error.value for error in at.error)
    assert not log_path.exists()


def test_each_method_records_component_scores():
    retriever = make_retriever()

    bm25 = retriever.retrieve(rag.BM25, "statistika", top_k=3).hits[0]
    assert set(bm25.komponen) == {"sparse"}
    assert bm25.peringkat["sparse"] == 1
    assert bm25.peringkat["final"] == 1

    dense = retriever.retrieve(rag.DENSE, "apa saja", top_k=3).hits[0]
    assert set(dense.komponen) == {"dense"}
    assert dense.peringkat["dense"] == 1

    hybrid = retriever.retrieve(rag.HYBRID, "peluang", top_k=3).hits
    assert all(set(hit.komponen) == {"dense", "sparse", "rrf"} for hit in hybrid)
    top = hybrid[0]
    assert top.peringkat["final"] == 1
    assert top.komponen["rrf"] == pytest.approx(top.score)
    assert any(hit.komponen["sparse"] == 0.0 and hit.peringkat["sparse"] == 0 for hit in hybrid)


def test_reformulated_hybrid_keeps_component_trace():
    def reformulate(query: str) -> str:
        return "statistika dasar"

    result = make_retriever().retrieve(rag.HYBRID_REFORMULATED, "peluang", top_k=2, reformulate=reformulate)

    assert result.reformulated_query == "statistika dasar"
    assert result.hits
    assert all({"dense", "sparse", "rrf", "final"} == set(hit.komponen) | {"final"} for hit in result.hits)
