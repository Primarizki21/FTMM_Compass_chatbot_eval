"""UI chatbot RAG FTMM COMPASS dengan dua mode: chat bebas dan evaluasi ground truth."""

import time
from datetime import datetime
from uuid import uuid4

import streamlit as st

from lib import chat_log, gt_eval, llm, rag
from lib.kb import PRODI_BY_ACRONYM, load_documents

MODE_CHAT = "Chat Bebas"
MODE_EVAL = "Evaluasi Ground Truth"
ALL_PRODI = "Semua prodi"
ERRORS = (rag.DenseIndexError, *llm.LLM_ERRORS)
GEMINI_DELAY_SECONDS = 6

st.set_page_config(page_title="Compass AI - RAG", page_icon="🧭", layout="wide")


@st.cache_resource
def get_retriever() -> rag.CourseRetriever:
    return rag.CourseRetriever(load_documents())


def start_new_chat() -> None:
    st.session_state.messages = []
    st.session_state.session_id = uuid4().hex[:8]


def show_sources(hits: list[rag.Hit], reformulated: str) -> None:
    with st.expander(f"Konteks yang diambil ({len(hits)})"):
        if reformulated:
            st.caption(f"Kueri hasil reformulasi: {reformulated}")
        for hit in hits:
            prodi = hit.course.prodi or "Regulasi"
            st.markdown(f"**{hit.course.kode} - {hit.course.nama}** ({prodi}) · skor {hit.score:.4f}")
            st.caption(f"File: {hit.course.sumber}")
            st.text(hit.course.chunk)


def show_stats(message: dict) -> None:
    st.caption(
        f"{message['method']} · retrieval {message['retrieval_ms']:.0f} ms · "
        f"token pertama {message['ttft_ms']:.0f} ms · generasi {message['generation_ms']:.0f} ms · "
        f"token {message['prompt_tokens']} + {message['answer_tokens']}"
    )


def warn_setup_problems(method: str, config: llm.LLMConfig) -> None:
    if config.provider == llm.GEMINI:
        problem = llm.gemini_setup_problem()
        if problem:
            st.warning(problem)
        st.info("Pertanyaan dan konteks knowledge base dikirim ke server Google.")

    required = set()
    if config.provider == llm.LOCAL:
        required.add(config.model)
    if method in rag.METHODS_NEEDING_EMBEDDING:
        required.add(llm.EMBED_MODEL)
    if not required:
        return
    try:
        missing = required - llm.installed_models()
    except llm.LLM_ERRORS:
        st.error("Ollama tidak merespons. Jalankan `ollama serve` lalu muat ulang halaman ini.")
        return
    if missing:
        st.warning(
            f"Model belum terbaca Ollama: {', '.join(sorted(missing))}. Cek dengan `ollama list`. "
            "Kalau modelnya ada di E:\\Ollama_models, tutup Ollama lalu buka lagi."
        )


def chat_page(method: str, top_k: int, config: llm.LLMConfig) -> None:
    if "messages" not in st.session_state:
        start_new_chat()

    with st.sidebar:
        prodi_choice = st.selectbox("Filter prodi", [ALL_PRODI, *PRODI_BY_ACRONYM.values()])
        st.button("Chat baru", on_click=start_new_chat, use_container_width=True)
        st.caption(f"Sesi: {st.session_state.session_id}")
        if chat_log.LOG_PATH.exists():
            st.download_button(
                "Unduh riwayat CSV",
                chat_log.LOG_PATH.read_bytes(),
                file_name="chat_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.caption(f"Tersimpan di: {chat_log.LOG_PATH}")

    st.header("Chat Bebas")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message["hits"], message["reformulated"])
                show_stats(message)

    question = st.chat_input("Tanyakan seputar mata kuliah dan regulasi FTMM...")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    prodi = None if prodi_choice == ALL_PRODI else prodi_choice

    with st.chat_message("assistant"):
        try:
            with st.spinner("Mencari konteks..."):
                result = get_retriever().retrieve(
                    method, question, top_k, prodi, reformulate=lambda q: llm.reformulate_query(q, config)
                )
            stats: dict = {}
            messages = rag.build_messages(question, result.hits, history)
            answer = st.write_stream(llm.stream_chat(messages, config, rag.keep_alive_for(method), stats))
        except ERRORS as error:
            st.error(str(error))
            return

        answer_message = {
            "role": "assistant",
            "content": answer,
            "hits": result.hits,
            "reformulated": result.reformulated_query,
            "method": method,
            "retrieval_ms": result.latency_ms,
            "ttft_ms": stats.get("ttft_ms", 0.0),
            "generation_ms": stats.get("generation_ms", 0.0),
            "prompt_tokens": stats.get("prompt_tokens", 0),
            "answer_tokens": stats.get("answer_tokens", 0),
        }
        show_sources(result.hits, result.reformulated_query)
        show_stats(answer_message)

    st.session_state.messages += [{"role": "user", "content": question}, answer_message]
    chat_log.append_turn(
        {
            "waktu": datetime.now().isoformat(timespec="seconds"),
            "sesi_id": st.session_state.session_id,
            "giliran": sum(m["role"] == "user" for m in st.session_state.messages),
            "metode": method,
            "top_k": top_k,
            "filter_prodi": prodi or "",
            "penyedia_llm": config.provider,
            "model_llm": config.model,
            "model_embedding": llm.EMBED_MODEL if method in rag.METHODS_NEEDING_EMBEDDING else "",
            "pertanyaan": question,
            "kueri_reformulasi": result.reformulated_query,
            "jawaban": answer,
            "kode_mk_diambil": "; ".join(h.course.kode for h in result.hits),
            "file_sumber": "; ".join(h.course.sumber for h in result.hits),
            "skor_retrieval": "; ".join(f"{h.score:.4f}" for h in result.hits),
            "konteks": "\n\n---\n\n".join(h.course.chunk for h in result.hits),
            "latensi_retrieval_ms": round(result.latency_ms, 1),
            "ttft_ms": round(answer_message["ttft_ms"], 1),
            "latensi_generasi_ms": round(answer_message["generation_ms"], 1),
            "token_prompt": answer_message["prompt_tokens"],
            "token_jawaban": answer_message["answer_tokens"],
        }
    )


def download_if_exists(label: str, path) -> None:
    if path.exists():
        st.download_button(label, path.read_bytes(), file_name=path.name, mime="text/csv")


def scoring_section(name: str) -> None:
    st.subheader("Skor")
    answers = gt_eval.read_rows(gt_eval.answers_path(name))
    if st.button("Hitung skor", disabled=not answers or st.session_state.eval_running):
        try:
            with st.spinner(f"Menghitung BERTScore dan ROUGE untuk {len(answers)} jawaban di CPU..."):
                scored = gt_eval.score_answers(answers)
                gt_eval.write_scores(name, scored, gt_eval.summarize(scored))
        except ImportError as error:
            st.error(f"Paket metrik belum terpasang: {error.name}. Lihat bagian instalasi di README.")
    summary = gt_eval.read_rows(gt_eval.summary_path(name))
    if summary:
        st.dataframe(summary, use_container_width=True)
        st.caption("Hit@1, Hit@k, dan MRR hanya dihitung untuk soal silabus (n_retrieval).")
    download_if_exists("Unduh jawaban per soal", gt_eval.answers_path(name))
    download_if_exists("Unduh skor per soal", gt_eval.scores_path(name))
    download_if_exists("Unduh ringkasan", gt_eval.summary_path(name))


def evaluation_page(method: str, top_k: int, config: llm.LLMConfig) -> None:
    ground_truth = gt_eval.load_ground_truth()
    name = gt_eval.run_name(config, method, top_k)
    running = st.session_state.eval_running

    st.header("Evaluasi Ground Truth")
    st.caption(
        f"{len(ground_truth)} soal dari GT_MASTER.csv. Tiap soal dijawab tanpa riwayat chat, "
        "dan hasilnya disimpan per kombinasi LLM, metode, dan top-k."
    )
    count = st.number_input(
        "Jumlah soal (kurangi untuk uji coba)", 1, len(ground_truth), len(ground_truth), disabled=running
    )
    default_delay = GEMINI_DELAY_SECONDS if config.provider == llm.GEMINI else 0
    delay = st.number_input("Jeda antar soal (detik)", 0, 120, default_delay, disabled=running)

    selected = gt_eval.sample_evenly(ground_truth, int(count))
    done_ids = {row["id"] for row in gt_eval.read_rows(gt_eval.answers_path(name))}
    remaining = [item for item in selected if item["id"] not in done_ids]
    finished = len(selected) - len(remaining)
    st.progress(finished / len(selected), text=f"{finished}/{len(selected)} soal selesai")
    st.caption(f"File hasil: {gt_eval.answers_path(name)}")

    if st.session_state.eval_error:
        st.error(st.session_state.eval_error)

    start_col, stop_col = st.columns(2)
    if start_col.button("Mulai / Lanjutkan", disabled=running or not remaining, use_container_width=True):
        st.session_state.eval_running = True
        st.session_state.eval_error = ""
        st.rerun()
    if stop_col.button("Berhenti", disabled=not running, use_container_width=True):
        st.session_state.eval_running = False
        st.rerun()

    if running and remaining:
        item = remaining[0]
        st.info(f"Sedang menjawab soal {item['id']}: {item['question']}")
        try:
            row = gt_eval.answer_question(get_retriever(), item, method, top_k, config)
        except ERRORS as error:
            st.session_state.eval_running = False
            st.session_state.eval_error = f"Evaluasi berhenti di soal {item['id']}: {error}. Jawaban sebelumnya tetap tersimpan."
            st.rerun()
        else:
            gt_eval.save_answer(name, row)
            if delay and len(remaining) > 1:
                time.sleep(delay)
            st.rerun()
    elif running:
        st.session_state.eval_running = False
        st.rerun()

    scoring_section(name)


st.session_state.setdefault("eval_running", False)
st.session_state.setdefault("eval_error", "")
is_running = st.session_state.eval_running

with st.sidebar:
    st.title("Compass AI")
    mode = st.radio("Mode", [MODE_CHAT, MODE_EVAL], disabled=is_running)
    method_choice = st.selectbox("Metode retrieval", rag.METHODS, disabled=is_running)
    provider = st.selectbox("Sumber LLM", llm.PROVIDERS, disabled=is_running)
    model = st.text_input("Model LLM", llm.DEFAULT_MODELS[provider], key=f"model_{provider}", disabled=is_running)
    top_k_choice = st.number_input("Jumlah dokumen (top-k)", 1, 10, 3, disabled=is_running)
    st.caption(f"Embedding: {llm.EMBED_MODEL}")
    llm_config = llm.LLMConfig(provider, model.strip())
    warn_setup_problems(method_choice, llm_config)
    st.divider()

if mode == MODE_CHAT:
    chat_page(method_choice, int(top_k_choice), llm_config)
else:
    evaluation_page(method_choice, int(top_k_choice), llm_config)
