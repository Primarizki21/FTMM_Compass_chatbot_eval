"""Akses LLM lokal (Ollama) atau Gemini API, dan embedding lokal.

Model lokal dimuat bergantian (keep_alive=0) karena VRAM 4 GB tidak muat untuk dua model sekaligus.
"""

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import numpy as np
import ollama

from lib.kb import ROOT

LOCAL = "Lokal (Ollama)"
GEMINI = "Gemini API"
PROVIDERS = [LOCAL, GEMINI]


def load_env_file(path=ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


load_env_file()

LOCAL_MODEL = os.getenv("LLM_MODEL_NAME", "llama3.2:3b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text-v2-moe")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Sama dengan PureRAGEngine di aplikasi (1/src/engine/pure_rag.py).
CHAT_OPTIONS = {"temperature": 0.2, "top_p": 0.9, "num_predict": 512, "num_ctx": 2048}
# Lebih longgar dari model lokal karena token "thinking" Gemini ikut dihitung dalam batas ini.
GEMINI_MAX_OUTPUT_TOKENS = 2048


class LLMError(RuntimeError):
    pass


LLM_ERRORS = (LLMError, ollama.ResponseError, ConnectionError, httpx.HTTPError)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str


DEFAULT_MODELS = {LOCAL: LOCAL_MODEL, GEMINI: GEMINI_MODEL}

_ollama_client = ollama.Client(host=OLLAMA_HOST)


def installed_models() -> set[str]:
    names = set()
    for item in _ollama_client.list().models:
        names.add(item.model)
        names.add(item.model.removesuffix(":latest"))
    return names


def embed_texts(texts: list[str], keep_alive: int | str = 0) -> np.ndarray:
    response = _ollama_client.embed(model=EMBED_MODEL, input=texts, keep_alive=keep_alive)
    vectors = np.array(response["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def gemini_setup_problem() -> str | None:
    if not os.getenv("GEMINI_API_KEY"):
        return "GEMINI_API_KEY belum diatur. Isi di file .env pada folder RAG."
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return "Paket google-genai belum terpasang. Lihat bagian instalasi di README."
    return None


def _gemini_client():
    problem = gemini_setup_problem()
    if problem:
        raise LLMError(problem)
    from google import genai

    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _gemini_request(messages: list[dict]) -> tuple[list, str]:
    from google.genai import types

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
        for m in messages
        if m["role"] != "system"
    ]
    return contents, system


def _stream_gemini(messages: list[dict], model: str, start: float, stats: dict) -> Iterator[str]:
    client = _gemini_client()
    from google.genai import errors, types

    contents, system = _gemini_request(messages)
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=CHAT_OPTIONS["temperature"],
        top_p=CHAT_OPTIONS["top_p"],
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
    )
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
            usage = chunk.usage_metadata
            if usage:
                stats["prompt_tokens"] = usage.prompt_token_count or 0
                stats["answer_tokens"] = usage.candidates_token_count or 0
            text = chunk.text or ""
            if text:
                stats.setdefault("ttft_ms", (time.perf_counter() - start) * 1000.0)
                yield text
    except errors.APIError as error:
        raise LLMError(f"Gemini API menolak permintaan ({error.code}): {error.message}") from error


def _stream_ollama(messages: list[dict], model: str, keep_alive: int | str, start: float, stats: dict) -> Iterator[str]:
    stream = _ollama_client.chat(model=model, messages=messages, options=CHAT_OPTIONS, keep_alive=keep_alive, stream=True)
    for part in stream:
        if part.done:
            stats["prompt_tokens"] = part.prompt_eval_count or 0
            stats["answer_tokens"] = part.eval_count or 0
        text = part.message.content or ""
        if text:
            stats.setdefault("ttft_ms", (time.perf_counter() - start) * 1000.0)
            yield text


def stream_chat(messages: list[dict], config: LLMConfig, keep_alive: int | str, stats: dict) -> Iterator[str]:
    """Mengalirkan jawaban dan mengisi stats dengan ttft_ms, generation_ms, prompt_tokens, answer_tokens."""
    start = time.perf_counter()
    if config.provider == GEMINI:
        yield from _stream_gemini(messages, config.model, start, stats)
    else:
        yield from _stream_ollama(messages, config.model, keep_alive, start, stats)
    stats["generation_ms"] = (time.perf_counter() - start) * 1000.0


def reformulate_query(query: str, config: LLMConfig) -> str:
    """Prompt dan opsi lokal sama dengan HybridCourseRetriever.reformulate_query di aplikasi."""
    prompt = (
        "Ubah kueri informal atau pertanyaan mahasiswa berikut menjadi kata kunci mata kuliah/akademik "
        "formal dalam Bahasa Indonesia tanpa tanda kutip atau penjelasan tambahan. "
        "Contoh input: 'belajar koding dasar python' -> Output: 'Algoritma dan Pemrograman Python'\n"
        f"Input: {query}\nOutput:"
    )
    if config.provider == GEMINI:
        client = _gemini_client()
        from google.genai import errors, types

        try:
            response = client.models.generate_content(
                model=config.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS),
            )
        except errors.APIError as error:
            raise LLMError(f"Gemini API menolak permintaan ({error.code}): {error.message}") from error
        text = response.text or ""
    else:
        response = _ollama_client.chat(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 30},
            keep_alive=0,
        )
        text = response.message.content
    return text.strip().split("\n")[0].replace('"', "")
