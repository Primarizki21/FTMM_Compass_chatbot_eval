"""Empat metode retrieval RAG dan penyusunan prompt, mengikuti logika aplikasi."""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib import llm
from lib.bm25 import BM25Okapi
from lib.kb import ROOT, Course

BM25 = "BM25 (Sparse)"
DENSE = "Dense (Embedding)"
HYBRID = "Hybrid RRF (Sparse + Dense)"
HYBRID_REFORMULATED = "Hybrid + Query Reformulation"
METHODS = [BM25, DENSE, HYBRID, HYBRID_REFORMULATED]
METHODS_NEEDING_EMBEDDING = {DENSE, HYBRID, HYBRID_REFORMULATED}

INDEX_DIR = ROOT / "index"
RRF_K = 60
HISTORY_MESSAGES = 4

SYSTEM_PROMPT = """Anda adalah Compass AI, asisten akademik FTMM Universitas Airlangga.
Jawab pertanyaan mahasiswa secara akurat dan informatif berdasarkan informasi konteks mata kuliah dan regulasi akademik yang diberikan di bawah ini.
Jika informasi tidak ada dalam konteks, katakan dengan sopan bahwa Anda tidak memiliki informasi spesifik tersebut.

[KONTEKS AKADEMIK RELEVAN]:
{context}
"""


class DenseIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    course: Course
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[Hit]
    latency_ms: float
    reformulated_query: str = ""


def corpus_hash(courses: list[Course]) -> str:
    return hashlib.sha256("\n\0".join(c.chunk for c in courses).encode("utf-8")).hexdigest()


def index_paths(model: str, index_dir: Path = INDEX_DIR) -> tuple[Path, Path]:
    stem = model.replace(":", "_").replace("/", "_")
    return index_dir / f"{stem}.npy", index_dir / f"{stem}.json"


def load_dense_index(courses: list[Course], model: str, index_dir: Path = INDEX_DIR) -> np.ndarray:
    matrix_path, meta_path = index_paths(model, index_dir)
    build_hint = "Jalankan: python -m uv run python build_index.py"
    if not matrix_path.exists() or not meta_path.exists():
        raise DenseIndexError(f"Index untuk model {model} belum dibuat. {build_hint}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["corpus_sha256"] != corpus_hash(courses):
        raise DenseIndexError(f"Knowledge base berubah setelah index dibuat. {build_hint}")
    return np.load(matrix_path)


class CourseRetriever:
    def __init__(
        self,
        courses: list[Course],
        embed_query: Callable[[str], np.ndarray] = llm.embed_query,
        load_matrix: Callable[[], np.ndarray] | None = None,
    ):
        self.courses = courses
        self.bm25 = BM25Okapi([c.chunk for c in courses])
        self.embed_query = embed_query
        self.load_matrix = load_matrix or (lambda: load_dense_index(courses, llm.EMBED_MODEL))
        self._matrix: np.ndarray | None = None

    def _allowed(self, doc_idx: int, prodi: str | None) -> bool:
        return prodi is None or self.courses[doc_idx].prodi in (prodi, "")

    def _sparse(self, query: str, limit: int, prodi: str | None) -> list[tuple[int, float]]:
        return [hit for hit in self.bm25.search(query) if self._allowed(hit[0], prodi)][:limit]

    def _dense(self, query: str, limit: int, prodi: str | None) -> list[tuple[int, float]]:
        if self._matrix is None:
            self._matrix = self.load_matrix()
        scores = self._matrix @ self.embed_query(query)
        order = np.argsort(-scores, kind="stable")
        ranked = [(int(i), float(scores[i])) for i in order if self._allowed(int(i), prodi)]
        return ranked[:limit]

    def _hybrid(self, query: str, top_k: int, prodi: str | None) -> list[tuple[int, float]]:
        pool = top_k * 3
        fused: dict[int, float] = {}
        for ranked in (self._dense(query, pool, prodi), self._sparse(query, pool, prodi)):
            for rank, (doc_idx, _) in enumerate(ranked, start=1):
                fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def retrieve(
        self,
        method: str,
        query: str,
        top_k: int = 3,
        prodi: str | None = None,
        reformulate: Callable[[str], str] | None = None,
    ) -> RetrievalResult:
        start = time.perf_counter()
        reformulated = ""
        if method == BM25:
            ranked = self._sparse(query, top_k, prodi)
        elif method == DENSE:
            ranked = self._dense(query, top_k, prodi)
        elif method == HYBRID:
            ranked = self._hybrid(query, top_k, prodi)
        elif method == HYBRID_REFORMULATED:
            if reformulate is None:
                raise ValueError("Metode ini membutuhkan fungsi reformulate")
            reformulated = reformulate(query)
            ranked = self._hybrid(f"{query} {reformulated}", top_k, prodi)
        else:
            raise ValueError(f"Metode tidak dikenal: {method}")
        latency = (time.perf_counter() - start) * 1000.0
        hits = [Hit(self.courses[i], score) for i, score in ranked]
        return RetrievalResult(hits, latency, reformulated)


def keep_alive_for(method: str) -> int | str:
    """Melepas LLM lokal dari VRAM kalau giliran berikutnya perlu memuat model embedding."""
    return 0 if method in METHODS_NEEDING_EMBEDDING else "10m"


def build_messages(query: str, hits: list[Hit], history: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(h.course.chunk for h in hits) or "Tidak ada dokumen relevan yang ditemukan."
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        *history[-HISTORY_MESSAGES:],
        {"role": "user", "content": query},
    ]
