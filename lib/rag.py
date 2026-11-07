"""Empat metode retrieval RAG dan penyusunan prompt, mengikuti logika aplikasi."""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
    komponen: dict[str, float] = field(default_factory=dict)
    peringkat: dict[str, int] = field(default_factory=dict)


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

    def _sparse_scores(self, query: str) -> dict[int, float]:
        """Skor BM25 setiap dokumen berskor > 0, urut seperti hasil pencarian."""
        return dict(self.bm25.search(query))

    def _dense_scores(self, query: str) -> dict[int, float]:
        if self._matrix is None:
            self._matrix = self.load_matrix()
        scores = self._matrix @ self.embed_query(query)
        order = np.argsort(-scores, kind="stable")
        return {int(i): float(scores[i]) for i in order}

    def _allowed_scores(self, scores: dict[int, float], prodi: str | None) -> list[tuple[int, float]]:
        return [(i, score) for i, score in scores.items() if self._allowed(i, prodi)]

    def _component_maps(
        self, scores: dict[int, float], prodi: str | None
    ) -> tuple[dict[int, float], dict[int, int]]:
        ranked = self._allowed_scores(scores, prodi)
        return dict(ranked), {i: rank for rank, (i, _) in enumerate(ranked, start=1)}

    def _rank(
        self, method: str, query: str, top_k: int, prodi: str | None
    ) -> tuple[list[tuple[int, float]], dict[str, dict[int, float]], dict[int, float]]:
        """Peringkat akhir plus skor setiap tahap retrieval yang dipakai metode ini."""
        sparse: dict[int, float] = {}
        dense: dict[int, float] = {}
        if method in (BM25, HYBRID, HYBRID_REFORMULATED):
            sparse = self._sparse_scores(query)
        if method in (DENSE, HYBRID, HYBRID_REFORMULATED):
            dense = self._dense_scores(query)

        fused: dict[int, float] = {}
        if method == BM25:
            ranked = self._allowed_scores(sparse, prodi)[:top_k]
            used = {"sparse": sparse}
        elif method == DENSE:
            ranked = self._allowed_scores(dense, prodi)[:top_k]
            used = {"dense": dense}
        else:
            pool = top_k * 3
            for scores in (dense, sparse):
                for rank, (doc_idx, _) in enumerate(self._allowed_scores(scores, prodi)[:pool], start=1):
                    fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (RRF_K + rank)
            ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
            used = {"dense": dense, "sparse": sparse}
        return ranked, used, fused

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
        if method == HYBRID_REFORMULATED:
            if reformulate is None:
                raise ValueError("Metode ini membutuhkan fungsi reformulate")
            reformulated = reformulate(query)
            search_query = f"{query} {reformulated}"
        elif method in METHODS:
            search_query = query
        else:
            raise ValueError(f"Metode tidak dikenal: {method}")

        ranked, used, fused = self._rank(method, search_query, top_k, prodi)
        latency = (time.perf_counter() - start) * 1000.0

        komponen: dict[str, dict[int, float]] = {}
        peringkat: dict[str, dict[int, int]] = {}
        for name, scores in used.items():
            komponen[name], peringkat[name] = self._component_maps(scores, prodi)
        if fused:
            komponen["rrf"] = fused
            peringkat["rrf"] = {doc_idx: rank for rank, (doc_idx, _) in enumerate(ranked, start=1)}
        peringkat["final"] = {doc_idx: rank for rank, (doc_idx, _) in enumerate(ranked, start=1)}

        hits = [
            Hit(
                self.courses[doc_idx],
                score,
                komponen={name: scores.get(doc_idx, 0.0) for name, scores in komponen.items()},
                peringkat={name: ranks.get(doc_idx, 0) for name, ranks in peringkat.items()},
            )
            for doc_idx, score in ranked
        ]
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
