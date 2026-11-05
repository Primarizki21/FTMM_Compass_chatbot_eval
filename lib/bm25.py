"""BM25 Okapi dengan tokenizer dan parameter yang sama seperti 1/src/knowledge/bm25.py."""

import math
import re

import numpy as np


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"\w+", text.lower()) if len(tok) > 1]


class BM25Okapi:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)

        tokenized = [tokenize(doc) for doc in corpus]
        self.doc_lens = np.array([len(tokens) for tokens in tokenized], dtype=np.float32)
        self.avg_doc_len = float(np.mean(self.doc_lens)) if self.corpus_size else 1.0

        self.doc_term_freqs: list[dict[str, int]] = []
        doc_freqs: dict[str, int] = {}
        for tokens in tokenized:
            freqs: dict[str, int] = {}
            for tok in tokens:
                freqs[tok] = freqs.get(tok, 0) + 1
            self.doc_term_freqs.append(freqs)
            for tok in freqs:
                doc_freqs[tok] = doc_freqs.get(tok, 0) + 1

        self.idf = {
            term: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freqs.items()
        }

    def search(self, query: str) -> list[tuple[int, float]]:
        """Mengembalikan semua dokumen berskor > 0, terurut menurun; skor seri mengikuti urutan korpus."""
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for tok in tokenize(query):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for doc_idx, freqs in enumerate(self.doc_term_freqs):
                tf = freqs.get(tok, 0)
                if tf > 0:
                    length_norm = 1.0 - self.b + self.b * (self.doc_lens[doc_idx] / self.avg_doc_len)
                    scores[doc_idx] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)

        order = np.argsort(-scores, kind="stable")
        return [(int(idx), float(scores[idx])) for idx in order if scores[idx] > 0]
