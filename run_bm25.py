"""Evaluasi retrieval: BM25, Dense, Hybrid RRF, atau Hybrid + Query Reformulation."""

import argparse
import sys
from pathlib import Path

from lib import llm, rag
from lib.evaluate import run
from lib.kb import ROOT, WEB_CAT_DIR, load_documents
from lib.runlog import catat_event, catat_run

# ponytail: nama berkas tetap run_bm25.py demi tautan README/Makefile; rename saat target web ikut memakainya.
PROVIDER = [llm.LOCAL, llm.NINE_ROUTER, llm.OPENAI]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "retrieval")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah kueri untuk uji cepat")
    parser.add_argument("--web", action="store_true", help="Sertakan web_kb_categorized/chunks.jsonl")
    parser.add_argument("--metode", choices=rag.METHODS, default=rag.BM25)
    parser.add_argument("--provider", choices=PROVIDER, default=llm.LOCAL, help="Hanya dipakai metode reformulasi")
    parser.add_argument("--model", default="", help="Kosong berarti model default provider")
    parser.add_argument("--top-k", type=int, default=5, help="Kandidat retrieval sebelum penilaian Top-5")
    parser.add_argument(
        "--keep-alive",
        default="0",
        help="Nilai keep_alive Ollama untuk embedding kueri; 0 = ikut perilaku aplikasi",
    )
    args = parser.parse_args()

    with catat_run(args.output, sys.argv):
        try:
            courses = load_documents(web_dir=WEB_CAT_DIR if args.web else None)
            catat_event(args.output, tahap="muat_korpus", dokumen=len(courses), web=args.web)
            model = args.model or llm.DEFAULT_MODELS[args.provider]
            config = llm.LLMConfig(provider=args.provider, model=model)
            retriever = rag.CourseRetriever(
                courses,
                embed_query=lambda teks: llm.embed_texts([teks], keep_alive=args.keep_alive)[0],
            )
            posisi = {course.uid: index for index, course in enumerate(courses)}
            catatan: dict[str, dict] = {}

            def cari(query: str) -> list[tuple[int, float, dict]]:
                hasil = retriever.retrieve(
                    args.metode,
                    query,
                    top_k=args.top_k,
                    reformulate=lambda teks: llm.reformulate_query(teks, config),
                )
                if hasil.reformulated_query:
                    catatan[query] = {"kueri_reformulasi": hasil.reformulated_query}
                return [(posisi[hit.course.uid], hit.score, hit.komponen) for hit in hasil.hits]

            embedding = llm.EMBED_MODEL if args.metode in rag.METHODS_NEEDING_EMBEDDING else ""
            butuh_llm = args.metode == rag.HYBRID_REFORMULATED
            punya_sparse = args.metode in (rag.BM25, rag.HYBRID, rag.HYBRID_REFORMULATED)
            meta = {
                "provider": args.provider if butuh_llm else "-",
                "model": model if butuh_llm else "-",
                "embedding": embedding,
                "embed_keep_alive": args.keep_alive if embedding else None,
                "top_k": args.top_k,
                "bm25_k1": retriever.bm25.k1 if punya_sparse else None,
                "bm25_b": retriever.bm25.b if punya_sparse else None,
                "varian_kueri": "reformulasi" if butuh_llm else "asli",
            }
            catat_event(
                args.output,
                tahap="mulai",
                metode=args.metode,
                limit=args.limit,
                dokumen=len(courses),
                **meta,
            )
            run(args.metode, cari, courses, args.output, args.limit, meta=meta, catatan=catatan)
        except BaseException as error:
            catat_event(args.output, tahap="gagal", error=f"{type(error).__name__}: {error}")
            raise
        catat_event(args.output, tahap="selesai")


if __name__ == "__main__":
    main()
