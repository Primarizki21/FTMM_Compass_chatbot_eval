"""Membuat ulang index Dense dari knowledge_base/, opsional ditambah web_kb_categorized/."""

import argparse
import json
import sys

import numpy as np

from lib import llm
from lib.kb import ROOT, WEB_CAT_DIR, load_documents
from lib.rag import INDEX_DIR, corpus_hash, index_paths
from lib.runlog import catat_event, catat_run

BATCH_SIZE = 32
LOG_DIR = ROOT / "results" / "build_index"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", action="store_true", help="Sertakan web_kb_categorized/chunks.jsonl")
    args = parser.parse_args()

    with catat_run(LOG_DIR, sys.argv, nama_log="build.log"):
        try:
            courses = load_documents(web_dir=WEB_CAT_DIR if args.web else None)
            chunks = [course.chunk for course in courses]
            catat_event(LOG_DIR, tahap="mulai", dokumen=len(chunks), model=llm.EMBED_MODEL, web=args.web)
            print(f"Membuat embedding {len(chunks)} dokumen dengan {llm.EMBED_MODEL} ...")

            batches = []
            for start in range(0, len(chunks), BATCH_SIZE):
                batches.append(llm.embed_texts(chunks[start : start + BATCH_SIZE], keep_alive="30s"))
                print(f"  {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")
            matrix = np.vstack(batches)

            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            matrix_path, meta_path = index_paths(llm.EMBED_MODEL)
            np.save(matrix_path, matrix)
            meta = {"model": llm.EMBED_MODEL, "dimensi": matrix.shape[1], "total": len(chunks), "corpus_sha256": corpus_hash(courses)}
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            catat_event(LOG_DIR, tahap="selesai", baris=matrix.shape[0], dimensi=matrix.shape[1], corpus_sha256=meta["corpus_sha256"])
            print(f"Selesai: {matrix_path} ({matrix.shape[0]} x {matrix.shape[1]})")
        except BaseException as error:
            catat_event(LOG_DIR, tahap="gagal", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
