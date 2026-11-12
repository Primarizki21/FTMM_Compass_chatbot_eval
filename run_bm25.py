"""Evaluasi retrieval BM25. Tidak memakai GPU maupun Ollama."""

import argparse
import sys
from pathlib import Path

from lib.bm25 import BM25Okapi
from lib.evaluate import run
from lib.kb import ROOT, WEB_CAT_DIR, load_courses, load_documents


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "bm25")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah kueri untuk uji cepat")
    parser.add_argument("--web", action="store_true", help="Sertakan web_kb_categorized/chunks.jsonl")
    args = parser.parse_args()

    courses = load_documents(web_dir=WEB_CAT_DIR) if args.web else load_courses()
    index = BM25Okapi([course.chunk for course in courses])
    run("BM25 (k1=1.5, b=0.75)", index.search, courses, args.output, args.limit)


if __name__ == "__main__":
    main()
