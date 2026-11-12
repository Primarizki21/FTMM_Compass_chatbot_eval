.DEFAULT_GOAL := help

.PHONY: help sync test index run streamlit bm25 ledger scrape kategori web-eval web-eval-run web-gold web-eval-gold clean-index

help: ## Tampilkan daftar perintah
	@grep -E '^[a-z0-9][a-z0-9 -]*:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

sync: ## Pasang dependensi (uv mengelola Python dan paket)
	uv sync

test: ## Jalankan seluruh test
	uv run pytest -q

index: ## Buat atau timpa index Dense untuk EMBED_MODEL_NAME aktif
	uv run python build_index.py

run streamlit: ## Jalankan UI Streamlit
	uv run streamlit run app.py

bm25: ## Benchmark retrieval BM25 tanpa LLM
	uv run python run_bm25.py

ledger: ## Susun ledger agregat dari results/
	uv run python scripts/ledger.py

scrape: ## Ambil informasi web FTMM ke web_kb/ (konservatif)
	uv run python scripts/scrape_ftmm.py --delay 4

kategori: ## Rapikan web_kb/pages.jsonl menjadi JSON kategori + chunks.jsonl
	uv run python scripts/categorize_web_kb.py

web-eval: ## Susun dataset evaluasi retrieval web (silver)
	uv run python scripts/build_web_eval.py

web-eval-run: ## Evaluasi retrieval web (silver) dengan BM25
	uv run python scripts/run_web_eval.py

web-gold: ## Susun gold set evaluasi retrieval web
	uv run python scripts/build_web_gold.py

web-eval-gold: web-gold ## Evaluasi gold set web dengan BM25
	uv run python scripts/run_web_eval.py --dataset web_kb_categorized/web_eval_gold.jsonl

clean-index: ## Hapus index Dense (dibuat ulang dengan make index)
	rm -f index/*.npy index/*.json
