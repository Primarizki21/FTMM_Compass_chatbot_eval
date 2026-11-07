.DEFAULT_GOAL := help

.PHONY: help sync test index run streamlit bm25 ledger clean-index

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

clean-index: ## Hapus index Dense (dibuat ulang dengan make index)
	rm -f index/*.npy index/*.json
