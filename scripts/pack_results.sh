#!/usr/bin/env bash
# Arsip hasil retrieval untuk Drive tim: satu zip berisi seluruh direktori run.
set -euo pipefail
cd "$(dirname "$0")/.."
stamp=$(date +%F)
mkdir -p results/exports
mapfile -t runs < <(find results/retrieval -maxdepth 1 -mindepth 1 -type d -name '*-uid' | sort)
out="results/exports/FTMM_retrieval_${stamp}.zip"
rm -f "$out"
zip -qr "$out" "${runs[@]}" -x '*.npy' -x '*.pyc'
echo "$out $(du -h "$out" | cut -f1) ${#runs[@]} run"
