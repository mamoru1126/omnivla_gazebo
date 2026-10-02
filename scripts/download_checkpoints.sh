#!/usr/bin/env bash
# OmniVLA の公式チェックポイントを Hugging Face から取得する (コンテナ内で実行)
#   bash scripts/download_checkpoints.sh                       # omnivla-original (約 16GB)
#   bash scripts/download_checkpoints.sh omnivla-finetuned-cast # CAST で追加学習された版
set -euo pipefail
DEST="${CHECKPOINT_ROOT:-/checkpoints}"
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(omnivla-original)
fi
mkdir -p "${DEST}"
for m in "${MODELS[@]}"; do
  echo "==> NHirose/${m} -> ${DEST}/${m}"
  huggingface-cli download "NHirose/${m}" --local-dir "${DEST}/${m}"
done
ls -lh "${DEST}"/*
