#!/bin/zsh
# Replay damitma (B0) -> mlx verisi -> baseline'lar B0, B1 (MLX 4B), B2 (Ollama 12B), B3 (Ollama 26B). Sira: model yeniden yuklemesin.
set -e
cd "$(dirname "$0")/.."; P=.venv/bin/python
SETS=(data/test_seen.jsonl data/test_unseen.jsonl data/test_insan.jsonl data/trap.jsonl data/flores/flores_tr-en.jsonl data/flores/flores_en-tr.jsonl)
$P -I scripts/translate_sys.py B0 data/train_replay_src.jsonl results/replay_train_B0.jsonl
$P -I scripts/translate_sys.py B0 data/valid_replay_src.jsonl results/replay_valid_B0.jsonl
$P -I scripts/make_mlx_data.py
for S in B0 B1 B2 B3; do
  for f in $SETS; do
    $P -I scripts/translate_sys.py $S $f results/$(basename $f .jsonl)_$S.jsonl
  done
  echo "[run] $S bitti $(date +%H:%M)"
done
echo "[run] BASELINE TAMAM"
