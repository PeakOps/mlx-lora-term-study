#!/bin/zsh
# L1/L2 degerlendirme (s0-s2, en dusuk val loss checkpoint'i) -> COMET -> puan. (orijinalde paralel bir yan testin bitmesi beklendi)
# (ayni GPU'yu paylasmasin; H6 hiz olcumu temiz kalsin).
cd "$(dirname "$0")/.."; P=.venv/bin/python
echo "[eval] basladi $(date +%H:%M)"
SETS=(data/test_seen.jsonl data/test_unseen.jsonl data/test_insan.jsonl data/trap.jsonl data/flores/flores_tr-en.jsonl data/flores/flores_en-tr.jsonl)
for S in 0 1 2; do
  [ -n "$THERMAL_GATE" ] && zsh "$THERMAL_GATE" 80 70
  for SYS in L1 L2; do
    for f in $SETS; do
      $P -I scripts/translate_sys.py $SYS $f results/$(basename $f .jsonl)_${SYS}_s$S.jsonl --adapter adapters/s${S}_best || { echo "[eval] DUR $SYS s$S $f"; exit 1 }
    done
  done
  echo "[eval] s$S bitti $(date +%H:%M)"
done
$P -I scripts/comet_score.py 'flores_*' || { echo "[eval] COMET DUR"; exit 1 }
$P -I scripts/score.py > logs/score.txt 2>&1
echo "[eval] TAMAM $(date +%H:%M)"
