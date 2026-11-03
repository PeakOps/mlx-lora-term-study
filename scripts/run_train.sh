#!/bin/zsh
# Ana egitimler s0, s1, s2 sirayla; her birinden once isi kapisi. Hata/DUR olursa zincir durur.
cd "$(dirname "$0")/.."; P=.venv/bin/python
for S in 0 1 2; do
  [ -f adapters/s$S/done.json ] && { echo "[train-run] s$S zaten bitmis"; continue }
  [ -n "$THERMAL_GATE" ] && zsh "$THERMAL_GATE" 80 70
  echo "[train-run] s$S basladi $(date +%H:%M) swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9,]+)M.*/\1/')"
  $P -I scripts/train_b.py --seed $S --out adapters/s$S > logs/train_s$S.log 2>&1 || { echo "[train-run] s$S HATA/DUR"; exit 1 }
  echo "[train-run] s$S bitti $(date +%H:%M)"
done
echo "[train-run] EGITIM TAMAM"
