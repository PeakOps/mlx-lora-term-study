#!/bin/zsh
# macmon her saniye -> logs/macmon_<etiket>.jsonl ; PID dosyasi logs/macmon_<etiket>.pid (durdurma: kill $(cat ...))
cd "$(dirname "$0")/.."
macmon pipe -i 1000 >> logs/macmon_$1.jsonl 2>/dev/null &
echo $! > logs/macmon_$1.pid
