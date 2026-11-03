"""data/terimler_raw.txt (| ayrimli taslak) -> data/terimler.tsv + oz-tutarlilik kontrolu."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from termlib import COLS, contains, load_terms  # noqa: E402

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for line in raw.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    parts = line.split("|")
    assert len(parts) == 9, line
    rows.append([f"T{len(rows)+1:03d}"] + [p.strip() for p in parts])
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(COLS)
    w.writerows(rows)
terms = load_terms(out)
bad = 0
for t in terms:
    if not contains(t["ornek_tr"], t["tr_hedef"], "tr"):
        print(f"UYARI {t['id']} {t['en']}: ornek hedef bicim icermiyor: {t['ornek_tr']}")
        bad += 1
    if t["tr_yasak"] and contains(t["ornek_tr"], t["tr_yasak"], "tr"):
        print(f"UYARI {t['id']} {t['en']}: ornek YASAK bicim iceriyor")
        bad += 1
    if not contains(t["en"], t["en_kabul"], "en"):
        print(f"UYARI {t['id']} {t['en']}: en_kabul kanonik terimi kapsamiyor")
        bad += 1
from collections import Counter  # noqa: E402
print(f"{len(terms)} terim, siniflar={dict(Counter(t['sinif'] for t in terms))}, "
      f"karar_bekleyen={sum(t['karar']=='?' for t in terms)}, uyari={bad}")
