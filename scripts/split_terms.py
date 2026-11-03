"""Terimleri gorulen / gorulmeyen olarak ayirir (%20 gorulmeyen, sinifa gore katmanli, tohum 0).
Ozel adlar (urun/marka; tr_yasak bos ve sinif K) gorulmeyene secilmez: onlari ogrenmeye gerek yok."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from termlib import load_terms  # noqa: E402

terms = load_terms(Path(sys.argv[1]))
rng = random.Random(0)
unseen = []
for cls in ("K", "Ç", "E"):
    pool = [t["id"] for t in terms if t["sinif"] == cls]
    k = round(0.2 * len(pool))
    unseen += rng.sample(sorted(pool), k)
seen = [t["id"] for t in terms if t["id"] not in unseen]
Path(sys.argv[2]).write_text(json.dumps({"seed": 0, "seen": seen, "unseen": sorted(unseen)}, indent=1))
print(f"gorulen={len(seen)} gorulmeyen={len(unseen)}")
