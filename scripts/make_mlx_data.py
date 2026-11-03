"""train_term/valid_term (terimli) + replay (kaynak + B0 ciktisi = oz-damitma) -> data/mlx/{train,valid}.jsonl.
Once: translate_sys.py B0 data/train_replay_src.jsonl results/replay_train_B0.jsonl (ayni valid icin)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import train_record  # noqa: E402

R = Path(__file__).resolve().parents[1]
rng = random.Random(0)
for split in ("train", "valid"):
    recs = [json.loads(x) for x in open(R / f"data/{split}_term.mlx.jsonl", encoding="utf-8")]
    rep = [json.loads(x) for x in open(R / f"results/replay_{split}_B0.jsonl", encoding="utf-8")]
    n_term = len(recs)
    for r in rep:
        if not r["hyp"] or len(r["hyp"]) > 4 * len(r["src"]) + 20:  # bos/gevezelik -> alma
            continue
        a, b = r["dir"].split("-")
        recs.append(train_record(r["src"], r["hyp"], a, b))
    rng.shuffle(recs)
    (R / "data/mlx").mkdir(exist_ok=True)
    with open(R / f"data/mlx/{split}.jsonl", "w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{split}: {len(recs)} kayit (terimli {n_term}, replay {len(recs)-n_term}, "
          f"replay orani {(len(recs)-n_term)/len(recs):.0%})")
