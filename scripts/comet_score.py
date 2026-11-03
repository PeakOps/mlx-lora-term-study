"""COMET-22 (wmt22-comet-da, CPU, cevrimdisi) segment puanlari. Protokol 5.1; H3 icin FLORES uzerinde L1-B0.

Her results/<set>_<SYS>[_sN].jsonl icin results/comet/<ayni ad>.json yazar (segment puanlari + sistem ortalamasi,
0-100). Varsa atlar. Referanssiz kumeler (test_insan) atlanir. Her dosyadan once guard (isi/ses/canli/swap).
Kullanim: comet_score.py [glob, varsayilan: 'flores_*']   ornek: comet_score.py 'test_seen_*'
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent / "compat"))  # pkg_resources uyumluluk katmani
sys.path.insert(0, str(Path(__file__).parent))
from guard import guard  # noqa: E402

R = Path(__file__).resolve().parents[1]
CKPT = R / "models/wmt22-comet-da/checkpoints/model.ckpt"


def main() -> None:
    pat = sys.argv[1] if len(sys.argv) > 1 else "flores_*"
    files = [p for p in sorted(R.glob(f"results/{pat}.jsonl")) if not p.name.startswith("test_insan")]
    todo = [p for p in files if not (R / "results/comet" / f"{p.stem}.json").exists()]
    if not todo:
        print("yapilacak yok"); return
    from comet import load_from_checkpoint
    model = load_from_checkpoint(str(CKPT))
    (R / "results/comet").mkdir(parents=True, exist_ok=True)
    for p in todo:
        guard(f"comet:{p.stem}")
        rows = [json.loads(x) for x in open(p, encoding="utf-8")]
        rows = [r for r in rows if r.get("ref")]
        data = [{"src": r["src"], "mt": r["hyp"] or "", "ref": r["ref"]} for r in rows]
        pred = model.predict(data, batch_size=16, gpus=0, num_workers=1, progress_bar=False)
        seg = [round(100 * s, 3) for s in pred.scores]
        out = {"file": p.name, "n": len(seg), "system_score": round(100 * pred.system_score, 3),
               "idx": [r["idx"] for r in rows], "segments": seg}
        (R / "results/comet" / f"{p.stem}.json").write_text(json.dumps(out))
        print(f"[comet] {p.stem} n={len(seg)} COMET={out['system_score']:.2f}", flush=True)


if __name__ == "__main__":
    main()
