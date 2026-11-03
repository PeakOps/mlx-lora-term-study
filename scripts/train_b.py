"""B deneyi egitim sarmalayicisi (mlx-lm 0.29.1 train_model'i dogrudan cagirir).

Neden sarmalayici: mlx_lm.lora.run() verilen training_callback'i yok sayiyor; isi molasini optimizer durumunu
ve LR takvimini SIFIRLAMADAN, ayni surec icinde vermek icin callback'te bekliyoruz (parca parca yeniden
baslatma Adam durumunu ve kosinus takvimini bozardi).

- mx.set_cache_limit (19 GB swap dersi), PID dosyasi, her raporda guard (ses/canli/swap/serin_bekle 80 70).
- Not: mlx-lm 0.29.1'de --iters MIKRO-batch sayar. Protokol: 600 optimizer adimi x etkin batch 8.
  proje kurali (kucuk batch) ile: batch 2 x grad_accum 4 = 8 -> iters = 2400 mikro-batch.
  Rapor/eval/kayit mikro-batch biriminde: 40 / 200 / 200 (= 10 / 50 / 50 optimizer adimi).
Kullanim: train_b.py --seed 0 --out adapters/s0 [--iters 2400] [--pilot]
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
from guard import a_running, guard, swap_used_mb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Cb:
    def __init__(self, log: Path, accum: int):
        self.log, self.accum = log, accum

    def _w(self, d: dict) -> None:
        d["ts"] = time.time()
        with open(self.log, "a") as f:
            f.write(json.dumps(d) + "\n")

    def on_train_loss_report(self, info: dict) -> None:
        info = dict(info, kind="train", opt_step=info["iteration"] // self.accum, swap_mb=swap_used_mb())
        self._w(info)
        if swap_used_mb() > 2048:
            raise SystemExit("swap > 2 GB -> DUR")
        m = guard(f"train it={info['iteration']}", hard_exit=True)  # sicaksa burada bekler
        self._w({"kind": "guard", "iteration": info["iteration"], **m})

    def on_val_loss_report(self, info: dict) -> None:
        self._w(dict(info, kind="val", opt_step=info["iteration"] // self.accum))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=2400)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=50, help="optimizer adimi")
    ap.add_argument("--cache-gb", type=float, default=4.0)
    ap.add_argument("--pilot", action="store_true", help="20 optimizer adimi, kayit yok")
    a = ap.parse_args()
    if a_running():
        sys.exit("A deneyi calisiyor; egitim baslatilmaz.")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.pid").write_text(str(os.getpid()))
    mx.set_cache_limit(int(a.cache_gb * 1024**3))
    guard("train-start")

    from mlx_lm.lora import CONFIG_DEFAULTS, load_dataset, train_model
    from mlx_lm.utils import load

    total_opt = (20 if a.pilot else a.iters // a.accum)
    iters = total_opt * a.accum
    cfg = dict(CONFIG_DEFAULTS)
    cfg.update(
        model=str(ROOT / "models/translategemma-4b-it-8bit"), train=True, fine_tune_type="lora",
        data=str(ROOT / "data/mlx"), seed=a.seed, num_layers=a.num_layers, batch_size=a.batch,
        iters=iters, val_batches=-1, learning_rate=a.lr, steps_per_report=10 * a.accum,
        steps_per_eval=(iters if a.pilot else 50 * a.accum), save_every=(10**9 if a.pilot else 50 * a.accum),
        adapter_path=str(out), max_seq_length=512, grad_checkpoint=False, grad_accumulation_steps=a.accum,
        mask_prompt=True, lora_parameters={"rank": a.rank, "dropout": 0.05, "scale": a.scale},
        lr_schedule={"name": "cosine_decay", "arguments": [a.lr, max(1, total_opt - a.warmup)],
                     "warmup": a.warmup if not a.pilot else 5},
        test=False, report_to=None, resume_adapter_file=None,
    )
    args = types.SimpleNamespace(**cfg)
    (out / "run_config.json").write_text(json.dumps(cfg, indent=1, default=str))
    model, tok = load(args.model)
    train_set, valid_set, _ = load_dataset(args, tok)
    t0 = time.time()
    train_model(args, model, train_set, valid_set, Cb(out / "train_log.jsonl", a.accum))
    dt = time.time() - t0
    (out / "done.json").write_text(json.dumps({"wall_s": dt, "peak_mem_gb": mx.get_peak_memory() / 1e9}))
    print(f"[train] bitti {dt/60:.1f} dk, tepe bellek {mx.get_peak_memory()/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
