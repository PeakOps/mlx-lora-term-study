"""Indirilen HF repolarini HF API'deki LFS sha256 / boyut ile karsilastirir."""
import hashlib
from pathlib import Path

from huggingface_hub import HfApi

api = HfApi()
for repo, d in [("mlx-community/translategemma-4b-it-8bit", "models/translategemma-4b-it-8bit"),
                ("Unbabel/wmt22-comet-da", "models/wmt22-comet-da")]:
    info = api.model_info(repo, files_metadata=True)
    print(f"## {repo} @ {info.sha}")
    ok = True
    for s in info.siblings:
        p = Path(d) / s.rfilename
        if not p.exists():
            print("  EKSIK", s.rfilename)
            ok = False
            continue
        size = p.stat().st_size
        if s.lfs:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            match = h.hexdigest() == s.lfs.sha256 and size == s.lfs.size
            print(f"  {s.rfilename:40s} {size/1e9:7.3f} GB sha256={'OK' if match else 'HATALI'}")
        else:
            match = size == s.size
            print(f"  {s.rfilename:40s} {size:>10d} B  boyut={'OK' if match else 'HATALI'}")
        ok &= match
    print("  SONUC:", "DOGRULANDI" if ok else "SORUN VAR")
