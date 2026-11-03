"""EOS duzeltmesi (protokol 2.2): <eos> + <end_of_turn> durdurma tokeni. Orijinaller *.orig olarak saklanir."""
import json
import shutil
import sys
from pathlib import Path

from transformers import AutoTokenizer

d = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(d)
ids = [tok.convert_tokens_to_ids("<eos>"), tok.convert_tokens_to_ids("<end_of_turn>")]
print("eos ids:", ids)
for name in ["config.json", "generation_config.json"]:
    p = d / name
    orig = d / (name + ".orig")
    if not orig.exists():
        shutil.copy2(p, orig)
    cfg = json.loads(orig.read_text())
    cfg["eos_token_id"] = ids
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    print(f"{name}: eos_token_id={cfg['eos_token_id']} (yedek: {orig.name})")
