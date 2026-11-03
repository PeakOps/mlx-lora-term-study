"""Sistem ciktilari: B0, B1, L1, L2 (MLX 4B) ve B2, B3 (Ollama). Protokol 6.

B1/B2/L2 'duzeltme tablosu' = canlidaki gibi ISTEME sozluk ekleme (canli servisin mt_prompt kopyasi,
sozluk = data/canli_snapshot/sozluk.tsv). B0/L1 = resmi TranslateGemma sablonu (common.inference_prompt).
Hepsi acgozlu (temp 0), max 256 token. Her 20 cumlede guard (isi/ses/canli/swap).
Kullanim: translate_sys.py <SISTEM> <girdi.jsonl> <cikti.jsonl> [--adapter adapters/s0]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from common import MODEL_4B, inference_prompt  # noqa: E402
from guard import a_running, guard  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OLLAMA = "http://127.0.0.1:11434"
LANG = {"tr": "Turkish", "en": "English"}


# ---- canli sistemin birebir kopyasi (server.py 2026-10-29, mt_prompt / glossary_hits / _stem) ----
def _trlower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def load_glossary() -> list[tuple[str, str]]:
    out = []
    for line in (ROOT / "data/canli_snapshot/sozluk.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or "\t" not in line:
            continue
        tr, en = line.split("\t", 1)
        out.append((tr.strip(), en.strip()))
    return out


GLOSSARY = load_glossary()


def _stem(term: str) -> str:
    t = _trlower(term)
    if len(t) >= 5 and t[-1] in "aeıioöuü":
        t = t[:-1]
    return t


def glossary_hits(text: str, src: str) -> list[tuple[str, str]]:
    low = _trlower(text) if src == "tr" else text.lower()
    hits = []
    for tr, en in GLOSSARY:
        key = _stem(tr) if src == "tr" else en.lower()
        if re.search(r"(?<!\w)" + re.escape(key), low):
            hits.append((tr, en))
    return hits


def mt_prompt(text: str, src: str) -> str:
    dst = "en" if src == "tr" else "tr"
    p = (f"You are a professional {LANG[src]} ({src}) to {LANG[dst]} ({dst}) translator. "
         f"Produce only the {LANG[dst]} translation, without any additional explanations. ")
    if src == "en":
        p += "Keep the meaning exactly; never add a negation that is not in the source. "
    hits = glossary_hits(text, src)
    if hits:
        pairs = "; ".join(f"{tr} = {en}" if src == "tr" else f"{en} = {tr}" for tr, en in hits)
        p += f"Glossary (always use these {LANG[dst]} terms): {pairs}.\n\n"
    return p + f"Please translate the following {LANG[src]} text into {LANG[dst]}:\n\n\n{text}"


def clean(out: str) -> str:  # canlidaki son islem
    out = out.strip()
    if len(out) > 1 and out[0] in "\"“'" and out[-1] in "\"”'":
        out = out[1:-1].strip()
    return out
# ---- kopya sonu ----


B3_TERMS = None


def b3_prompt(text: str, src: str) -> str:
    """B3: gemma4:26b-a4b + terim listesi + 'terimleri koru'. Sozluk = terimler.tsv (onayli), tum terimler."""
    global B3_TERMS
    if B3_TERMS is None:
        from termlib import load_terms
        B3_TERMS = load_terms(ROOT / "data/terimler.tsv")
    dst = "en" if src == "tr" else "tr"
    lines = "\n".join(f"- {t['en']} = {t['tr_hedef'][0]}" for t in B3_TERMS)
    return (f"Translate the following {LANG[src]} text into {LANG[dst]}. Output only the translation.\n"
            f"Use these terms exactly (English = Turkish). English terms kept in Turkish take Turkish suffixes "
            f"after an apostrophe (e.g. node'a). Keep product names unchanged.\n{lines}\n\nText:\n{text}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["B0", "B1", "B2", "B3", "L1", "L2"])
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    if a_running():
        sys.exit("A deneyi calisiyor; kosu baslatilmaz.")
    rows = [json.loads(x) for x in open(a.inp, encoding="utf-8")]
    done = set()
    if Path(a.out).exists():
        done = {json.loads(x)["idx"] for x in open(a.out, encoding="utf-8")}
    mlx_sys = a.system in ("B0", "B1", "L1", "L2")
    if mlx_sys:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
        mx.set_cache_limit(4 * 1024**3)
        if a.system in ("L1", "L2") and not a.adapter:
            sys.exit("L1/L2 icin --adapter gerekli")
        model, tok = load(str(MODEL_4B), adapter_path=(str(ROOT / a.adapter) if a.adapter else None))
        sampler = make_sampler(temp=0.0)
    with open(a.out, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if i in done:
                continue
            if i % 20 == 0:
                guard(f"{a.system}:{Path(a.inp).stem}:{i}")
            src, tgt = r["dir"].split("-")
            text = r["src_text"]
            t0 = time.perf_counter()
            if mlx_sys:
                if a.system in ("B0", "L1"):
                    prompt = inference_prompt(tok, text, src, tgt)
                else:
                    prompt = ("<bos><start_of_turn>user\n" + mt_prompt(text, src)
                              + "<end_of_turn>\n<start_of_turn>model\n")
                ids = tok.encode(prompt, add_special_tokens=False)
                mx.reset_peak_memory()
                hyp = generate(model, tok, ids, max_tokens=a.max_tokens, sampler=sampler, verbose=False)
                n_gen = len(tok.encode(hyp, add_special_tokens=False))
                peak = mx.get_peak_memory() / 1e9
                if a.system in ("B1", "L2"):
                    hyp = clean(hyp)
            else:
                if a.system == "B2":
                    body = {"model": "translategemma:12b", "prompt": mt_prompt(text, src), "stream": False,
                            "keep_alive": "30m", "options": {"num_ctx": 8192, "temperature": 0,
                                                             "num_predict": a.max_tokens}}
                    url = f"{OLLAMA}/api/generate"
                else:
                    body = {"model": "gemma4:26b-a4b", "stream": False, "think": False, "keep_alive": -1,
                            "messages": [{"role": "user", "content": b3_prompt(text, src)}],
                            "options": {"num_ctx": 8192, "temperature": 0, "num_predict": a.max_tokens}}
                    url = f"{OLLAMA}/api/chat"
                j = httpx.post(url, json=body, timeout=300).json()
                hyp = clean(j["response"] if a.system == "B2" else j["message"]["content"])
                n_gen = j.get("eval_count", 0)
                peak = None
            dt = time.perf_counter() - t0
            f.write(json.dumps({"idx": i, "system": a.system, "dir": r["dir"], "src": text, "ref": r.get("ref"),
                                "hyp": hyp.strip(), "gen_tokens": n_gen, "wall_s": round(dt, 3),
                                "peak_mem_gb": peak}, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
