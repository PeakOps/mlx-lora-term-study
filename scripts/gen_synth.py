"""Sentetik paralel cumle uretimi (yerel gemma4:26b-a4b, Ollama). Protokol 3.1-3.3.

Oturumlar (sizinti onleme: farkli istem, farkli alan, farkli tohum; test, train'den SONRA uretilir):
  train   : gorulen terimler, alanlar A, tohum 1000+i      -> data/raw/train_pairs.jsonl
  valid   : gorulen terimler, alanlar A, tohum 5000+i      -> data/raw/valid_pairs.jsonl
  generic : terimsiz genel cumleler (replay kaynagi)       -> data/raw/generic_src.jsonl
  test_seen / test_unseen : alanlar B, istem B, tohum 9000+i
  trap    : terimin teknik OLMAYAN anlami (tuzak)

keep_alive=-1: canli sistemin gemma4 "Forever" durumunu bozmamak icin (A e3 ile ayni).
Her parti oncesi guard() (ses, canli istemci, swap, serin_bekle 80 70). A calisirken BASLAMAZ.
Kullanim: gen_synth.py <oturum> <hedef_cift_sayisi>
"""
import json
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from guard import a_running, guard  # noqa: E402
from termlib import contains, load_terms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4:26b-a4b"
TERMS = load_terms(ROOT / "data" / "terimler.tsv")
BYID = {t["id"]: t for t in TERMS}
SPLIT = json.loads((ROOT / "data" / "split_terms.json").read_text())

DOMAINS_A = ["toplantıda söylenen bir cümle", "Slack mesajı", "runbook adımı", "kısa bir e-posta cümlesi",
             "müşterinin sorduğu bir soru"]
DOMAINS_B = ["telefon görüşmesinde söylenen cümle", "destek talebi (ticket) açıklaması",
             "haftalık durum raporundan bir cümle", "koridorda kısa konuşma", "yöneticiye verilen kısa özet"]

SCHEMA = {"type": "object", "properties": {"pairs": {"type": "array", "items": {
    "type": "object", "properties": {"tr": {"type": "string"}, "en": {"type": "string"}},
    "required": ["tr", "en"]}}}, "required": ["pairs"]}

PROMPT_A = """Bir HPC/bulut altyapı şirketinde (PeakOps) çalışan ekip için {n} adet KISA, konuşma dilinde
Türkçe–İngilizce paralel cümle çifti yaz. Tür: {domain}.
Her cümle aşağıdaki terim grubundan birini kullansın (grup başına en az bir cümle):
{groups}
KURALLAR:
- Türkçe cümlede terimi TAM OLARAK verilen Türkçe biçimiyle yaz; İngilizce kalanlara Türkçe eki kesme işaretiyle ekle (ör. node'a, GPU'lar).
- İngilizce cümlede terimin İngilizce biçimini kullan.
- 6-20 kelime. Doğal, günlük iş dili. Rakam, isim, zaman çeşitlendir. Aynı kalıbı tekrar etme.
- Çeviri birebir anlam taşısın; ekleme/çıkarma yok.
JSON döndür: {{"pairs":[{{"tr":"...","en":"..."}}]}}"""

PROMPT_B = """You are writing an evaluation set for a Turkish<->English technical interpreter.
Write {n} short sentence pairs. Context: {domain} at a company running GPU clusters and cloud HPC.
Each item must use the term group given below (one item per group, in order):
{groups}
Rules: Turkish side uses exactly the given Turkish form (English loanwords take Turkish suffixes after an
apostrophe); English side uses the English term. Vary sentence structure: questions, negations, conditionals,
past tense. 5-22 words each. Faithful translations.
Return JSON: {{"pairs":[{{"tr":"...","en":"..."}}]}}"""

PROMPT_GENERIC = """{n} adet KISA, günlük ve iş hayatından Türkçe–İngilizce paralel cümle çifti yaz.
Tür: {domain}. HİÇBİR teknik/BT terimi kullanma (bilgisayar, sunucu, yazılım, bulut, veri vb. YOK).
6-18 kelime, çeşitli konular (hava, yemek, yol, aile, planlar, alışveriş, spor).
JSON döndür: {{"pairs":[{{"tr":"...","en":"..."}}]}}"""

PROMPT_TRAP = """Write {n} short Turkish–English sentence pairs where the Turkish word below is used ONLY in its
everyday, NON-technical meaning (nothing about computers, IT, servers, clusters or software):
{groups}
Examples of the idea: "kravat düğümü", "bankadaki kuyruk", "iş görüşmesi", "uçağın pilotu".
5-18 words each. Return JSON: {{"pairs":[{{"tr":"...","en":"..."}}]}}"""


def group_text(group: list[dict]) -> str:
    return "; ".join(f"[TR: {t['tr_hedef'][0]} | EN: {t['en']}]" for t in group)


def all_forms(t: dict) -> tuple[list[str], list[str]]:
    tr = t["tr_hedef"] + t["tr_kaynak"] + t["tr_yasak"]
    en = t["en_kabul"] + [t["en"]]
    return tr, en


def mentions(pair: dict, t: dict) -> bool:
    tr, en = all_forms(t)
    return contains(pair["tr"], tr, "tr") or contains(pair["en"], en, "en")


def validate(pair: dict, group: list[dict], forbid_ids: list[str]) -> tuple[bool, list[str]]:
    """Cift gecerli mi: grubun en az bir terimi dogru bicimde iki tarafta da var, yasak yok, gorulmeyen yok."""
    used = [t["id"] for t in group
            if contains(pair["tr"], t["tr_hedef"], "tr") and contains(pair["en"], t["en_kabul"], "en")]
    if not used:
        return False, []
    if any(t["tr_yasak"] and contains(pair["tr"], t["tr_yasak"], "tr") for t in group):
        return False, []
    if any(mentions(pair, BYID[i]) for i in forbid_ids):
        return False, []
    # cumlede gecen diger (gorulen) terimleri de kaydet: TAcc hepsini sayar
    extra = [t["id"] for t in TERMS if t["id"] not in used and t["id"] not in forbid_ids
             and contains(pair["tr"], t["tr_hedef"], "tr") and contains(pair["en"], t["en_kabul"], "en")]
    return True, used + extra


def call(prompt: str, seed: int, temperature: float) -> list[dict]:
    body = {"model": MODEL, "stream": False, "think": False, "format": SCHEMA, "keep_alive": -1,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "seed": seed, "num_predict": 1500, "num_ctx": 4096}}
    r = httpx.post(OLLAMA, json=body, timeout=300)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"]).get("pairs", [])


def main() -> None:
    session, target = sys.argv[1], int(sys.argv[2])
    if a_running():
        sys.exit("A deneyi calisiyor; B veri uretimi baslatilmaz.")
    out = ROOT / "data" / "raw" / f"{session}_pairs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    have = sum(1 for _ in open(out)) if out.exists() else 0
    seed_base = {"train": 1000, "valid": 5000, "generic": 7000, "test_seen": 9000,
                 "test_unseen": 9500, "trap": 9900}[session]
    seen = [BYID[i] for i in SPLIT["seen"]]
    unseen = [BYID[i] for i in SPLIT["unseen"]]
    rng = random.Random(seed_base)
    i = 0
    with open(out, "a", encoding="utf-8") as f:
        while have < target:
            if i % 5 == 0:
                guard(f"gen:{session}:{have}/{target}")
            seed = seed_base + i
            i += 1
            n = 8
            if session in ("train", "valid"):
                pool, forbid, tmpl, doms, temp = seen, SPLIT["unseen"], PROMPT_A, DOMAINS_A, 0.9
            elif session == "test_seen":
                pool, forbid, tmpl, doms, temp = seen, SPLIT["unseen"], PROMPT_B, DOMAINS_B, 0.7
            elif session == "test_unseen":
                pool, forbid, tmpl, doms, temp = unseen, [], PROMPT_B, DOMAINS_B, 0.7
            else:
                pool, forbid, tmpl, doms, temp = [], [], None, DOMAINS_A + DOMAINS_B, 0.9
            if session == "generic":
                groups, prompt = [], PROMPT_GENERIC.format(n=n, domain=rng.choice(doms))
            elif session == "trap":
                amb = [t for t in TERMS if t["tr_yasak"] or t["sinif"] == "Ç"]
                groups = [[rng.choice(amb)] for _ in range(n)]
                prompt = PROMPT_TRAP.format(n=n, groups="\n".join(t[0]["tr_hedef"][0] for t in groups))
            else:
                groups = [rng.sample(pool, rng.choice([1, 1, 2, 3])) for _ in range(n)]
                prompt = tmpl.format(n=n, domain=rng.choice(doms),
                                     groups="\n".join(f"{k+1}. {group_text(g)}" for k, g in enumerate(groups)))
            t0 = time.time()
            try:
                pairs = call(prompt, seed, temp)
            except Exception as e:  # noqa: BLE001
                print("[gen] hata:", repr(e), flush=True)
                time.sleep(10)
                continue
            kept = 0
            for k, p in enumerate(pairs):
                if not p.get("tr") or not p.get("en"):
                    continue
                rec = {"session": session, "seed": seed, "tr": p["tr"].strip(), "en": p["en"].strip()}
                if session == "generic":
                    if any(mentions(rec, t) for t in TERMS):
                        continue
                    rec["terms"] = []
                elif session == "trap":
                    if k >= len(groups):
                        continue
                    rec["trap_term"] = groups[k][0]["id"]
                    rec["terms"] = []
                else:
                    g = groups[k] if k < len(groups) else []
                    ok, used = validate(rec, g, forbid)
                    if not ok:
                        continue
                    rec["terms"] = used
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            f.flush()
            have += kept
            print(f"[gen] {session} seed={seed} kept={kept}/{len(pairs)} toplam={have} {time.time()-t0:.1f}s",
                  flush=True)


if __name__ == "__main__":
    main()
