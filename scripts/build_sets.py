"""Ham ciftler -> train/valid/test kumeleri + sizinti taramasi + MANIFEST (protokol 3.2, 3.3).

Adimlar:
 1. Normalize tam-eslesme tekrarlarini sil (kucuk harf, noktalama, Turkce karakter katlama).
 2. Gorulmeyen terimler train/valid'de gecmiyor mu (regex) - geciyorsa o satir train'den atilir.
 3. Test cumlesi ile train arasinda 8-gram ortusmesi > 0 ise test cumlesi atilir.
 4. Karakter 5-gram Jaccard > 0.7 olan test-train ciftlerinde test cumlesi atilir (MinHash yerine tam Jaccard;
    n kucuk oldugu icin kesin hesap). Train ici yakin kopyalar da atilir.
 5. Yon atamasi: her cift tek yone (tohum 0) -> yarisi TR->EN, yarisi EN->TR.
 6. Replay (%25) kaynak cumleleri ayri dosyaya; hedefleri tabanin kendi ciktisi (distill_replay.py, GPU, A'dan sonra).
Kullanim: build_sets.py (dosya yollari sabit, <repo>/data)
"""
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import train_record  # noqa: E402
from termlib import contains, load_terms  # noqa: E402

D = Path(__file__).resolve().parents[1] / "data"
RAW = D / "raw"
TERMS = {t["id"]: t for t in load_terms(D / "terimler.tsv")}
SPLIT = json.loads((D / "split_terms.json").read_text())
SIZES = {"train_term": 1500, "train_replay": 500, "valid_term": 150, "valid_replay": 50,
         "test_seen": 200, "test_unseen": 100, "trap": 30}
FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def norm(s: str) -> str:
    s = s.replace("I", "ı").replace("İ", "i").lower().translate(FOLD)
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def ngrams(s: str, n: int) -> set:
    w = norm(s).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def shingles(s: str, n: int = 5) -> set:
    t = norm(s)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jacc(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def load(name: str) -> list[dict]:
    p = RAW / f"{name}_pairs.jsonl"
    return [json.loads(x) for x in open(p, encoding="utf-8")] if p.exists() else []


def dedup(rows: list[dict], seen_keys: set) -> list[dict]:
    out = []
    for r in rows:
        k = norm(r["tr"]) + "||" + norm(r["en"])
        if k in seen_keys or norm(r["tr"]) in seen_keys:
            continue
        seen_keys.update({k, norm(r["tr"])})
        out.append(r)
    return out


def mentions_unseen(r: dict) -> bool:
    for i in SPLIT["unseen"]:
        t = TERMS[i]
        if contains(r["tr"], t["tr_hedef"] + t["tr_kaynak"] + t["tr_yasak"], "tr") or \
                contains(r["en"], t["en_kabul"] + [t["en"]], "en"):
            return True
    return False


def main() -> None:
    rng = random.Random(0)
    keys: set = set()
    report = {}
    train = dedup(load("train"), keys)
    valid = dedup(load("valid"), keys)
    generic = dedup(load("generic"), keys)
    n0 = len(train) + len(valid)
    train = [r for r in train if not mentions_unseen(r)]
    valid = [r for r in valid if not mentions_unseen(r)]
    report["gorulmeyen_terim_icin_atilan_train_valid"] = n0 - len(train) - len(valid)

    train_side = train + valid + generic
    tr8 = set().union(*[ngrams(r["tr"], 8) | ngrams(r["en"], 8) for r in train_side]) if train_side else set()
    tr_sh = [shingles(r["tr"]) for r in train_side] + [shingles(r["en"]) for r in train_side]

    def clean_test(rows: list[dict], name: str) -> list[dict]:
        out, d8, dj = [], 0, 0
        for r in dedup(rows, keys):
            if (ngrams(r["tr"], 8) | ngrams(r["en"], 8)) & tr8:
                d8 += 1
                continue
            a, b = shingles(r["tr"]), shingles(r["en"])
            if any(jacc(a, s) > 0.7 or jacc(b, s) > 0.7 for s in tr_sh):
                dj += 1
                continue
            out.append(r)
        report[f"{name}_atilan_8gram"] = d8
        report[f"{name}_atilan_jaccard"] = dj
        return out

    tests = {n: clean_test(load(n), n) for n in ("test_seen", "test_unseen", "trap")}
    # test-gorulmeyen: gercekten gorulmeyen terim icermeli
    tests["test_unseen"] = [r for r in tests["test_unseen"] if set(r["terms"]) & set(SPLIT["unseen"])]

    def take(rows: list[dict], n: int, name: str) -> list[dict]:
        rows = rows[:]
        rng.shuffle(rows)
        if len(rows) < n:
            report[f"EKSIK_{name}"] = f"{len(rows)}/{n}"
        return rows[:n]

    sets = {
        "train_term": take(train, SIZES["train_term"], "train_term"),
        "valid_term": take(valid, SIZES["valid_term"], "valid_term"),
        "test_seen": take(tests["test_seen"], SIZES["test_seen"], "test_seen"),
        "test_unseen": take(tests["test_unseen"], SIZES["test_unseen"], "test_unseen"),
        "trap": take(tests["trap"], SIZES["trap"], "trap"),
    }
    g = take(generic, SIZES["train_replay"] + SIZES["valid_replay"], "generic")
    sets["train_replay_src"], sets["valid_replay_src"] = g[:SIZES["train_replay"]], g[SIZES["train_replay"]:]

    # Yon atamasi: listenin yarisi tr->en, yarisi en->tr (karistirilmis sirada)
    for name, rows in sets.items():
        for k, r in enumerate(rows):
            r["dir"] = "tr-en" if k % 2 == 0 else "en-tr"
            r["src_text"], r["ref"] = (r["tr"], r["en"]) if r["dir"] == "tr-en" else (r["en"], r["tr"])
        with open(D / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[f"n_{name}"] = len(rows)

    # Terimli egitim kayitlari (replay hedefleri distill_replay.py ile eklenir)
    for name in ("train_term", "valid_term"):
        with open(D / f"{name}.mlx.jsonl", "w", encoding="utf-8") as f:
            for r in sets[name]:
                a, b = r["dir"].split("-")
                f.write(json.dumps(train_record(r["src_text"], r["ref"], a, b), ensure_ascii=False) + "\n")
    (D / "build_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


def manifest() -> None:
    """Test dosyalarini dondur: SHA-256 -> MANIFEST.txt (egitimden ONCE)."""
    root = Path(__file__).resolve().parents[1]
    files = sorted(D.glob("test_*.jsonl")) + [D / "trap.jsonl", D / "terimler.tsv", D / "split_terms.json",
                                                 root / "PREREG.md"] + sorted((D / "flores").glob("*"))
    files += sorted(D.glob("test_insan*.jsonl"))
    lines = []
    for p in files:
        if p.exists() and p.is_file():
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}")
    (root / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    manifest() if sys.argv[1:] == ["manifest"] else main()
