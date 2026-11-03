"""B deneyi puanlama (protokol 5.1, 5.4; PREREG). Girdi: results/<set>_<SYS>[_sN].jsonl + data/<set>.jsonl (idx ile).

Olcutler (gecis = kumedeki bir cumlede beklenen bir terim):
 TAcc      : gecislerin kac tanesinde hipotez kabul edilen bir bicimi iceriyor (termlib.contains). Gecis agirlikli.
 yasak     : en-tr gecislerinde hipotezde terimin yasak bicimi var mi (tr_yasak).
 tutarlilik: kumede >= 3 gecisi olan terimlerden, her geciste AYNI kabul bicimiyle (ilk eslesen alternatif)
             cevrilenlerin orani; bir gecis bile kacirildiysa tutarsiz sayilir.
 tuzak     : trap kumesinde, referansta olmayan teknik terim biciminin hipoteze eklenme orani (cumle basina).
 chrF++    : sacrebleu CHRF(word_order=2), yon basina korpus duzeyi.
Istatistik: eslestirilmis bootstrap (1.000, tohum 12345, cumle duzeyinde yeniden ornekleme, %95 GA);
TAcc icin gecis duzeyinde McNemar (kesin binom). L1/L2 = tohumlarin ortalamasi (her yeniden ornekte de).
Kullanim: score.py  -> results/scores.json + konsol tablosu
"""
import json
import sys
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from termlib import accepted_forms, contains, forbidden_forms, load_terms  # noqa: E402

R = Path(__file__).resolve().parents[1]
TERMS = {t["id"]: t for t in load_terms(R / "data/terimler.tsv")}
SPLIT = json.loads((R / "data/split_terms.json").read_text())
SETS = ["test_seen", "test_unseen", "test_insan", "trap", "flores_tr-en", "flores_en-tr"]
SYSTEMS = ["B0", "B1", "B2", "B3", "L1", "L2"]
N_BOOT, BOOT_SEED = 1000, 12345
CHRF_PP = CHRF(word_order=2)


def set_path(name: str) -> Path:
    return R / "data" / ("flores" if name.startswith("flores") else "") / f"{name}.jsonl"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in open(p, encoding="utf-8")]


def outputs(name: str, system: str) -> list[list[dict]]:
    """Sistem ciktilari; L1/L2 icin tohum basina bir liste."""
    if system in ("L1", "L2"):
        files = sorted(R.glob(f"results/{name}_{system}_s*.jsonl"))
    else:
        files = [p for p in [R / f"results/{name}_{system}.jsonl"] if p.exists()]
    return [sorted(load_jsonl(p), key=lambda r: r["idx"]) for p in files]


def term_ids(row: dict, name: str) -> list[str]:
    ids = row.get("terms") or []
    if name == "test_unseen":  # H5: yalniz gorulmeyen terimler
        ids = [i for i in ids if i in SPLIT["unseen"]]
    return ids


def first_match(text: str, forms: list[str], lang: str) -> int:
    for k, fm in enumerate(forms):
        if contains(text, [fm], lang):
            return k
    return -1


def per_sentence(rows: list[dict], outs: list[dict], name: str) -> list[dict]:
    """Her cumle icin gecis sonuclari (TAcc/yasak/tutarlilik), tuzak bayragi."""
    res = []
    for r, o in zip(rows, outs):
        tgt = r["dir"].split("-")[1]
        hyp = o["hyp"] or ""
        occ = []
        for tid in term_ids(r, name):
            t = TERMS[tid]
            acc = accepted_forms(t, tgt)
            forb = forbidden_forms(t, tgt)
            occ.append({"term": tid, "ok": contains(hyp, acc, tgt), "form": first_match(hyp, acc, tgt),
                        "forbidden": bool(forb) and contains(hyp, forb, tgt), "has_forb": bool(forb)})
        trap_hit = None
        if name == "trap" and r.get("trap_term"):
            t = TERMS[r["trap_term"]]
            forms = accepted_forms(t, tgt)
            ref = r.get("ref") or ""
            trap_hit = contains(hyp, forms, tgt) and not contains(ref, forms, tgt)
        res.append({"occ": occ, "trap": trap_hit})
    return res


def agg(ps: list[dict], idx=None) -> dict:
    sel = ps if idx is None else [ps[i] for i in idx]
    occ = [o for s in sel for o in s["occ"]]
    out = {"n_sent": len(sel), "n_occ": len(occ)}
    if occ:
        out["tacc"] = 100 * np.mean([o["ok"] for o in occ])
        fo = [o for o in occ if o["has_forb"]]
        out["yasak"] = 100 * np.mean([o["forbidden"] for o in fo]) if fo else None
        by = {}
        for o in occ:
            by.setdefault(o["term"], []).append(o["form"])
        multi = {k: v for k, v in by.items() if len(v) >= 3}
        out["tutarlilik"] = 100 * np.mean([len(set(v)) == 1 and v[0] >= 0 for v in multi.values()]) if multi else None
        out["n_tutarlilik_terim"] = len(multi)
    traps = [s["trap"] for s in sel if s["trap"] is not None]
    if traps:
        out["tuzak"] = 100 * np.mean(traps)
    return out


def chrf(outs: list[dict], rows: list[dict], idx=None) -> float | None:
    ii = range(len(rows)) if idx is None else idx
    pairs = [(outs[i]["hyp"] or "", rows[i]["ref"]) for i in ii if rows[i].get("ref")]
    if not pairs:
        return None
    return CHRF_PP.corpus_score([h for h, _ in pairs], [[r for _, r in pairs]]).score


def system_metrics(name: str, system: str, rows: list[dict], idx=None) -> dict | None:
    runs = outputs(name, system)
    runs = [o for o in runs if len(o) == len(rows)]
    if not runs:
        return None
    per = [agg(per_sentence(rows, o, name), idx) for o in runs]
    out = {"n_runs": len(runs)}
    for k in per[0]:
        vals = [p[k] for p in per if p.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
        if k in ("tacc", "tuzak", "yasak") and len(vals) > 1:
            out[k + "_std"] = float(np.std(vals, ddof=1))
    c = [chrf(o, rows, idx) for o in runs]
    if c[0] is not None:
        out["chrf"] = float(np.mean(c))
    return out


def compare(name: str, a: str, b: str, rows: list[dict], metric: str) -> dict | None:
    """a - b farki: nokta tahmini + eslestirilmis bootstrap %95 GA (+ TAcc icin McNemar)."""
    ra, rb = outputs(name, a), outputs(name, b)
    ra, rb = [o for o in ra if len(o) == len(rows)], [o for o in rb if len(o) == len(rows)]
    if not ra or not rb:
        return None
    pa = [per_sentence(rows, o, name) for o in ra]
    pb = [per_sentence(rows, o, name) for o in rb]

    def val(idx):
        if metric == "chrf":
            va = np.mean([chrf(o, rows, idx) for o in ra]); vb = np.mean([chrf(o, rows, idx) for o in rb])
        else:
            xa = [agg(p, idx).get(metric) for p in pa]; xb = [agg(p, idx).get(metric) for p in pb]
            if any(x is None for x in xa + xb):
                return None
            va, vb = np.mean(xa), np.mean(xb)
        return float(va - vb)

    point = val(None)
    if point is None:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    n = len(rows)
    boots = [val(list(rng.integers(0, n, n))) for _ in range(N_BOOT)]
    boots = [x for x in boots if x is not None]
    out = {"a": a, "b": b, "metric": metric, "diff": point, "ci95": [float(np.percentile(boots, 2.5)),
                                                                   float(np.percentile(boots, 97.5))],
           "n_boot": len(boots)}
    if metric == "tacc":
        # McNemar: gecis duzeyinde, tohumlar icin cogunluk oyu (L1 tek tohumsa dogrudan)
        oa = np.array([[o["ok"] for s in p for o in s["occ"]] for p in pa]).mean(0) >= 0.5
        ob = np.array([[o["ok"] for s in p for o in s["occ"]] for p in pb]).mean(0) >= 0.5
        b01, b10 = int(np.sum(oa & ~ob)), int(np.sum(~oa & ob))
        out["mcnemar"] = {"a_dogru_b_yanlis": b01, "a_yanlis_b_dogru": b10,
                          "p": float(binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0}
    return out


def comet_segments(name: str, system: str) -> list[np.ndarray]:
    stems = ([p.stem for p in sorted(R.glob(f"results/{name}_{system}_s*.jsonl"))] if system in ("L1", "L2")
             else [f"{name}_{system}"])
    out = []
    for s in stems:
        p = R / "results/comet" / f"{s}.json"
        if p.exists():
            out.append(np.array(json.loads(p.read_text())["segments"]))
    return out


def compare_comet(name: str, a: str, b: str) -> dict | None:
    """H3: COMET (0-100) a - b; tohum ortalamasi, eslestirilmis bootstrap (segment duzeyi)."""
    sa, sb = comet_segments(name, a), comet_segments(name, b)
    if not sa or not sb or len({len(x) for x in sa + sb}) != 1:
        return None
    ma, mb = np.mean(sa, axis=0), np.mean(sb, axis=0)
    d = ma - mb
    rng = np.random.default_rng(BOOT_SEED)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    return {"a": a, "b": b, "metric": "comet", "a_mean": float(ma.mean()), "b_mean": float(mb.mean()),
            "diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_boot": N_BOOT}


def main() -> None:
    res = {"tablolar": {}, "karsilastirma": {}}
    for name in SETS:
        p = set_path(name)
        if not p.exists():
            continue
        rows = load_jsonl(p)
        res["tablolar"][name] = {s: system_metrics(name, s, rows) for s in SYSTEMS}
    # PREREG karsilastirmalari
    plan = [("test_seen", "L1", "B0", "tacc"), ("test_seen", "L1", "B2", "tacc"),
            ("test_unseen", "L1", "B0", "tacc"), ("test_insan", "L1", "B0", "tacc"),
            ("test_insan", "L1", "B2", "tacc"), ("trap", "L1", "B0", "tuzak"),
            ("flores_tr-en", "L1", "B0", "chrf"), ("flores_en-tr", "L1", "B0", "chrf")]
    for name, a, b, m in plan:
        p = set_path(name)
        if p.exists():
            res["karsilastirma"][f"{name}:{a}-{b}:{m}"] = compare(name, a, b, load_jsonl(p), m)
    for name in ("flores_tr-en", "flores_en-tr"):
        res["karsilastirma"][f"{name}:L1-B0:comet"] = compare_comet(name, "L1", "B0")
    (R / "results/scores.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for name, t in res["tablolar"].items():
        print(f"\n== {name}")
        for s, m in t.items():
            if m:
                print(f"  {s}: " + ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in m.items() if v is not None))
    print("\n== karsilastirmalar")
    for k, v in res["karsilastirma"].items():
        if v:
            print(f"  {k}: {v['diff']:+.2f} GA[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
                  + (f" McNemar p={v['mcnemar']['p']:.3g}" if "mcnemar" in v else ""))


if __name__ == "__main__":
    main()
