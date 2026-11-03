"""Terim sozlugu okuma + eslestirme (TAcc, yasak bicim). Tek kaynak: data/terimler.tsv."""
import csv
import re
from pathlib import Path

COLS = ["id", "en", "sinif", "tr_hedef", "tr_yasak", "en_kabul", "tr_kaynak", "ek_ornek", "ornek_tr", "karar"]


def trlower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def norm_apos(s: str) -> str:
    return s.replace("’", "'").replace("ʼ", "'").replace("`", "'")


def _alts(field: str) -> list[str]:
    return [a.strip() for a in field.split(";") if a.strip()]


def load_terms(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k in ("tr_hedef", "tr_yasak", "en_kabul", "tr_kaynak", "ek_ornek"):
            r[k] = _alts(r[k])
    return rows


# Kisa kokler (son kelime <= 4 harf: is, es, pay, kap, kota...) yalniz cekim ekiyle eslesir; yoksa
# "is" -> "islemci", "es" -> "esik", "pay" -> "paylasim" gibi on ek eslesmesi olur.
TR_SUFFIX = (r"(?:'?(?:l[ae]r)?(?:[ıiuü]m[ıiuü]z|[ıiuü]n[ıiuü]z|[ıiuü]m|[ıiuü]n|s[ıiuü]|[ıiuü]|l[ae]r[ıi])?"
             r"(?:n?[dt][ae]ki|n?[dt][ae]n|n?[dt][ae]|n?y?[ıiuü]n|n?y?[ıiuü]|n?y?[ae]|y?l[ae]|ki)?)"
             r"(?![\wçğıöşü])")
SHORT_ROOT = 4


def is_acronym(form: str) -> bool:
    """Tamami buyuk harf kisaltma (AZ, AI, GPU, EC2): buyuk/kucuk harfe duyarli aranir ("az once", "aile" eslesmesin)."""
    return any(c.isalpha() for c in form) and form == form.upper()


def form_regex(form: str, lang: str, dotless: bool = False, keep_case: bool = False) -> re.Pattern:
    """Kelime basinda baslayan bicim. TR: uzun kokte ek serbest, kisa kokte yalniz cekim eki; EN: cogul/iyelik serbest."""
    if keep_case:
        f = norm_apos(form)
    else:
        f = trlower(norm_apos(form)) if lang == "tr" else norm_apos(form).lower()
    body = r"\s+".join(re.escape(p) for p in f.split())
    if dotless:
        body = body.replace("ı", "[ıi]")
    if lang == "tr":
        tail = TR_SUFFIX if len(f.split()[-1]) <= SHORT_ROOT else ""
        return re.compile(r"(?<![\wçğıöşü])" + body + tail)
    return re.compile(r"(?<![\w])" + body + r"(?:'?s|es|')?(?![\w])")


_SOFT = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}


def expand_tr(form: str) -> list[str]:
    """Unsuz yumusamasi: kuyruk->kuyruğ, öncelik->önceliğ, hesap->hesab, metrik->metriğ."""
    out = [form]
    if form and form[-1] in _SOFT and " " not in form[-3:] and form[-1].islower():
        repl = "g" if form.endswith("nk") else _SOFT[form[-1]]
        out.append(form[:-1] + repl)
    return out


def contains(text: str, forms: list[str], lang: str) -> bool:
    raw = norm_apos(text)
    pats = []  # (desen, metin varyantlari)
    if lang == "tr":
        # Ingilizce kokenli kelimeler (Instance) icin hem Turkce hem duz kucuk harf
        variants = {trlower(raw), raw.lower()}
        for fm in forms:
            if is_acronym(fm):
                pats.append((form_regex(fm, "tr", keep_case=True), {raw}))
                continue
            pats += [(form_regex(x, "tr"), variants) for x in expand_tr(fm)]
            pats.append((form_regex(fm, "tr", dotless=True), variants))
    else:
        for fm in forms:
            pats.append((form_regex(fm, "en", keep_case=True), {raw}) if is_acronym(fm)
                        else (form_regex(fm, "en"), {raw.lower()}))
    return any(p.search(t) for p, vs in pats for t in vs)


def accepted_forms(term: dict, tgt_lang: str) -> list[str]:
    return term["tr_hedef"] if tgt_lang == "tr" else term["en_kabul"]


def forbidden_forms(term: dict, tgt_lang: str) -> list[str]:
    # Yasak listesi Turkce hedef icin tanimli; EN hedefte yasak = kabul edilmeyen TR kaynak bicimin cevirisi yok
    return term["tr_yasak"] if tgt_lang == "tr" else []
