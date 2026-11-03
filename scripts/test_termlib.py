import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from termlib import contains  # noqa: E402

cases = [
    ("Baş düğüme bağlan.", ["baş düğüm"], "tr", True),
    ("Head node'a bağlan.", ["head node"], "tr", True),
    ("HEAD NODE’da sorun var", ["head node"], "tr", True),
    ("Kuyruğa ekledim", ["kuyruk"], "tr", True),
    ("Instance'ları kapat", ["instance"], "tr", True),
    ("İşlemci yüzde 100", ["iş"], "tr", False),  # kisa kok: yalniz cekim eki
    ("İşi bitirdim", ["iş"], "tr", True),
    ("İşlerinizi kuyruğa atın", ["iş"], "tr", True),
    ("İşte durum bu", ["iş"], "tr", True),          # bilinen sinir: "iste" (zarf) cekimden ayirt edilemez
    ("İşçi geldi", ["iş"], "tr", False),
    ("Eşik değerini düşür", ["eş"], "tr", False),
    ("Paylaşımlı disk", ["pay"], "tr", False),
    ("Kotası doldu", ["kota"], "tr", True),
    ("Kotasında yer yok", ["kota"], "tr", True),
    ("Kapıyı kapat", ["kap"], "tr", True),     # bilinen sinir: kap+ı+yı bicimbilimsel olarak gecerli
    ("Akünün voltajı düştü", ["akü"], "tr", True),
    ("Az önce baktım", ["AZ"], "tr", False),
    ("AZ'yi değiştir", ["AZ"], "tr", True),
    ("Aile toplantısı", ["AI"], "tr", False),
    ("GPU'ları boşalt", ["GPU"], "tr", True),
    ("EC2'de açtım", ["EC2"], "tr", True),
    ("Node'una bağlan", ["node"], "tr", True),
    ("Kümeyle ilgili", ["küme"], "tr", True),
    ("Kümelenme analizi", ["küme"], "tr", False),
    ("The air is fresh", ["AI"], "en", False),
    ("Use AI here", ["AI"], "en", True),
    ("bankadaki kuyrukta bekledim", ["head node"], "tr", False),
    ("Check the head nodes.", ["head node"], "en", True),
    ("The cluster's queue", ["queue"], "en", True),
    ("The clusters' queues", ["cluster"], "en", True),
    ("headnode is down", ["head node"], "en", False),
    ("GPUs are idle", ["GPU"], "en", True),
    ("subqueue", ["queue"], "en", False),
]
ok = True
for text, forms, lang, exp in cases:
    got = contains(text, forms, lang)
    ok &= got == exp
    print(("OK  " if got == exp else "FAIL"), lang, text, forms, got)
sys.exit(0 if ok else 1)
