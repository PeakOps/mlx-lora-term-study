"""Sablon duzeltmesi: asistan icerigindeki '\\n        ' onekini kaldirir ({{ -> {{-).
Yalniz asistan dali degisir; cikarim istemi (kullanici dali + generation prompt) bayt bayt ayni kalir.
Orijinal chat_template.jinja.orig olarak saklanir. Dogrulama template_check.py ile yapilir."""
import shutil
import sys
from pathlib import Path

d = Path(sys.argv[1])
p, orig = d / "chat_template.jinja", d / "chat_template.jinja.orig"
if not orig.exists():
    shutil.copy2(p, orig)
src = orig.read_text()
old = "        {{ message[\"content\"] | trim }}"
assert src.count(old) == 1, "beklenen satir bulunamadi"
p.write_text(src.replace(old, "        {{- message[\"content\"] | trim }}"))
print("sablon yamandi:", p)
