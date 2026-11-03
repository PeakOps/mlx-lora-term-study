# PREREG eki (2 Kasım 2026)

PREREG.md dondurulmuş olduğu için değiştirilmedi. Bu ek ayrı bir dosyadır.

## Sözlük: "Jarvi onaylı varsayılan"
- Çağlar sözlüğe satır satır bakamadı. 2 Kasım kararıyla (KARARLAR.md) Kaya'nın önerileri, değişiklik yapılmadan kabul edildi.
- 65 "karar gereken" terimin hepsi SOZLUK_ONAY_caglar.md'deki ilk biçimle kullanılıyor.
- data/terimler.tsv: 238 terim (E 66, Ç 104, K 68). split_terms.json: 190 görülen, 48 görülmeyen, tohum 0.
- terimler.tsv sha256: cf956a30ccbae1f0a124c23791409d82ec48354f707b0806607005f52ad6b6c9
- Çağlar kör kontrolde sözlük kararlarını düzeltebilir. Düzeltme olursa yeniden puanlama yapılır ve raporda ayrıca belirtilir; eğitim tekrarlanmaz.

## termlib düzeltmesi (eşleştirme; veri üretiminden ve puanlamadan ÖNCE yapıldı)
- Sorun: Türkçe kalıplarda sağ sınır yoktu. Kısa kökler başka kelimelerin başında da eşleşiyordu (iş→işlemci, eş→eşik, pay→paylaşım). Büyük harfli kısaltmalar küçük harfle aranıyordu (AZ→"az önce", AI→"aile").
- Düzeltme 1: Son kelimesi ≤4 harf olan Türkçe biçimlerde yalnız çekim eklerine izin veriliyor (-ler, iyelik, hâl ekleri, -ki).
- Düzeltme 2: Tamamı büyük harfli kısaltmalar büyük/küçük harfe duyarlı aranıyor.
- Bilinen sınırlar: "işte" (zarf) ile "kapı" (kap+ı) çekimden ayırt edilemiyor.
- Doğrulama: Sözlükteki 238 örnek cümlenin hiçbirinde eşleşme kaybı yok. test_termlib.py 34/34 geçti.
- Eski sürüm: ~/deney-devir/b_yedek/termlib.py.orig

## Pilot notu (2 Kas 2026, 17:28)
- Pilot (20 optimizer adımı) koştu: 3.8 dk, tepe bellek 6.19 GB, swap sabit 1.08 GB, val loss 6.149 → 0.699, ~27 tok/s (eğitim, yalnız hedef tokenları).
- train_b.py'de yalnız pilotu etkileyen bir kusur var: cosine uzunluğu `total_opt - a.warmup` = max(1, 20-50) = 1 adım çıkıyor, bu yüzden pilotta LR 6. adımdan sonra 0. Ana koşularda takvim doğrulandı (0 → 50. adımda 5e-5, sonra cosine ile 600. adımda ≈0). Kod değiştirilmedi; pilot sonucu raporlanan bir çıktı değil.
