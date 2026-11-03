# B deneyi — ön kayıt (DONDURULDU 2026-10-31, eğitimden önce)

Kaynak: docs/research/2026-11-mlx-lora-protokol.md §1. Bu dosyanın SHA-256'sı MANIFEST.txt'ye yazılır; sonradan değiştirilmez.

| # | Hipotez | Ölçüt | Eşik |
|---|---|---|---|
| H1 | LoRA görülen terimlerde doğruluğu artırır | TAcc(test-görülen), L1 − B0 | ≥ +15 puan VE eşleştirilmiş bootstrap %95 GA alt sınırı > 0 |
| H2 | 4B+LoRA canlı sistem kadar iyi | TAcc(test-görülen), L1 − B2 | ≥ −2 puan (nokta tahmini) |
| H3 | Genel kalite düşmez | FLORES+ devtest, L1 − B0 | ΔCOMET ≥ −0,5 (0–100) VE ΔchrF++ ≥ −1,0 |
| H4 | Aşırı genelleme yok | Tuzak setinde yanlış terim ekleme oranı | L1 ≤ B0 + 3 puan |
| H5 (keşif) | Görülmeyen terimlere genelleşir | TAcc(test-görülmeyen), L1 − B0 | ≥ +5 puan |
| H6 | Hız kazancı | Üretim token/sn, L1 / B2 | ≥ 2× |

Kararlar (sabit):
- L1 = 3 tohumun (0,1,2) ortalaması; her tohum için kontrol noktası = en düşük valid kaybı (son adım değil).
- Bootstrap: 1.000 yeniden örnekleme, eşleştirilmiş, tohum 12345. TAcc için geçiş düzeyinde McNemar ayrıca.
- Çözümleme: açgözlü (temperature 0), max 256 yeni token, aynı normalizasyon, bütün sistemler.
- test-görülen / test-görülmeyen / test-insan ayrı raporlanır; birleştirilmez. Asıl iddia test-insan'a dayanır.
- FLORES+ indirilemezse H3 protokol §2.3 yedeğiyle (200 kendi genel cümle) koşulur ve yazıda "zayıf sinyal" denir.

Ön kayıtla ilgili uygulama notları (Kaya, dondurmadan önce):
1. Canlı sistemdeki "düzeltme tablosu" (~/ceviri/sozluk.tsv) bir SON İŞLEM değil, İSTEME eklenen sözlüktür
   (server.py mt_prompt: "Glossary (always use these ... terms): ..."). B1, B2 ve L2 bu yüzden canlıdaki gibi
   "isteme sözlük ekleme" yöntemiyle koşulur; canlı mt_prompt metni bire bir kopyalanır.
2. TranslateGemma sohbet şablonu asistan içeriğinin önüne "\n" + 8 boşluk koyuyordu (kalite kapısı yakaladı).
   Yalnız asistan dalı yamalandı; çıkarım istemi orijinal şablonla bayt bayt aynı (logs/template_check.txt).
3. EOS: config.json ve generation_config.json'a eos_token_id=[1,106] (<eos>, <end_of_turn>) eklendi; bütün 4B kolları aynı dosyayı kullanır.
