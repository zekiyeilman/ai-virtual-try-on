---
name: vton-beden-testi
description: VTON beden taraması — giydirme/uyum çıktısını XS→XXL boyunca ölçüp montajla doğrular. Kullan: engine.py, mask_generator.py, heatmap.py veya category_router.py'de beden/kalıp/maske/ısı haritası değişikliğinden sonra; kullanıcı "bedenler arası fark yok", ısı haritası veya alt giyim görüntüsü şikayet ettiğinde.
---

# VTON beden taraması

Testlerin yakalayamadığı görsel regresyonları sayı + görüntü ile yakalar. Değişikliği "bitti" saymadan önce bu taramanın **yeşil** olması gerekir.

## Adımlar

1. **Hızlı testler**: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests -q` → hepsi geçer. Isı haritası veya maske koduna dokunulduysa `VTON_SLOW_TESTS=1` ile tekrar çalıştır.
2. **Tarama**: değişikliğin dokunduğu her kategori için çalıştır (Windows'ta `PYTHONIOENCODING=utf-8` önek):
   - Üst giyim: `.venv/Scripts/python.exe scripts/size_sweep.py`
   - Alt giyim: `.venv/Scripts/python.exe scripts/size_sweep.py --preset lower --user u2`
   - Açıya özgü değişiklikte `--angle arka` / `--angle yan`; kalıba özgü değişiklikte `--fit slim` ile de çalıştır. HD yolu için `--ai` (önbellekli HF master yalnız U1 üst giyimde var).
3. **Sayısal kontrol** — konsoldaki `== <etiket>` satırları ve `Isı Haritası Bölgesel Kesitler`:
   - Kıyafet `genişlik@30/60/90%` ve `alan` beden büyüdükçe **monoton artar** (XS < M < XXL).
   - Isı haritası `Vücut=…px` değerleri **her bedende aynı** (±%3) — değişiyorsa vücut sınırı kıyafetten türetiliyor demektir (eski hata).
   - Slim, oversize'dan dar; U2 + M pantolonda bel kırmızı (negatif cm).
4. **Görsel kontrol**: `out/size_sweep/montage_result.jpg` ve `montage_heatmap.jpg` dosyalarını Read ile aç. Hayalet eski kıyafet, açık renk şerit, kolun kıyafetin altında kalması, bel/paçada V veya testere dişi kenar ara.
5. **Rapor**: kullanıcıya beden başına ölçülen genişlikleri (px) ve görsel bulguları ver; önce/sonra karşılaştırması varsa iki sayıyı yan yana yaz.

**Yeşil** = 1–4'ün hepsi sağlandı. Biri sağlanmazsa `systematic-debugging` skill'i ile kök nedeni bul; sabit/tavan ekleyerek sayıyı zorlamak yerine fizik katmanının (`CategoryRouter.extra_metrics['vis_ratio_*']`) görüntü katmanına ulaşıp ulaşmadığını izle — 2026-11-19'daki üç hatanın ortak kökü buydu (bkz. `docs/ANALIZ_VE_YOL_HARITASI.md` §3).

## Referans

- Test kullanıcıları: U1 = 164/92/60/89 cm, U2 = 170/100/84/114 cm (boy/göğüs/bel/basen). `--user u2` için `goruntu/user2_*.png` yerelde olmalı (gizlilik nedeniyle git dışı).
- Bir beden ~14 sn (yerel mod, CPU). Çıktılar `out/` altında, git dışı.
