# VTON — Mühendislik Analizi, Yapılan Düzeltmeler ve Yol Haritası

> Kayıt tarihi: 2026-11-19. Bu belge, projenin baştan analiz edilmesine gerek kalmasın diye o günkü
> kapsamlı inceleme oturumunun kalıcı kaydıdır. Yeni bir oturumda önce `CLAUDE.md`, sonra bu belge okunmalı.
> Kodla çelişen bir ifade görülürse kod esastır; bu belgeyi güncelleyin.

## 1. Amaç ve ürün tezi

"Bu beden bana nasıl durur, nereden sıkar?" sorusuna görsel + sayısal cevap veren sanal kabin.
Rakip VTON sistemleri kıyafeti bedenden bağımsız vücuda sündürür; projenin farkı **ANSUR II + beden tablosu +
kalıp + kumaş esnekliği** katmanıdır. Ticari değer bu fizik/uyum katmanında; render modeli değiştirilebilir bir parçadır.

Kullanıcı hedefi: TÜBİTAK 2209-A / akademik tez **ve** satılabilir, şirket kurulabilir bir ürün.

## 2. Teknoloji değerlendirmesi

| Katman | Teknoloji | Değerlendirme |
| :--- | :--- | :--- |
| Poz / landmark | MediaPipe Pose | Yeterli, hızlı. Ölçüyü fotoğraftan çıkarmak için SMPL-X/HMR'ye yükseltilebilir. |
| Segmentasyon | SegFormer `mattmdjaga/segformer_b2_clothes` | İyi. Sınıflar: 4 üst, 5 etek, 6 pantolon, 7 elbise, 11 yüz, 14/15 kollar, 12/13 bacaklar. |
| Antropometri | ANSUR II + Ramanujan elips | Projenin en güçlü, savunulabilir kısmı. |
| Yerel render | TPS + prosedürel gölgelendirme | Önizleme kalitesinde; kollar köşeli, kumaş prosedürel. |
| HD render | IDM-VTON (HF Space / Colab) | **Lisans CC BY-NC-SA (ticari değil)**; yalnız üst giyim (VITON-HD) ile eğitilmiş. |
| Arayüz | Gradio | Demo için iyi; ürün için web/mobil ön yüz gerekecek. |

**Mimari zayıflıklar:** `mask_generator.py` ~3.400 satır, çok sayıda elle ayarlanmış sihirli sabit;
fizik katmanı doğru sayıyı hesaplayıp görüntü katmanı bunu kullanmıyordu (aşağıdaki üç hatanın ortak kökü).

## 3. Kullanıcının üç şikayeti — kök neden ve çözüm (2026-11-19)

### 3.1 Bedenler arası görüntü farkı yoktu
- **Kök neden (üst):** `synthesize_vectorized_upper_vton` genişliği fotoğraftaki eski kıyafet × `(1+0.044·adım)`
  ile alıyor, üstelik `min(...photo*1.03)` tavanı ve kol sınırı (arm clamp) büyümeyi kesiyordu → XS→XXL **+%2.6**.
- **Kök neden (alt):** kalça/uyluk çarpanları `clip(...,0.22)` ile doyuyordu → M ve XXL aynı.
- **Kök neden (HD):** eski `_apply_size_fit_warp_to_hd` tüm bandı ±%5 esnetiyordu (kollar + baskı da geriliyordu).
- **Kök neden (çift sayım):** drape düzleştirme faktörü bedene bağlıydı; artık sabit `GARMENT_DRAPE_FLATTEN = 1.04`.
- **Çözüm — fiziksel, doymayan ölçekleme:** görünen silüet = vücut × max(1, kıyafet_2B / vücut_2B)
  (`CategoryRouter.visible_width_ratio`, `extra_metrics['vis_ratio_*']`). Kıyafet vücuttan darsa silüet vücut kalır.
- **Çözüm — HD:** `VTONEngine._retarget_hd_to_size` (SegFormer kıyafet maskesi → satır bazlı genişlik oranı,
  merkezdeki %60 çekirdek birebir korunur → baskı bozulmaz, pantolonda ağın altında bacak başına ölçek,
  etek/paça ucu boy farkıyla kayar, daralmada açılan alan declothed plaka ile doldurulur; plaka 61 px halka
  üzerinden düşük frekanslı renk eşlemesiyle HD'ye uydurulur; kollar Z-index ile üste konur, 9×9 erozyon).
- **Ölçülen sonuç:** HD göğüs genişliği XS/M/XXL = **591 → 641 → 747 px**; alt giyim diz satırı 493 → 522 → 583 px;
  slim pantolon oversize'dan belirgin dar.

### 3.2 Isı haritası yanlıştı
- **Kök neden:** mavi "vücut" çizgisi kıyafet maskesinin içe aşındırılmasıyla çiziliyordu → vücut genişliği
  bedene göre değişiyordu (488 → 337 → 162 px). Yani harita kendi kendisiyle karşılaştırma yapıyordu.
- **Çözüm (`vton/nodes/heatmap.py` baştan yazıldı):**
  - Gerçek vücut: orijinal kullanıcı fotoğrafında SegFormer (sınıflar 4,5,6,7,8,11,12,13,14,15,17), gövde ANSUR yarı
    genişliği × 1.12 ile kırpılır → bedenden bağımsız (göğüs her bedende **572 px**).
  - Bölgesel bolluk (cm) landmark satırları arasında enterpole edilir; sürekli skala `_EASE_STOPS`
    (-8, -3, 0, 3, 7, 12, 22 cm) kırmızı → sarı → yeşil; kumaş–vücut arası hava boşluğu taralı.
  - `ease_label`: PAÇA / ETEK UCU "Tasarım genişliği (serbest)" (renk için 12 cm'de kırpılır); diğerleri
    Dar/Tam/Rahat/Bol/Çok Bol, kalıba göre `loose_lim` (skinny 6, slim 10, regular 16, oversize 32).
  - Türkçe etiketler PIL ile (`_font()` Windows/DejaVu yedekli), özet kartı, renk çubuğu, lejant.
  - Konfor = 100 − ağırlıklı cezalar; beden önerisi `CategoryRouter.recommend_size`
    (darlık cezası 2.0, bolluk 0.6, esneklik darlığı tolere eder: `stretch_tolerance_cm` 10/6/3/0).
  - `last_heatmap_stats`: `comfort_score`, `ideal_size`, `size_costs`, `body_mask_px`, `garment_mask_px`,
    `overhang_px`, `zones{y, body_px, garment_px, ease_px, ease_cm, effective_ease_cm, status}`.
- **Doğrulama:** User 2 + M pantolon → bel −6 cm kırmızı; User 2 + regular üst → göğüs dar. Arka ve yan çalışıyor.

### 3.3 Alt giyim düzgün çalışmıyordu
- Tavanlar kaldırıldı. `CategoryRouter.lower_leg_metrics` uyluk/diz/paça çevrelerini
  `LEG_BODY_RATIOS = {thigh 0.60, knee 0.385, ankle 0.235}` × kalça ve `PANTS_LEG_EASE` (kalıba göre
  uyluk/diz bolluğu + paça çevresi: skinny −1.5/−1.0/25, slim 3/3/33, regular 8/9/42, oversize 18/22/56) ile hesaplar.
- `synthesize_vectorized_lower_vton`: `r_waist/r_hips/r_thigh/r_knee/r_hem` oranları; bel ×(1+0.35·(r−1));
  paça ≥ diz×0.78; kaynakta flare > 1.10 ise korunur; yan profil kalça × derinlik oranları 0.62/0.46/0.33.
- `waistband_reliable` kontrolü: ürün fotoğrafında bel bandı tişört/el ile kapalıysa doku bant genişliği 0 →
  V-biçimli bel artefaktı giderildi.

### 3.4 Yolda bulunan hata
"Regular / Standart Kalıp" Slim sınıflanıyordu: "stan**dar**t" içinde `"dar"` geçiyor. Tüm kontroller `"dar kal"`
olarak düzeltildi (`category_router.py`, `mask_generator.py`, `inference_api.py`). Yeni kalıp metinlerinde dikkat!

## 4. Bilinen, henüz çözülmemiş sınırlamalar
1. **`inference_api.py`** kişi görüntüsü ve maskeyi en-boy oranını korumadan 768×1024'e sıkıştırıyor (~%12 yatay bozulma).
   Uzak servis test edilemediği için dokunulmadı → pad/letterbox + geri kırpma ile düzeltilmeli.
2. **HD alt giyim / elbise**: `yisol/IDM-VTON` yalnız üst giyim. DressCode destekli model gerekir (bkz. §5).
3. **Yerel mod**: kollar köşeli, kumaş prosedürel; ürün içinde "önizleme" olarak konumlanmalı.
4. **`mask_generator.py` ~satır 2112**: omuz eğimi için sütun bazlı Python döngüsü (CLAUDE.md kuralına aykırı) duruyor.
5. `mask_generator.py` modüllere bölünmeli (üst / alt / elbise / maske / TPS).

## 5. Ticari yol haritası (öncelik sırasıyla)
1. **Lisans:** IDM-VTON ve FitDiT ticari değil. Satış için ticari lisanslı model ya da kendi eğitimin şart.
2. **Render arka ucu:** alt giyim/elbise için CatVTON (<8 GB VRAM, hafif) veya Leffa (daha kaliteli) — lisansları kontrol et.
   Render'ı `inference_api` arkasında değiştirilebilir arayüz olarak tut.
3. **Akademik konumlanma:** beden duyarlı VTON çalışmaları — SV-VTON (arXiv 2504.00562), FitVTON (arXiv 2606.12012,
   FLUX.1 Kontext), FIT veri seti (UW/Google, 1.13M örnek). TÜBİTAK raporunda farkını bunlara göre anlat.
4. **Ölçü girişi:** cm istemek dönüşümü düşürür → fotoğraftan SMPL-X/HMR ile ölçü tahmini; MAE/RMSE ile raporla.
5. **Değerlendirme betiği** `vton/evaluation.py`: beden×kalıp kombinasyonlarında SSIM/FID/MAE tablosu.
6. **Hukuk:** vücut fotoğrafları KVKK kapsamında hassas veri (açık rıza, saklama/silme politikası).
   Trendyol kazıma ToS riski → resmi API / satıcı ürün akışı.
7. **Altyapı:** GPU dağıtımı (kalıcı sunucu), önbellek, kuyruk.

Kaynaklar: [FASHN karşılaştırma](https://fashn.ai/blog/comparing-the-top-4-open-source-virtual-try-on-viton-models) ·
[FitDiT](https://boyuanjiang-fitdit.hf.space/) · [CatVTON](https://awesome.ecosyste.ms/projects/github.com%2FZheng-Chong%2FCatVTON) ·
[MC-VTON](https://arxiv.org/pdf/2501.03630) · [Re-CatVTON/Leffa](https://arxiv.org/html/2511.18775v1) ·
[SV-VTON](https://arxiv.org/pdf/2504.00562) · [FIT](https://johannakarras.github.io/FIT/static/pdf/paper.pdf) ·
[FitVTON](https://arxiv.org/pdf/2606.12012)

## 6. Doğrulama araçları
- `pytest tests -q` — hızlı testler (`tests/test_fit_physics.py`: kalıp sınıflama, monoton beden ölçekleme,
  doymama, slim<oversize, beden önerisi U1 üst=M / U2 pantolon=XXL / U2 üst∈{L,XL}, ısı haritası renkleri).
- `VTON_SLOW_TESTS=1 pytest tests -q` — gerçek modellerle: vücut konturu bedenden bağımsız (±%3), maske alanı XS<M<XXL.
- `python scripts/size_sweep.py [--preset lower] [--user u2] [--angle arka|yan] [--fit slim] [--ai]` —
  beden taraması + `out/size_sweep/montage_*.jpg`. Windows'ta `PYTHONIOENCODING=utf-8` gerekebilir.
- Test kullanıcıları: U1 = 164/92/60/89 cm (boy/göğüs/bel/basen), U2 = 170/100/84/114 cm.
- Test görselleri: `goruntu/temp_garment_extracted_{on,arka,yan}.png` (üst), `goruntu/test_pants_garment.png` (alt, `olcu_ref.jpg`'den).

## 7. Önerilen Claude Code skill'leri
`git-guardrails-claude-code` (çok sayıda commit edilmemiş iş), `setup-pre-commit` (pytest + pyflakes),
`tdd` / `verification-before-completion`, `systematic-debugging`, `codebase-design` (mask_generator bölme),
`karpathy-guidelines`, `research` / `deep-research`, `anthropic-skills:docx` (TÜBİTAK belgeleri),
`anthropic-skills:skill-creator` ile `scripts/size_sweep.py` etrafında projeye özel "beden testi" skill'i.
