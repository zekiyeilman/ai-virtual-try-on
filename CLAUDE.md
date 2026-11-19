# CLAUDE.md — VTON 360° Beden, Kalıp ve Kumaş Duyarlı Sanal Giydirme Ekosistemi Rehberi

Bu belge, projeyi devralan yapay zeka asistanının (Claude) gereksiz dosya okumalarıyla token harcamasını önlemek, mimariyi tek seferde kavramasını sağlamak ve mevcut çalışan yapıyı bozmadan geliştirmeye devam etmesi için hazırlanmıştır.

---

## 1. Biz Ne Yapıyoruz? (Proje Özeti)

Standart sanal giydirme (Virtual Try-On / VTON) sistemleri kıyafeti kullanıcının beden ölçülerinden bağımsız olarak vücuda "tam oturacak şekilde" sündürür. **Bu projenin temel farkı; antropometrik ölçülere, seçilen bedene (XS–XXL), kıyafet kalıbına (Oversize, Regular, Slim Fit, Skinny/Bodycon) ve kumaş esnekliğine (%0 Dokuma – %10+ Likra) duyarlı, 360° (Ön, Arka, Yan) çalışan hibrit bir Sanal Kabin ve Biyomekanik Konfor/Isı Haritası sistemi olmasıdır.**

Sistem 6 ana katmandan oluşur:
1. **Otonom E-Ticaret Kazıma (`vton/scraper/`)**: Trendyol ürün linkinden ürün görsellerini (Ön, Arka, Yan, Düz Zemin/Flat-lay), açıklamadaki manken ölçülerini (Ground Truth) ve beden tablosunu çeker.
2. **Biyometrik Kamera Kalibrasyonu (`vton/calibration.py` & `vton/nodes/body_mesh.py`)**: MediaPipe Pose ve 3B Yaw/Pitch açı düzeltmeli kafatası biyometrisi (göz bebekleri arası mesafe IPD ve yüz boyu) ile fotoğraftaki `cm / piksel` ölçeğini çıkarır.
3. **ANSUR II 3B → 2B Antropometrik Dönüşüm (`vton/models/ansur.py`)**: Kullanıcının 3B çevre ölçülerini (Göğüs, Bel, Basen - cm) ABD Ordusu ANSUR II derinlik/genişlik elips oranları ve Ramanujan elips çevre formülüyle 2B ön genişlik ($W_{2D}$) ve yan derinliğe ($D_{2D}$) dönüştürür.
4. **Kategori Yönlendirici ve Anatomik TPS + Agnostik Maskeleme (`vton/category_router.py` & `vton/mask_generator.py`)**:
   - Üst giyim, alt giyim (pantolon/etek) ve tam boy (elbise) kategorilerini yönlendirir.
   - Thin-Plate Spline (TPS) ve silindirik derinlik gölgelendirmesiyle kıyafeti beden/kalıp ölçeğine uygun olarak vücuda giydirir (`build_diffusion_inputs`).
   - Saç, el, ön kol, yüz/boyun ve telefon gibi ön plan nesnelerini (Z-index occlusion) korur.
5. **Hibrit Render Motoru (`vton/engine.py` & `vton/inference_api.py`)**:
   - **Hızlı Yerel Mod**: Harici GPU gerektirmeden anatomik TPS, hayalet eski kıyafet temizleme (ghost inpainting) ve 3B gölgelendirme ile anında fotogerçekçi sonuç üretir.
   - **HD AI Difüzyon Modu**: Hugging Face (`yisol/IDM-VTON`) veya Google Colab T4/L4 GPU sunucusuna bağlanarak difüzyon tabanlı render alır. Difüzyonun bozduğu tişört baskılarını/yazılarını `restore_garment_print` ile 1:1 vektörel netlikte geri yükler ve yalnız maske bölgesini orijinal yüksek çözünürlüklü fotoğrafa harmanlar (`_composite_diffusion_into_original`).
6. **Biyomekanik Uyum & Isı Haritası (`vton/nodes/heatmap.py`)**: Giydirilmiş kıyafet silüeti ile kullanıcının gerçek anatomik vücut sınırları arasındaki mesafeyi ve çevre bolluk/darlık farkını ($\Delta W$ + kumaş esneklik toleransı) analiz eder. Dar bölgeleri Kırmızı, tam kalıp bölgeleri Sarı, dökümlü/oversize bölgeleri Yeşil olarak çizer; kullanıcının iç vücut hatlarını kesintisiz mavi anatomik eğrilerle gösterir.

---

## 2. Bu Zamana Kadar Neleri Başardık? (Tamamlanan & Doğrulananlar)

- **%100 Vektörel Görüntü İşleme**: `mask_generator.py`, `inference_api.py` ve `heatmap.py` içerisindeki yavaş Python `for y in range(h)` piksel döngüleri tamamen kaldırıldı; tüm işlemler NumPy/OpenCV matris operasyonlarıyla vektörel hale getirildi.
- **360° (Ön, Arka, Yan) Doğru Yön ve Maske Yönetimi**:
  - Arka duruşta ön yaka/göğüs oyuntularının sırtta delik açması sorunu giderildi (`orientation == "back"`).
  - Yan duruşta (`orientation == "side"`) silindirik perspektif daralması ve Z-ekseni derinlik ölçeklemesi oturtuldu.
- **Doğal Vücut Hatları ve Temiz Yan Profil Isı Haritası (`vton/nodes/heatmap.py`)**:
  - Mavi vücut sınırı çizgileri yapay kutu/poligon yerine SegFormer'dan çıkarılan gerçek anatomik sınırlarla (`real_body_full`, `isClosed=False` açık eğriler) boyun, omuz, kol, bel kavisini doğal takip edecek şekilde düzeltildi.
  - Yan profilde mankenin sırt/kol arkasında (sağ tarafta) oluşan gereksiz yeşil taşmalar temizlendi; bolluk analizi yalnızca ön Z-ekseni göğüs/karın döküm bölgesine odaklandı.
- **Beden (XS–XXL) ve Kalıp (Oversize → Skinny) Dinamik Ölçekleme**:
  - Kullanıcı arayüzden beden veya kalıp değiştirdiğinde, hem sıfırdan üretimde hem de HD Render üzerinde (`VTONEngine._retarget_hd_to_size`: SegFormer kıyafet maskesi + satır bazlı görünür genişlik oranı, baskıyı koruyan %60 çekirdek, pantolonda bacak başına ölçekleme, temiz plaka + renk eşlemeli dolgu) gerçek zamanlı genişlik/boy farkı uygulanıyor.
- **Fiziksel Beden Ölçekleme (doymayan)**: Görünen silüet = vücut × max(1, kıyafet_2B / vücut_2B) (`CategoryRouter.visible_width_ratio`, `extra_metrics['vis_ratio_*']`). Pantolonda uyluk/diz/paça çevreleri `lower_leg_metrics` ile hesaplanır. Eski `clip(...,0.22)` tavanları kaldırıldı.
- **Testler**: `pytest tests -q` (hızlı) ve `VTON_SLOW_TESTS=1 pytest tests -q` (gerçek modellerle entegrasyon).
- **Gradio Web Arayüzü (`app.py`)**:
  - Manken/Kullanıcı çekmecesi (User 1, User 2, Erkek Manken ve Yeni Profil Kaydetme), otomatik URL'den kategori/kalıp/esneklik tespiti, özel kıyafet yükleme desteği ve senkronize varsayılan sonuç galerisi çalışır durumda.

---

## 3. Neyi Hedefliyoruz & Neler İyileştirilebilir? (Geliştirme Alanları)

1. **Alt Giyim (Pantolon / Etek) ve Elbise Kategorilerinde Çok Açılı (Arka/Yan) İnce Ayar**:
   - Üst giyim (tişört vb.) Ön, Arka ve Yan açılarda çok yüksek kalitede çalışıyor.
   - Alt giyimde (`lower_body`) iki bacaklı (bipedal) TPS bükme mevcut ancak yan ve arka duruşlarda paça genişliği, ağ (crotch) hizalaması ve bel kemeri geçişleri farklı pantolon kalıplarında (Mom Jean, Wide Leg, Skinny) daha da kusursuzlaştırılabilir.
2. **Erkek Manken Görsel Setinin Tamamlanması**:
   - `app.py` içinde Erkek Manken profili tanımlı ancak `goruntu/` klasöründe `user_erkek_on.jpg`, `user_erkek_arka.jpg`, `user_erkek_yan.jpg` dosyaları henüz eklenmedi (kullanıcı arayüzden yükleyebiliyor, fakat varsayılan yüksek çözünürlüklü erkek stüdyo fotoğrafları eklenebilir).
3. **Kumaş Fizik Motoru (Drape & Kıvrım Sentezi)**:
   - Yerel TPS modunda (`use_ai_render=False`), bedene göre genişleyen kıyafette (ör. XL veya Oversize seçildiğinde) kumaşın kendi üzerine düşen dikey mikro-kıvrım gölgeleri (procedural fabric folds / normal map) geliştirilerek difüzyonsuz modun gerçekçiliği artırılabilir.
4. **Scraper Dayanıklılığı ve Çoklu Platform Desteği**:
   - Şu an Trendyol kazıyıcı (`vton/scraper/trendyol.py`) aktif. Zara, H&M, Mavi gibi diğer e-ticaret siteleri için esnek adaptörler veya beden tablosu parser'ları genişletilebilir.
5. **Birim Testleri ve Nicel Metrik Raporlayıcı (SSIM / FID / MAE)**:
   - Akademik tez ve TÜBİTAK 2209-A raporu için farklı beden/kalıp kombinasyonlarında otomatik SSIM, FID ve metrik kalibrasyon sapması (MAE/RMSE) tablosu üreten bir değerlendirme betiği (`vton/evaluation.py`) eklenebilir.

---

## 3b. Güncel Durum ve Analiz Kaydı (son güncelleme: 2026-11-19)

> Projeyi baştan analiz ETMEYİN. Tam mühendislik analizi, üç ana şikayetin (beden farkı yok / ısı haritası yanlış / alt giyim bozuk) kök nedenleri, ölçülmüş sonuçlar, lisans uyarıları ve ticari yol haritası **[`docs/ANALIZ_VE_YOL_HARITASI.md`](docs/ANALIZ_VE_YOL_HARITASI.md)** içindedir. Önce onu okuyun.

- **Çözüldü (2026-11-19):** fiziksel doymayan beden ölçekleme (yerel + HD `_retarget_hd_to_size`), gerçek vücut tabanlı ısı haritası, alt giyim bacak metrikleri, "standart"→slim sınıflama hatası (`"dar kal"` kontrolü).
- **Açık işler (öncelik sırası):** (1) `inference_api.py` 768×1024 en-boy bozulması; (2) alt giyim/elbise için ticari lisanslı HD model (IDM-VTON CC BY-NC-SA + yalnız üst giyim); (3) `mask_generator.py` ~2112 sütun döngüsü + dosyayı bölme; (4) `vton/evaluation.py`.
- **Hızlı doğrulama:** `pytest tests -q` ve `python scripts/size_sweep.py --preset upper|lower --user u1|u2` (çıktı `out/size_sweep/`, git dışı).
- **Dikkat:** Kalıp metni kontrollerinde asla yalın `"dar"` aramayın ("stan**dar**t" eşleşir).
- Değişiklik öncesi dosya yedekleri: `out/backup_pre_2026-11-19/` (git dışı).

---

## 4. Dosya Haritası: Hangi Dosyalar Sürekli İşlenmeli, Hangilerine Bir Kez Bakmak Yeterli?

> **ÖNEMLİ TOKEN KURALI**: Tüm projeyi her seferinde baştan sona okumayın. Aşağıdaki sınıflandırmaya göre yalnızca ilgili modülü okuyun.

### 🔥 A. Sürekli İşlenen / Aktif Geliştirme Dosyaları (Core Active Files)
Görüntü kalitesi, giydirme fiziği, maskeleme, ısı haritası veya arayüz değişikliklerinde **doğrudan bu dosyalar üzerinde çalışılır**:

| Dosya | Rolü ve Ne Zaman Düzenleneceği |
| :--- | :--- |
| [`vton/engine.py`](vton/engine.py) | **Ana Orkestratör (`VTONEngine`)**. Beden/kalıp/esneklik hesapları, `resize_garment_for_user`, HD render önbellek/bükme (`_apply_size_fit_warp_to_hd`) ve boru hattı akışı buradadır. |
| [`vton/mask_generator.py`](vton/mask_generator.py) | **TPS Bükme ve Anatomik Maske Motoru (`AnatomicalMaskGenerator`)**. En büyük dosyadır (~168 KB). Tamamını okumak yerine fonksiyon bazlı (`prewarp_garment_tps`, `generate_category_agnostic_mask`, `build_diffusion_inputs`, `extract_anatomical_z_masks`) arama yaparak okuyun. |
| [`vton/nodes/heatmap.py`](vton/nodes/heatmap.py) | **Node-03: Isı Haritası (`FitHeatmapNode`)**. Gerçek vücut silüeti (orijinal fotoğraf SegFormer + ANSUR kırpma, bedenden bağımsız), bölgesel bolluk profili (cm), sürekli kırmızı→sarı→yeşil skala, taralı hava boşluğu, Türkçe etiketler (PIL) ve beden önerisi (`CategoryRouter.recommend_size`). |
| [`vton/inference_api.py`](vton/inference_api.py) | **IDM-VTON İstemcisi ve Post-Processing**. Colab/HF API çağrısı (`render_with_idm_vton`), 768x1024 ön ölçekleme ve vektörel baskı/logo restorasyonu (`restore_garment_print`) buradadır. |
| [`vton/category_router.py`](vton/category_router.py) | **Kategori Yönlendirici ve Tekil Model Kayıt Defteri (`CategoryRouter`, `GlobalModelRegistry`)**. Üst/alt/elbise landmark hedefleri ve `AsymmetricScaleSpec` ölçek kuralları buradadır. |
| [`app.py`](app.py) | **Gradio Web Arayüzü**. Kullanıcı profilleri, URL'den otomatik kategori algılama, `islem_baslat` döngüsü ve galeriler buradadır. |

---

### 🔒 B. Kararlı Altyapı Dosyaları (Bir Kez Bakıp Geçilecek / Dokunulmayacak Dosyalar)
Bu dosyalar görevini eksiksiz yapmaktadır. **Özel bir hata veya yeni bir özellik doğrudan bu modülü ilgilendirmiyorsa bu dosyaları tekrar tekrar okumayın**:

| Dosya | Neden Tekrar Okunmasına Gerek Yok? |
| :--- | :--- |
| [`vton/__init__.py`](vton/__init__.py) | Yalnızca `lazy import` dışa aktarımlarını tutar. |
| [`vton/paths.py`](vton/paths.py) | `PROJECT_ROOT`, `GORUNTU_DIR` ve `goruntu("dosya.jpg")` yol çözücüsüdür. Tamamen sabittir. |
| [`vton/models/ansur.py`](vton/models/ansur.py) | `Ansur3DTo2DConverter`: Ramanujan elips formülüyle 3B çevreyi 2B genişlik/derinliğe çevirir (`circumference_to_width`, `circumference_to_depth`). Matematiksel olarak sabittir. |
| [`vton/nodes/body_mesh.py`](vton/nodes/body_mesh.py) | `BodyMeshNode`: MediaPipe Pose ile 3B vücut tensörü, duruş yönü (`Ön`, `Arka`, `Yan`) ve omuz/kalça piksel genişliğini çıkarır. Kararlıdır. |
| [`vton/nodes/segmentation.py`](vton/nodes/segmentation.py) | `SegFormerNode`: Kıyafet fotoğrafından hedef kategoriyi (üst/alt/elbise) ayıklar ve RGBA olarak kırpar. Kararlıdır. |
| [`vton/calibration.py`](vton/calibration.py) | Kafatası biyometrisi (IPD + yüz uzunluğu) ve RandomForest derinlik kalibrasyonu yardımcısıdır. Kararlıdır. |
| [`vton/extractor.py`](vton/extractor.py) & [`vton/renderer.py`](vton/renderer.py) | Hafif yardımcı sarmalayıcılardır (asıl iş `segmentation.py` ve `inference_api.py` içindedir). |
| [`vton/scraper/trendyol.py`](vton/scraper/trendyol.py), [`vton/scraper/pipeline.py`](vton/scraper/pipeline.py), [`vton/scraper/calibration.py`](vton/scraper/calibration.py) | Selenium tabanlı Trendyol veri kazıma hattıdır. Yalnızca scraper ile ilgili bir talep olduğunda incelenmelidir. |

---

### 📁 C. Veri, Doküman ve Görsel Klasörleri (Okunmaması Gerekenler)
- **`goruntu/`**: Yalnızca aktif manken fotoğraflarını (`user_{on,arka,yan}.jpg`, `user2_{on,arka,yan}.png`), kazınan ürün görsellerini (`on.jpg`, `arka.jpg`, `yan.jpg`, `olcu_ref.jpg`), ayıklanan kıyafet PNG'lerini ve Gradio varsayılan sonuçlarını (`temp_result_*`, `hf_master_u1_*`) barındırır.
- **`dataset/ansur/`**: Ham ANSUR II CSV veri setidir. Asla doğrudan okunmamalıdır.
- **`VTON_Chrome_Profil/`**: Selenium tarayıcı önbelleğidir. Asla taranmamalıdır.
- **`generate_tubitak_docx.py`, `*.docx`**: Akademik tez ve TÜBİTAK 2209-A başvuru belgeleridir. Kod geliştirmesiyle ilgisi yoktur.

---

## 5. Kritik Mühendislik Kuralları

1. **Modeller Tekil (Singleton) Yüklenmeli**: SegFormer ve MediaPipe modellerini asla fonksiyon içinde tekrar yüklemeyin; her zaman `from vton.category_router import get_global_models` üzerinden `get_global_models()` singleton yapısını kullanın.
2. **Piksel Döngüsü Yasak**: Görüntü işlemede Python `for` döngüsüyle piksel gezmeyin; `np.where`, `cv2.remap`, `cv2.distanceTransform`, `cv2.morphologyEx` ve yayınlama (broadcasting) kullanın.
3. **Dosya Yolları**: Göreli `"goruntu/..."` yerine her zaman `from vton.paths import goruntu` fonksiyonunu kullanın. Dosya adlarında Türkçe karakter (`Ö`, `ş`, `ı` vb.) kullanmayın (Windows OpenCV `cv2.imread` uyumluluğu için yalnızca `on`, `arka`, `yan` anahtarlarını kullanın).
