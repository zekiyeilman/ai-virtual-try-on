import gradio as gr
import os
import re
from dotenv import load_dotenv
load_dotenv()
import cv2
import json
import numpy as np
from vton import VTONEngine, SegFormerNode, vton_otonom_boru_hatti
from vton.paths import goruntu, PROJECT_ROOT

USER_PROFILES_JSON = str(PROJECT_ROOT / "user_profiles.json")


def _varsa(yol):
    """Dosya varsa yolunu, yoksa None döndürür (Gradio varsayılan görselleri için)."""
    return yol if (yol and os.path.exists(yol)) else None


DEFAULT_PROFILES = {
    "User 1 — İnce Bel Kadın Manken (Boy: 164 | Göğüs: 92 | Bel: 60 | Basen: 89)": {
        "front": "user_on.jpg",
        "back": "user_arka.jpg",
        "side": "user_yan.jpg",
        "height": 164.0,
        "chest": 92.0,
        "waist": 60.0,
        "hips": 89.0,
        "gender": "Kadın",
    },
    "User 2 — Kıvrımlı Kadın Manken (Boy: 170 | Göğüs: 100 | Bel: 84 | Basen: 114)": {
        "front": "user2_on.png",
        "back": "user2_arka.png",
        "side": "user2_yan.png",
        "height": 170.0,
        "chest": 100.0,
        "waist": 84.0,
        "hips": 114.0,
        "gender": "Kadın",
    },
    "Erkek Manken — Standart/Atletik (Boy: 182 | Göğüs: 102 | Bel: 84 | Basen: 98)": {
        "front": "user_erkek_on.jpg",
        "back": "user_erkek_arka.jpg",
        "side": "user_erkek_yan.jpg",
        "height": 182.0,
        "chest": 102.0,
        "waist": 84.0,
        "hips": 98.0,
        "gender": "Erkek",
    },
    "➕ Özel Kullanıcı (Manuel Fotoğraf & Ölçü Yükle)": {
        "front": "",
        "back": "",
        "side": "",
        "height": 170.0,
        "chest": 94.0,
        "waist": 76.0,
        "hips": 96.0,
        "gender": "Kadın",
    },
}


def load_user_profiles():
    profiles = dict(DEFAULT_PROFILES)
    if os.path.exists(USER_PROFILES_JSON):
        try:
            with open(USER_PROFILES_JSON, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                profiles.update(saved)
        except Exception:
            pass
    return profiles


def save_user_profile(new_name, u_front, u_back, u_side, height, chest, waist, hips, gender):
    profiles = load_user_profiles()
    clean_name = (new_name or "").strip()
    if not clean_name:
        clean_name = f"User {len(profiles)} — Özel"
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", clean_name.lower())[:24]

    front_rel, back_rel, side_rel = "", "", ""
    if u_front is not None:
        front_rel = f"custom_{slug}_on.png"
        cv2.imwrite(goruntu(front_rel), cv2.cvtColor(u_front, cv2.COLOR_RGB2BGR))
    if u_back is not None:
        back_rel = f"custom_{slug}_arka.png"
        cv2.imwrite(goruntu(back_rel), cv2.cvtColor(u_back, cv2.COLOR_RGB2BGR))
    if u_side is not None:
        side_rel = f"custom_{slug}_yan.png"
        cv2.imwrite(goruntu(side_rel), cv2.cvtColor(u_side, cv2.COLOR_RGB2BGR))

    full_label = f"{clean_name} (Boy: {int(height)} | Göğüs: {int(chest)} | Bel: {int(waist)} | Basen: {int(hips)})"
    profiles[full_label] = {
        "front": front_rel,
        "back": back_rel,
        "side": side_rel,
        "height": float(height),
        "chest": float(chest),
        "waist": float(waist),
        "hips": float(hips),
        "gender": gender or "Kadın",
    }

    try:
        with open(USER_PROFILES_JSON, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return gr.update(), f"❌ Profil kaydedilemedi: {e}"

    return (
        gr.update(choices=list(profiles.keys()), value=full_label),
        f"✅ '{full_label}' çekmeceye başarıyla kaydedildi!",
    )


def select_user_profile(profile_key):
    profiles = load_user_profiles()
    p = profiles.get(profile_key)
    if not p:
        return gr.update(), gr.update(), gr.update(), 164.0, 92.0, 60.0, 89.0, "Kadın"

    f_path = _varsa(goruntu(p["front"])) if p.get("front") else None
    b_path = _varsa(goruntu(p["back"])) if p.get("back") else None
    s_path = _varsa(goruntu(p["side"])) if p.get("side") else None

    return (
        f_path,
        b_path,
        s_path,
        float(p.get("height", 164.0)),
        float(p.get("chest", 92.0)),
        float(p.get("waist", 60.0)),
        float(p.get("hips", 89.0)),
        p.get("gender", "Kadın"),
    )


engine = VTONEngine()
segformer_node = SegFormerNode()
engine.segformer_node = segformer_node

GARMENT_CATEGORIES = [
    "Üst Giyim (Tişört / Gömlek / Kazak / Ceket)",
    "Alt Giyim - Pantolon / Jean / Eşofman",
    "Alt Giyim - Etek / Şort",
    "Elbise / Tulum",
]

FIT_TYPES = [
    "Oversize / Rahat Kalıp",
    "Regular / Standart Kalıp",
    "Slim Fit / Dar Kalıp",
    "Skinny / Bodycon (Saran)",
]

STRETCH_TYPES = [
    "Esnek Değil (%0 Dokuma/Kot)",
    "Az Esnek (%2-3 Likra/Süprem)",
    "Orta Esnek (%5-8 Elastan/Triko)",
    "Yüksek Esnek (%10+ Likra/Tayt)",
]

SIZES = ["XS", "S", "M", "L", "XL", "XXL"]


def _detect_garment_metadata_from_url(url: str):
    u = (url or "").lower()
    # Kategori tespiti
    if any(k in u for k in ["pantolon", "jean", "kot", "esofman", "jogger", "tayt"]):
        cat = GARMENT_CATEGORIES[1]
    elif any(k in u for k in ["etek", "sort", "bermuda"]):
        cat = GARMENT_CATEGORIES[2]
    elif any(k in u for k in ["elbise", "tulum", "salopet"]):
        cat = GARMENT_CATEGORIES[3]
    else:
        cat = GARMENT_CATEGORIES[0]

    # Kalıp tespiti
    if any(k in u for k in ["oversize", "genis-kalip", "boyfriend", "baggy", "loose", "rahat-kalip"]):
        fit = FIT_TYPES[0]
    elif any(k in u for k in ["slim", "dar-kalip", "fitted"]):
        fit = FIT_TYPES[2]
    elif any(k in u for k in ["skinny", "bodycon", "extra-dar", "korse"]):
        fit = FIT_TYPES[3]
    else:
        fit = FIT_TYPES[1]

    # Esneklik tespiti
    if any(k in u for k in ["tayt", "likrali", "elastan", "toparlayici"]):
        stretch = STRETCH_TYPES[3]
    elif any(k in u for k in ["triko", "kazak", "hirka", "ribana", "kaskorse"]):
        stretch = STRETCH_TYPES[2]
    elif any(k in u for k in ["orme", "t-shirt", "tisort", "sweatshirt", "penye", "suprem"]):
        stretch = STRETCH_TYPES[1]
    elif any(k in u for k in ["dokuma", "gomlek", "ceket", "kaban", "jean", "denim", "keten"]):
        stretch = STRETCH_TYPES[0]
    else:
        stretch = STRETCH_TYPES[1]

    return cat, fit, stretch


def linkten_veri_cek(url):
    url = (url or "").strip().strip("'\"")
    if not url or "trendyol" not in url:
        return (
            None,
            "❌ Geçerli bir Trendyol linki girmelisiniz.",
            "M",
            gr.update(),
            [],
            gr.update(),
            gr.update(),
            gr.update(),
        )

    try:
        cat_auto, fit_auto, stretch_auto = _detect_garment_metadata_from_url(url)
        sonuc = vton_otonom_boru_hatti(url)
        if sonuc is None:
            return (
                None,
                "❌ Scraper ürün verilerini çekemedi. Lütfen geçerli bir Trendyol ürün linki girdiğinizden emin olun.",
                "M",
                gr.update(),
                [],
                gr.update(value=cat_auto),
                gr.update(value=fit_auto),
                gr.update(value=stretch_auto),
            )

        on_img_path = sonuc.get("on_image_path")
        garment_img = None
        if on_img_path and os.path.exists(on_img_path):
            garment_img = cv2.cvtColor(cv2.imread(on_img_path), cv2.COLOR_BGR2RGB)
            import shutil

            if not sonuc.get("arka_image_path"):
                try:
                    shutil.copyfile(on_img_path, goruntu("arka.jpg"))
                except Exception:
                    pass
            if not sonuc.get("yan_image_path"):
                try:
                    shutil.copyfile(on_img_path, goruntu("yan.jpg"))
                except Exception:
                    pass

        gt = sonuc.get("ground_truth") or {}
        bt = sonuc.get("beden_tablosu") or {}

        info_md = "### 📊 Linkten Çekilen Manken, Kalıp ve Beden Tablosu Analizi\n"
        info_md += f"- **Otomatik Tespit Edilen Kategori:** `{cat_auto}`\n"
        info_md += f"- **Otomatik Tespit Edilen Kalıp:** `{fit_auto}`\n"
        info_md += f"- **Otomatik Tespit Edilen Kumaş Esnekliği:** `{stretch_auto}`\n\n"
        if gt:
            info_md += "**Manken Gerçek Verileri (Ground Truth):**\n"
            for k, v in gt.items():
                info_md += f"- **{k.replace('_', ' ').title()}:** {v}\n"
        else:
            info_md += "Manken ölçüleri açıklamalarda bulunamadı.\n"

        info_md += "\n**Beden Tablosu:**\n"
        if bt and "veriler" in bt:
            info_md += "| Beden | Göğüs | Bel | Basen |\n|---|---|---|---|\n"
            for satir in bt["veriler"]:
                info_md += (
                    f"| {satir.get('beden')} | "
                    f"{satir.get('gogus',[0,0])[0]}-{satir.get('gogus',[0,0])[1]} | "
                    f"{satir.get('bel',[0,0])[0]}-{satir.get('bel',[0,0])[1]} | "
                    f"{satir.get('basen',[0,0])[0]}-{satir.get('basen',[0,0])[1]} |\n"
                )

        manken_beden = "M"
        if gt and "manken_bedeni" in gt:
            manken_beden = str(gt["manken_bedeni"]).upper()
            if manken_beden not in SIZES:
                manken_beden = "M"

        galeri_resimleri = []
        for path_key, label in [
            ("ref_image_path", "Referans (İlk)"),
            ("on_image_path", "Ön Duruş"),
            ("arka_image_path", "Arka Duruş"),
            ("yan_image_path", "Yan Duruş"),
        ]:
            img_p = sonuc.get(path_key)
            if img_p and os.path.exists(img_p):
                galeri_resimleri.append((img_p, label))

        return (
            garment_img,
            info_md,
            manken_beden,
            gr.update(value=manken_beden),
            galeri_resimleri,
            gr.update(value=cat_auto),
            gr.update(value=fit_auto),
            gr.update(value=stretch_auto),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            None,
            f"❌ Hata: {str(e)}",
            "M",
            gr.update(),
            [],
            gr.update(),
            gr.update(),
            gr.update(),
        )


def islem_baslat(
    user_front,
    user_back,
    user_side,
    garment_img,
    height,
    chest,
    waist,
    hips,
    gender,
    garment_type,
    fit_type,
    fabric_stretch,
    model_worn_size,
    desired_size,
    use_ai,
    colab_url,
    progress=gr.Progress(track_tqdm=True),
):
    user_measurements = {
        "height": float(height),
        "chest": float(chest),
        "waist": float(waist),
        "hips": float(hips),
        "gender": gender or "Kadın",
    }

    vton_results = []
    heatmap_results = []
    mesajlar = []

    custom_g_path = None
    is_custom_upload = False
    if garment_img is not None:
        custom_g_path = goruntu("temp_uploaded_garment.jpg")
        custom_bgr = cv2.cvtColor(garment_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(custom_g_path, custom_bgr)
        default_on_p = goruntu("on.jpg")
        if os.path.exists(default_on_p):
            def_on_bgr = cv2.imread(default_on_p)
            if def_on_bgr is None or def_on_bgr.shape != custom_bgr.shape:
                is_custom_upload = True
            else:
                diff = float(np.mean(np.abs(def_on_bgr.astype(np.float32) - custom_bgr.astype(np.float32))))
                if diff > 4.0:
                    is_custom_upload = True
        else:
            is_custom_upload = True

    front_g = custom_g_path if (custom_g_path and os.path.exists(custom_g_path)) else goruntu("on.jpg")
    if is_custom_upload and custom_g_path and os.path.exists(custom_g_path):
        back_g = custom_g_path
        side_g = custom_g_path
    else:
        back_g = goruntu("arka.jpg") if os.path.exists(goruntu("arka.jpg")) else front_g
        side_g = goruntu("yan.jpg") if os.path.exists(goruntu("yan.jpg")) else front_g

    angles = [
        (user_front, front_g, "on", "Ön"),
        (user_back, back_g, "arka", "Arka"),
        (user_side, side_g, "yan", "Yan"),
    ]

    for idx, (u_img, g_path, angle_key, label) in enumerate(angles):
        progress((idx) / len(angles), desc=f"{label} duruşu analiz ediliyor ve giydiriliyor...")
        try:
            if u_img is None:
                continue

            if not os.path.exists(g_path):
                mesajlar.append(f"{label} yönü için kıyafet fotoğrafı bulunamadı. Atlanıyor.")
                continue

            u_path = goruntu(f"temp_user_{angle_key}.jpg")
            cv2.imwrite(u_path, cv2.cvtColor(u_img, cv2.COLOR_RGB2BGR))

            g_extracted_path = goruntu(f"temp_garment_extracted_{angle_key}.png")
            cropped_garment, mask = segformer_node.process_image(g_path, garment_type=garment_type)

            if cropped_garment is None:
                mesajlar.append(f"Hata: {label} yönü için segmentasyon başarısız oldu.")
                continue

            cv2.imwrite(g_extracted_path, cropped_garment)

            out_name = goruntu(f"temp_result_{angle_key}.jpg")

            mesaj = engine.run(
                user_image_path=u_path,
                garment_image_path=g_extracted_path,
                original_on_jpg_path=g_path,
                output_name=out_name,
                user_measurements=user_measurements,
                desired_size=desired_size,
                model_worn_size=model_worn_size,
                garment_type=garment_type,
                use_ai_render=use_ai,
                colab_url=colab_url,
                fit_type=fit_type,
                fabric_stretch=fabric_stretch,
                gender=gender or "Kadın",
            )
            mesajlar.append(f"[{label} Duruş]:\n" + mesaj)

            heatmap_name = out_name.replace(".jpg", "_heatmap.jpg")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"CRITICAL ERROR in loop: {error_trace}")
            mesajlar.append(f"Kritik Hata Oluştu ({label}): {e}")
            continue

        # engine.run() hem doğrulanmış HD Render'ı (beden/kalıp ölçekli) hem de yeni kullanıcı sonucunu out_name'e yazar
        if os.path.exists(out_name):
            final_img = cv2.cvtColor(cv2.imread(out_name), cv2.COLOR_BGR2RGB)
            vton_results.append((final_img, f"{label} ({desired_size})"))

        if os.path.exists(heatmap_name):
            h_img = cv2.cvtColor(cv2.imread(heatmap_name), cv2.COLOR_BGR2RGB)
            heatmap_results.append((h_img, f"{label} Isı Haritası ({desired_size})"))

    progress(1.0, desc="İşlem tamamlandı!")

    if not vton_results:
        return None, None, "İşlenecek geçerli bir fotoğraf eşleşmesi bulunamadı. Hem kullanıcı hem kıyafetin aynı yönleri olmalı."

    return vton_results, heatmap_results, "\n\n".join(mesajlar)


def _varsayilan_sonuclari_getir():
    vton_list = []
    heatmap_list = []
    for angle_key, label in [("on", "Ön"), ("arka", "Arka"), ("yan", "Yan")]:
        hd_p = goruntu(f"temp_result_{angle_key}_HD_RENDER.jpg")
        std_p = goruntu(f"temp_result_{angle_key}.jpg")
        hm_p = goruntu(f"temp_result_{angle_key}_heatmap.jpg")

        candidates = [p for p in [hd_p, std_p] if os.path.exists(p)]
        if candidates:
            best_p = max(candidates, key=os.path.getmtime)
            vton_list.append((best_p, label))

        if os.path.exists(hm_p):
            heatmap_list.append((hm_p, label))
    return vton_list, heatmap_list


initial_profiles = load_user_profiles()
default_profile_key = list(initial_profiles.keys())[0]

with gr.Blocks(title="VTON 360° Beden, Kalıp & Kumaş Duyarlı Sanal Kabin") as demo:
    gr.Markdown("# 👕 Otonom 360° Size-Aware & Fit-Aware VTON Ekosistemi")
    gr.Markdown(
        "Kadın ve Erkek modellerde **Üst Giyim, Pantolon/Jean, Etek/Şort ve Elbise** kategorilerini; "
        "**XS - XXL** bedenlerini, **Oversize / Regular / Slim Fit / Bodycon** kalıplarını ve **Kumaş Esnekliğini (%0 - %10+ Likra)** "
        "3D ANSUR II antropometrik ısı haritasıyla analiz eder."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Ürün & Kalıp Ekosistemi")
            url_in = gr.Textbox(
                value="https://www.trendyol-milla.com/trendyolmilla/antrasit-100-pamuk-oversize-genis-kalip-baskili-orme-t-shirt-twoss25ts00148-p-896485014",
                placeholder="Trendyol Ürün Linki Yapıştırın (Tişört, Pantolon, Etek, Elbise, Gömlek...)",
                label="Ürün Linki",
            )
            fetch_btn = gr.Button("🔗 Linkten Veri, Kalıp ve Beden Tablosu Çek", variant="secondary")

            scraper_info = gr.Markdown("Manken, Kalıp ve Beden Tablosu Verileri Burada Gösterilecek...")
            garment_gallery = gr.Gallery(label="Kazınan Ürün Görselleri (Referans, Ön, Arka, Yan)", height=220, columns=4)
            garment_in = gr.Image(
                value=_varsa(goruntu("on.jpg")),
                label="Giydirilecek Kıyafet (Galeriden seçip veya manuel yükleyebilirsiniz)",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 2. Kullanıcı Çekmecesi & Antropometrik Veriler")

            with gr.Accordion("🗄️ Manken & Kullanıcı Profil Çekmecesi (User 1 / User 2 / Erkek / Özel)", open=True):
                profile_dropdown = gr.Dropdown(
                    choices=list(initial_profiles.keys()),
                    value=default_profile_key,
                    label="Kayıtlı Kullanıcı / Manken Seçin",
                )
                with gr.Accordion("➕ Yeni Kullanıcı Profili Ekle / Çekmeceye Kaydet", open=False):
                    with gr.Row():
                        new_profile_name = gr.Textbox(
                            placeholder="Örn: User 3 - Zeynep / Ahmet",
                            label="Yeni Profil Adı",
                            scale=2,
                        )
                        save_profile_btn = gr.Button("💾 Çekmeceye Kaydet", variant="secondary", scale=1)
                    save_status_md = gr.Markdown("")

            with gr.Row():
                user_front = gr.Image(value=_varsa(goruntu("user_on.jpg")), label="Ön Fotoğraf")
                user_back = gr.Image(value=_varsa(goruntu("user_arka.jpg")), label="Arka Fotoğraf")
                user_side = gr.Image(value=_varsa(goruntu("user_yan.jpg")), label="Yan Fotoğraf")

            with gr.Row():
                gender_in = gr.Radio(choices=["Kadın", "Erkek"], value="Kadın", label="Cinsiyet (ANSUR II Modeli)")
                height_in = gr.Number(value=164.0, label="Boy (cm)")
                chest_in = gr.Number(value=92.0, label="Göğüs Çevresi (cm)")
                waist_in = gr.Number(value=60.0, label="Bel Çevresi (cm)")
                hips_in = gr.Number(value=89.0, label="Basen Çevresi (cm)")

            gr.Markdown("#### 🧵 Kıyafet Kategorisi, Kalıp (Fit), Kumaş Esnekliği ve Beden")
            with gr.Row():
                garment_type_in = gr.Dropdown(
                    choices=GARMENT_CATEGORIES,
                    value=GARMENT_CATEGORIES[0],
                    label="Kıyafet Kategorisi",
                )
                fit_type_in = gr.Dropdown(
                    choices=FIT_TYPES,
                    value=FIT_TYPES[0],
                    label="Kalıp Tipi (Fit)",
                )
            with gr.Row():
                fabric_stretch_in = gr.Dropdown(
                    choices=STRETCH_TYPES,
                    value=STRETCH_TYPES[1],
                    label="Kumaş Esnekliği (Likra / Elastan)",
                )
                model_worn_in = gr.Dropdown(
                    choices=SIZES,
                    value="M",
                    label="Mankenin Giydiği Beden",
                )
                size_in = gr.Dropdown(
                    choices=SIZES,
                    value="M",
                    label="Denemek İstediğin Beden",
                )

            use_ai_in = gr.Checkbox(value=True, label="🚀 Hugging Face (yisol/IDM-VTON) / Colab HD AI Render Kullan")
            colab_url_in = gr.Textbox(
                placeholder="https://xxxx.gradio.live (Colab T4 GPU adresinizi girin)",
                label="Colab API URL (AI Render işaretlendiğinde kullanılır)",
                visible=False,
            )

            def toggle_colab_url(is_checked):
                return gr.update(visible=is_checked)

            use_ai_in.change(fn=toggle_colab_url, inputs=[use_ai_in], outputs=[colab_url_in])

            submit_btn = gr.Button("✨ Analiz Et ve Giydir (Ön, Arka, Yan + Isı Haritası)", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 3. Sonuçlar & Kalıp/Esneklik Isı Haritası")
            _init_vton, _init_heatmap = _varsayilan_sonuclari_getir()
            out_img = gr.Gallery(
                value=_init_vton,
                label="Giydirilmiş Sonuçlar (Ön, Arka, Yan)",
                columns=3,
                height=340,
            )
            out_heatmap = gr.Gallery(
                value=_init_heatmap,
                label="Isı Haritaları (Darlık / Tam Kalıp / Oversize Bolluk & Konfor)",
                columns=3,
                height=340,
            )
            out_text = gr.Textbox(
                label="Detaylı Antropometrik & Kalıp Analiz Raporu (Doğrulama & Telemetri Çıktısı)",
                lines=22,
            )

    # Çekmeceden profil seçildiğinde fotoğrafları ve ölçüleri otomatik doldur
    profile_dropdown.change(
        fn=select_user_profile,
        inputs=[profile_dropdown],
        outputs=[user_front, user_back, user_side, height_in, chest_in, waist_in, hips_in, gender_in],
    )

    # Yeni profili çekmeceye kaydet
    save_profile_btn.click(
        fn=save_user_profile,
        inputs=[new_profile_name, user_front, user_back, user_side, height_in, chest_in, waist_in, hips_in, gender_in],
        outputs=[profile_dropdown, save_status_md],
    )

    demo.load(
        fn=_varsayilan_sonuclari_getir,
        inputs=None,
        outputs=[out_img, out_heatmap],
    )

    fetch_btn.click(
        fn=linkten_veri_cek,
        inputs=[url_in],
        outputs=[
            garment_in,
            scraper_info,
            model_worn_in,
            size_in,
            garment_gallery,
            garment_type_in,
            fit_type_in,
            fabric_stretch_in,
        ],
    )

    submit_btn.click(
        fn=islem_baslat,
        inputs=[
            user_front,
            user_back,
            user_side,
            garment_in,
            height_in,
            chest_in,
            waist_in,
            hips_in,
            gender_in,
            garment_type_in,
            fit_type_in,
            fabric_stretch_in,
            model_worn_in,
            size_in,
            use_ai_in,
            colab_url_in,
        ],
        outputs=[out_img, out_heatmap, out_text],
    )

if __name__ == "__main__":
    share_mode = os.environ.get("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=False, share=share_mode)

