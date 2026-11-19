"""
Beden / kalıp fiziği ve ısı haritası regresyon testleri.

Hızlı testler (model yüklemez):  pytest tests -q
Entegrasyon testleri (SegFormer + MediaPipe, ~1 dk):  VTON_SLOW_TESTS=1 pytest tests -q
"""
import os

import numpy as np
import pytest

from vton.category_router import CategoryRouter
from vton.nodes.heatmap import ease_to_bgr

SIZES = ["XS", "S", "M", "L", "XL", "XXL"]
U1 = {"height": 164.0, "chest": 92.0, "waist": 60.0, "hips": 89.0}  # ince bel kadın
U2 = {"height": 170.0, "chest": 100.0, "waist": 84.0, "hips": 114.0}  # kıvrımlı kadın
UPPER = "Üst Giyim (Tişört / Gömlek / Kazak / Ceket)"
PANTS = "Alt Giyim - Pantolon / Jean / Eşofman"
FITS = ["Oversize / Rahat Kalıp", "Regular / Standart Kalıp", "Slim Fit / Dar Kalıp", "Skinny / Bodycon (Saran)"]


def _spec(size, fit, cat=UPPER, user=U1):
    router = CategoryRouter()
    garment = np.zeros((400, 300, 4), np.uint8)
    garment[40:360, 30:270, 3] = 255
    return router.compute_asymmetric_drape_scaling(
        garment_bgra=garment,
        garment_cm_width=50.0,
        garment_length_cm=66.0,
        user_cm_per_pixel=0.08,
        user_measurements=user,
        category_or_garment_type=cat,
        fit_type=fit,
        fabric_stretch="Az Esnek (%2-3 Likra/Süprem)",
        desired_size=size,
    )


def test_fit_classification_standart_is_not_slim():
    # "stanDARt" eskiden "dar" alt-dizgisi yüzünden Slim sayılıyordu
    assert CategoryRouter.fit_key_of("Regular / Standart Kalıp") == "regular"
    assert CategoryRouter.get_fit_ease_spec("upper_body", "top", "Regular / Standart Kalıp")["chest_ease"] == 10.0
    assert CategoryRouter.fit_key_of("Slim Fit / Dar Kalıp") == "slim"
    assert CategoryRouter.fit_key_of("Skinny / Bodycon (Saran)") == "skinny"
    assert CategoryRouter.fit_key_of("Oversize / Rahat Kalıp") == "oversize"


@pytest.mark.parametrize("fit", FITS)
def test_upper_size_grading_is_monotonic_and_not_saturated(fit):
    widths = [_spec(s, fit).target_width_px for s in SIZES]
    assert all(a < b for a, b in zip(widths, widths[1:])), widths
    chest = [_spec(s, fit).extra_metrics["vis_ratio_chest"] for s in SIZES]
    assert all(a <= b for a, b in zip(chest, chest[1:])), chest
    # Bol kalıpta M -> XXL arası görünür genişlik en az %10 artmalı (eski sürüm ~%2.6 idi)
    if fit.startswith("Oversize"):
        m, xxl = chest[2], chest[5]
        assert xxl / m > 1.10, (m, xxl)


def test_lower_body_ratios_do_not_saturate_between_m_and_xxl():
    # Eski sürüm kalça/uyluk çarpanını 1.22'de kırpıyordu → M ve XXL özdeşti
    m = _spec("M", FITS[0], PANTS).extra_metrics
    xxl = _spec("XXL", FITS[0], PANTS).extra_metrics
    for key in ("vis_ratio_hips", "vis_ratio_thigh", "vis_ratio_knee"):
        assert xxl[key] > m[key] * 1.05, (key, m[key], xxl[key])


def test_slim_pants_narrower_than_oversize():
    slim = _spec("M", FITS[2], PANTS).extra_metrics
    over = _spec("M", FITS[0], PANTS).extra_metrics
    for key in ("vis_ratio_thigh", "vis_ratio_knee", "vis_ratio_hem"):
        assert slim[key] < over[key], key


def test_visible_ratio_never_below_body():
    # Kıyafet vücuttan dar olsa bile silüet vücuttan içeri giremez (kumaş gerilir)
    assert CategoryRouter.visible_width_ratio(20.0, 30.0) == 1.0
    assert CategoryRouter.visible_width_ratio(45.0, 30.0) == pytest.approx(1.5)


def test_size_recommendation():
    r = CategoryRouter()
    assert r.recommend_size(U1, "Kadın", UPPER, "Az Esnek")[0] == "M"
    assert r.recommend_size(U2, "Kadın", PANTS, "Esnek Değil")[0] == "XXL"
    assert r.recommend_size(U2, "Kadın", UPPER, "Esnek Değil")[0] in ("L", "XL")


def test_heatmap_colors_red_tight_green_loose():
    tight, fit, loose = ease_to_bgr(np.array([-6.0, 1.0, 14.0]))
    assert tight[2] > 200 and tight[1] < 80  # BGR: kırmızı baskın
    assert fit[2] > 200 and fit[1] > 150  # sarı/turuncu
    assert loose[1] > 180 and loose[2] < 120  # yeşil baskın


@pytest.mark.skipif(os.environ.get("VTON_SLOW_TESTS") != "1", reason="VTON_SLOW_TESTS=1 ile çalışır (modeller yüklenir)")
def test_integration_body_contour_invariant_and_garment_grows(tmp_path):
    import cv2

    from vton.engine import VTONEngine
    from vton.paths import goruntu

    user, garment = goruntu("user_on.jpg"), goruntu("temp_garment_extracted_on.png")
    if not (os.path.exists(user) and os.path.exists(garment)):
        pytest.skip("Test görselleri yok")
    eng = VTONEngine()
    body, area = {}, {}
    for sz in ("XS", "M", "XXL"):
        out = str(tmp_path / f"r_{sz}.jpg")
        eng.run(user, garment, goruntu("on.jpg"), out, user_measurements=U1, desired_size=sz, use_ai_render=False)
        body[sz] = eng.node_03.last_heatmap_stats["zones"]["GÖĞÜS"]["body_px"]
        m = cv2.imread(out.replace(".jpg", "_mask.png"), cv2.IMREAD_UNCHANGED)
        area[sz] = int(np.count_nonzero(m[..., 3] if m.ndim == 3 else m))
    # Gerçek vücut silüeti bedenden bağımsız (±%3)
    assert max(body.values()) - min(body.values()) <= 0.03 * max(body.values()), body
    # Kıyafet alanı beden büyüdükçe artar
    assert area["XS"] < area["M"] < area["XXL"], area
