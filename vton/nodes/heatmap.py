"""
Node-03: Fiziksel Tabanlı Biyomekanik Uyum (Fit) Isı Haritası.

Tasarım ilkeleri (eski sürümdeki "kıyafet maskesini içe aşındırıp vücut diye çizme" hatası giderildi):

1. GERÇEK VÜCUT SİLÜETİ
   Mavi vücut çizgisi, kullanıcının ORİJİNAL fotoğrafının SegFormer insan silüetinden gelir. Kullanıcının üzerindeki
   eski bol kıyafet vücudu şişirmesin diye gövde satırları ANSUR II'den beklenen 2B genişliğin (yan profilde derinliğin)
   %12 fazlasıyla sınırlanır. Vücut silüeti seçilen bedenden BAĞIMSIZDIR (beden değişince değişmez).

2. BÖLGESEL BOLLUK PROFİLİ (cm)
   Her anatomik bölgede (göğüs, bel, kalça / uyluk, diz, paça) kıyafet 3B çevresi − vücut 3B çevresi + kumaş esnekliği
   toleransı hesaplanır ve landmark satırları arasında yumuşak enterpolasyonla her piksel satırına dağıtılır.

3. SÜREKLİ RENK SKALASI
   Kırmızı (dar / gergin) → Turuncu → Sarı (tam oturur) → Açık yeşil (rahat) → Yeşil/Turkuaz (bol).
   Kumaş ile vücut arasındaki hava boşluğu taralı gösterilir; darlıkta gerilim vücut kenarlarında yoğunlaşır.

4. OKUNUR ÇIKTI
   Bölge etiketleri (Türkçe karakterli, cm değerli), renk skalası çubuğu ve kompakt özet kartı.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..category_router import CategoryRouter, get_global_models

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None

# BGR renk durakları: (bolluk_cm, renk)
_EASE_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
    (-8.0, (30, 20, 200)),
    (-3.0, (40, 40, 245)),
    (0.0, (20, 150, 255)),
    (3.0, (40, 215, 245)),
    (7.0, (90, 225, 150)),
    (12.0, (70, 200, 60)),
    (22.0, (150, 170, 20)),
]

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "DejaVuSans.ttf",
]
_FONT_BOLD_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "DejaVuSans-Bold.ttf",
]
_font_cache: Dict[Tuple[int, bool], object] = {}


def _font(size: int, bold: bool = False):
    key = (int(size), bool(bold))
    if key in _font_cache:
        return _font_cache[key]
    f = None
    if Image is not None:
        for p in (_FONT_BOLD_CANDIDATES if bold else _FONT_CANDIDATES):
            try:
                f = ImageFont.truetype(p, int(size))
                break
            except Exception:
                continue
        if f is None:
            try:
                f = ImageFont.load_default()
            except Exception:
                f = None
    _font_cache[key] = f
    return f


def ease_to_bgr(ease_cm: np.ndarray) -> np.ndarray:
    """Vektörel bolluk(cm) → BGR renk (N,3) float32."""
    e = np.asarray(ease_cm, dtype=np.float32)
    xs = np.array([s[0] for s in _EASE_STOPS], dtype=np.float32)
    cols = np.array([s[1] for s in _EASE_STOPS], dtype=np.float32)
    out = np.stack([np.interp(e, xs, cols[:, c]) for c in range(3)], axis=-1)
    return out.astype(np.float32)


def ease_label(ease_cm: float, fit_key: str, zone: str = "") -> str:
    if zone in ("PAÇA", "ETEK UCU"):
        return "Tasarım genişliği (serbest)"
    if ease_cm < -2.0:
        return "Dar / Gergin"
    if ease_cm < 2.0:
        return "Tam / Saran"
    loose_lim = {"skinny": 6.0, "slim": 10.0, "regular": 16.0, "oversize": 32.0}.get(fit_key, 16.0)
    if ease_cm <= loose_lim * 0.55:
        return "Rahat"
    if ease_cm <= loose_lim:
        return "Bol (kalıba uygun)"
    return "Çok Bol"


class FitHeatmapNode:
    def __init__(self):
        self.last_heatmap_stats: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------
    @staticmethod
    def get_stretch_tolerance_cm(fabric_stretch: str = "Az Esnek (%2-3 Likra)"):
        tol = CategoryRouter.stretch_tolerance_cm(fabric_stretch)
        names = {10.0: "%10+ Yüksek Likra", 6.0: "%5-8 Orta Elastan", 3.0: "%2-3 Az Esnek", 0.0: "%0 Esnemez (Dokuma)"}
        return tol, names.get(tol, "")

    @staticmethod
    def _row_extents(mask_bool: np.ndarray):
        h, w = mask_bool.shape[:2]
        has = mask_bool.any(axis=1)
        L = np.where(has, np.argmax(mask_bool, axis=1), 0).astype(np.float32)
        R = np.where(has, w - 1 - np.argmax(mask_bool[:, ::-1], axis=1), 0).astype(np.float32)
        return has, L, R

    @staticmethod
    def _smooth1d(v: np.ndarray, k: int) -> np.ndarray:
        k = max(3, int(k) | 1)
        return cv2.GaussianBlur(v.reshape(-1, 1).astype(np.float32), (1, k), k * 0.25).ravel()

    @staticmethod
    def _draw_contour_layer(mask_u8: np.ndarray, color_bgr, thickness: int, h: int, w: int, dashed: bool = False) -> np.ndarray:
        """Alt-piksel yumuşatılmış kontur katmanı (BGR + alfa)."""
        layer = np.zeros((h, w, 3), dtype=np.uint8)
        alpha = np.zeros((h, w), dtype=np.uint8)
        contours, _ = cv2.findContours(np.where(mask_u8 > 127, 255, 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if len(cnt) < 25:
                continue
            pts = cnt[:, 0, :].astype(np.float32)
            pad = min(20, len(pts) // 2)
            pts_pad = np.vstack([pts[-pad:], pts, pts[:pad]])
            k = 15
            pts_s = cv2.GaussianBlur(pts_pad.reshape(-1, 1, 2), (1, k), 3.0).reshape(-1, 2)[pad:-pad]
            p8 = np.round(pts_s * 8.0).astype(np.int32)
            if not dashed:
                cv2.polylines(layer, [p8.reshape(-1, 1, 2)], True, color_bgr, thickness, cv2.LINE_AA, shift=3)
                cv2.polylines(alpha, [p8.reshape(-1, 1, 2)], True, 255, thickness, cv2.LINE_AA, shift=3)
            else:
                step, dash = 22, 13
                for s in range(0, len(p8), step):
                    seg = p8[s : s + dash]
                    if len(seg) >= 2:
                        cv2.polylines(layer, [seg.reshape(-1, 1, 2)], False, color_bgr, thickness, cv2.LINE_AA, shift=3)
                        cv2.polylines(alpha, [seg.reshape(-1, 1, 2)], False, 255, thickness, cv2.LINE_AA, shift=3)
        return np.dstack([layer, alpha])

    @staticmethod
    def _blend_layer(base: np.ndarray, layer_bgra: np.ndarray, clip_mask: Optional[np.ndarray] = None, opacity: float = 1.0) -> np.ndarray:
        a = layer_bgra[:, :, 3].astype(np.float32) / 255.0 * opacity
        if clip_mask is not None:
            a = a * clip_mask.astype(np.float32)
        a = a[..., None]
        return np.clip(base.astype(np.float32) * (1.0 - a) + layer_bgra[:, :, :3].astype(np.float32) * a, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # 1) Gerçek vücut silüeti
    # ------------------------------------------------------------------
    def _real_body_mask(
        self,
        user_original_bgr: np.ndarray,
        out_shape: Tuple[int, int],
        zone_rows: List[float],
        body_half_px: List[float],
    ) -> np.ndarray:
        h, w = out_shape
        src = user_original_bgr
        if src.shape[:2] != (h, w):
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
        seg = get_global_models().get_semantic_map(src)
        # İnsan gövdesi: kıyafetler, yüz, bacaklar, kollar, kemer, atkı (saç/şapka/çanta/arka plan/ayakkabı hariç)
        body = np.isin(seg, [4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 17])
        arms = np.isin(seg, [14, 15])
        torso = body & ~arms

        if len(zone_rows) >= 2:
            ys = np.arange(h, dtype=np.float32)
            order = np.argsort(zone_rows)
            zr = np.asarray(zone_rows, dtype=np.float32)[order]
            zh = np.asarray(body_half_px, dtype=np.float32)[order]
            half_allow = np.interp(ys, zr, zh) * 1.12 + 3.0
            band = (ys >= zr[0] - 10) & (ys <= zr[-1] + 10)
            has, L, R = self._row_extents(torso)
            cx = self._smooth1d(np.where(has, 0.5 * (L + R), w * 0.5), 31)
            xx = np.arange(w, dtype=np.float32)[None, :]
            inside = np.abs(xx - cx[:, None]) <= half_allow[:, None]
            torso = np.where(band[:, None], torso & inside, torso)

        body = torso | arms
        body_u8 = np.where(body, 255, 0).astype(np.uint8)
        body_u8 = cv2.morphologyEx(body_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        body_u8 = cv2.GaussianBlur(body_u8, (7, 7), 2.0)
        return np.where(body_u8 >= 128, 255, 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # 2) Bölgesel bolluk profili
    # ------------------------------------------------------------------
    @staticmethod
    def _zones(route, landmarks, h: int, w: int, em: Dict[str, float], um: Dict[str, float], stretch_tol: float, orientation: str):
        lm = landmarks.landmark
        sh_y = 0.5 * (lm[11].y + lm[12].y) * h
        hip_y = 0.5 * (lm[23].y + lm[24].y) * h
        knee_y = 0.5 * (lm[25].y + lm[26].y) * h
        ank_y = 0.5 * (lm[27].y + lm[28].y) * h
        T = max(hip_y - sh_y, 40.0)
        cm_px = float(em.get("user_cm_per_pixel", 0.1))
        side = orientation == "side"

        def half_px(part: str) -> float:
            key = f"user_{part}_depth_cm" if side else f"user_{part}_2d_cm"
            return 0.5 * float(em.get(key, 30.0)) / max(cm_px, 1e-4)

        u_ch, u_wa, u_hi = float(um.get("chest", 92)), float(um.get("waist", 74)), float(um.get("hips", 96))
        g_ch = float(em.get("garment_chest_3d_cm", u_ch + 8))
        g_wa = float(em.get("garment_waist_3d_cm", u_wa + 8))
        g_hi = float(em.get("garment_hips_3d_cm", u_hi + 8))
        tol = stretch_tol * 0.5

        def eff(g: float, u: float) -> float:
            d = g - u
            return d + tol if d < 0 else d  # esneklik yalnızca darlığı tolere eder

        cat = route.canonical_category
        zones = []  # (ad, y, ease_eff_cm, raw_diff_cm, body_half_px)
        if cat == "lower_body":
            zones.append(("BEL", hip_y - 0.25 * T, eff(g_wa, u_wa), g_wa - u_wa, half_px("waist")))
            zones.append(("KALÇA", hip_y + 0.05 * T, eff(g_hi, u_hi), g_hi - u_hi, half_px("hips")))
            if route.sub_category == "pants":
                for nm, y, gk, uk in (
                    ("UYLUK", hip_y + 0.30 * T, "garment_thigh_3d_cm", "user_thigh_3d_cm"),
                    ("DİZ", knee_y, "garment_knee_3d_cm", "user_knee_3d_cm"),
                    ("PAÇA", ank_y - 0.08 * T, "garment_hem_3d_cm", "user_ankle_3d_cm"),
                ):
                    g, u = float(em.get(gk, 0.0)), float(em.get(uk, 0.0))
                    if g > 0 and u > 0:
                        e = eff(g, u)
                        if nm == "PAÇA":
                            e = min(e, 12.0)  # paça ağzı tasarım ölçüsüdür, konforu belirlemez
                        zones.append((nm, y, e, g - u, 0.0))
            else:
                zones.append(("ETEK UCU", knee_y, min(max(eff(g_hi, u_hi), 0.0) * 1.3 + 4.0, 12.0), (g_hi - u_hi) * 1.3 + 4.0, 0.0))
        else:
            zones.append(("GÖĞÜS", sh_y + 0.26 * T, eff(g_ch, u_ch), g_ch - u_ch, half_px("chest")))
            zones.append(("BEL", sh_y + 0.62 * T, eff(g_wa, u_wa), g_wa - u_wa, half_px("waist")))
            zones.append(("KALÇA", hip_y + 0.08 * T, eff(g_hi, u_hi), g_hi - u_hi, half_px("hips")))
            if cat == "full_body":
                zones.append(("ETEK UCU", knee_y, min(max(eff(g_hi, u_hi), 0.0) * 1.3 + 4.0, 12.0), (g_hi - u_hi) * 1.3 + 4.0, 0.0))
        return zones, (sh_y, hip_y, knee_y, ank_y, T)

    # ------------------------------------------------------------------
    # Ana fonksiyon
    # ------------------------------------------------------------------
    def apply_heatmap_overlay(
        self,
        image: np.ndarray,
        body_tensor: dict,
        garment_width_cm: float,
        user_measurements: dict,
        garment_type: str = "Üst Giyim",
        user_original_bgr=None,
        segformer_node=None,
        landmarks=None,
        warped_garment_mask=None,
        fit_type: str = "Oversize / Rahat Kalıp",
        fabric_stretch: str = "Az Esnek (%2-3 Likra)",
        desired_size: str = "M",
        scale_spec=None,
        gender: str = "Kadın",
        orientation: str = "front",
        **_ignored,
    ) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        route = CategoryRouter.resolve_route(garment_type)
        em = (getattr(scale_spec, "extra_metrics", None) or {}) if scale_spec is not None else {}
        um = user_measurements or {}
        fit_key = CategoryRouter.fit_key_of(fit_type)
        stretch_tol, stretch_name = self.get_stretch_tolerance_cm(fabric_stretch)
        if "user_cm_per_pixel" not in em:
            ppc = float((body_tensor or {}).get("px_per_cm", 0.0)) or (w / 140.0)
            em = dict(em, user_cm_per_pixel=1.0 / ppc)

        # --- Kıyafet maskesi ---
        if warped_garment_mask is not None and np.count_nonzero(np.asarray(warped_garment_mask) > 127) > 400:
            gmask = np.where(np.asarray(warped_garment_mask) > 127, 255, 0).astype(np.uint8)
            if gmask.shape[:2] != (h, w):
                gmask = cv2.resize(gmask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            seg_d = get_global_models().get_semantic_map(out)
            gmask = np.where(np.isin(seg_d, list(route.segformer_fallback_labels)), 255, 0).astype(np.uint8)
        gmask = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        G = gmask > 0

        if landmarks is None or not hasattr(landmarks, "landmark") or not np.any(G):
            self.last_heatmap_stats = {}
            return out

        zones, (sh_y, hip_y, knee_y, ank_y, T) = self._zones(route, landmarks, h, w, em, um, stretch_tol, orientation)

        # --- Gerçek vücut ---
        torso_zone_rows = [z[1] for z in zones if z[4] > 0]
        torso_halves = [z[4] for z in zones if z[4] > 0]
        if user_original_bgr is not None:
            body_u8 = self._real_body_mask(user_original_bgr, (h, w), torso_zone_rows, torso_halves)
        else:
            body_u8 = np.zeros((h, w), np.uint8)
        B = body_u8 > 0

        # --- Satır bazlı bolluk profili ---
        ys = np.arange(h, dtype=np.float32)
        zy = np.array([z[1] for z in zones], dtype=np.float32)
        ze = np.array([z[2] for z in zones], dtype=np.float32)
        order = np.argsort(zy)
        ease_row = np.interp(ys, zy[order], ze[order]).astype(np.float32)
        ease_row = self._smooth1d(ease_row, max(15, int(T * 0.15)))

        # --- Boyama ---
        row_col = ease_to_bgr(ease_row)  # (h,3)
        color_layer = np.broadcast_to(row_col[:, None, :], (h, w, 3)).astype(np.float32)

        gap = G & ~B
        dist_in_body = cv2.distanceTransform(np.where(B, 255, 0).astype(np.uint8), cv2.DIST_L2, 5)
        tight_strength = np.clip(-ease_row / 6.0, 0.0, 1.0)[:, None]
        edge_w = np.exp(-dist_in_body / max(6.0, w * 0.012))
        alpha = np.where(G, 0.40, 0.0).astype(np.float32)
        alpha = alpha + np.where(G & B, 0.30 * tight_strength * edge_w, 0.0)

        # Hava boşluğu: diyagonal tarama
        xx, yy = np.meshgrid(np.arange(w, dtype=np.int32), np.arange(h, dtype=np.int32))
        period = max(10, int(w * 0.009))
        hatch = (((xx + yy) % period) < max(2, period // 3)).astype(np.float32)
        alpha = alpha + np.where(gap, 0.10 + 0.22 * hatch, 0.0)
        alpha = np.clip(alpha, 0.0, 0.75)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 1.2) * cv2.GaussianBlur(G.astype(np.float32), (3, 3), 0.8)
        blended = np.clip(
            out.astype(np.float32) * (1.0 - alpha[..., None]) + color_layer * alpha[..., None], 0, 255
        ).astype(np.uint8)

        # --- Konturlar ---
        th = max(2, int(round(w * 0.0022)))
        g_layer = self._draw_contour_layer(gmask, (255, 255, 255), th, h, w)
        blended = self._blend_layer(blended, g_layer, opacity=0.85)
        if np.any(B):
            clip = cv2.dilate(gmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * th + 3, 2 * th + 3))) > 0
            b_layer = self._draw_contour_layer(body_u8, (255, 200, 30), th + 1, h, w)
            blended = self._blend_layer(blended, b_layer, clip_mask=clip, opacity=1.0)

        # --- İstatistikler ---
        recommended, size_costs = CategoryRouter().recommend_size(um, gender, garment_type, fabric_stretch)
        weights = {"GÖĞÜS": 1.0, "BEL": 0.6, "KALÇA": 0.8, "UYLUK": 0.7, "DİZ": 0.4, "PAÇA": 0.2, "ETEK UCU": 0.2}
        loose_lim = {"skinny": 6.0, "slim": 10.0, "regular": 16.0, "oversize": 32.0}[fit_key]
        penalty = 0.0
        for nm, _, e_eff, _, _ in zones:
            wgt = weights.get(nm, 0.5)
            penalty += wgt * (6.0 * max(0.0, -e_eff) + 1.0 * max(0.0, e_eff - loose_lim))
        comfort = int(np.clip(100.0 - penalty, 15.0, 100.0))

        zone_telemetry = {}
        callouts = []
        for nm, y, e_eff, raw, _ in zones:
            yi = int(np.clip(round(y), 0, h - 1))
            g_row = G[yi]
            b_row = B[yi] & g_row
            g_px = int(np.count_nonzero(g_row))
            b_px = int(np.count_nonzero(b_row))
            zone_telemetry[nm] = {
                "y": yi,
                "body_px": b_px,
                "garment_px": g_px,
                "ease_px": g_px - b_px,
                "ease_cm": round(float(raw), 1),
                "effective_ease_cm": round(float(e_eff), 1),
                "status": ease_label(e_eff, fit_key, nm),
            }
            if g_px > 0:
                xs_g = np.where(g_row)[0]
                callouts.append((nm, yi, int(xs_g.max()), float(raw), float(e_eff)))

        self.last_heatmap_stats = {
            "comfort_score": comfort,
            "ideal_size": recommended,
            "size_costs": size_costs,
            "body_mask_px": int(np.count_nonzero(B)),
            "garment_mask_px": int(np.count_nonzero(G)),
            "overhang_px": int(np.count_nonzero(gap)),
            "zones": zone_telemetry,
        }

        return self._annotate(
            blended, callouts, fit_key, fit_type, desired_size, recommended, comfort, stretch_name
        )

    # ------------------------------------------------------------------
    # Etiketler, renk skalası, özet kartı (PIL ile Türkçe karakter desteği)
    # ------------------------------------------------------------------
    def _annotate(self, img, callouts, fit_key, fit_type, desired_size, recommended, comfort, stretch_name):
        h, w = img.shape[:2]
        s = max(0.55, min(w / 1400.0, 1.6))
        f_title = _font(int(26 * s), True)
        f_txt = _font(int(21 * s), False)
        f_lab = _font(int(22 * s), True)
        f_small = _font(int(17 * s), False)
        if f_txt is None or Image is None:
            return img

        # Kartlar ve çizgiler OpenCV ile (yarı saydam), metin PIL ile
        overlay = img.copy()
        texts = []

        # Bölge etiketleri: sağ kenar boşluğuna dikey yığın
        box_w = int(300 * s)
        box_h = int(58 * s)
        x_box = w - box_w - int(16 * s)
        min_gap = box_h + int(8 * s)
        last_y = -1e9
        for nm, yi, x_edge, raw, e_eff in sorted(callouts, key=lambda c: c[1]):
            y_box = max(yi - box_h // 2, last_y + min_gap)
            y_box = int(min(y_box, h - box_h - 4))
            last_y = y_box
            col = tuple(int(c) for c in ease_to_bgr(np.array([e_eff]))[0])
            anchor = (int(x_edge), int(yi))
            cv2.circle(overlay, anchor, int(7 * s), col, -1, cv2.LINE_AA)
            cv2.circle(overlay, anchor, int(7 * s), (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
            if x_box > x_edge + 20:
                cv2.line(overlay, anchor, (x_box, y_box + box_h // 2), (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
            cv2.rectangle(overlay, (x_box, y_box), (x_box + box_w, y_box + box_h), (25, 25, 25), -1)
            cv2.rectangle(overlay, (x_box, y_box), (x_box + int(8 * s), y_box + box_h), col, -1)
            texts.append(((x_box + int(16 * s), y_box + int(5 * s)), f"{nm}  {raw:+.1f} cm", f_lab, (255, 255, 255)))
            texts.append(((x_box + int(16 * s), y_box + int(31 * s)), ease_label(e_eff, fit_key, nm), f_small, (220, 220, 220)))

        # Özet kartı (sol üst)
        pad = int(16 * s)
        card_w, card_h = int(430 * s), int(150 * s)
        cv2.rectangle(overlay, (pad, pad), (pad + card_w, pad + card_h), (25, 25, 25), -1)
        fit_short = str(fit_type).split("/")[0].strip()
        ok_size = str(desired_size).upper() == recommended
        texts.append(((pad + int(14 * s), pad + int(10 * s)), "Uyum Analizi", f_title, (255, 255, 255)))
        texts.append((
            (pad + int(14 * s), pad + int(48 * s)),
            f"Seçilen: {str(desired_size).upper()}   ·   Önerilen: {recommended}",
            f_txt,
            (140, 235, 140) if ok_size else (120, 200, 255),
        ))
        texts.append(((pad + int(14 * s), pad + int(80 * s)), f"Kalıp: {fit_short}   ·   Kumaş: {stretch_name}", f_small, (220, 220, 220)))
        texts.append(((pad + int(14 * s), pad + int(110 * s)), f"Hareket konforu: {comfort}/100", f_txt, (255, 255, 255)))

        # Renk skalası (sol alt)
        bar_h = int(min(h * 0.32, 380 * s))
        bar_w = int(22 * s)
        bx = pad
        by = h - bar_h - int(60 * s)
        e_vals = np.linspace(20.0, -8.0, bar_h, dtype=np.float32)
        bar = np.repeat(ease_to_bgr(e_vals)[:, None, :], bar_w, axis=1).astype(np.uint8)
        cv2.rectangle(overlay, (bx - int(8 * s), by - int(44 * s)), (bx + int(190 * s), by + bar_h + int(14 * s)), (25, 25, 25), -1)
        overlay[by : by + bar_h, bx : bx + bar_w] = bar
        texts.append(((bx, by - int(38 * s)), "Bolluk (cm)", f_small, (255, 255, 255)))
        for tv, tl in ((20, "+20  çok bol"), (10, "+10  bol"), (3, "+3  rahat"), (0, "0  tam oturur"), (-4, "-4  dar")):
            ty = by + int((20.0 - tv) / 28.0 * bar_h)
            cv2.line(overlay, (bx + bar_w, ty), (bx + bar_w + int(6 * s), ty), (255, 255, 255), 1, cv2.LINE_AA)
            texts.append(((bx + bar_w + int(10 * s), ty - int(10 * s)), tl, f_small, (235, 235, 235)))

        # Açıklama (alt)
        texts.append((
            (bx, h - int(36 * s)),
            "Mavi çizgi: gerçek vücut sınırı   ·   Taralı alan: kumaş–vücut arası boşluk",
            f_small,
            (255, 255, 255),
        ))
        tw = int(720 * s)
        cv2.rectangle(overlay, (bx - int(8 * s), h - int(42 * s)), (min(w - 1, bx + tw), h - int(8 * s)), (25, 25, 25), -1)

        img = cv2.addWeighted(overlay, 0.82, img, 0.18, 0)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for (x, y), txt, fnt, col_bgr in texts:
            draw.text((int(x), int(y)), txt, font=fnt, fill=(int(col_bgr[2]), int(col_bgr[1]), int(col_bgr[0])))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
