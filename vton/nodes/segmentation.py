"""
Node-01: Semantic Garment & Flat-Lay Segmentation (Global Singleton & Category-Routed).

Resolves:
- Directive 1 (Latency Optimization): Uses globally initialized SegFormer, U2Net (rembg),
  and MediaPipe Pose from `GlobalModelRegistry` mapped to CUDA, with 100% vectorized
  NumPy masking (`np.isin`, `np.where`) and zero Python pixel loops.
- Directive 2 (Dynamic Category Routing): Routes 'upper_body', 'lower_body' (pants/skirt),
  and 'full_body' (dress) directly via `CategoryRouter`.
"""

import os
import cv2
import numpy as np
import rembg
from PIL import Image

from ..category_router import CategoryRouter, get_global_models


class SegFormerNode:
    def __init__(self, model_id: str = "mattmdjaga/segformer_b2_clothes"):
        models = get_global_models()
        self.processor = models.segformer_processor
        self.model = models.segformer_model
        self.device = models.device
        self.mp_pose = models.mp_pose
        self.pose = models.pose_detector
        self.rembg_session = models.u2net_session
        self.garment_class_map = {
            "Üst Giyim": [4],
            "Üst Giyim (Tişört / Gömlek / Kazak / Ceket)": [4],
            "Alt Giyim": [5, 6],
            "Alt Giyim - Pantolon": [6, 5],
            "Alt Giyim - Pantolon / Jean / Eşofman": [6, 5],
            "Alt Giyim - Etek": [5, 6],
            "Alt Giyim - Etek / Şort": [5, 6],
            "Elbise": [7, 4, 5],
            "Elbise / Tulum": [7, 4, 5, 6],
            "upper_body": [4],
            "lower_body": [5, 6],
            "full_body": [7],
        }

    def resolve_target_classes(self, garment_type: str):
        if garment_type in self.garment_class_map:
            return self.garment_class_map[garment_type]
        return CategoryRouter.get_segformer_target_labels(garment_type, include_fallback=False)

    def _is_tshirt_flatlay_mask(self, binary_mask: np.ndarray, has_wood_hanger: bool = False) -> bool:
        """Detect whether a flat-lay foreground mask has T-shirt/shirt geometry (wide top sleeves, single bottom hem)."""
        ys, xs = np.where(binary_mask > 127)
        if len(ys) < 100:
            return False
        y0, y1 = int(np.min(ys)), int(np.max(ys))
        x0, x1 = int(np.min(xs)), int(np.max(xs))
        bh = max(1, y1 - y0)
        bw = max(1, x1 - x0)
        crop = binary_mask[y0:y1 + 1, x0:x1 + 1] > 127

        # Measure width in upper sleeve zone (20%-45%) vs lower torso zone (65%-90%)
        u_slice = crop[int(bh * 0.20):max(int(bh * 0.20) + 1, int(bh * 0.45)), :]
        l_slice = crop[int(bh * 0.65):max(int(bh * 0.65) + 1, int(bh * 0.90)), :]
        u_widths = np.sum(u_slice, axis=1)
        l_widths = np.sum(l_slice, axis=1)
        u_w = float(np.percentile(u_widths, 90)) if len(u_widths) > 0 else float(bw)
        l_w = float(np.percentile(l_widths, 75)) if len(l_widths) > 0 else float(bw)

        # Check if lower region has a central split between two pant legs
        mid_x = bw // 2
        band_w = max(2, int(bw * 0.08))
        lower_center = crop[int(bh * 0.60):int(bh * 0.92), max(0, mid_x - band_w):min(bw, mid_x + band_w)]
        has_crotch_split = float(np.mean(lower_center)) < 0.35 if lower_center.size > 0 else False

        if has_crotch_split:
            return False
        if has_wood_hanger and (u_w >= l_w * 1.05):
            return True
        if (u_w >= l_w * 1.18) and (bh / float(bw) < 1.35):
            return True
        return False

    def _synthesize_template_from_fabric(self, rgba_crop: np.ndarray, route_key: str) -> tuple[np.ndarray, np.ndarray]:
        """
        When an upper-body flat-lay (e.g. T-shirt) is inadvertently passed for a skirt/pants/dress target,
        extract its clean central torso fabric (stripping out sleeves) and shape it into an anatomical
        category template so sleeves are never warped onto legs.
        """
        h, w = rgba_crop.shape[:2]
        # Take clean central fabric swatch (30%..85% height, 28%..72% width)
        y0, y1 = int(h * 0.28), int(h * 0.88)
        x0, x1 = int(w * 0.28), int(w * 0.72)
        swatch = rgba_crop[y0:y1, x0:x1].copy()
        if swatch.size == 0 or np.sum(swatch[:, :, 3] > 30) < 100:
            swatch = rgba_crop.copy()

        out_h, out_w = 700, 460
        tex_bgr = cv2.resize(swatch[:, :, :3], (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        mask = np.zeros((out_h, out_w), dtype=np.uint8)

        yy = np.arange(out_h, dtype=np.float32)
        t = yy / float(max(1, out_h - 1))
        xx = np.arange(out_w, dtype=np.float32)[None, :]

        if route_key == "lower_body_pants":
            # Waistband -> Hips -> Crotch split (t=0.28) -> Left & Right pant legs
            half_w = np.interp(t, [0.0, 0.22, 0.28, 1.0], [out_w * 0.34, out_w * 0.42, out_w * 0.43, out_w * 0.44])
            mid = out_w * 0.5
            in_outer = (xx >= (mid - half_w)[:, None]) & (xx <= (mid + half_w)[:, None])
            # Crotch gap below t = 0.28
            gap_half = np.where(t < 0.28, 0.0, np.interp(t, [0.28, 0.45, 1.0], [2.0, out_w * 0.06, out_w * 0.09]))
            in_gap = (t[:, None] >= 0.28) & (xx > (mid - gap_half)[:, None]) & (xx < (mid + gap_half)[:, None])
            valid = in_outer & (~in_gap)
            mask = np.where(valid, 255, 0).astype(np.uint8)
        elif route_key == "lower_body_skirt":
            # A-line skirt silhouette
            half_w = np.interp(t, [0.0, 0.25, 1.0], [out_w * 0.30, out_w * 0.38, out_w * 0.46])
            mid = out_w * 0.5
            valid = (xx >= (mid - half_w)[:, None]) & (xx <= (mid + half_w)[:, None])
            mask = np.where(valid, 255, 0).astype(np.uint8)
        else:
            # Full-body dress silhouette
            half_w = np.interp(t, [0.0, 0.18, 0.38, 0.65, 1.0], [out_w * 0.36, out_w * 0.34, out_w * 0.29, out_w * 0.40, out_w * 0.46])
            mid = out_w * 0.5
            valid = (xx >= (mid - half_w)[:, None]) & (xx <= (mid + half_w)[:, None])
            mask = np.where(valid, 255, 0).astype(np.uint8)

        mask = cv2.GaussianBlur(mask, (5, 5), 1.2)
        out_rgba = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2BGRA)
        out_rgba[:, :, 3] = mask
        return out_rgba, mask

    def process_image(self, image_path: str, garment_type: str = "Üst Giyim"):
        """
        Segment only the target garment category from `image_path` using globally loaded
        MediaPipe + SegFormer + U2Net, with automatic category verification and
        stripped anatomy (never including shoes, bare legs, bare arms, or T-shirt sleeves on pants).
        """
        print(f"[Node-01] SegFormer Görüntü Analizi Başladı: {image_path} (Hedef: {garment_type})")

        image = Image.open(image_path).convert("RGB")
        image_rgb = np.array(image)
        image_cv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = image_cv.shape[:2]
        total_px = float(max(1, h_img * w_img))

        models = get_global_models()
        route = CategoryRouter.get_route(garment_type)
        target_classes = self.resolve_target_classes(garment_type)
        pred_seg = models.get_semantic_map(image_cv)

        pose_results = models.detect_pose(image_rgb, is_rgb=True)
        has_pose = pose_results.pose_landmarks is not None

        # Detect wooden hanger in upper 25% of flat-lay image
        top_h_chk = int(h_img * 0.25)
        hsv_top = cv2.cvtColor(image_cv[:top_h_chk, :, :3], cv2.COLOR_BGR2HSV)
        is_wood_top = (hsv_top[:, :, 0] >= 5) & (hsv_top[:, :, 0] <= 35) & (hsv_top[:, :, 1] >= 35)
        has_wood_hanger_top = int(np.sum(is_wood_top)) > int(total_px * 0.002)

        # Detect human anatomy classes (face, shoes, bare legs) for cropped e-commerce photos
        anatomy_px = int(np.sum(np.isin(pred_seg, [9, 10, 11, 12, 13])))
        has_upper_px = int(np.sum(pred_seg == 4)) > int(total_px * 0.02)
        has_lower_px = int(np.sum(np.isin(pred_seg, [5, 6]))) > int(total_px * 0.02)
        is_person = has_pose or (
            not has_wood_hanger_top
            and ((anatomy_px > int(total_px * 0.015)) or (has_upper_px and has_lower_px))
        )

        # Check for category mismatch when user selects lower_body (Pants/Skirt) but default `on.jpg` is a flat-lay T-shirt
        target_seg_px = int(np.sum(np.isin(pred_seg, target_classes)))
        used_ref_fallback = False
        if route.category == "lower_body" and target_seg_px < int(total_px * 0.005):
            # Check if olcu_ref.jpg in the same directory has an on-model photo wearing the target lower_body category
            ref_candidate = os.path.join(os.path.dirname(os.path.abspath(image_path)), "olcu_ref.jpg")
            base_name = os.path.basename(image_path).lower()
            if (
                route.key == "lower_body_pants"
                and os.path.exists(ref_candidate)
                and base_name in ("on.jpg", "arka.jpg", "yan.jpg")
            ):
                ref_img = cv2.imread(ref_candidate)
                if ref_img is not None:
                    ref_seg = models.get_semantic_map(ref_img)
                    ref_pants_px = int(np.sum(np.isin(ref_seg, [6, 5])))
                    if ref_pants_px > int(ref_img.shape[0] * ref_img.shape[1] * 0.02):
                        print(f"[Node-01] '{base_name}' içinde alt giyim bulunamadı; '{ref_candidate}' içindeki gerçek pantolon segmenti kullanılıyor.")
                        image_cv = ref_img
                        pred_seg = ref_seg
                        h_img, w_img = image_cv.shape[:2]
                        total_px = float(max(1, h_img * w_img))
                        target_seg_px = ref_pants_px
                        is_person = True
                        used_ref_fallback = True

        if not is_person:
            print("[Node-01] Manken bulunamadı (Flat-Lay). Global U2Net ile tüm obje ayıklanıyor...")
            rgba_image = rembg.remove(image_cv, session=self.rembg_session)
            rgba_image = np.array(rgba_image, copy=True)

            # Vectorized wooden/metal hanger removal in upper 25% of flat-lay image
            h, w = rgba_image.shape[:2]
            top_h = int(h * 0.25)
            hsv = cv2.cvtColor(rgba_image[:top_h, :, :3], cv2.COLOR_BGR2HSV)
            is_wood = (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] >= 30)
            y_wood = np.where(is_wood)[0]
            has_wood_hanger = len(y_wood) > 25 or has_wood_hanger_top
            if len(y_wood) > 0:
                max_y = min(top_h - 1, int(np.percentile(y_wood, 95)))
                rgba_image[: max_y + 8, :, 3] = np.where(
                    is_wood[: max_y + 8, :], 0, rgba_image[: max_y + 8, :, 3]
                )
                mid_x = w // 2
                rgba_image[: max_y + 8, mid_x - int(w * 0.08) : mid_x + int(w * 0.08), 3] = 0

            # Strip any residual human anatomy/shoes if SegFormer spotted any
            non_garment = np.isin(pred_seg, [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16])
            rgba_image[:, :, 3] = np.where(non_garment, 0, rgba_image[:, :, 3])

            y_indices, x_indices = np.where(rgba_image[:, :, 3] > 20)
            if len(y_indices) > 0:
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                cropped_garment = rgba_image[y_min : y_max + 1, x_min : x_max + 1]
                binary_mask = np.where(rgba_image[:, :, 3] > 20, 255, 0).astype(np.uint8)

                # Guard against warping T-shirt sleeves onto legs/skirts when a T-shirt flat-lay is loaded
                if route.category in ("lower_body", "full_body") and (
                    target_seg_px < int(total_px * 0.005)
                    or self._is_tshirt_flatlay_mask(binary_mask, has_wood_hanger)
                ):
                    print(f"[Node-01] UYARI: Üst giyim (T-shirt) flat-lay görseli '{route.key}' kategorisiyle seçildi. Kollar kesilerek kategori şablonu oluşturuluyor...")
                    return self._synthesize_template_from_fabric(cropped_garment, route.key)

                return cropped_garment, binary_mask
            return None, None

        print("[Node-01] Manken algılandı. Global SegFormer ile kategoriye duyarlı segmentasyon yapılıyor...")
        # Strictly exclude non-garment anatomy & footwear classes [1,2,3,9,10,11,12,13,14,15,16]
        mask = np.where(np.isin(pred_seg, target_classes), 255, 0).astype(np.uint8)

        if not np.any(mask):
            fallback_classes = CategoryRouter.get_segformer_target_labels(garment_type, include_fallback=True)
            mask = np.where(np.isin(pred_seg, fallback_classes), 255, 0).astype(np.uint8)
        if not np.any(mask):
            mask = np.where(np.isin(pred_seg, [4, 5, 6, 7]), 255, 0).astype(np.uint8)
        if not np.any(mask):
            print(f"[Node-01] UYARI: {garment_type} bulunamadı! SegFormer başarısız.")
            return None, None

        # Remove disconnected small noise components while keeping both pant legs if separated
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = int(np.max(areas))
            keep_ids = np.where(areas >= max(64, int(max_area * 0.12)))[0] + 1
            mask = np.where(np.isin(labels, keep_ids), 255, 0).astype(np.uint8)

        # Anti-alias extracted garment alpha edges (Directive 5)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask_smooth = cv2.GaussianBlur(mask, (5, 5), 1.2)

        rgba_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2BGRA)
        rgba_image[:, :, 3] = mask_smooth

        y_indices, x_indices = np.where(mask_smooth > 10)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        cropped_garment = rgba_image[y_min : y_max + 1, x_min : x_max + 1]

        if (
            route.category in ("lower_body", "full_body")
            and not used_ref_fallback
            and target_seg_px < int(total_px * 0.005)
        ):
            print(f"[Node-01] UYARI: Görselde '{route.key}' bulunamadı (sadece üst giyim mevcut). Kumaş dokusundan '{route.key}' şablonu oluşturuluyor...")
            return self._synthesize_template_from_fabric(cropped_garment, route.key)

        return cropped_garment, mask


if __name__ == "__main__":
    from vton.paths import goruntu

    node = SegFormerNode()
    if os.path.exists(goruntu("temp_garment.jpg")):
        result, _ = node.process_image(goruntu("temp_garment.jpg"), "Üst Giyim")
        if result is not None:
            cv2.imwrite(goruntu("test_segformer_output.png"), result)
            print("[TEST] Çıktı goruntu/test_segformer_output.png olarak kaydedildi.")
