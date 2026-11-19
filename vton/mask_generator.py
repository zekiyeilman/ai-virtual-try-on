"""
Category-Conditional Anatomical Z-Indexing, Orientation-Aware (Front/Side/Back) Masking,
Thin-Plate Spline (TPS) Garment Warping, and Anti-Aliased Mask Generation Engine.

Resolves:
- Lower-Body Anchor (Solution 1): Locks the top-center anchor of lower-body garments
  (pants/skirts) strictly to the midpoint of MediaPipe landmarks 23 & 24 in (X, Y) space,
  stretching strictly downward toward knees (25, 26) and ankles (27, 28), while covering
  any high-waist legging remnants and eliminating side-profile inter-leg gaps.
- Orientation Directives & Back Solid Block Mode (Solution 2): Integrates
  `CategoryRouter.get_masking_directives(orientation, category)`. Only carves `face_neck`
  and `chest` when `orientation == "front"`. When `orientation == "back"`, disables all
  SegFormer skin/background intersections and uses a solid, uninterrupted garment block.
- 3D Cylindrical Shading & Edge Decontamination: Replaces raw `user_gray` blurring (which
  caused dark racerback bra ghosting) with pure 3D cylindrical surface shading, decontaminates
  transparent garment borders (eliminating black shoulder triangles and white halos), and
  prevents double chest prints in side view.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from vton.category_router import (
    AsymmetricScaleSpec,
    CategoryRouter,
    SEG_LABELS,
    get_global_models,
)


# ---------------------------------------------------------------------------
# Thin-Plate Spline (TPS) 2D Non-Rigid Warping Solver
# ---------------------------------------------------------------------------
class ThinPlateSplineWarper:
    """
    100% vectorized Radial Basis Function Thin-Plate Spline (TPS) 2D warper.
    """

    @staticmethod
    def _tps_kernel(r_sq: np.ndarray) -> np.ndarray:
        return np.where(r_sq > 1e-10, 0.5 * r_sq * np.log(np.maximum(r_sq, 1e-10)), 0.0)

    @classmethod
    def solve_tps_weights(
        cls,
        dst_pts: np.ndarray,
        src_pts: np.ndarray,
        regularization: float = 1e-3,
    ) -> np.ndarray:
        n = dst_pts.shape[0]
        diff = dst_pts[:, None, :] - dst_pts[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        K = cls._tps_kernel(dist_sq)
        K[np.diag_indices(n)] += regularization

        P = np.hstack([np.ones((n, 1), dtype=np.float64), dst_pts.astype(np.float64)])
        L = np.zeros((n + 3, n + 3), dtype=np.float64)
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T

        V = np.zeros((n + 3, 2), dtype=np.float64)
        V[:n, :] = src_pts.astype(np.float64)

        try:
            W = np.linalg.solve(L, V)
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(L, V, rcond=None)[0]
        return W.astype(np.float32)

    @classmethod
    def warp_image_tps(
        cls,
        garment_bgra: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        out_shape: Tuple[int, int],
        grid_step: int = 8,
        regularization: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        out_h, out_w = out_shape
        W = cls.solve_tps_weights(dst_pts, src_pts, regularization=regularization)
        n = dst_pts.shape[0]

        gh_coarse = max(8, out_h // grid_step + 2)
        gw_coarse = max(8, out_w // grid_step + 2)
        ys = np.linspace(0.0, float(out_h - 1), gh_coarse, dtype=np.float32)
        xs = np.linspace(0.0, float(out_w - 1), gw_coarse, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

        diff = grid_pts[:, None, :] - dst_pts[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        U = cls._tps_kernel(dist_sq).astype(np.float32)

        P_grid = np.hstack([np.ones((grid_pts.shape[0], 1), dtype=np.float32), grid_pts])
        mapped = U @ W[:n, :] + P_grid @ W[n:, :]

        map_x_coarse = mapped[:, 0].reshape(gh_coarse, gw_coarse)
        map_y_coarse = mapped[:, 1].reshape(gh_coarse, gw_coarse)

        map_x = cv2.resize(map_x_coarse, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(map_y_coarse, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        warped = cv2.remap(
            garment_bgra,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return warped, map_x, map_y


# ---------------------------------------------------------------------------
# Orientation-Aware Anatomical Z-Indexing & Mask Generator
# ---------------------------------------------------------------------------
class AnatomicalMaskGenerator:
    """
    Production-grade category-conditional and orientation-aware ('front', 'side', 'back')
    mask generator, Z-index occlusion resolver, and TPS garment pre-warper.
    """

    def __init__(self, gender: str = "Kadın") -> None:
        self.router = CategoryRouter(gender=gender)

    # -----------------------------------------------------------------------
    # Adım 1: MediaPipe Omuz Landmarkları (11 ve 12) ile Yön (front, side, back) Tespiti
    # -----------------------------------------------------------------------
    @staticmethod
    def detect_pose_orientation(
        pose_landmarks,
        image_w: int = 1,
        image_h: int = 1,
        side_x_threshold: float = 0.12,
    ) -> str:
        if pose_landmarks is None:
            return "front"

        lm = pose_landmarks.landmark
        l_sh = lm[11]  # LEFT_SHOULDER
        r_sh = lm[12]  # RIGHT_SHOULDER
        l_hip = lm[23]  # LEFT_HIP
        r_hip = lm[24]  # RIGHT_HIP

        dx_norm = abs(float(l_sh.x) - float(r_sh.x))
        dx_px = dx_norm * float(image_w)

        sh_mid_y = 0.5 * (float(l_sh.y) + float(r_sh.y)) * float(image_h)
        hip_mid_y = 0.5 * (float(l_hip.y) + float(r_hip.y)) * float(image_h)
        torso_h_px = max(abs(hip_mid_y - sh_mid_y), 1.0)
        sh_torso_ratio = dx_px / torso_h_px if image_w > 1 and image_h > 1 else (dx_norm / 0.28)

        if dx_norm < side_x_threshold or sh_torso_ratio < 0.45:
            return "side"

        if float(l_sh.x) < float(r_sh.x):
            return "back"

        return "front"

    @staticmethod
    def decontaminate_garment_edges(garment_bgra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eliminate black [0,0,0] or white [255,255,255] studio background halos from
        extracted RGBA garments and return `(clean_bgra, base_color_bgr)`.
        """
        if garment_bgra.ndim == 3 and garment_bgra.shape[2] == 4:
            alpha = garment_bgra[:, :, 3]
            bgr = garment_bgra[:, :, :3].astype(np.float32)
        else:
            bgr = garment_bgra[:, :, :3].astype(np.float32)
            alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)

        gh, gw = alpha.shape[:2]
        core_mask = cv2.erode(
            np.where(alpha > 180, 255, 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        sample_roi = bgr[int(gh * 0.55) : int(gh * 0.82), int(gw * 0.30) : int(gw * 0.70)]
        sample_alpha = core_mask[int(gh * 0.55) : int(gh * 0.82), int(gw * 0.30) : int(gw * 0.70)] > 0

        if np.any(sample_alpha):
            base_color_bgr = np.median(sample_roi[sample_alpha], axis=0).astype(np.float32)
        elif np.any(core_mask > 0):
            base_color_bgr = np.median(bgr[core_mask > 0], axis=0).astype(np.float32)
        else:
            base_color_bgr = np.array([58.0, 56.0, 56.0], dtype=np.float32)

        # Replace transparent/halo pixels and bright white neck-tag artifacts with base_color_bgr
        clean_bgr = np.where(core_mask[..., None] > 0, bgr, base_color_bgr[None, None, :])

        # Remove white neck label / hanger artifacts in top 12% of garment
        top_band_h = max(4, int(gh * 0.12))
        top_lum = np.mean(clean_bgr[:top_band_h, :, :], axis=2)
        base_lum = float(np.mean(base_color_bgr))
        is_tag_or_hanger = top_lum > (base_lum + 38.0)
        clean_bgr[:top_band_h, :, :] = np.where(
            is_tag_or_hanger[..., None], base_color_bgr[None, None, :], clean_bgr[:top_band_h, :, :]
        )

        # Erode alpha slightly and smooth to prevent white fringe on pants/tops
        eroded_alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        smooth_alpha = cv2.GaussianBlur(eroded_alpha, (5, 5), 1.2)

        clean_bgra = np.dstack([np.clip(clean_bgr, 0, 255).astype(np.uint8), smooth_alpha])
        return clean_bgra, base_color_bgr

    @staticmethod
    def refine_and_antialias_mask(
        binary_mask: np.ndarray,
        close_ksize: int = 19,
        open_ksize: int = 9,
        blur_ksize: int = 9,
        blur_sigma: float = 2.4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask_u8 = np.where(binary_mask > 127, 255, 0).astype(np.uint8)

        if close_ksize > 1:
            kc = close_ksize if close_ksize % 2 == 1 else close_ksize + 1
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)

        if open_ksize > 1:
            ko = open_ksize if open_ksize % 2 == 1 else open_ksize + 1
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open)

        kb = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        feathered_u8 = cv2.GaussianBlur(mask_u8, (kb, kb), blur_sigma)
        soft_alpha = feathered_u8.astype(np.float32) / 255.0
        clean_binary = np.where(feathered_u8 >= 115, 255, 0).astype(np.uint8)
        return clean_binary, soft_alpha

    # -----------------------------------------------------------------------
    # Adım 2: Standart Maske Ayrıştırma ve Düğüm (Node) Tespit İşlemleri
    # -----------------------------------------------------------------------
    def extract_anatomical_z_masks(
        self,
        user_bgr: np.ndarray,
        seg_map: np.ndarray,
        pose_landmarks,
        orientation: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        h, w = user_bgr.shape[:2]
        if orientation is None:
            orientation = self.detect_pose_orientation(pose_landmarks, w, h)

        user_gray = cv2.cvtColor(user_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        lm_dict = self.router.extract_category_landmarks(pose_landmarks, w, h, "full_body")
        coords = lm_dict["all_coords"]
        sh_mid = lm_dict["shoulder_mid"]
        hip_mid = lm_dict["hip_mid"]
        torso_h = max(abs(float(hip_mid[1] - sh_mid[1])), 80.0)
        sh_span = max(abs(float(coords[12, 0] - coords[11, 0])), torso_h * 0.32)

        y_collar = max(6.0, float(sh_mid[1] - torso_h * (0.14 if orientation == "back" else 0.12)))
        collar_drop = torso_h * (0.035 if orientation == "back" else 0.105)
        collar_half_w = max(sh_span * 0.21, torso_h * 0.12)
        neck_cx = float(sh_mid[0])

        # 1. HAIR MASK (Class 2)
        raw_hair = (seg_map == SEG_LABELS["hair"]) & (user_gray < 48.0)
        hair_mask = np.where(raw_hair, 255, 0).astype(np.uint8)
        if np.any(hair_mask):
            hair_dil = cv2.dilate(hair_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)))
            dark_strands = (
                (hair_dil > 0)
                & (user_gray < 30.0)
                & (~np.isin(seg_map, [4, 5, 6, 7, 14, 15]))
                & (grid_y < (hip_mid[1] - torso_h * 0.12))
            )
            hair_mask = np.where((hair_mask > 0) | dark_strands, 255, 0).astype(np.uint8)
            hair_mask = cv2.morphologyEx(
                hair_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15))
            )
            hair_mask = cv2.medianBlur(hair_mask, 11)

        # 2. FACE & NECK MASK (Classes 1, 3, 11)
        # SegFormer class 11 ("face") also labels bare shoulder/clavicle skin when the user wears a tank top.
        # Never let bare shoulder skin below y_collar be classified as face_neck!
        if orientation in ("side", "back"):
            face_cutoff_y = float(sh_mid[1] - torso_h * 0.135)
            face_skin_cond = (seg_map == SEG_LABELS["face"]) & (grid_y < face_cutoff_y)
        else:
            face_cutoff_y = y_collar + collar_drop * 0.82
            in_neck_col = np.abs(grid_x - neck_cx) < (collar_half_w * 0.85)
            in_head_col = np.abs(grid_x - neck_cx) < (collar_half_w * 1.15)
            face_skin_cond = (seg_map == SEG_LABELS["face"]) & (
                (grid_y < (y_collar - torso_h * 0.06))
                | ((grid_y < y_collar) & in_head_col)
                | ((grid_y < face_cutoff_y) & in_neck_col)
            )
        face_cond = face_skin_cond | np.isin(seg_map, [SEG_LABELS["hat"], SEG_LABELS["sunglasses"]])
        face_neck_mask = np.where(face_cond, 255, 0).astype(np.uint8)

        # 3. CHEST / FRONT NECKLINE CUTOUT MASK (Ön yüz yaka/göğüs açıklığı)
        dx_neck = (grid_x - neck_cx) / max(collar_half_w, 1.0)
        in_neck = np.abs(dx_neck) < 1.0
        neck_curve_y = float(y_collar) + collar_drop * (1.0 - np.clip(dx_neck, -1.0, 1.0) ** 2)
        chest_cutout_cond = (
            in_neck
            & (grid_y >= float(max(0, y_collar - 25.0)))
            & (grid_y < neck_curve_y)
            & ((seg_map == SEG_LABELS["face"]) | (~np.isin(seg_map, [4, 5, 6, 7])))
        )
        chest_mask = np.where(chest_cutout_cond, 255, 0).astype(np.uint8)

        # 4. ARMS, FOREARMS, HANDS & FOREGROUND OBJECTS
        arms_mask = np.where(
            np.isin(seg_map, [SEG_LABELS["left_arm"], SEG_LABELS["right_arm"]]), 255, 0
        ).astype(np.uint8)

        # Estimate user skin color for accurate wrist/hand segmentation when overlapping dark clothes
        skin_ref_mask = np.isin(seg_map, [11, 14, 15]) & (grid_y < float(hip_mid[1]))
        skin_med_bgr = (
            np.median(user_bgr[skin_ref_mask], axis=0).astype(np.float32)
            if np.any(skin_ref_mask)
            else np.array([145.0, 165.0, 200.0], dtype=np.float32)
        )
        skin_dist = np.linalg.norm(user_bgr.astype(np.float32) - skin_med_bgr[None, None, :], axis=2)
        is_skin_like = (skin_dist < 56.0) & (user_gray > 75.0) & (~np.isin(seg_map, [0, 2]))

        forearms_bool = np.zeros((h, w), dtype=bool)
        hands_bool = np.zeros((h, w), dtype=bool)
        phone_bool = np.zeros((h, w), dtype=bool)

        arm_pairs = [
            (coords[11, :2], coords[13, :2], coords[15, :2], coords[[15, 17, 19, 21], :2], SEG_LABELS["left_arm"]),
            (coords[12, :2], coords[14, :2], coords[16, :2], coords[[16, 18, 20, 22], :2], SEG_LABELS["right_arm"]),
        ]
        fg_side_arm_label = (
            SEG_LABELS["left_arm"]
            if int(np.sum(seg_map == SEG_LABELS["left_arm"])) >= int(np.sum(seg_map == SEG_LABELS["right_arm"]))
            else SEG_LABELS["right_arm"]
        )
        for sh_pt, el_pt, wr_pt, hand_pts, arm_seg_id in arm_pairs:
            if orientation == "side" and arm_seg_id != fg_side_arm_label:
                # In side profile, only the near (foreground) arm is in front of the garment;
                # the far arm behind the torso must never carve holes into the front hem.
                continue
            this_arm_u8 = np.where(seg_map == arm_seg_id, 255, 0).astype(np.uint8) if orientation == "side" else arms_mask
            up_vec = el_pt - sh_pt
            up_len_sq = max(float(np.dot(up_vec, up_vec)), 1.0)
            t_up = np.clip(((grid_x - sh_pt[0]) * up_vec[0] + (grid_y - sh_pt[1]) * up_vec[1]) / up_len_sq, 0.0, 1.0)
            dist_up = np.hypot(grid_x - (sh_pt[0] + t_up * up_vec[0]), grid_y - (sh_pt[1] + t_up * up_vec[1]))

            fore_vec = wr_pt - el_pt
            fore_len_sq = max(float(np.dot(fore_vec, fore_vec)), 1.0)
            t_fore = np.clip(
                ((grid_x - el_pt[0]) * fore_vec[0] + (grid_y - el_pt[1]) * fore_vec[1]) / fore_len_sq, 0.0, 1.0
            )
            dist_fore = np.hypot(grid_x - (el_pt[0] + t_fore * fore_vec[0]), grid_y - (el_pt[1] + t_fore * fore_vec[1]))

            hand_center = np.mean(hand_pts, axis=0)
            hand_radius = max(torso_h * 0.25, 34.0)
            dist_hand = np.hypot(grid_x - hand_center[0], grid_y - hand_center[1])
            dist_wrist = np.hypot(grid_x - wr_pt[0], grid_y - wr_pt[1])
            in_hand_radius = (dist_hand <= hand_radius) | (dist_wrist <= torso_h * 0.18)

            is_raised_selfie_hand = (
                (orientation == "front")
                & (wr_pt[1] < el_pt[1] - 12.0)
                & (wr_pt[1] > y_collar - torso_h * 0.22)
                & (abs(float(wr_pt[0] - sh_mid[0])) < sh_span * 0.88)
            )
            is_crossed_in_front = (
                (orientation == "front")
                & (abs(float(wr_pt[0] - sh_mid[0])) < sh_span * 0.56)
                & (wr_pt[1] < float(hip_mid[1] + torso_h * 0.08))
            )
            if orientation == "side":
                cuff_cut_y = float(sh_pt[1] + torso_h * 0.36)
                is_this_forearm = (
                    (this_arm_u8 > 0)
                    & (dist_fore < (dist_up + 8.0))
                    & (dist_fore < torso_h * 0.11)
                    & (t_fore > 0.05)
                    & (grid_y >= cuff_cut_y)
                )
                forearms_bool |= is_this_forearm
            elif orientation == "front":
                if is_raised_selfie_hand or is_crossed_in_front:
                    is_this_forearm = (arms_mask > 0) & (dist_fore < (dist_up + 6.0)) & (t_fore > 0.06)
                    forearms_bool |= is_this_forearm
                else:
                    # Arms hanging at sides: mark bare forearm below elbow so torso never swallows forearm
                    is_this_forearm = (arms_mask > 0) & (dist_fore < (dist_up + 5.0)) & (t_fore > 0.12)
                    forearms_bool |= is_this_forearm
            if is_raised_selfie_hand:
                fingers_zone = (
                    (grid_y >= (wr_pt[1] - torso_h * 0.42))
                    & (grid_y <= (sh_pt[1] - 4.0))
                    & (grid_x >= (wr_pt[0] - torso_h * 0.18))
                    & (grid_x <= (wr_pt[0] + torso_h * 0.20))
                )
                wrist_outer_zone = (
                    (grid_y >= (sh_pt[1] - 6.0))
                    & (grid_y <= (wr_pt[1] + 20.0))
                    & (grid_x >= (wr_pt[0] - 2.0))
                    & (grid_x <= (wr_pt[0] + torso_h * 0.24))
                )
                hands_bool |= (arms_mask > 0) & (fingers_zone | wrist_outer_zone)

                phone_zone = (
                    (grid_y >= (wr_pt[1] - torso_h * 0.42))
                    & (grid_y <= (sh_pt[1] - 2.0))
                    & (grid_x >= (wr_pt[0] - torso_h * 0.18))
                    & (grid_x <= (wr_pt[0] + torso_h * 0.06))
                    & (user_gray < 105.0)
                    & (~np.isin(seg_map, [4, 5, 6, 7]))
                )
                phone_bool |= phone_zone
            else:
                if orientation == "side":
                    is_pocket_hand = (this_arm_u8 > 0) & (dist_hand <= torso_h * 0.12) & (t_fore > 0.65)
                else:
                    low_hand_zone = grid_y >= float(hip_mid[1] - torso_h * 0.14)
                    is_pocket_hand = (
                        ((this_arm_u8 > 0) | (is_skin_like & low_hand_zone & (seg_map != 0)))
                        & (dist_hand <= torso_h * 0.16)
                        & (t_fore > 0.55)
                    )
                hands_bool |= is_pocket_hand

        forearms_mask = np.where(forearms_bool, 255, 0).astype(np.uint8)
        hands_mask = np.where(hands_bool, 255, 0).astype(np.uint8)

        # 5. LEGS & SHOES MASKS
        legs_mask = np.where(
            np.isin(seg_map, [SEG_LABELS["left_leg"], SEG_LABELS["right_leg"]]),
            255,
            0,
        ).astype(np.uint8)
        shoes_mask = np.where(
            np.isin(seg_map, [SEG_LABELS["left_shoe"], SEG_LABELS["right_shoe"]]),
            255,
            0,
        ).astype(np.uint8)

        # 6. FOREGROUND OBJECTS MASK
        fg_obj_cond = np.isin(seg_map, [SEG_LABELS["bag"], SEG_LABELS["scarf"]]) | phone_bool
        foreground_objects_mask = np.where(fg_obj_cond, 255, 0).astype(np.uint8)

        return {
            "orientation": orientation,
            "hair": hair_mask,
            "face_neck": face_neck_mask,
            "chest": chest_mask,
            "arms": arms_mask,
            "forearms": forearms_mask,
            "hands": hands_mask,
            "legs": legs_mask,
            "shoes": shoes_mask,
            "foreground_objects": foreground_objects_mask,
        }

    # -----------------------------------------------------------------------
    # Solution 1: Lower-Body Anchor at Landmarks 23 & 24 + Downward Stretching
    # -----------------------------------------------------------------------
    def build_tps_control_points(
        self,
        cropped_garment: np.ndarray,
        pose_landmarks,
        user_shape: Tuple[int, int],
        scale_spec: AsymmetricScaleSpec,
        is_side_profile: bool = False,
        is_back_view: bool = False,
        seg_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct paired (src_pts, dst_pts) TPS control points:
        - For `lower_body`: Locks the top-center of the pants/skirt (`[gw * 0.50, 0.0]`)
          to the Lower-Body Anchor `0.5 * (LM_23 + LM_24)` in (X, Y) coordinates, and
          stretches strictly downward through knees `(25, 26)` to ankles `(27, 28)`.
        """
        uh, uw = user_shape[:2]
        gh, gw = cropped_garment.shape[:2]

        lm = self.router.extract_category_landmarks(
            pose_landmarks, uw, uh, scale_spec.canonical_category
        )
        p_l_sh, p_r_sh = lm["left_shoulder"], lm["right_shoulder"]
        p_l_hip, p_r_hip = lm["left_hip"], lm["right_hip"]
        p_l_knee, p_r_knee = lm["left_knee"], lm["right_knee"]
        p_l_ankle, p_r_ankle = lm["left_ankle"], lm["right_ankle"]

        if p_l_sh[0] <= p_r_sh[0]:
            sh_L, sh_R = p_l_sh.copy(), p_r_sh.copy()
            hip_L, hip_R = p_l_hip.copy(), p_r_hip.copy()
            knee_L, knee_R = p_l_knee.copy(), p_r_knee.copy()
            ank_L, ank_R = p_l_ankle.copy(), p_r_ankle.copy()
        else:
            sh_L, sh_R = p_r_sh.copy(), p_l_sh.copy()
            hip_L, hip_R = p_r_hip.copy(), p_l_hip.copy()
            knee_L, knee_R = p_r_knee.copy(), p_l_knee.copy()
            ank_L, ank_R = p_r_ankle.copy(), p_l_ankle.copy()

        sh_mid = 0.5 * (sh_L + sh_R)
        # Exact (X, Y) midpoint of MediaPipe Landmarks 23 & 24
        anchor_23_24 = lm["lower_body_anchor"]
        knee_mid = 0.5 * (knee_L + knee_R)
        ank_mid = 0.5 * (ank_L + ank_R)
        torso_h = max(abs(float(anchor_23_24[1] - sh_mid[1])), 80.0)

        em = getattr(scale_spec, "extra_metrics", None) or {}
        size_step = float(em.get("size_step", 0.0))  # XS=-2, S=-1, M=0, L=+1, XL=+2, XXL=+3
        step_scale_w = 1.0 + 0.068 * size_step       # ~6.8% width change per size step (~34% XS->XXL)
        step_scale_len = 1.0 + 0.028 * size_step     # ~2.8% length change per size step

        half_torso_w = max(scale_spec.torso_width_px * 0.5, torso_h * 0.22)
        if is_side_profile:
            half_torso_w = max(scale_spec.target_width_px * 0.44, torso_h * 0.20)

        cat = scale_spec.canonical_category
        sub = scale_spec.sub_category

        if cat == "upper_body":
            ft_low = str(getattr(scale_spec, "fit_type", "")).lower()
            is_oversize = ("oversize" in ft_low) or ("rahat" in ft_low) or ("geniş" in ft_low)
            is_slim = ("slim" in ft_low) or ("dar kal" in ft_low) or ("skinny" in ft_low)

            # Measure actual torso and sleeve cuff bounds in the source garment (100% vectorized)
            g_alpha = (
                cropped_garment[:, :, 3] > 20
                if (cropped_garment.ndim == 3 and cropped_garment.shape[2] == 4)
                else np.ones((gh, gw), dtype=bool)
            )
            band_t = g_alpha[int(gh * 0.56) : int(gh * 0.92), :]
            valid_t = np.sum(band_t, axis=1) > 10
            if np.any(valid_t):
                vb = band_t[valid_t]
                src_torso_l = float(np.median(np.argmax(vb, axis=1)))
                src_torso_r = float(np.median(gw - 1 - np.argmax(vb[:, ::-1], axis=1)))
            else:
                src_torso_l, src_torso_r = gw * 0.17, gw * 0.83
            src_torso_w = max(src_torso_r - src_torso_l, gw * 0.45)

            band_s = g_alpha[int(gh * 0.24) : int(gh * 0.48), :]
            valid_s = np.sum(band_s, axis=1) > 10
            if np.any(valid_s):
                vs = band_s[valid_s]
                src_cuff_l = float(np.min(np.argmax(vs, axis=1)))
                src_cuff_r = float(np.max(gw - 1 - np.argmax(vs[:, ::-1], axis=1)))
            else:
                src_cuff_l, src_cuff_r = gw * 0.02, gw * 0.98

            cx_top = float(sh_mid[0])
            cx_chest = float(0.70 * sh_mid[0] + 0.30 * anchor_23_24[0])
            cx_waist = float(0.45 * sh_mid[0] + 0.55 * anchor_23_24[0])
            cx_hem = float(anchor_23_24[0])

            y_top = max(8.0, float(sh_mid[1] - torso_h * 0.115))
            # Clear vertical hem progression across XS..XXL and fit types
            hem_extra_ratio = 0.11 + 0.032 * size_step + (0.045 if is_oversize else (-0.02 if is_slim else 0.0))
            hem_y = min(
                float(uh - 10),
                float(anchor_23_24[1] + torso_h * hem_extra_ratio + scale_spec.gravity_drop_px * 0.35),
            )
            cuff_ratio = float(np.clip(0.40 + 0.014 * size_step + (0.03 if is_oversize else 0.0), 0.36, 0.48))
            y_cuff = y_top + (hem_y - y_top) * cuff_ratio
            waist_y = y_top + (hem_y - y_top) * 0.68

            if is_side_profile:
                # Side profile: map the central torso cylinder [src_torso_l, src_torso_r] cleanly
                # onto the side torso + upper-arm thickness, scaled by XS..XXL size_step.
                side_half_w = max(scale_spec.target_width_px * 0.40, torso_h * 0.22 * step_scale_w)
                ease_side = (12.0 if is_oversize else (3.0 if is_slim else 7.0)) + size_step * 4.5

                nose_x = float(lm["all_coords"][0, 0])
                facing_left = nose_x <= float(sh_mid[0])

                if facing_left:
                    side_lx_top = cx_top - torso_h * 0.17
                    side_rx_top = cx_top + torso_h * 0.22
                else:
                    side_lx_top = cx_top - torso_h * 0.22
                    side_rx_top = cx_top + torso_h * 0.17

                side_lx_chest = cx_chest - side_half_w * 1.02 - ease_side
                side_rx_chest = cx_chest + side_half_w * 0.92 + ease_side * 0.6

                # Ensure front bust (seg_map==4) and upper shoulder/arm (14,15) are covered at chest
                if seg_map is not None:
                    y0_c = int(max(0, sh_mid[1] + torso_h * 0.08))
                    y1_c = int(min(uh, y_cuff + 10))
                    band_side = np.isin(seg_map[y0_c:y1_c, :], [4, 14, 15])
                    rows_valid = np.sum(band_side, axis=1) > 8
                    if np.any(rows_valid):
                        vb_s = band_side[rows_valid]
                        seg_l_min = float(np.percentile(np.argmax(vb_s, axis=1), 10))
                        seg_r_max = float(np.percentile(uw - 1 - np.argmax(vb_s[:, ::-1], axis=1), 90))
                        side_lx_chest = min(side_lx_chest, seg_l_min - max(0.0, ease_side * 0.5))
                        side_rx_chest = max(side_rx_chest, seg_r_max + max(2.0, ease_side * 0.4))

                    y0_top = int(max(0, sh_mid[1] - torso_h * 0.06))
                    y1_top = int(min(uh, sh_mid[1] + torso_h * 0.04))
                    band_top = np.isin(seg_map[y0_top:y1_top, :], [4, 11, 14, 15])
                    rows_top = np.sum(band_top, axis=1) > 6
                    if np.any(rows_top):
                        vb_t = band_top[rows_top]
                        t_l_min = float(np.percentile(np.argmax(vb_t, axis=1), 20))
                        t_r_max = float(np.percentile(uw - 1 - np.argmax(vb_t[:, ::-1], axis=1), 80))
                        side_lx_top = float(np.clip(t_l_min - 2.0, side_lx_chest + 12.0, cx_top - torso_h * 0.10))
                        side_rx_top = float(np.clip(t_r_max + 2.0, cx_top + torso_h * 0.10, side_rx_chest - 4.0))
                else:
                    side_lx_top = max(side_lx_top, side_lx_chest + 12.0)
                    side_rx_top = min(side_rx_top, side_rx_chest - 4.0)

                taper_px = (4.0 if is_oversize else (14.0 if is_slim else 10.0)) - size_step * 2.2
                side_lx_waist = side_lx_chest + taper_px
                side_rx_waist = side_rx_chest - taper_px * 0.4
                side_lx_hem = side_lx_chest + taper_px * 0.35
                side_rx_hem = side_rx_chest - taper_px * 0.2

                min_side_span = max(torso_h * 0.32, 62.0)
                if (side_rx_waist - side_lx_waist) < min_side_span:
                    mid_w = 0.5 * (side_lx_waist + side_rx_waist)
                    side_lx_waist, side_rx_waist = mid_w - 0.5 * min_side_span, mid_w + 0.5 * min_side_span
                if (side_rx_hem - side_lx_hem) < min_side_span:
                    mid_h = 0.5 * (side_lx_hem + side_rx_hem)
                    side_lx_hem, side_rx_hem = mid_h - 0.5 * min_side_span, mid_h + 0.5 * min_side_span

                span_top = max(side_rx_top - side_lx_top, torso_h * 0.30)
                span_chest = side_rx_chest - side_lx_chest
                span_waist = side_rx_waist - side_lx_waist
                span_hem = side_rx_hem - side_lx_hem

                y_sh_side = float(sh_mid[1]) - torso_h * 0.065

                if facing_left:
                    print_r_x = min(gw * 0.69, src_torso_r - max(18.0, gw * 0.08))
                    src_x_cols = [src_torso_l, gw * 0.50, print_r_x, src_torso_r]
                    dst_w_fracs = [0.00, 0.30, 0.58, 1.00]
                    top_row_ys = [
                        float(sh_mid[1]) - torso_h * 0.045,
                        float(sh_mid[1]) - torso_h * 0.105,
                        float(sh_mid[1]) - torso_h * 0.125,
                        float(sh_mid[1]) - torso_h * 0.088,
                    ]
                else:
                    print_l_x = max(gw * 0.31, src_torso_l + max(18.0, gw * 0.08))
                    src_x_cols = [src_torso_l, print_l_x, gw * 0.50, src_torso_r]
                    dst_w_fracs = [0.00, 0.42, 0.70, 1.00]
                    top_row_ys = [
                        float(sh_mid[1]) - torso_h * 0.088,
                        float(sh_mid[1]) - torso_h * 0.125,
                        float(sh_mid[1]) - torso_h * 0.105,
                        float(sh_mid[1]) - torso_h * 0.045,
                    ]

                src_y_rows = [gh * 0.10, gh * 0.42, gh * 0.68, gh * 0.97]
                dst_bounds_rows = [
                    (side_lx_top, span_top, y_sh_side),
                    (side_lx_chest, span_chest, y_cuff),
                    (side_lx_waist, span_waist, waist_y),
                    (side_lx_hem, span_hem, hem_y),
                ]
                src_pts = np.array(
                    [[sx, sy] for sy in src_y_rows for sx in src_x_cols],
                    dtype=np.float32,
                )
                dst_pts_list = []
                for r_idx, (lx, sp, dy) in enumerate(dst_bounds_rows):
                    for c_idx, wf in enumerate(dst_w_fracs):
                        row_y = top_row_ys[c_idx] if r_idx == 0 else dy
                        dst_pts_list.append([lx + wf * sp, row_y])
                dst_pts = np.array(dst_pts_list, dtype=np.float32)
            else:
                # Front & Back views: Real ANSUR + Size-Table Driven Widths (XS -> XXL Distinct Progression)
                sh_left_x = min(float(sh_L[0]), float(sh_R[0]))
                sh_right_x = max(float(sh_L[0]), float(sh_R[0]))
                sh_left_y = float(sh_L[1]) if float(sh_L[0]) <= float(sh_R[0]) else float(sh_R[1])
                sh_right_y = float(sh_R[1]) if float(sh_L[0]) <= float(sh_R[0]) else float(sh_L[1])
                sh_span = max(sh_right_x - sh_left_x, torso_h * 0.42)

                # Shoulder drop / pad scales visibly across XS (-2) .. XXL (+3)
                base_pad = sh_span * (0.085 if is_oversize else (0.025 if is_slim else 0.050))
                shoulder_pad = max(4.0, base_pad + size_step * (sh_span * 0.032))

                min_safe_half_w = max(torso_h * 0.20, sh_span * 0.40)

                # Use multi-zone garment pixel widths from CategoryRouter when available
                g_chest_half = 0.5 * float(em.get("garment_chest_px", half_torso_w * 2.0))
                g_waist_half = 0.5 * float(em.get("garment_waist_px", g_chest_half * 2.0 * scale_spec.waist_taper_ratio))
                g_hips_half = 0.5 * float(em.get("garment_hips_px", g_chest_half * 2.0 * scale_spec.hem_flare_ratio))

                w_top = max(0.5 * sh_span + shoulder_pad, g_chest_half * 0.96, min_safe_half_w)
                w_chest = max(g_chest_half, 0.5 * sh_span * (0.86 + 0.065 * (size_step + 2)), min_safe_half_w)
                if is_back_view:
                    w_waist = max(g_waist_half * 0.96, w_chest * (0.91 if is_oversize else 0.84), min_safe_half_w * 0.84)
                    w_hem = max(g_hips_half * 0.97, w_chest * (0.95 if is_oversize else 0.89), min_safe_half_w * 0.88)
                else:
                    w_waist = max(g_waist_half, w_chest * (0.94 if is_oversize else 0.86), min_safe_half_w * 0.86)
                    w_hem = max(g_hips_half, w_chest * (0.98 if is_oversize else 0.91), min_safe_half_w * 0.90)

                sh_dst_lx = min(sh_left_x - shoulder_pad, cx_top - w_top)
                sh_dst_rx = max(sh_right_x + shoulder_pad, cx_top + w_top)
                w_top_actual = 0.5 * (sh_dst_rx - sh_dst_lx)
                cx_sh_mid = 0.5 * (sh_dst_lx + sh_dst_rx)

                scale_x_top = (2.0 * w_top_actual) / src_torso_w
                sleeve_size_mult = float(np.clip(0.95 + 0.06 * size_step + (0.08 if is_oversize else 0.0), 0.80, 1.30))
                overhang_l_px = max(8.0, (src_torso_l - src_cuff_l) * scale_x_top * sleeve_size_mult)
                overhang_r_px = max(8.0, (src_cuff_r - src_torso_r) * scale_x_top * sleeve_size_mult)

                src_pts = np.array(
                    [
                        [gw * 0.50, gh * 0.04],       # Collar
                        [src_torso_l, gh * 0.12],     # Left shoulder seam
                        [src_torso_r, gh * 0.12],     # Right shoulder seam
                        [src_cuff_l, gh * 0.42],      # Left outer sleeve cuff
                        [src_cuff_r, gh * 0.42],      # Right outer sleeve cuff
                        [src_torso_l, gh * 0.44],     # Left underarm torso seam
                        [src_torso_r, gh * 0.44],     # Right underarm torso seam
                        [src_torso_l, gh * 0.68],     # Left waist seam
                        [src_torso_r, gh * 0.68],     # Right waist seam
                        [src_torso_l, gh * 0.97],     # Left hem
                        [gw * 0.50, gh * 0.98],       # Center hem
                        [src_torso_r, gh * 0.97],     # Right hem
                    ],
                    dtype=np.float32,
                )

                dst_pts = np.array(
                    [
                        [cx_sh_mid, y_top],
                        [sh_dst_lx, sh_left_y - torso_h * 0.095],
                        [sh_dst_rx, sh_right_y - torso_h * 0.095],
                        [cx_chest - w_chest - overhang_l_px, y_cuff + 2.0],
                        [cx_chest + w_chest + overhang_r_px, y_cuff + 2.0],
                        [cx_chest - w_chest, y_cuff],
                        [cx_chest + w_chest, y_cuff],
                        [cx_waist - w_waist, waist_y],
                        [cx_waist + w_waist, waist_y],
                        [cx_hem - w_hem, hem_y],
                        [cx_hem, hem_y + 6.0],
                        [cx_hem + w_hem, hem_y],
                    ],
                    dtype=np.float32,
                )

        elif cat == "lower_body":
            anchor_x = float(anchor_23_24[0])
            anchor_y = float(anchor_23_24[1])

            if seg_map is not None:
                existing_lower = np.isin(seg_map, [5, 6, 8])
                band_at_anchor = existing_lower[
                    int(max(0, anchor_y - 10)) : int(min(uh, anchor_y + 20)), :
                ]
                if np.any(band_at_anchor):
                    xs_top = np.where(band_at_anchor)[1]
                    anchor_x = 0.5 * float(np.min(xs_top) + np.max(xs_top))

            coords_2d = lm["all_coords"][:, :2]
            leg_chains = sorted(
                [(23, 25, 27), (24, 26, 28)],
                key=lambda t: float(coords_2d[t[0], 0]) + float(coords_2d[t[1], 0]),
            )
            hip_L_pt, knee_L_pt, ank_L_pt = (
                coords_2d[leg_chains[0][0]],
                coords_2d[leg_chains[0][1]],
                coords_2d[leg_chains[0][2]],
            )
            hip_R_pt, knee_R_pt, ank_R_pt = (
                coords_2d[leg_chains[1][0]],
                coords_2d[leg_chains[1][1]],
                coords_2d[leg_chains[1][2]],
            )

            y_hip_l = max(anchor_y + 22.0, float(hip_L_pt[1] + torso_h * 0.10))
            y_hip_r = max(anchor_y + 22.0, float(hip_R_pt[1] + torso_h * 0.10))
            len_step_px = size_step * (torso_h * 0.025)
            if sub == "pants":
                y_knee_l = max(y_hip_l + 30.0, float(knee_L_pt[1]))
                y_knee_r = max(y_hip_r + 30.0, float(knee_R_pt[1]))
                y_hem_l = min(float(uh - 8), max(y_knee_l + 35.0, float(ank_L_pt[1]) + scale_spec.gravity_drop_px * 0.10 + len_step_px))
                y_hem_r = min(float(uh - 8), max(y_knee_r + 35.0, float(ank_R_pt[1]) + scale_spec.gravity_drop_px * 0.10 + len_step_px))
            else:
                y_hem_shared = min(
                    float(uh - 15),
                    max(
                        float(knee_mid[1] + scale_spec.gravity_drop_px * 0.25 + len_step_px * 1.4),
                        anchor_y + scale_spec.target_height_px * 0.78 + len_step_px,
                    ),
                )
                y_knee_l = y_knee_r = 0.5 * (0.5 * (y_hip_l + y_hip_r) + y_hem_shared)
                y_hem_l = y_hem_r = y_hem_shared

            cx_hip = float(anchor_x)
            cx_knee = 0.5 * float(knee_L_pt[0] + knee_R_pt[0])
            cx_hem = 0.5 * float(ank_L_pt[0] + ank_R_pt[0]) if sub == "pants" else cx_knee

            g_waist_half = 0.5 * float(em.get("garment_waist_px", half_torso_w * 2.0 * scale_spec.waist_taper_ratio))
            g_hips_half = 0.5 * float(em.get("garment_hips_px", half_torso_w * 2.0 * scale_spec.hip_flare_ratio))
            w_waist = max(g_waist_half, torso_h * 0.20)
            w_hip = max(g_hips_half, w_waist * 1.04)
            w_knee = w_hip * (0.88 if sub == "pants" else 1.12)
            w_hem = w_hip * (scale_spec.hem_flare_ratio / max(scale_spec.hip_flare_ratio, 0.5))
            if sub == "pants":
                fit_p = self.pants_fit_profile(scale_spec, is_side_profile=is_side_profile)
                w_knee *= max(0.85, fit_p["knee_mult"])
                w_hem *= max(0.82, fit_p["hem_mult"])

            if is_side_profile and sub == "pants":
                src_pts = np.array(
                    [
                        [gw * 0.52, 0.0],
                        [gw * 0.68, 0.0],
                        [gw * 0.84, 0.0],
                        [gw * 0.52, gh * 0.24],
                        [gw * 0.88, gh * 0.24],
                        [gw * 0.54, gh * 0.60],
                        [gw * 0.86, gh * 0.60],
                        [gw * 0.54, gh * 0.98],
                        [gw * 0.70, gh * 0.98],
                        [gw * 0.86, gh * 0.98],
                    ],
                    dtype=np.float32,
                )
            else:
                src_pts = np.array(
                    [
                        [gw * 0.16, 0.0],
                        [gw * 0.50, 0.0],
                        [gw * 0.84, 0.0],
                        [gw * 0.10, gh * 0.24],
                        [gw * 0.90, gh * 0.24],
                        [gw * 0.14, gh * 0.60],
                        [gw * 0.86, gh * 0.60],
                        [gw * 0.14, gh * 0.98],
                        [gw * 0.50, gh * 0.98],
                        [gw * 0.86, gh * 0.98],
                    ],
                    dtype=np.float32,
                )

            leg_w_knee = max(w_knee * 0.52, torso_h * 0.12)
            leg_w_hem = max(w_hem * 0.52, torso_h * 0.12)
            if sub == "pants" and not is_side_profile:
                raw_knee_lx = float(knee_L_pt[0]) - leg_w_knee
                raw_knee_rx = float(knee_R_pt[0]) + leg_w_knee
                raw_hem_lx = float(ank_L_pt[0]) - leg_w_hem
                raw_hem_rx = float(ank_R_pt[0]) + leg_w_hem
            else:
                raw_knee_lx, raw_knee_rx = cx_knee - w_knee * (0.65 if is_side_profile else 1.0), cx_knee + w_knee * (0.65 if is_side_profile else 1.0)
                raw_hem_lx, raw_hem_rx = cx_hem - w_hem * (0.65 if is_side_profile else 1.0), cx_hem + w_hem * (0.65 if is_side_profile else 1.0)

            min_lower_span = max(torso_h * (0.24 if is_side_profile else 0.34), 48.0)
            knee_lx = float(min(raw_knee_lx, raw_knee_rx))
            knee_rx = float(max(raw_knee_lx, raw_knee_rx))
            if (knee_rx - knee_lx) < min_lower_span:
                k_mid = 0.5 * (knee_lx + knee_rx)
                knee_lx, knee_rx = k_mid - 0.5 * min_lower_span, k_mid + 0.5 * min_lower_span

            hem_lx = float(min(raw_hem_lx, raw_hem_rx))
            hem_rx = float(max(raw_hem_lx, raw_hem_rx))
            if (hem_rx - hem_lx) < min_lower_span:
                h_mid = 0.5 * (hem_lx + hem_rx)
                hem_lx, hem_rx = h_mid - 0.5 * min_lower_span, h_mid + 0.5 * min_lower_span

            cx_hem_clamped = 0.5 * (hem_lx + hem_rx)
            y_hem_mid = 0.5 * (y_hem_l + y_hem_r)
            w_waist_eff = w_waist * (0.68 if is_side_profile else 1.0)
            w_hip_eff = w_hip * (0.72 if is_side_profile else 1.0)

            dst_pts = np.array(
                [
                    [anchor_x - w_waist_eff, anchor_y],
                    [anchor_x, anchor_y],
                    [anchor_x + w_waist_eff, anchor_y],
                    [cx_hip - w_hip_eff, y_hip_l],
                    [cx_hip + w_hip_eff, y_hip_r],
                    [knee_lx, y_knee_l],
                    [knee_rx, y_knee_r],
                    [hem_lx, y_hem_l],
                    [cx_hem_clamped, y_hem_mid + 3.0],
                    [hem_rx, y_hem_r],
                ],
                dtype=np.float32,
            )

        else:  # full_body
            len_step_px = size_step * (torso_h * 0.032)
            y_top = max(8.0, float(sh_mid[1] + torso_h * scale_spec.anchor_y_offset_ratio))
            y_waist = float(sh_mid[1] + torso_h * 0.66)
            y_hip = float(anchor_23_24[1])
            y_hem = min(
                float(uh - 14),
                max(
                    float(knee_mid[1] + torso_h * 0.24 + scale_spec.gravity_drop_px * 0.30 + len_step_px),
                    y_top + scale_spec.target_height_px * 0.82 + len_step_px,
                ),
            )

            cx_top = float(sh_mid[0])
            cx_waist = float(0.4 * sh_mid[0] + 0.6 * anchor_23_24[0])
            cx_hip = float(anchor_23_24[0])
            cx_hem = float(knee_mid[0])

            g_chest_half = 0.5 * float(em.get("garment_chest_px", half_torso_w * 2.0))
            g_waist_half = 0.5 * float(em.get("garment_waist_px", g_chest_half * 2.0 * scale_spec.waist_taper_ratio))
            g_hips_half = 0.5 * float(em.get("garment_hips_px", g_chest_half * 2.0 * scale_spec.hip_flare_ratio))

            w_sh = max(g_chest_half * 0.96, torso_h * 0.22)
            w_waist = max(g_waist_half, torso_h * 0.19)
            w_hip = max(g_hips_half, w_waist * 1.06)
            w_hem = max(w_hip * (scale_spec.hem_flare_ratio / max(scale_spec.hip_flare_ratio, 0.5)), w_hip * 1.08)

            src_pts = np.array(
                [
                    [gw * 0.50, gh * 0.03],
                    [gw * 0.20, gh * 0.10],
                    [gw * 0.80, gh * 0.10],
                    [gw * 0.22, gh * 0.36],
                    [gw * 0.78, gh * 0.36],
                    [gw * 0.16, gh * 0.56],
                    [gw * 0.84, gh * 0.56],
                    [gw * 0.10, gh * 0.97],
                    [gw * 0.50, gh * 0.98],
                    [gw * 0.90, gh * 0.97],
                ],
                dtype=np.float32,
            )

            dst_pts = np.array(
                [
                    [cx_top, y_top],
                    [cx_top - w_sh, float(sh_L[1])],
                    [cx_top + w_sh, float(sh_R[1])],
                    [cx_waist - w_waist, y_waist],
                    [cx_waist + w_waist, y_waist],
                    [cx_hip - w_hip, y_hip],
                    [cx_hip + w_hip, y_hip],
                    [cx_hem - w_hem, y_hem],
                    [cx_hem, y_hem + 8.0],
                    [cx_hem + w_hem, y_hem],
                ],
                dtype=np.float32,
            )

        return src_pts, dst_pts

    # -----------------------------------------------------------------------
    # Lower-Body Bipedal (Two Independent Legs) TPS Warping
    # -----------------------------------------------------------------------
    # Number of 5x5 elliptical dilation iterations applied to the lower-body diffusion
    # mask so the old garment (e.g. black leggings) can never ghost out of the new trousers.
    LOWER_BODY_DILATE_ITERATIONS = 7
    LOWER_BODY_DILATE_KSIZE = 5

    @staticmethod
    def _lower_waist_anchor(lm: Dict[str, np.ndarray], seg_map: Optional[np.ndarray], h: int) -> Tuple[float, float, float]:
        """
        Waistband anchor: horizontally pinned to the midpoint of landmarks 23 & 24 (refined to the
        centre of the existing lower garment's top band) and vertically leveled at the top edge of
        the existing pants/leggings so no old waistband peeks out above the new trousers.
        """
        sh_mid = lm["shoulder_mid"]
        anchor = lm["lower_body_anchor"]
        torso_h = max(abs(float(anchor[1] - sh_mid[1])), 80.0)
        anchor_x = float(anchor[0])
        anchor_y = float(anchor[1] - torso_h * 0.22)

        if seg_map is not None:
            existing = np.isin(seg_map, [5, 6, 8])
            ys_ex, _ = np.where(existing)
            if len(ys_ex) > 50:
                seg_top_y = float(np.min(ys_ex))
                anchor_y = max(float(sh_mid[1] + torso_h * 0.38), seg_top_y - 3.0)
                top_band = existing[int(max(0, anchor_y)) : int(min(h, anchor[1] + 20)), :]
                if np.any(top_band):
                    xs_top = np.where(top_band)[1]
                    anchor_x = 0.5 * float(np.min(xs_top) + np.max(xs_top))
        return anchor_x, anchor_y, torso_h

    @staticmethod
    def _level_and_clean_flat_pants(crop_bgra: np.ndarray) -> np.ndarray:
        """
        Prepare a flat-lay trouser crop for warping (all matrix ops):
        - removes dark hanger-clip / label artifacts from the top-left waistband,
        - vertically levels the (often tilted) waistband column-by-column so the top edge is horizontal.
        """
        gh, gw = crop_bgra.shape[:2]
        rgb = crop_bgra[:, :, :3].astype(np.float32)
        alpha_core = cv2.erode(
            np.where(crop_bgra[:, :, 3] > 180, 255, 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ) > 0

        gx_grid, gy_grid = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32))
        lum = np.mean(rgb, axis=2)

        leg_zone = alpha_core & (gy_grid > gh * 0.20) & (gy_grid < gh * 0.65)
        fabric_med = (
            np.median(rgb[leg_zone & (lum >= np.percentile(lum[leg_zone], 50))], axis=0)
            if np.any(leg_zone)
            else np.median(rgb[alpha_core], axis=0) if np.any(alpha_core) else np.array([60.0, 50.0, 40.0], np.float32)
        ).astype(np.float32)

        wb_zone = alpha_core & (gy_grid > gh * 0.03) & (gy_grid < gh * 0.14) & (gx_grid > gw * 0.46) & (gx_grid < gw * 0.76)
        wb_med = np.median(rgb[wb_zone], axis=0).astype(np.float32) if np.any(wb_zone) else fabric_med
        clip_artifact = (
            (gy_grid < gh * 0.16) & (gx_grid < gw * 0.44) & (lum < float(np.mean(wb_med)) * 0.72)
        ).astype(np.float32)
        if np.any(clip_artifact > 0):
            wc = cv2.GaussianBlur(
                cv2.dilate(clip_artifact, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))), (15, 15), 4.0
            )[..., None]
            rgb = rgb * (1.0 - wc) + wb_med[None, None, :] * wc

        col_has = np.sum(alpha_core, axis=0) > 10
        col_top = np.where(col_has, np.argmax(alpha_core, axis=0), 0).astype(np.float32)
        col_top = cv2.GaussianBlur(col_top.reshape(1, -1), (31, 1), 7.0).ravel()
        col_top = np.clip(col_top, 0.0, gh * 0.12)
        level_y = (
            col_top[None, :] + (gy_grid / max(float(gh - 1), 1.0)) * (float(gh - 1) - col_top[None, :])
        ).astype(np.float32)

        bgra = np.dstack([np.clip(rgb, 0, 255), crop_bgra[:, :, 3].astype(np.float32)]).astype(np.float32)
        leveled = cv2.remap(bgra, gx_grid, level_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return np.clip(leveled, 0, 255).astype(np.uint8)

    @staticmethod
    def pants_fit_profile(scale_spec: AsymmetricScaleSpec, is_side_profile: bool = False) -> Dict[str, float]:
        """
        Fit-aware & size-graded trouser leg drape profile:
        - `knee_mult` / `hem_mult`: knee / hem width multipliers relative to the hip leg width.
        - `straight`: 0..1; how much the leg axis straightens toward a vertical wide-leg column vs hugging knees/ankles.
        - `seg_relax`: 0..1; how much the tight bond to the user's old (narrow) leg segmentation is relaxed.
        """
        ft = str(getattr(scale_spec, "fit_type", "") or "").lower()
        em = getattr(scale_spec, "extra_metrics", None) or {}
        size_step = float(em.get("size_step", 0.0))  # XS=-2 .. XXL=+3
        step_delta = 0.045 * size_step

        if "oversize" in ft or "rahat" in ft or "wide" in ft or "geniş" in ft:
            prof = {
                "knee_mult": max(1.30, 1.32 + step_delta),
                "hem_mult": max(1.40, 1.42 + step_delta * 1.15),
                "straight": 0.72,
                "seg_relax": 1.0,
            }
        elif "slim" in ft or "bodycon" in ft or "skinny" in ft or "dar kal" in ft or "tayt" in ft:
            prof = {
                "knee_mult": max(0.84, 1.0 + step_delta * 0.75),
                "hem_mult": max(0.82, 1.0 + step_delta * 0.75),
                "straight": 0.0,
                "seg_relax": 0.0,
            }
        else:  # regular / classic
            prof = {
                "knee_mult": max(0.92, 1.12 + step_delta),
                "hem_mult": max(0.90, 1.18 + step_delta * 1.05),
                "straight": 0.25,
                "seg_relax": 0.6,
            }
        if is_side_profile:
            prof["knee_mult"] = 1.0 + (prof["knee_mult"] - 1.0) * 0.55
            prof["hem_mult"] = 1.0 + (prof["hem_mult"] - 1.0) * 0.55
        return prof

    def prewarp_lower_body_bipedal(
        self,
        garment_bgra: np.ndarray,
        pose_landmarks,
        user_shape: Tuple[int, int],
        scale_spec: AsymmetricScaleSpec,
        seg_map: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Bipedal warping with Dynamic Bent-Knee Pose Tracking, XS-XXL Size Grading, and Crotch-to-Ankle Barrier:
        1. Splits the flat-lay trousers at the crotch gap into two independent leg panels (Left & Right).
        2. Uses 4 smooth anatomical knot rows (Waist, Crotch, Knee, Hem = 12 control points per leg) so TPS
           never oscillates or pinches the knees.
        3. Enforces a mathematical Crotch-to-Ankle Inter-Leg Barrier (`barrier_x`) while preserving full outer
           leg width so left and right legs never cross or pinch.
        """
        uh, uw = user_shape[:2]
        ys, xs = np.where(garment_bgra[:, :, 3] > 15)
        if len(ys) < 100:
            return None
        crop = garment_bgra[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        crop = self._level_and_clean_flat_pants(crop)
        gh, gw = crop.shape[:2]
        if gh < 40 or gw < 40:
            return None
        alpha_bool = crop[:, :, 3] > 60

        rows_any = np.where(alpha_bool.any(axis=1))[0]
        if len(rows_any) < 40:
            return None
        row_first, row_last = int(rows_any[0]), int(rows_any[-1])
        row_span = float(max(row_last - row_first, 1))

        lm = self.router.extract_category_landmarks(pose_landmarks, uw, uh, "lower_body")
        coords = lm["all_coords"][:, :2]

        legs = sorted(
            [(23, 25, 27), (24, 26, 28)],
            key=lambda t: float(coords[t[0], 0]) + float(coords[t[1], 0]),
        )

        em = getattr(scale_spec, "extra_metrics", None) or {}
        size_step = float(em.get("size_step", 0.0))
        step_scale_w = 1.0 + 0.068 * size_step

        anchor_x, anchor_y, torso_h = self._lower_waist_anchor(lm, seg_map, uh)
        g_hips_half = 0.5 * float(em.get("garment_hips_px", scale_spec.torso_width_px * scale_spec.hip_flare_ratio))
        w_hip = max(g_hips_half, torso_h * 0.24 * step_scale_w)

        hip_y_lm = float(lm["hip_mid"][1])
        y_hip = max(anchor_y + 24.0, hip_y_lm + torso_h * 0.10)
        y_crotch = max(y_hip + 10.0, anchor_y + 56.0, hip_y_lm + torso_h * 0.15)

        low_band = alpha_bool[int(gh * 0.68) : int(gh * 0.96), :]
        col_any = low_band.any(axis=0)
        cols_present = np.where(col_any)[0]
        mid_x = gw * 0.5
        if len(cols_present) > 20:
            span_cols = np.arange(int(cols_present.min()), int(cols_present.max()) + 1)
            gap_cols = span_cols[~col_any[span_cols]]
            central = gap_cols[(gap_cols > gw * 0.30) & (gap_cols < gw * 0.70)]
            if len(central) >= 3:
                mid_x = float(np.mean(central))
        mid = int(round(mid_x))
        pad = max(3, int(gw * 0.015))

        centre_open = ~alpha_bool[:, max(0, mid - 2) : min(gw, mid + 3)].any(axis=1)
        centre_open[: int(row_first + row_span * 0.18)] = False
        crotch_candidates = np.where(centre_open)[0]
        crotch_frac = (
            float(np.clip((crotch_candidates[0] - row_first) / row_span, 0.22, 0.42))
            if len(crotch_candidates) > 0
            else 0.28
        )
        knee_frac = float(min(0.84, max(0.56, crotch_frac + 0.28)))
        top_frac, hem_frac = 0.03, 0.96
        # Use 4 clean anatomical knot rows to prevent RBF spline oscillation/pinching
        fracs = np.array([top_frac, crotch_frac, knee_frac, hem_frac], dtype=np.float32)
        src_rows = row_first + fracs * row_span
        band_half = max(5, int(row_span * 0.035))

        def row_extents_lr(row_y: float, side: int) -> Tuple[float, float]:
            r = int(np.clip(round(row_y), 0, gh - 1))
            band = alpha_bool[max(0, r - band_half) : min(gh, r + band_half + 1)]
            if side == 0:
                cols = np.where(band[:, :mid].any(axis=0))[0]
                if len(cols) == 0:
                    return gw * 0.10, float(mid)
                lx, rx = float(np.percentile(cols, 3)), float(np.percentile(cols, 97))
                rx = max(rx, float(mid - 4), lx + gw * 0.30)
            else:
                cols = np.where(band[:, mid:].any(axis=0))[0]
                if len(cols) == 0:
                    return float(mid), gw * 0.90
                lx, rx = float(np.percentile(cols, 3) + mid), float(np.percentile(cols, 97) + mid)
                lx = min(lx, float(mid + 4), rx - gw * 0.30)
            lx_safe = min(lx, rx)
            rx_safe = max(lx, rx, lx_safe + gw * 0.28)
            return lx_safe, rx_safe

        r_hip_src = row_first + 0.5 * (top_frac + crotch_frac) * row_span
        l0_lx, _ = row_extents_lr(float(r_hip_src), 0)
        _, r1_rx = row_extents_lr(float(r_hip_src), 1)
        scale_x = (2.0 * w_hip) / max(r1_rx - l0_lx, gw * 0.55)

        knots_f = [top_frac, crotch_frac, knee_frac, hem_frac]
        leg_hip_w = w_hip
        fit = self.pants_fit_profile(scale_spec)
        knee_floor = max(0.76, fit["knee_mult"] * 0.85)
        hem_floor = max(0.74, fit["hem_mult"] * 0.85)
        width_cap_mult = max(1.45, fit["hem_mult"] * 1.15)
        len_step_px = size_step * (torso_h * 0.025)

        leg_centres_x = []
        leg_dst_ys = []
        for side, (hip_i, knee_i, ank_i) in enumerate(legs):
            sgn = -1.0 if side == 0 else 1.0
            seam_waist_x = anchor_x + sgn * 0.48 * leg_hip_w
            crotch_c_x = 0.55 * seam_waist_x + 0.45 * float(coords[hip_i, 0])
            knee_c_x = float(coords[knee_i, 0])
            ank_c_x = float(coords[ank_i, 0])

            cx_traj = np.interp(
                fracs,
                knots_f,
                [seam_waist_x, crotch_c_x, knee_c_x, ank_c_x],
            ).astype(np.float32)

            y_knee_leg = max(y_crotch + 28.0, float(coords[knee_i, 1]))
            y_hem_leg = min(
                float(uh - 8),
                max(y_knee_leg + 35.0, float(coords[ank_i, 1]) + scale_spec.gravity_drop_px * 0.12 + len_step_px),
            )
            dy_traj = np.interp(
                fracs,
                knots_f,
                [anchor_y, y_crotch, y_knee_leg, y_hem_leg],
            ).astype(np.float32)

            leg_centres_x.append(cx_traj)
            leg_dst_ys.append(dy_traj)

        c_left = np.minimum(leg_centres_x[0], leg_centres_x[1])
        c_right = np.maximum(leg_centres_x[0], leg_centres_x[1])
        min_centre_sep = max(leg_hip_w * 0.62, 32.0)
        too_close = (c_right - c_left) < min_centre_sep
        mid_centres = 0.5 * (c_left + c_right)
        c_left = np.where(too_close, mid_centres - 0.5 * min_centre_sep, c_left)
        c_right = np.where(too_close, mid_centres + 0.5 * min_centre_sep, c_right)
        leg_centres_x = [c_left, c_right]

        barrier_x = 0.5 * (c_left + c_right)
        min_inseam_gap = max(4.0, leg_hip_w * 0.05)

        warped_legs, src_all, dst_all = [], [], []
        for side in range(2):
            cx_traj = leg_centres_x[side]
            dy_traj = leg_dst_ys[side]
            src_list, dst_list = [], []

            for i in range(len(fracs)):
                src_lx, src_rx = row_extents_lr(float(src_rows[i]), side)
                width_s = max(src_rx - src_lx, gw * 0.30)
                src_cx = 0.5 * (src_lx + src_rx)

                if fracs[i] <= top_frac + 0.02:
                    width_d = leg_hip_w * 0.96
                elif fracs[i] <= crotch_frac + 0.02:
                    width_d = leg_hip_w * 1.02
                elif fracs[i] <= knee_frac + 0.02:
                    width_d = float(np.clip(leg_hip_w * 0.88 * fit["knee_mult"], leg_hip_w * knee_floor, leg_hip_w * width_cap_mult))
                else:
                    width_d = float(np.clip(leg_hip_w * 0.86 * fit["hem_mult"], leg_hip_w * hem_floor, leg_hip_w * width_cap_mult))

                centre_d = float(cx_traj[i])
                dst_lx = centre_d - 0.5 * width_d
                dst_rx = centre_d + 0.5 * width_d

                if fracs[i] >= crotch_frac:
                    if side == 0:
                        max_rx = float(barrier_x[i]) - 0.5 * min_inseam_gap
                        if dst_rx > max_rx:
                            dst_rx = max_rx
                            dst_lx = min(dst_lx, dst_rx - width_d * 0.92)
                    else:
                        min_lx = float(barrier_x[i]) + 0.5 * min_inseam_gap
                        if dst_lx < min_lx:
                            dst_lx = min_lx
                            dst_rx = max(dst_rx, dst_lx + width_d * 0.92)

                min_w_leg = max(leg_hip_w * 0.65, 28.0)
                dst_lx_safe = float(min(dst_lx, dst_rx))
                dst_rx_safe = float(max(dst_lx, dst_rx, dst_lx_safe + min_w_leg))
                dst_cx_safe = 0.5 * (dst_lx_safe + dst_rx_safe)
                y_d = float(dy_traj[i])

                src_list.extend([[src_lx, src_rows[i]], [src_cx, src_rows[i]], [src_rx, src_rows[i]]])
                dst_list.extend([[dst_lx_safe, y_d], [dst_cx_safe, y_d], [dst_rx_safe, y_d]])

            src_k = np.array(src_list, dtype=np.float32)
            dst_k = np.array(dst_list, dtype=np.float32)

            A = np.hstack([src_k, np.ones((src_k.shape[0], 1), dtype=np.float32)])
            coef, _, _, _ = np.linalg.lstsq(A, dst_k, rcond=None)
            frame_src = np.array(
                [[-gw * 0.6, -gh * 0.25], [gw * 1.6, -gh * 0.25], [gw * 1.6, gh * 1.25], [-gw * 0.6, gh * 1.25]],
                dtype=np.float32,
            )
            frame_dst = np.hstack([frame_src, np.ones((4, 1), dtype=np.float32)]) @ coef
            src_k = np.vstack([src_k, frame_src])
            dst_k = np.vstack([dst_k, frame_dst.astype(np.float32)])

            panel = crop.copy()
            if side == 0:
                panel[:, min(gw, mid + pad) :, 3] = 0
            else:
                panel[:, : max(0, mid - pad), 3] = 0

            warped_k, _, _ = ThinPlateSplineWarper.warp_image_tps(
                panel, src_k, dst_k, out_shape=(uh, uw), regularization=3e-2
            )
            warped_legs.append(warped_k)
            src_all.append(src_k)
            dst_all.append(dst_k)

        gx_u, gy_u = np.meshgrid(np.arange(uw, dtype=np.float32), np.arange(uh, dtype=np.float32))
        mean_dy = 0.5 * (leg_dst_ys[0] + leg_dst_ys[1])
        order_idx = np.argsort(mean_dy)
        barrier_row_x = np.interp(
            np.arange(uh, dtype=np.float32),
            mean_dy[order_idx],
            barrier_x[order_idx],
        ).astype(np.float32)[:, None]
        below_crotch = gy_u >= (y_crotch + 6.0)
        warped_legs[0][:, :, 3] = np.where(
            below_crotch & (gx_u > (barrier_row_x - 1.0)), 0, warped_legs[0][:, :, 3]
        )
        warped_legs[1][:, :, 3] = np.where(
            below_crotch & (gx_u < (barrier_row_x + 1.0)), 0, warped_legs[1][:, :, 3]
        )

        a_l = warped_legs[0][:, :, 3].astype(np.float32) / 255.0
        a_r = warped_legs[1][:, :, 3].astype(np.float32) / 255.0
        alpha_u = a_r + a_l * (1.0 - a_r)
        premult = (
            warped_legs[1][:, :, :3].astype(np.float32) * a_r[..., None]
            + warped_legs[0][:, :, :3].astype(np.float32) * (a_l * (1.0 - a_r))[..., None]
        )
        color = premult / np.maximum(alpha_u, 1e-3)[..., None]

        warped_bgra = np.dstack(
            [np.clip(color, 0, 255).astype(np.uint8), np.clip(alpha_u * 255.0, 0, 255).astype(np.uint8)]
        )
        warped_bgra[: max(0, int(anchor_y) - 2), :, 3] = 0

        outside_pelvis = (gy_u < y_crotch) & (np.abs(gx_u - anchor_x) > w_hip * 1.28)
        warped_bgra[:, :, 3] = np.where(outside_pelvis, 0, warped_bgra[:, :, 3])
        return warped_bgra, np.vstack(src_all), np.vstack(dst_all)

    def prewarp_garment_tps(
        self,
        garment_bgra: np.ndarray,
        pose_landmarks,
        user_shape: Tuple[int, int],
        scale_spec: AsymmetricScaleSpec,
        is_side_profile: bool = False,
        is_back_view: bool = False,
        seg_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        clean_garment, base_color_bgr = self.decontaminate_garment_edges(garment_bgra)
        ys, xs = np.where(clean_garment[:, :, 3] > 15)
        if len(ys) > 0:
            cropped = clean_garment[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        else:
            cropped = clean_garment

        src_pts, dst_pts = self.build_tps_control_points(
            cropped,
            pose_landmarks,
            user_shape,
            scale_spec,
            is_side_profile=is_side_profile,
            is_back_view=is_back_view,
            seg_map=seg_map,
        )
        # Lower-body trousers (front/back views): two independent leg panels (Bipedal Warping)
        # so the crotch/inseam region follows the real leg landmarks instead of being flattened
        # into a single bounding-box block.
        if (
            scale_spec.canonical_category == "lower_body"
            and scale_spec.sub_category == "pants"
            and not is_side_profile
        ):
            bip = self.prewarp_lower_body_bipedal(
                garment_bgra=clean_garment,
                pose_landmarks=pose_landmarks,
                user_shape=user_shape,
                scale_spec=scale_spec,
                seg_map=seg_map,
            )
            if bip is not None:
                return bip

        if is_side_profile and scale_spec.canonical_category == "upper_body":
            ch, cw = cropped.shape[:2]
            # Trim outer sleeves and level the top collar/shoulder line at 0.10*ch so:
            # 1) the front flat-lay collar never juts out under the chin, and
            # 2) the sloped flat-lay shoulder corners are 100% opaque fabric across [src_min_x, src_max_x]
            src_min_x = int(max(0, round(float(np.min(src_pts[:, 0])) + 6.0)))
            src_max_x = int(min(cw, round(float(np.max(src_pts[:, 0])) - 6.0)))
            cropped = cropped.copy()
            cropped[:, :src_min_x, 3] = 0
            cropped[:, src_max_x:, 3] = 0
            y_neck_line = int(ch * 0.10)
            y_sh_base = int(min(ch - 1, max(y_neck_line + 2, round(ch * 0.19))))
            cropped[:y_neck_line, :, 3] = 0
            ref_row = cropped[y_sh_base : y_sh_base + 1, src_min_x:src_max_x, :]
            top_band = cropped[y_neck_line:y_sh_base, src_min_x:src_max_x, :]
            transparent_shoulder_corners = top_band[:, :, 3:4] < 180
            cropped[y_neck_line:y_sh_base, src_min_x:src_max_x, :] = np.where(
                transparent_shoulder_corners, ref_row, top_band
            )

        # Affine-consistent outer frame prevents radial-basis spline overshoot/oscillation
        ch, cw = cropped.shape[:2]
        A = np.hstack([src_pts, np.ones((src_pts.shape[0], 1), dtype=np.float32)])
        coef, _, _, _ = np.linalg.lstsq(A, dst_pts, rcond=None)
        frame_src = np.array(
            [[-cw * 0.5, -ch * 0.25], [cw * 1.5, -ch * 0.25], [cw * 1.5, ch * 1.25], [-cw * 0.5, ch * 1.25]],
            dtype=np.float32,
        )
        frame_dst = np.hstack([frame_src, np.ones((4, 1), dtype=np.float32)]) @ coef
        src_framed = np.vstack([src_pts, frame_src])
        dst_framed = np.vstack([dst_pts, frame_dst.astype(np.float32)])

        warped_bgra, map_x, map_y = ThinPlateSplineWarper.warp_image_tps(
            cropped, src_framed, dst_framed, out_shape=user_shape[:2], regularization=2e-2
        )

        # Enforce downward-only stretching below the top anchor Y for lower_body
        if scale_spec.canonical_category == "lower_body":
            anchor_top_y = max(0, int(np.min(dst_pts[:3, 1])) - 2)
            warped_bgra[:anchor_top_y, :, 3] = 0

        return warped_bgra, src_pts, dst_pts

    # -----------------------------------------------------------------------
    # Solution 2: Orientation-Gated (`get_masking_directives`) Agnostic Mask
    # -----------------------------------------------------------------------
    def generate_category_agnostic_mask(
        self,
        user_bgr: np.ndarray,
        pose_landmarks,
        warped_garment_bgra: np.ndarray,
        scale_spec: AsymmetricScaleSpec,
        seg_map: Optional[np.ndarray] = None,
        is_side_profile: bool = False,
        is_back_view: bool = False,
        for_diffusion: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Build the category-conditional agnostic (inpainting) mask.

        `for_diffusion=True` produces the mask that is sent to IDM-VTON:
        - upper_body: generous, organic union of the user's real old clothing and the TPS-warped
          garment silhouette (NOT the synthetic render silhouette), so the diffusion model is free to
          create natural sleeve/hem drape. The front neckline (`chest`) is not carved, so the model
          paints its own collar.
        - lower_body: additionally dilated with `cv2.dilate` (LOWER_BODY_DILATE_ITERATIONS) to kill old
          garment ghosting, then hands/shoes/foreground are carved back out and the waist top is
          clamped so the bare midriff is never repainted.
        """
        h, w = user_bgr.shape[:2]
        if seg_map is None:
            seg_map = get_global_models().get_semantic_map(user_bgr)

        orientation = self.detect_pose_orientation(pose_landmarks, w, h)
        if is_side_profile:
            orientation = "side"
        elif is_back_view:
            orientation = "back"

        route = CategoryRouter.resolve_route(
            scale_spec.canonical_category if scale_spec.sub_category != "skirt" else "skirt"
        )
        directives = CategoryRouter.get_masking_directives(orientation, route.canonical_category)
        z_masks = self.extract_anatomical_z_masks(
            user_bgr, seg_map, pose_landmarks, orientation=orientation
        )

        lm = self.router.extract_category_landmarks(pose_landmarks, w, h, route.canonical_category)
        sh_mid = lm["shoulder_mid"]
        hip_mid = lm["lower_body_anchor"]
        knee_mid = lm["knee_mid"]
        ank_mid = lm["ankle_mid"]
        torso_h = max(abs(float(hip_mid[1] - sh_mid[1])), 80.0)

        warped_alpha = (
            warped_garment_bgra[:, :, 3]
            if (
                warped_garment_bgra is not None
                and warped_garment_bgra.ndim == 3
                and warped_garment_bgra.shape[2] == 4
            )
            else np.zeros((h, w), dtype=np.uint8)
        )
        garment_silhouette = np.where(warped_alpha > 15, 255, 0).astype(np.uint8)

        existing_cat_clothes = np.where(
            np.isin(seg_map, route.user_clothes_to_replace), 255, 0
        ).astype(np.uint8)

        ys_w = np.where(garment_silhouette > 0)[0]
        if route.canonical_category == "upper_body":
            y_top_bound = max(0, int(sh_mid[1] - torso_h * 0.18))
            y_bot_bound = (
                int(np.max(ys_w) + 10)
                if len(ys_w) > 0
                else min(h, int(hip_mid[1] + torso_h * 0.18 + scale_spec.gravity_drop_px * 0.35))
            )
            row_idx = np.arange(h)[:, None]
            existing_cat_clothes = np.where(
                (row_idx >= y_top_bound) & (row_idx <= y_bot_bound + 12),
                existing_cat_clothes,
                0,
            ).astype(np.uint8)

        elif route.canonical_category == "lower_body":
            # Anchor top bound to the (23, 24) waist/hip anchor
            y_top_bound = (
                int(np.min(ys_w) - 4)
                if len(ys_w) > 0
                else max(0, int(hip_mid[1] - torso_h * 0.16))
            )
            if route.sub_category == "pants":
                y_bot_bound = min(
                    h,
                    int(max(ank_mid[1], np.max(ys_w) if len(ys_w) > 0 else ank_mid[1]) + 12),
                )
            else:
                y_bot_bound = (
                    int(np.max(ys_w) + 8)
                    if len(ys_w) > 0
                    else min(h, int(knee_mid[1] + scale_spec.gravity_drop_px * 0.25))
                )
            row_idx = np.arange(h)[:, None]
            existing_cat_clothes = np.where(
                (row_idx >= max(0, y_top_bound - 12)) & (row_idx <= y_bot_bound + 10),
                existing_cat_clothes,
                0,
            ).astype(np.uint8)

        else:  # full_body
            y_top_bound = max(0, int(sh_mid[1] - torso_h * 0.18))
            y_bot_bound = (
                int(np.max(ys_w) + 12)
                if len(ys_w) > 0
                else min(h, int(knee_mid[1] + torso_h * 0.32 + scale_spec.gravity_drop_px * 0.30))
            )

        raw_agnostic = cv2.bitwise_or(garment_silhouette, existing_cat_clothes)
        raw_agnostic = cv2.dilate(
            raw_agnostic, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        if for_diffusion and route.canonical_category == "lower_body":
            # Dilation (mask inflation): old leggings/pants must never ghost out of the new garment.
            raw_agnostic = cv2.dilate(
                raw_agnostic,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (self.LOWER_BODY_DILATE_KSIZE, self.LOWER_BODY_DILATE_KSIZE),
                ),
                iterations=self.LOWER_BODY_DILATE_ITERATIONS,
            )

        # -------------------------------------------------------------------
        # Solution 2: Apply `get_masking_directives` (Back Pose Solid Block vs Front/Side Carving)
        # -------------------------------------------------------------------
        carved_mask = raw_agnostic.copy()

        if directives["solid_block_mode"]:
            # Back pose: Do NOT intersect or subtract SegFormer skin/neck/chest/arms/hands!
            # Keep the garment mask as a solid, unbroken block, only preserving hair strands.
            if directives["carve_hair"]:
                carved_mask = cv2.subtract(carved_mask, z_masks["hair"])
            carved_mask = cv2.subtract(carved_mask, z_masks["shoes"])
        else:
            if directives["carve_face_neck"]:
                carved_mask = cv2.subtract(carved_mask, z_masks["face_neck"])
            if directives["carve_chest_neckline"] and not for_diffusion:
                carved_mask = cv2.subtract(carved_mask, z_masks["chest"])
            if directives["carve_foreground_objects"]:
                carved_mask = cv2.subtract(carved_mask, z_masks["foreground_objects"])
            if directives["carve_hair"]:
                carved_mask = cv2.subtract(carved_mask, z_masks["hair"])
            if directives["carve_hands"]:
                carved_mask = cv2.subtract(carved_mask, z_masks["hands"])
            if directives["carve_forearms"]:
                carved_mask = cv2.subtract(carved_mask, z_masks["forearms"])

            if route.canonical_category == "upper_body":
                carved_mask = cv2.subtract(carved_mask, z_masks["legs"])
                carved_mask = cv2.subtract(carved_mask, z_masks["shoes"])
            elif route.canonical_category == "lower_body":
                carved_mask = cv2.subtract(carved_mask, z_masks["shoes"])
                upper_clothes_above_waist = np.where(
                    (seg_map == SEG_LABELS["upper_clothes"]) & (np.arange(h)[:, None] < y_top_bound + 4),
                    255,
                    0,
                ).astype(np.uint8)
                carved_mask = cv2.subtract(carved_mask, upper_clothes_above_waist)
                if directives["preserve_bare_lower_legs"]:
                    bare_lower_legs = np.where(
                        (z_masks["legs"] > 0) & (np.arange(h)[:, None] > y_bot_bound),
                        255,
                        0,
                    ).astype(np.uint8)
                    carved_mask = cv2.subtract(carved_mask, bare_lower_legs)
            else:
                carved_mask = cv2.subtract(carved_mask, z_masks["shoes"])
                bare_lower_legs = np.where(
                    (z_masks["legs"] > 0) & (np.arange(h)[:, None] > y_bot_bound),
                    255,
                    0,
                ).astype(np.uint8)
                carved_mask = cv2.subtract(carved_mask, bare_lower_legs)

        carved_mask[: max(0, y_top_bound - 6), :] = 0
        carved_mask[min(h, y_bot_bound + 10) :, :] = 0

        if for_diffusion and route.canonical_category == "lower_body" and len(ys_w) > 0:
            # Never repaint the bare midriff above the waistband (dilation must only grow sideways/down).
            carved_mask[: max(0, int(np.min(ys_w)) - 2), :] = 0
            # Re-protect foreground anatomy that dilation may have re-entered (shoes, hands, objects).
            guard = cv2.bitwise_or(z_masks["shoes"], z_masks["foreground_objects"])
            if directives["carve_hands"] and not directives["solid_block_mode"]:
                guard = cv2.bitwise_or(guard, z_masks["hands"])
            carved_mask = cv2.subtract(carved_mask, guard)

        clean_binary, soft_alpha = self.refine_and_antialias_mask(
            carved_mask,
            close_ksize=21,
            open_ksize=9,
            blur_ksize=11,
            blur_sigma=2.8,
        )

        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[clean_binary > 0] = [255, 255, 255, 255]
        return mask_rgba, soft_alpha, z_masks

    # -----------------------------------------------------------------------
    # 100% Vectorized Anatomical Upper-Body Synthesis (Solid Back, Clean Side & Front)
    # -----------------------------------------------------------------------
    # Universal De-Clothing & Clean-Plate Anatomical Reconstructor
    # -----------------------------------------------------------------------
    def reconstruct_clean_plate(
        self,
        user_bgr: np.ndarray,
        seg_map: np.ndarray,
        pose_landmarks,
        scale_spec: AsymmetricScaleSpec,
        orientation: str = "front",
        garment_hem_y: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """
        100% vectorized De-Clothing & Clean-Plate engine:
        1. Builds `clean_bg_bgr` from the immediate exterior background strips adjacent to the person
           silhouette (plus OpenCV fast Telea inpainting fallback), so indoor rooms (pillars/sofas) and
           studio walls are reconstructed seamlessly without any skin/dark halo around the hips.
        2. Builds `bare_body_bgr` & `anatomical_body_mask` strictly for exposed anatomical limbs (e.g.,
           bare lower legs below skirts/dresses and upper neck/clavicle), NEVER painting lateral skin
           halos outside the waist or hips.
        3. Replaces all conflicting old clothes (`seg_map in [4, 5, 6, 7, 8]`) outside kept garments with
           `bare_body_bgr` (inside exposed limbs) and `clean_bg_bgr` (outside anatomical body).
        """
        h, w = user_bgr.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        ys_vec = np.arange(h, dtype=np.float32)

        # 1. Pure Background Plate Sampled Immediately Outside the Person Silhouette
        fg_u8 = np.where(seg_map > 0, 255, 0).astype(np.uint8)
        fg_dil_u8 = cv2.dilate(fg_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        fg_dil = fg_dil_u8 > 0
        bg_pure = ~fg_dil

        user_f = user_bgr.astype(np.float32)
        global_bg_med = (
            np.median(user_f[bg_pure], axis=0).astype(np.float32)
            if np.any(bg_pure)
            else np.median(user_f[:, :10], axis=(0, 1)).astype(np.float32)
        )

        # Find row-wise left & right silhouette boundaries of fg_dil
        row_has_fg = np.any(fg_dil, axis=1)
        xl_fg = np.where(row_has_fg, np.argmax(fg_dil, axis=1), int(w * 0.25)).astype(np.float32)
        xr_fg = np.where(row_has_fg, w - 1 - np.argmax(fg_dil[:, ::-1], axis=1), int(w * 0.75)).astype(np.float32)

        # Sample background strips immediately adjacent to the person's left & right edges
        near_l_mask = bg_pure & (grid_x >= np.maximum(0.0, xl_fg[:, None] - 32.0)) & (grid_x <= np.maximum(2.0, xl_fg[:, None] - 2.0))
        near_r_mask = bg_pure & (grid_x >= np.minimum(float(w - 3), xr_fg[:, None] + 2.0)) & (grid_x <= np.minimum(float(w - 1), xr_fg[:, None] + 32.0))

        border_w = max(12, int(w * 0.14))
        far_l_mask = bg_pure & (grid_x < border_w)
        far_r_mask = bg_pure & (grid_x >= w - border_w)

        def _row_mean_color(mask_2d: np.ndarray, fallback_2d: np.ndarray) -> np.ndarray:
            wts = mask_2d.astype(np.float32)
            w_sum = np.sum(wts, axis=1, keepdims=True)
            fb_wts = fallback_2d.astype(np.float32)
            fb_sum = np.sum(fb_wts, axis=1, keepdims=True)
            near_rgb = np.sum(user_f * wts[..., None], axis=1) / np.maximum(w_sum, 1.0)
            fb_rgb = np.sum(user_f * fb_wts[..., None], axis=1) / np.maximum(fb_sum, 1.0)
            out = np.where(w_sum > 1.0, near_rgb, np.where(fb_sum > 1.0, fb_rgb, global_bg_med[None, :]))
            return cv2.GaussianBlur(out.astype(np.float32).reshape(h, 1, 3), (1, 35), 8.0).reshape(h, 3)

        row_bg_l = _row_mean_color(near_l_mask, far_l_mask)
        row_bg_r = _row_mean_color(near_r_mask, far_r_mask)

        mid_fg_x = 0.5 * (xl_fg + xr_fg)
        span_fg_x = np.maximum(xr_fg - xl_fg, 40.0)
        u_bg = np.clip((grid_x - xl_fg[:, None]) / span_fg_x[:, None], 0.0, 1.0)[..., None]
        # Sharp transition near the body midline so left hip gets exact left background and right hip gets exact right background
        u_bg_smooth = np.clip(0.5 + 1.8 * (u_bg - 0.5), 0.0, 1.0)
        clean_bg_f = row_bg_l[:, None, :] * (1.0 - u_bg_smooth) + row_bg_r[:, None, :] * u_bg_smooth
        rng_bg = np.random.default_rng(77)
        bg_grain = cv2.GaussianBlur(rng_bg.normal(0.0, 1.0, (h, w)).astype(np.float32), (3, 3), 0)[..., None]
        clean_bg_f = np.clip(clean_bg_f + bg_grain, 0.0, 255.0)
        clean_bg_f = np.where(bg_pure[..., None], user_f, clean_bg_f)

        # 2. Authentic Skin Tone & Strict Anatomical Limb Reconstructor (NO lateral hip/waist skin halo!)
        skin_Valid = np.isin(seg_map, [11, 14, 15])
        if np.sum(skin_Valid) < 80:
            skin_Valid = np.isin(seg_map, [11, 12, 13, 14, 15])
        if np.any(skin_Valid):
            s_px = user_f[skin_Valid]
            s_lum = np.mean(s_px, axis=1)
            mid_band = (s_lum >= np.percentile(s_lum, 25)) & (s_lum <= np.percentile(s_lum, 85))
            skin_med_bgr = (
                np.median(s_px[mid_band], axis=0).astype(np.float32)
                if np.any(mid_band)
                else np.median(s_px, axis=0).astype(np.float32)
            )
        else:
            skin_med_bgr = np.array([150.0, 172.0, 205.0], dtype=np.float32)

        lm = self.router.extract_category_landmarks(pose_landmarks, w, h, "full_body")
        coords = lm["all_coords"][:, :2]
        sh_mid = lm["shoulder_mid"]
        hip_mid = lm["hip_mid"]
        torso_h = max(abs(float(hip_mid[1] - sh_mid[1])), 80.0)
        sh_span = max(abs(float(lm["right_shoulder"][0] - lm["left_shoulder"][0])), torso_h * 0.34)

        cat = scale_spec.canonical_category
        sub = scale_spec.sub_category

        # Upper neck / clavicle / upper-torso skin zone inside person silhouette
        y_sh_top = float(sh_mid[1] - torso_h * 0.14)
        y_clavicle_bot = float(sh_mid[1] + torso_h * 0.24)
        r_neck_sh = max(sh_span * 0.48, torso_h * 0.22)
        u_neck = (grid_x - float(sh_mid[0])) / max(r_neck_sh, 1.0)
        in_bare_neck = (grid_y >= y_sh_top) & (grid_y <= y_clavicle_bot) & (np.abs(u_neck) <= 1.0) & (fg_u8 > 0)
        neck_shade = np.clip(1.03 - 0.14 * (np.clip(u_neck, -1.2, 1.2) ** 2), 0.80, 1.05)[..., None]
        bare_neck_rgb = np.clip(skin_med_bgr[None, None, :] * neck_shade, 0.0, 255.0)

        # Bare Lower Limb Cylinders (Left Leg 23->25->27 and Right Leg 24->26->28)
        # Only reconstruct bare lower legs when wearing a skirt or dress (where lower legs are exposed below hem)
        legs = sorted(
            [(23, 25, 27), (24, 26, 28)],
            key=lambda t: float(coords[t[0], 0]) + float(coords[t[1], 0]),
        )
        in_bare_legs = np.zeros((h, w), dtype=bool)
        bare_legs_rgb = np.broadcast_to(skin_med_bgr, (h, w, 3)).copy().astype(np.float32)

        expose_lower_legs = (cat == "full_body") or (cat == "lower_body" and sub == "skirt")
        if expose_lower_legs:
            for side, (hip_i, knee_i, ank_i) in enumerate(legs):
                hy, ky, ay = float(coords[hip_i, 1]), float(coords[knee_i, 1]), float(coords[ank_i, 1])
                hx, kx, ax = float(coords[hip_i, 0]), float(coords[knee_i, 0]), float(coords[ank_i, 0])
                ky = max(hy + 25.0, ky)
                ay = max(ky + 30.0, ay)

                t_leg = np.clip((ys_vec - hy) / max(ay - hy, 1.0), 0.0, 1.0)
                k_frac = float(np.clip((ky - hy) / max(ay - hy, 1.0), 0.35, 0.68))
                cx_leg = np.interp(t_leg, [0.0, k_frac, 1.0], [hx, kx, ax])
                r_scale = 0.78 if orientation == "side" else 1.0
                r_leg = np.interp(
                    t_leg,
                    [0.0, 0.32, k_frac, 0.74, 1.0],
                    [
                        torso_h * 0.125 * r_scale,
                        torso_h * 0.115 * r_scale,
                        torso_h * 0.088 * r_scale,
                        torso_h * 0.090 * r_scale,
                        torso_h * 0.055 * r_scale,
                    ],
                )
                u_leg = (grid_x - cx_leg[:, None]) / np.maximum(r_leg[:, None], 1.0)
                # Start bare leg cylinder from mid-thigh downward so hips never get a skin halo
                leg_start_y = hy + (ky - hy) * 0.32
                this_leg = (grid_y >= leg_start_y) & (grid_y <= ay + 4.0) & (np.abs(u_leg) <= 1.0)
                leg_shade = np.clip(1.04 - 0.20 * (np.clip(u_leg, -1.2, 1.2) ** 2), 0.75, 1.06)[..., None]
                this_rgb = np.clip(skin_med_bgr[None, None, :] * leg_shade, 0.0, 255.0)
                bare_legs_rgb = np.where(this_leg[..., None], this_rgb, bare_legs_rgb)
                in_bare_legs |= this_leg

        # Preserve real bare skin pixels where present (face, neck, bare arms, and ankles/exposed legs)
        ank_min_y = min(float(coords[27, 1]), float(coords[28, 1]))
        if cat == "lower_body" and sub == "pants":
            # For full-length trousers, bare thighs/calves above the ankle outside the trouser silhouette
            # must be erased to clean background rather than leaving a skin halo around the trousers.
            real_skin_mask = np.isin(seg_map, [11, 14, 15]) | (
                np.isin(seg_map, [12, 13]) & (grid_y >= ank_min_y - 18.0)
            )
        else:
            real_skin_mask = np.isin(seg_map, [11, 12, 13, 14, 15])

        anatomical_body_mask = in_bare_neck | in_bare_legs | real_skin_mask
        bare_body_f = np.where(in_bare_legs[..., None], bare_legs_rgb, bare_neck_rgb)
        bare_body_f = np.where(real_skin_mask[..., None], user_f, bare_body_f)

        # 3. Category-Routed Conflicting Old Clothing Erasure
        z_masks = self.extract_anatomical_z_masks(user_bgr, seg_map, pose_landmarks, orientation=orientation)
        protected_fg = (
            (z_masks["hair"] > 0)
            | (z_masks["face_neck"] > 0)
            | (z_masks["hands"] > 0)
            | (z_masks["forearms"] > 0)
            | (z_masks["shoes"] > 0)
        )

        if cat == "upper_body":
            # Erase old upper_clothes (4) and only the upper shoulder/chest strap of any dress (7)
            dress_cut_y = float(sh_mid[1] + torso_h * 0.28)
            target_old = (seg_map == 4) | ((seg_map == 7) & (grid_y <= dress_cut_y))
        elif cat == "lower_body":
            # Erase old skirt (5), pants (6), belt (8), and lower skirt of any dress (7) below hourglass waist
            waist_cut_y = float(garment_hem_y) if garment_hem_y is not None else float(hip_mid[1] - torso_h * 0.24)
            target_old = np.isin(seg_map, [5, 6, 8]) | ((seg_map == 7) & (grid_y >= waist_cut_y))
            if sub == "pants":
                target_old |= np.isin(seg_map, [12, 13]) & (grid_y >= waist_cut_y) & (grid_y < ank_min_y - 18.0)
            protected_fg |= (seg_map == 4) & (grid_y < waist_cut_y)
        else:  # full_body
            target_old = np.isin(seg_map, [4, 5, 6, 7, 8])

        old_u8 = np.where(target_old & (~protected_fg), 255, 0).astype(np.uint8)
        old_dil_u8 = cv2.dilate(old_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        old_dil_u8 = np.where(protected_fg, 0, old_dil_u8).astype(np.uint8)
        if cat == "upper_body":
            dress_keep_y = float(sh_mid[1] + torso_h * 0.28)
            old_dil_u8[grid_y > dress_keep_y] = np.where(
                seg_map[grid_y > dress_keep_y] == 7,
                0,
                old_dil_u8[grid_y > dress_keep_y],
            )
        elif cat == "lower_body":
            w_cut = float(garment_hem_y) if garment_hem_y is not None else float(hip_mid[1] - torso_h * 0.24)
            old_dil_u8[grid_y < w_cut] = np.where(
                np.isin(seg_map[grid_y < w_cut], [4, 7]),
                0,
                old_dil_u8[grid_y < w_cut],
            )
        erase_alpha = cv2.GaussianBlur(old_dil_u8.astype(np.float32) / 255.0, (9, 9), 2.0)[..., None]

        # Composite bare_body_f inside anatomical_body_mask and clean_bg_f outside anatomical_body_mask
        body_soft = cv2.GaussianBlur(anatomical_body_mask.astype(np.float32), (7, 7), 1.5)[..., None]
        reconstructed_plate = clean_bg_f * (1.0 - body_soft) + bare_body_f * body_soft
        declothed_bgr = np.clip(user_f * (1.0 - erase_alpha) + reconstructed_plate * erase_alpha, 0.0, 255.0).astype(np.uint8)

        # Record De-Clothing Telemetry
        class_names = {4: "Üst Giyim(4)", 5: "Etek(5)", 6: "Pantolon(6)", 7: "Elbise(7)", 8: "Kemer(8)"}
        detected_classes = [
            f"{lbl}:{ int(np.sum(seg_map == cid)) }px"
            for cid, lbl in class_names.items()
            if int(np.sum(seg_map == cid)) > 50
        ]
        erase_bool = old_dil_u8 > 64
        erased_skin_px = int(np.sum(erase_bool & anatomical_body_mask))
        erased_bg_px = int(np.sum(erase_bool & (~anatomical_body_mask)))

        stats = {
            "detected_old_classes": ", ".join(detected_classes) if detected_classes else "Yok",
            "old_clothes_px": int(np.sum(target_old)),
            "erased_to_skin_px": erased_skin_px,
            "erased_to_bg_px": erased_bg_px,
            "clean_plate_verified": True,
        }
        self.last_declothing_stats = stats
        return declothed_bgr, stats

    def synthesize_vectorized_upper_vton(
        self,
        user_bgr: np.ndarray,
        garment_bgra: np.ndarray,
        pose_landmarks,
        seg_map: np.ndarray,
        scale_spec: AsymmetricScaleSpec,
        is_side_profile: bool = False,
        is_back_view: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = user_bgr.shape[:2]

        orientation = self.detect_pose_orientation(pose_landmarks, w, h)
        if is_side_profile:
            orientation = "side"
        elif is_back_view:
            orientation = "back"

        directives = CategoryRouter.get_masking_directives(orientation, "upper_body")

        lm = self.router.extract_category_landmarks(pose_landmarks, w, h, "upper_body")
        p_l_sh, p_r_sh = lm["left_shoulder"], lm["right_shoulder"]
        p_l_el, p_r_el = lm["left_elbow"], lm["right_elbow"]
        p_l_wr, p_r_wr = lm["left_wrist"], lm["right_wrist"]
        p_l_hip, p_r_hip = lm["left_hip"], lm["right_hip"]

        if p_l_sh[0] <= p_r_sh[0]:
            sh_L, el_L, wr_L, hip_L = p_l_sh.copy(), p_l_el, p_l_wr, p_l_hip
            sh_R, el_R, wr_R, hip_R = p_r_sh.copy(), p_r_el, p_r_wr, p_r_hip
        else:
            sh_L, el_L, wr_L, hip_L = p_r_sh.copy(), p_r_el, p_r_wr, p_r_hip
            sh_R, el_R, wr_R, hip_R = p_l_sh.copy(), p_l_el, p_l_wr, p_l_hip

        if wr_L[1] < sh_L[1] + 25.0 and sh_L[1] < sh_R[1] - 8.0:
            sh_L[1] = max(sh_L[1], sh_R[1] - 6.0)
        if wr_R[1] < sh_R[1] + 25.0 and sh_R[1] < sh_L[1] - 8.0:
            sh_R[1] = max(sh_R[1], sh_L[1] - 6.0)

        sh_mid = 0.5 * (sh_L + sh_R)
        hip_mid = 0.5 * (hip_L + hip_R)
        torso_h = max(abs(float(hip_mid[1] - sh_mid[1])), 80.0)
        sh_span = max(abs(float(sh_R[0] - sh_L[0])), torso_h * 0.34)

        em = getattr(scale_spec, "extra_metrics", None) or {}
        size_step = float(em.get("size_step", 0.0))  # XS=-2, S=-1, M=0, L=+1, XL=+2, XXL=+3
        ft_low = str(getattr(scale_spec, "fit_type", "")).lower()
        is_oversize = ("oversize" in ft_low) or ("rahat" in ft_low) or ("geniş" in ft_low)
        is_slim = ("slim" in ft_low) or ("dar kal" in ft_low) or ("skinny" in ft_low)

        y_collar = max(6, int(sh_mid[1] - torso_h * (0.145 if orientation == "back" else 0.130)))
        ys_existing_top = np.where(seg_map == SEG_LABELS["upper_clothes"])[0]
        if len(ys_existing_top) > 30:
            y_collar = max(6, min(y_collar, int(np.percentile(ys_existing_top, 1)) - 2))

        hem_extra_ratio = 0.06 + 0.038 * size_step + (0.045 if is_oversize else (-0.025 if is_slim else 0.0))
        y_hem_target = int(hip_mid[1] + torso_h * hem_extra_ratio + scale_spec.gravity_drop_px * 0.25)
        y_hem = min(h - 4, max(y_collar + int(torso_h * 0.85), y_hem_target))

        # Reconstruct de-clothed plate with exact y_hem so dress-wearing mannequins keep lower dress below y_hem
        declothed_bgr, _ = self.reconstruct_clean_plate(
            user_bgr, seg_map, pose_landmarks, scale_spec, orientation=orientation, garment_hem_y=float(y_hem)
        )

        clean_garment, base_color_bgr = self.decontaminate_garment_edges(garment_bgra)
        gy, gx = np.where(clean_garment[:, :, 3] > 15)
        crop_g = (
            clean_garment[np.min(gy) : np.max(gy) + 1, np.min(gx) : np.max(gx) + 1]
            if len(gy) > 0
            else clean_garment
        )
        gh, gw = crop_g.shape[:2]

        person_mask = np.where(seg_map > 0, 255, 0).astype(np.uint8)
        canvas = declothed_bgr.astype(np.float32).copy()

        # 2. Multi-Zone Anatomical Torso Cylinder Grounded on Mannequin Silhouette + XS..XXL Grading
        ys_vec = np.arange(h, dtype=np.float32)
        t_vec = np.clip((ys_vec - float(y_collar)) / max(float(y_hem - y_collar), 1.0), 0.0, 1.0)
        cx_interp = sh_mid[0] * (1.0 - t_vec) + hip_mid[0] * t_vec

        torso_seg = np.isin(seg_map, [4, 5, 6, 7, 8])
        y_ch_idx = int(np.clip(sh_mid[1] + torso_h * 0.26, 0, h - 1))
        y_wa_idx = int(np.clip(sh_mid[1] + torso_h * 0.62, 0, h - 1))
        y_hm_idx = int(np.clip(hip_mid[1], 0, h - 1))

        def _measure_half_w(y_c: int, default_hw: float) -> float:
            band = torso_seg[max(0, y_c - 8) : min(h, y_c + 9), :]
            if np.any(band):
                xs_b = np.where(band)[1]
                return max(float(np.percentile(xs_b, 97) - np.percentile(xs_b, 3)) * 0.5, default_hw * 0.78)
            return default_hw

        g_chest_half = 0.5 * float(em.get("garment_chest_px", scale_spec.torso_width_px))
        g_waist_half = 0.5 * float(em.get("garment_waist_px", g_chest_half * 2.0 * scale_spec.waist_taper_ratio))
        g_hips_half = 0.5 * float(em.get("garment_hips_px", g_chest_half * 2.0 * scale_spec.hem_flare_ratio))

        if orientation == "side":
            photo_side_ch = _measure_half_w(y_ch_idx, torso_h * 0.20)
            photo_side_wa = _measure_half_w(y_wa_idx, torso_h * 0.18)
            photo_side_hm = _measure_half_w(y_hm_idx, torso_h * 0.20)
            # Yan profilde fazla kumaş çoğunlukla öne (göğüs/karın önünde) dökülür: derinlik artışı genişlik oranının ~%70'i
            r_ch_side = 1.0 + 0.70 * (float(em.get("vis_ratio_chest", 1.0 + 0.05 * size_step)) - 1.0)
            r_wa_side = 1.0 + 0.70 * (float(em.get("vis_ratio_waist", 1.0 + 0.05 * size_step)) - 1.0)
            r_hi_side = 1.0 + 0.70 * (float(em.get("vis_ratio_hips", 1.0 + 0.05 * size_step)) - 1.0)
            side_chest = np.clip(photo_side_ch * r_ch_side, torso_h * 0.18, torso_h * 0.52)
            side_waist = np.clip(max(photo_side_wa * r_wa_side, side_chest * (0.92 if is_oversize else 0.82)), torso_h * 0.16, side_chest * 1.02)
            side_hem = np.clip(max(photo_side_hm * r_hi_side, side_chest * (0.95 if is_oversize else 0.86)), torso_h * 0.18, side_chest * 1.08)
            half_w_profile = np.interp(t_vec, [0.0, 0.30, 0.66, 1.0], [side_chest * 0.90, side_chest, side_waist, side_hem])

            # Align side torso centerline & outer bounds with actual side-profile torso silhouette so front bust & midriff are 100% covered
            side_body_seg = np.isin(seg_map, [4, 5, 6, 7])
            row_sb_counts = np.sum(side_body_seg, axis=1)
            med_sb_w = float(np.median(row_sb_counts[row_sb_counts > 10])) if np.any(row_sb_counts > 10) else 0.0
            row_has_sb = np.where(ys_vec <= y_ch_idx, row_sb_counts > 6, row_sb_counts > max(10.0, med_sb_w * 0.82))
            if np.any(row_has_sb):
                valid_rows = np.where(row_has_sb)[0]
                sb_lx_raw = np.argmax(side_body_seg[valid_rows], axis=1).astype(np.float32)
                sb_rx_raw = (w - 1 - np.argmax(side_body_seg[valid_rows, ::-1], axis=1)).astype(np.float32)
                # Interpolate smoothly across any bare midriff gap between crop-top (4) and leggings (6)
                sb_lx = np.interp(ys_vec, valid_rows.astype(np.float32), sb_lx_raw)
                sb_rx = np.interp(ys_vec, valid_rows.astype(np.float32), sb_rx_raw)
                k_sb = max(35, int(torso_h * 0.22) | 1)
                sb_lx = cv2.GaussianBlur(sb_lx.reshape(-1, 1), (1, k_sb), k_sb * 0.22).ravel()
                sb_rx = cv2.GaussianBlur(sb_rx.reshape(-1, 1), (1, k_sb), k_sb * 0.22).ravel()
                sb_mid = 0.5 * (sb_lx + sb_rx)
                cx_interp = 0.75 * sb_mid + 0.25 * cx_interp
                row_L = np.minimum(cx_interp - half_w_profile, sb_lx - 2.0)
                row_R = np.maximum(cx_interp + half_w_profile, sb_rx + 2.0)
            else:
                row_L = cx_interp - half_w_profile
                row_R = cx_interp + half_w_profile
        else:
            sh_left_x = min(float(sh_L[0]), float(sh_R[0]))
            sh_right_x = max(float(sh_L[0]), float(sh_R[0]))
            body_sh_half = max(0.5 * (sh_right_x - sh_left_x), torso_h * 0.24)

            # PHYSICAL SIZE GRADING (doymayan):
            # Görünen yarı genişlik = fotoğraftaki VÜCUT yarı genişliği × max(1, kıyafet_2B / vücut_2B).
            # Vücut genişliği fotoğraftan ölçülür ama eski bol kıyafet / çıplak bel hatasına karşı ANSUR tahminiyle sınırlanır.
            ansur_ch_half = 0.5 * float(em.get("body_chest_px", body_sh_half * 1.8))
            ansur_wa_half = 0.5 * float(em.get("body_waist_px", body_sh_half * 1.4))
            ansur_hm_half = 0.5 * float(em.get("body_hips_px", body_sh_half * 1.7))
            photo_ch_half = float(np.clip(_measure_half_w(y_ch_idx, ansur_ch_half), ansur_ch_half * 0.85, ansur_ch_half * 1.15))
            photo_wa_half = float(np.clip(_measure_half_w(y_wa_idx, ansur_wa_half), ansur_wa_half * 0.85, ansur_wa_half * 1.15))
            photo_hm_half = float(np.clip(_measure_half_w(y_hm_idx, ansur_hm_half), ansur_hm_half * 0.85, ansur_hm_half * 1.15))

            r_ch = float(em.get("vis_ratio_chest", 1.0 + 0.044 * size_step))
            r_wa = float(em.get("vis_ratio_waist", 1.0 + 0.044 * size_step))
            r_hi = float(em.get("vis_ratio_hips", 1.0 + 0.044 * size_step))

            w_chest_half = photo_ch_half * r_ch
            # Omuz dikişi: bol kıyafette omuz düşer ve dışa taşar (göğüs bolluğunun ~%55'i omuza yansır)
            w_sh_half = max(body_sh_half * 1.01, body_sh_half + 0.55 * (w_chest_half - photo_ch_half))
            # Kumaş göğüsten aşağı serbest düşer: bel, göğüs genişliğinin altına çok inemez (dökümlü kumaş içe kıvrılmaz)
            drape_floor = 0.93 if is_oversize else (0.80 if is_slim else 0.87)
            w_waist_half = max(photo_wa_half * r_wa, w_chest_half * drape_floor)
            w_hem_half = max(photo_hm_half * r_hi, w_chest_half * (drape_floor + 0.03))

            half_w_profile = np.interp(
                t_vec,
                [0.0, 0.28, 0.66, 1.0],
                [w_sh_half, w_chest_half, w_waist_half, w_hem_half],
            )
            row_L = cx_interp - half_w_profile
            row_R = cx_interp + half_w_profile

            # Bound lower torso width inside the inner edge of hanging arms below the sleeve cuff
            armpit_y = float(sh_mid[1] + torso_h * 0.35)
            below_armpit = ys_vec > armpit_y
            arm_mask_all = np.isin(seg_map, [14, 15])
            mid_x_int = int(np.clip(sh_mid[0], 20, w - 20))
            arm_overrun_px = float(max(0.0, w_waist_half - photo_wa_half) * 0.85)

            if el_L[1] > sh_L[1] + torso_h * 0.22 and wr_L[1] > el_L[1] - 10.0:
                left_arm_half = arm_mask_all[:, :mid_x_int]
                has_la = np.any(left_arm_half, axis=1)
                la_inner_x = np.where(
                    has_la,
                    mid_x_int - 1 - np.argmax(left_arm_half[:, ::-1], axis=1),
                    row_L,
                ).astype(np.float32)
                la_inner_x = np.minimum(la_inner_x, cx_interp - torso_h * 0.17)
                la_inner_x = cv2.GaussianBlur(la_inner_x.reshape(-1, 1), (1, 27), 6.0).ravel()
                # Bol kıyafet kolun ARKASINA uzanır (kol sonradan Z-index ile üste oyulur); yalnızca fazlalık kadar taşmaya izin ver
                row_L = np.where(below_armpit & has_la, np.maximum(row_L, la_inner_x + 1.0 - arm_overrun_px), row_L)

            if el_R[1] > sh_R[1] + torso_h * 0.22 and wr_R[1] > el_R[1] - 10.0:
                right_arm_half = arm_mask_all[:, mid_x_int:]
                has_ra = np.any(right_arm_half, axis=1)
                ra_inner_x = np.where(
                    has_ra,
                    mid_x_int + np.argmax(right_arm_half, axis=1),
                    row_R,
                ).astype(np.float32)
                ra_inner_x = np.maximum(ra_inner_x, cx_interp + torso_h * 0.17)
                ra_inner_x = cv2.GaussianBlur(ra_inner_x.reshape(-1, 1), (1, 27), 6.0).ravel()
                row_R = np.where(below_armpit & has_ra, np.minimum(row_R, ra_inner_x - 1.0 + arm_overrun_px), row_R)

            # On dress-wearing mannequins, ensure torso covers the dress silhouette where not occluded by arms
            if int(np.sum(seg_map == 7)) > 100:
                dress_only = (seg_map == 7) & (~arm_mask_all)
                row_has_d = np.sum(dress_only, axis=1) > 10
                d_lx = np.where(row_has_d, np.argmax(dress_only, axis=1), row_L).astype(np.float32)
                d_rx = np.where(row_has_d, w - 1 - np.argmax(dress_only[:, ::-1], axis=1), row_R).astype(np.float32)
                in_d_band = (ys_vec >= y_collar) & (ys_vec <= y_hem) & row_has_d
                row_L = np.where(in_d_band, np.minimum(row_L, d_lx), row_L)
                row_R = np.where(in_d_band, np.maximum(row_R, d_rx), row_R)

        # -----------------------------------------------------------------
        # RULE 1: Geometric Collision & Inversion Hard Clamp
        # -----------------------------------------------------------------
        min_torso_w = max(torso_h * 0.36, 56.0) if orientation != "side" else max(torso_h * 0.26, 42.0)
        center_row = 0.5 * (row_L + row_R)
        actual_w = np.maximum(row_R - row_L, min_torso_w)
        row_L = np.clip(center_row - actual_w * 0.5, 0.0, w - min_torso_w - 4.0)
        row_R = np.clip(row_L + actual_w, row_L + min_torso_w, w - 1.0)

        k_row = max(25, int(torso_h * 0.15) | 1)
        row_L = cv2.GaussianBlur(row_L.reshape(-1, 1), (1, k_row), k_row * 0.22).ravel()
        row_R = cv2.GaussianBlur(row_R.reshape(-1, 1), (1, k_row), k_row * 0.22).ravel()

        actual_w = np.maximum(row_R - row_L, min_torso_w)
        center_row = 0.5 * (row_L + row_R)
        row_L = np.clip(center_row - actual_w * 0.5, 0.0, w - min_torso_w - 4.0)
        row_R = np.clip(center_row + actual_w * 0.5, row_L + min_torso_w, w - 1.0)

        # 3. Vectorized 2D Torso UV Mapping
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        xl_2d = row_L[:, None]
        xr_2d = row_R[:, None]
        width_2d = np.maximum(xr_2d - xl_2d, min_torso_w)
        u_2d = (grid_x - xl_2d) / width_2d
        v_norm_2d = np.clip((grid_y - float(y_collar)) / max(float(y_hem - y_collar), 1.0), 0.0, 1.0)

        hem_arc_2d = 5.0 * np.sin(np.clip(u_2d, 0.0, 1.0) * math.pi)
        local_hem_y_2d = float(y_hem) + hem_arc_2d

        neck_cx = float(sh_mid[0])
        collar_half_w = max(sh_span * 0.21, torso_h * 0.12)
        if orientation == "front" and directives.get("carve_chest_neckline", True):
            collar_drop = torso_h * 0.105
            dx_neck_2d = (grid_x - neck_cx) / max(collar_half_w, 1.0)
            in_neck_2d = np.abs(dx_neck_2d) < 1.0
            neck_curve_y_2d = float(y_collar) + collar_drop * (1.0 - np.clip(dx_neck_2d, -1.0, 1.0) ** 2)
            collar_cond = (~in_neck_2d) | (grid_y >= neck_curve_y_2d)
        else:
            dx_neck_2d = np.zeros_like(grid_x)
            in_neck_2d = np.zeros_like(grid_x, dtype=bool)
            neck_curve_y_2d = float(y_collar)
            collar_cond = (grid_y >= float(y_collar))

        # Natural trapezius/acromion top-shoulder slope from collar edge to outer shoulders
        if orientation != "side":
            sh_slope_drop = max(2.0, float(sh_mid[1] - y_collar) * 0.22)
            dist_from_collar = np.maximum(0.0, np.abs(grid_x - neck_cx) - collar_half_w)
            slope_span = max(0.5 * sh_span - collar_half_w + torso_h * 0.12, 22.0)
            slope_1d = float(y_collar) + sh_slope_drop * np.clip(dist_from_collar[0, :] / slope_span, 0.0, 1.15)
            for x_col in range(w):
                if abs(x_col - neck_cx) > collar_half_w * 0.90:
                    ys_sh_skin = np.where(
                        np.isin(seg_map[:, x_col], [4, 5, 11, 14, 15])
                        & (np.arange(h) >= max(4, y_collar - int(torso_h * 0.04)))
                        & (np.arange(h) <= int(sh_mid[1] + torso_h * 0.06))
                    )[0]
                    if len(ys_sh_skin) > 0:
                        top_sh_y = float(np.min(ys_sh_skin)) - 2.0
                        slope_1d[x_col] = min(slope_1d[x_col], max(float(y_collar - int(torso_h * 0.03)), top_sh_y))
            k_sh = max(31, int(sh_span * 0.35) | 1)
            slope_1d = cv2.GaussianBlur(slope_1d.reshape(1, -1).astype(np.float32), (k_sh, 1), k_sh * 0.22).ravel()
            y_shoulder_slope = np.broadcast_to(slope_1d[None, :], (h, w))
        else:
            y_shoulder_slope = np.full_like(grid_x, float(y_collar))

        valid_torso = (
            (grid_y >= y_shoulder_slope)
            & (grid_y <= np.minimum(float(h - 1), float(y_hem + 8)))
            & (grid_x >= xl_2d)
            & (grid_x <= xr_2d)
            & (grid_y <= local_hem_y_2d)
            & collar_cond
        )

        torso_mask = np.where(valid_torso, 1.0, 0.0).astype(np.float32)
        g_torso_x0 = gw * 0.24
        g_torso_w = max(gw * 0.52, 1.0)
        map_x = np.where(valid_torso, g_torso_x0 + np.clip(u_2d, 0.0, 1.0) * g_torso_w, 0.0).astype(np.float32)
        map_y = np.where(valid_torso, (0.12 + v_norm_2d * 0.82) * (gh - 1.0), 0.0).astype(np.float32)

        rng_tex = np.random.default_rng(101)
        cotton_grain = cv2.GaussianBlur(
            rng_tex.normal(0.0, 1.35, (gh, gw)).astype(np.float32), (3, 3), 0
        )[..., None]
        clean_fabric = np.clip(base_color_bgr[None, None, :] + cotton_grain, 0.0, 255.0).astype(np.float32)

        sampled_torso = cv2.remap(
            clean_fabric,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sampled_torso[torso_mask <= 0.01] = base_color_bgr

        # 4. High-Fidelity Un-Thresholded Graphic Print Placement
        torso_rgb = sampled_torso.copy()
        if directives["render_front_print"]:
            chest_cy = float(y_collar + (y_hem - y_collar) * 0.36)
            chest_row_idx = int(min(h - 1, max(0, chest_cy)))
            chest_cx = 0.5 * (row_L[chest_row_idx] + row_R[chest_row_idx])
            print_box_size = min(
                max(torso_h * 0.52, sh_span * 0.55),
                (row_R[chest_row_idx] - row_L[chest_row_idx]) * 0.72,
            )
            p_half_w = print_box_size * 0.50
            p_half_h = print_box_size * 0.50

            py0, py1 = max(y_collar, int(chest_cy - p_half_h)), min(y_hem, int(chest_cy + p_half_h))
            px0, px1 = max(0, int(chest_cx - p_half_w)), min(w, int(chest_cx + p_half_w))

            if py1 > py0 + 20 and px1 > px0 + 20:
                print_source_g = crop_g
                src_patch_test = crop_g[int(gh * 0.16) : int(gh * 0.62), int(gw * 0.27) : int(gw * 0.73), :3].astype(np.float32)
                # If a 3/4 front view (like user2_yan.png) was passed a side/back garment image without the front print,
                # pull the front print from the extracted front garment if its base fabric color matches.
                if src_patch_test.size == 0 or float(np.std(src_patch_test)) < 8.5:
                    try:
                        from vton.paths import goruntu

                        for cand in [goruntu("temp_garment_extracted_on.png"), goruntu("ayiklanmis_on.png")]:
                            if os.path.exists(cand):
                                f_bgra = cv2.imread(cand, cv2.IMREAD_UNCHANGED)
                                if f_bgra is not None and f_bgra.ndim == 3 and f_bgra.shape[2] == 4:
                                    f_clean, f_base = self.decontaminate_garment_edges(f_bgra)
                                    if float(np.linalg.norm(f_base - base_color_bgr)) < 28.0:
                                        f_gy, f_gx = np.where(f_clean[:, :, 3] > 15)
                                        if len(f_gy) > 0:
                                            print_source_g = f_clean[np.min(f_gy) : np.max(f_gy) + 1, np.min(f_gx) : np.max(f_gx) + 1]
                                            break
                    except Exception:
                        pass

                p_gh, p_gw = print_source_g.shape[:2]
                src_patch_bgra = print_source_g[int(p_gh * 0.16) : int(p_gh * 0.62), int(p_gw * 0.27) : int(p_gw * 0.73)]
                if src_patch_bgra.size > 0:
                    pw, ph = px1 - px0, py1 - py0
                    resized_print = cv2.resize(src_patch_bgra[:, :, :3].astype(np.float32), (pw, ph), interpolation=cv2.INTER_LANCZOS4)
                    resized_alpha = cv2.resize(src_patch_bgra[:, :, 3].astype(np.float32), (pw, ph), interpolation=cv2.INTER_LINEAR)

                    # Align border fabric median of patch to base_color_bgr so the entire print is preserved without thresholding
                    b_margin = max(2, min(6, ph // 10, pw // 10))
                    border_mask = np.zeros((ph, pw), dtype=bool)
                    border_mask[:b_margin, :] = True
                    border_mask[-b_margin:, :] = True
                    border_mask[:, :b_margin] = True
                    border_mask[:, -b_margin:] = True
                    valid_border = border_mask & (resized_alpha > 200)
                    border_med = (
                        np.median(resized_print[valid_border], axis=0)
                        if np.any(valid_border)
                        else base_color_bgr
                    )
                    aligned_print = np.clip(
                        resized_print + (base_color_bgr[None, None, :] - border_med[None, None, :]),
                        0.0,
                        255.0,
                    )

                    # Raised-cosine Tukey window: 1.0 across inner 80% of patch, smooth feather at outer 10% border
                    uy = np.linspace(0.0, 1.0, ph, dtype=np.float32)
                    ux = np.linspace(0.0, 1.0, pw, dtype=np.float32)
                    wy = np.where(uy < 0.10, 0.5 - 0.5 * np.cos(np.pi * uy / 0.10), np.where(uy > 0.90, 0.5 - 0.5 * np.cos(np.pi * (1.0 - uy) / 0.10), 1.0))
                    wx = np.where(ux < 0.10, 0.5 - 0.5 * np.cos(np.pi * ux / 0.10), np.where(ux > 0.90, 0.5 - 0.5 * np.cos(np.pi * (1.0 - ux) / 0.10), 1.0))
                    win_2d = (wy[:, None] * wx[None, :] * np.clip((resized_alpha - 180.0) / 60.0, 0.0, 1.0))[..., None]
                    torso_rgb[py0:py1, px0:px1] = (
                        torso_rgb[py0:py1, px0:px1] * (1.0 - win_2d) + aligned_print * win_2d
                    )

        # 5. Smooth Anatomical Sleeves Graded by Size (XS..XXL) & Fit + Strict Bare-Forearm Separation
        sleeve_mask = np.zeros((h, w), dtype=np.float32)
        sleeve_shade_boost = np.ones((h, w), dtype=np.float32)
        below_cuff_arms_bool = np.zeros((h, w), dtype=bool)

        if orientation != "side":
            arms_to_draw = [(sh_L, el_L, wr_L), (sh_R, el_R, wr_R)]
        else:
            near_is_left = (
                pose_landmarks.landmark[11].visibility >= pose_landmarks.landmark[12].visibility
            )
            arms_to_draw = [(p_l_sh, p_l_el, p_l_wr) if near_is_left else (p_r_sh, p_r_el, p_r_wr)]

        arm_seg_mask = np.where(np.isin(seg_map, [14, 15]), 255, 0).astype(np.uint8)
        arm_seg_dil = cv2.dilate(arm_seg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        sleeve_size_mult = float(np.clip(1.0 + 0.050 * size_step + (0.08 if is_oversize else (-0.05 if is_slim else 0.0)), 0.84, 1.28))

        for sh_pt, el_pt, wr_pt in arms_to_draw:
            arm_vec = el_pt - sh_pt
            arm_len = max(float(np.linalg.norm(arm_vec)), 1.0)
            u_dir = arm_vec / arm_len
            n_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float32)
            if n_dir[1] > 0:
                n_dir = -n_dir

            is_raised_arm = (el_pt[1] < sh_pt[1] + torso_h * 0.08) or (wr_pt[1] < sh_pt[1] + torso_h * 0.05)
            base_reach = 0.56 if is_raised_arm else (0.50 if orientation == "side" else 0.56)
            reach_ratio = float(np.clip(base_reach * sleeve_size_mult, 0.42, 0.74))
            sleeve_len = min(arm_len * reach_ratio, torso_h * (0.38 if is_raised_arm else 0.44) * sleeve_size_mult)
            s_start = sh_pt - u_dir * (torso_h * 0.07)
            s_end = sh_pt + u_dir * sleeve_len

            up_thick_top = torso_h * (0.095 if is_raised_arm else 0.090) * sleeve_size_mult
            down_thick_top = torso_h * (0.18 if is_raised_arm else 0.195) * sleeve_size_mult
            up_thick_end = torso_h * (0.090 if is_raised_arm else 0.095) * sleeve_size_mult
            down_thick_end = torso_h * (0.125 if is_raised_arm else 0.145) * sleeve_size_mult

            poly = np.array(
                [
                    s_start + n_dir * up_thick_top,
                    s_start - n_dir * down_thick_top,
                    s_end - n_dir * down_thick_end,
                    s_end + n_dir * up_thick_end,
                ],
                dtype=np.int32,
            )
            this_sleeve = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(this_sleeve, [poly], 255)
            if not is_raised_arm:
                this_sleeve[grid_y < y_shoulder_slope] = 0

            t_up = np.clip(
                ((grid_x - sh_pt[0]) * arm_vec[0] + (grid_y - sh_pt[1]) * arm_vec[1]) / max(arm_len * arm_len, 1.0),
                0.0,
                1.0,
            )
            dist_upper = np.hypot(grid_x - (sh_pt[0] + t_up * arm_vec[0]), grid_y - (sh_pt[1] + t_up * arm_vec[1]))

            fore_vec = wr_pt - el_pt
            fore_len_sq = max(float(np.dot(fore_vec, fore_vec)), 1.0)
            t_fore = np.clip(
                ((grid_x - el_pt[0]) * fore_vec[0] + (grid_y - el_pt[1]) * fore_vec[1]) / fore_len_sq,
                0.0,
                1.0,
            )
            dist_forearm = np.hypot(
                grid_x - (el_pt[0] + t_fore * fore_vec[0]), grid_y - (el_pt[1] + t_fore * fore_vec[1])
            )
            is_on_forearm = (dist_forearm < (dist_upper + 6.0)) & (t_fore > 0.05)

            rel_x = grid_x - sh_pt[0]
            rel_y = grid_y - sh_pt[1]
            along_dist = rel_x * u_dir[0] + rel_y * u_dir[1]
            perp_dist = rel_x * n_dir[0] + rel_y * n_dir[1]
            in_upper_arm_band = (
                (along_dist >= -torso_h * 0.16)
                & (along_dist <= sleeve_len)
                & (np.abs(perp_dist) <= torso_h * 0.25 * sleeve_size_mult)
                & (~is_on_forearm)
            )

            this_sleeve[is_on_forearm | (along_dist > sleeve_len)] = 0
            this_sleeve = np.maximum(
                this_sleeve, np.where(in_upper_arm_band & (arm_seg_dil > 0), 255, 0).astype(np.uint8)
            )
            if not is_raised_arm:
                this_sleeve[grid_y < y_shoulder_slope] = 0

            # Mark bare arm below the sleeve cuff so torso_mask never paints over the forearm/elbow
            this_below_cuff = (
                (arm_seg_mask > 0)
                & ((along_dist > sleeve_len + 2.0) | is_on_forearm)
                & (dist_forearm < (torso_h * 0.11 if orientation == "side" else torso_h * 0.24))
            )
            below_cuff_arms_bool |= this_below_cuff

            norm_perp = np.clip(perp_dist / max(down_thick_top, 1.0), -1.0, 1.0)
            cyl_shade = 1.02 - 0.10 * (norm_perp ** 2)
            sleeve_shade_boost = np.where(this_sleeve > 0, cyl_shade, sleeve_shade_boost)
            sleeve_mask = np.maximum(sleeve_mask, this_sleeve.astype(np.float32) / 255.0)

        full_garment_mask = np.clip(np.maximum(torso_mask, sleeve_mask), 0.0, 1.0)

        if orientation == "front" and not directives.get("disable_skin_bg_intersection", False):
            person_upper_dil = cv2.dilate(person_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            bg_above_shoulders = (person_upper_dil == 0) & (grid_y < float(sh_mid[1] + torso_h * 0.08)) & (sleeve_mask <= 0.05)
            full_garment_mask[bg_above_shoulders] = 0.0

        if orientation == "front" and directives.get("carve_chest_neckline", False):
            neck_cut_mask = in_neck_2d & (grid_y < neck_curve_y_2d) & (grid_y >= float(max(0, y_collar - 25)))
            full_garment_mask[neck_cut_mask] = 0.0

        rib_thickness = max(8.0, torso_h * 0.040)
        dx_rib_2d = (grid_x - neck_cx) / max(collar_half_w + rib_thickness * 0.6, 1.0)
        in_rib_2d = np.abs(dx_rib_2d) < 1.05
        active_neck_curve = neck_curve_y_2d if (orientation == "front" and directives.get("carve_chest_neckline", False)) else (
            float(y_collar) + (torso_h * 0.025) * (1.0 - np.clip(dx_rib_2d, -1.0, 1.0) ** 2)
        )
        rib_band = np.where(
            in_rib_2d & (grid_y >= active_neck_curve) & (grid_y <= active_neck_curve + rib_thickness),
            1.0,
            0.0,
        ).astype(np.float32)
        rib_band = cv2.GaussianBlur(rib_band, (7, 7), 2.0) * full_garment_mask

        # 6. Pure 3D Cylindrical Anatomical Shading
        u_centered = np.clip((u_2d - 0.5) * 2.0, -1.2, 1.2)
        cyl_torso_shade = 1.04 - 0.13 * (u_centered ** 2)
        vertical_grad = 1.02 - 0.05 * v_norm_2d
        fold_wave = 1.0 + 0.020 * np.sin((grid_x - neck_cx) / max(sh_span * 0.15, 12.0)) * np.clip(
            (grid_y - sh_mid[1]) / torso_h, 0.0, 1.0
        )
        armhole_seam = cv2.GaussianBlur(np.abs(cv2.Sobel(torso_mask, cv2.CV_32F, 1, 0, ksize=3)), (15, 15), 4.0)
        seam_shadow = 1.0 - np.clip(armhole_seam * 0.04, 0.0, 0.03)

        total_shading = (cyl_torso_shade * vertical_grad * fold_wave * seam_shadow * sleeve_shade_boost - rib_band * 0.07)[..., None]
        shaded_garment = np.clip(torso_rgb * total_shading, 0.0, 255.0)

        if directives["render_side_print"]:
            try:
                from vton.inference_api import restore_side_garment_print
                from vton.paths import goruntu

                for cand in [goruntu("temp_garment_extracted_on.png"), goruntu("ayiklanmis_on.png")]:
                    if os.path.exists(cand):
                        front_bgra = cv2.imread(cand, cv2.IMREAD_UNCHANGED)
                        if front_bgra is not None:
                            _, f_base = self.decontaminate_garment_edges(front_bgra)
                            if float(np.linalg.norm(f_base - base_color_bgr)) < 28.0:
                                shaded_garment = np.clip(
                                    restore_side_garment_print(
                                        shaded_garment, front_bgra, h, w, shirt_mask=torso_mask
                                    ),
                                    0.0,
                                    255.0,
                                )
                                break
            except Exception:
                pass

        # 7. Orientation-Gated Z-Index Carving (with Below-Cuff Bare Forearm Protection)
        z_masks = self.extract_anatomical_z_masks(user_bgr, seg_map, pose_landmarks, orientation=orientation)
        raw_garment_u8 = np.clip(full_garment_mask * 255.0, 0, 255).astype(np.uint8)
        fg_occlusion_u8 = np.where(below_cuff_arms_bool, 255, 0).astype(np.uint8)

        if orientation in ("side", "back"):
            if orientation == "back" and directives.get("carve_hair", False) and np.any(z_masks["hair"] > 0):
                hair_clean = cv2.morphologyEx(
                    z_masks["hair"], cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21))
                )
                hair_clean = cv2.medianBlur(hair_clean, 19)
                fg_occlusion_u8 = cv2.bitwise_or(fg_occlusion_u8, hair_clean)
            if orientation == "side" and np.any(z_masks["forearms"] > 0):
                fg_occlusion_u8 = cv2.bitwise_or(fg_occlusion_u8, z_masks["forearms"])
                fg_occlusion_u8 = cv2.bitwise_or(fg_occlusion_u8, z_masks["hands"])
            carved_garment_u8 = cv2.subtract(raw_garment_u8, fg_occlusion_u8)
        else:
            if directives.get("carve_hair", False) and np.any(z_masks["hair"] > 0):
                hair_clean = cv2.morphologyEx(
                    z_masks["hair"], cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21))
                )
                hair_clean = cv2.medianBlur(hair_clean, 19)
                fg_occlusion_u8 = cv2.bitwise_or(fg_occlusion_u8, hair_clean)

            if not directives.get("solid_block_mode", False):
                if directives.get("carve_face_neck", False):
                    fg_occlusion_u8 = cv2.bitwise_or(fg_occlusion_u8, z_masks["face_neck"])
                if directives.get("carve_chest_neckline", False):
                    fg_occlusion_u8 = cv2.bitwise_or(fg_occlusion_u8, z_masks["chest"])
                if directives.get("carve_forearms", False):
                    fg_occlusion_u8 = cv2.bitwise_or(fg_occlusion_u8, z_masks["forearms"])
                if directives.get("carve_hands", False):
                    fg_occlusion_u8 = cv2.bitwise_or(fg_occlusion_u8, z_masks["hands"])
                if directives.get("carve_foreground_objects", False):
                    fg_occlusion_u8 = cv2.bitwise_or(fg_occlusion_u8, z_masks["foreground_objects"])

            carved_garment_u8 = cv2.subtract(raw_garment_u8, fg_occlusion_u8)

        _, effective_alpha = self.refine_and_antialias_mask(
            carved_garment_u8, close_ksize=9, open_ksize=5, blur_ksize=7, blur_sigma=1.6
        )

        hem_shadow = cv2.GaussianBlur(effective_alpha, (15, 25), 6.0)
        hem_shadow_shifted = np.zeros_like(hem_shadow)
        hem_shadow_shifted[7:, :] = hem_shadow[:-7, :]
        under_hem_shadow = (
            np.clip(hem_shadow_shifted - effective_alpha, 0.0, 1.0)
            * (1.0 - (fg_occlusion_u8 > 0).astype(np.float32))
            * 0.22
        )
        canvas = canvas * (1.0 - under_hem_shadow[..., None])

        final_bgr = np.clip(
            canvas * (1.0 - effective_alpha[..., None]) + shaded_garment * effective_alpha[..., None],
            0.0,
            255.0,
        ).astype(np.uint8)

        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[carved_garment_u8 > 64] = [255, 255, 255, 255]
        return final_bgr, mask_rgba

    # -----------------------------------------------------------------------
    # Solution 1: Vectorized Lower-Body Synthesis with Bipedal Legs, XS-XXL Grading & De-Clothing
    # -----------------------------------------------------------------------
    def synthesize_vectorized_lower_vton(
        self,
        user_bgr: np.ndarray,
        garment_bgra: np.ndarray,
        pose_landmarks,
        seg_map: np.ndarray,
        scale_spec: AsymmetricScaleSpec,
        is_side_profile: bool = False,
        is_back_view: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        100% vectorized lower-body (pants/trousers/skirt) synthesis:
        - First de-clothes the mannequin via `reconstruct_clean_plate` so any old dresses (`User 2`) or
          black leggings (`User 1`) are replaced with realistic 3D-shaded bare legs and clean studio background.
        - For `pants` in front/back views: synthesizes the Pelvic Block + Two Independent Anatomical Leg Cylinders
          with a Crotch-to-Ankle inseam separation barrier and per-leg 3D cylindrical shading.
        - For `pants` in side view: follows the side skeletal leg chain (`Waist -> Hip -> Knee -> Ankle`) with
          ANSUR sagittal depths.
        - For `skirt`: synthesizes an A-line / fitted skirt silhouette graded across XS..XXL, revealing bare lower legs.
        """
        h, w = user_bgr.shape[:2]
        orientation = self.detect_pose_orientation(pose_landmarks, w, h)
        if is_side_profile:
            orientation = "side"
        elif is_back_view:
            orientation = "back"

        directives = CategoryRouter.get_masking_directives(orientation, "lower_body")
        lm = self.router.extract_category_landmarks(pose_landmarks, w, h, "lower_body")
        coords = lm["all_coords"][:, :2]
        sh_mid = lm["shoulder_mid"]
        anchor_23_24 = lm["lower_body_anchor"]
        knee_L, knee_R = lm["left_knee"], lm["right_knee"]
        ank_L, ank_R = lm["left_ankle"], lm["right_ankle"]
        knee_mid = lm["knee_mid"]
        ank_mid = lm["ankle_mid"]
        torso_h = max(abs(float(anchor_23_24[1] - sh_mid[1])), 80.0)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        ys_vec = np.arange(h, dtype=np.float32)

        # 1. Anchor top-center of pants/skirt to true waist contour (existing pants/skirt OR hourglass dress waist pinch)
        anchor_x = float(anchor_23_24[0])
        anchor_y = float(anchor_23_24[1] - torso_h * 0.25)
        existing_lower_all = np.isin(seg_map, [5, 6, 8])
        ys_ex, _ = np.where(existing_lower_all)
        if len(ys_ex) > 50:
            seg_top_y = float(np.percentile(ys_ex, 1))
            anchor_y = max(float(sh_mid[1] + torso_h * 0.38), seg_top_y - 2.0)
            top_band = existing_lower_all[int(max(0, anchor_y)) : int(min(h, anchor_y + 18)), :]
            if np.any(top_band):
                xs_top = np.where(top_band)[1]
                anchor_x = 0.5 * float(np.percentile(xs_top, 3) + np.percentile(xs_top, 97))
        elif int(np.sum(seg_map == 7)) > 100:
            # Mannequin wears a dress (e.g. User 2): locate exact hourglass waist pinch row
            dress_mask = seg_map == 7
            y_win0 = int(np.clip(sh_mid[1] + torso_h * 0.44, 0, h - 10))
            y_win1 = int(np.clip(anchor_23_24[1] - torso_h * 0.14, y_win0 + 6, h - 1))
            band_d = dress_mask[y_win0:y_win1, :]
            row_valid = np.sum(band_d, axis=1) > 16
            if np.any(row_valid):
                xl_d = np.where(row_valid, np.argmax(band_d, axis=1), 0).astype(np.float32)
                xr_d = np.where(row_valid, w - 1 - np.argmax(band_d[:, ::-1], axis=1), float(w)).astype(np.float32)
                w_d = np.where(row_valid, xr_d - xl_d, 1e6)
                pinch_rel = int(np.argmin(w_d))
                anchor_y = float(y_win0 + pinch_rel)
                anchor_x = 0.5 * float(xl_d[pinch_rel] + xr_d[pinch_rel])

        # Reconstruct de-clothed plate using exact waist anchor_y so lower dress below anchor_y is cleanly erased
        declothed_bgr, _ = self.reconstruct_clean_plate(
            user_bgr,
            seg_map,
            pose_landmarks,
            scale_spec,
            orientation=orientation,
            garment_hem_y=float(anchor_y + 2.0),
        )
        canvas = declothed_bgr.astype(np.float32).copy()

        # 2. Extract Clean Garment Crop, Restore Bilateral Symmetry & Level Waistband
        clean_garment, base_color_bgr = self.decontaminate_garment_edges(garment_bgra)
        gy, gx = np.where(clean_garment[:, :, 3] > 25)
        crop_g = (
            clean_garment[np.min(gy) : np.max(gy) + 1, np.min(gx) : np.max(gx) + 1]
            if len(gy) > 0
            else clean_garment
        )
        gh, gw = crop_g.shape[:2]
        g_alpha_eroded = cv2.erode(
            np.where(crop_g[:, :, 3] > 180, 255, 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ) > 0
        g_rgb = crop_g[:, :, :3].astype(np.float32)

        leg_sample = crop_g[int(gh * 0.22) : int(gh * 0.68), int(gw * 0.56) : int(gw * 0.84)]
        leg_valid = leg_sample[:, :, 3] > 220
        if np.any(leg_valid):
            p_bgr = leg_sample[:, :, :3][leg_valid].astype(np.float32)
            p_lum = np.mean(p_bgr, axis=1)
            fabric_med_bgr = np.median(p_bgr[p_lum >= np.percentile(p_lum, 50)], axis=0).astype(np.float32)
        else:
            fabric_med_bgr = base_color_bgr.copy()

        gx_grid, gy_grid = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32))
        g_lum = np.mean(g_rgb, axis=2)
        wb_sample = g_rgb[int(gh * 0.04) : int(gh * 0.14), int(gw * 0.46) : int(gw * 0.76)]
        wb_med_bgr = np.median(wb_sample.reshape(-1, 3), axis=0).astype(np.float32) if wb_sample.size > 0 else fabric_med_bgr
        wb_lum = float(np.mean(wb_med_bgr))
        waist_clip_mask = ((gy_grid < gh * 0.24) & (gx_grid < gw * 0.45) & (g_lum < wb_lum * 0.75)).astype(np.float32)
        if np.any(waist_clip_mask > 0):
            wc_alpha = cv2.GaussianBlur(
                cv2.dilate(waist_clip_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))),
                (15, 15),
                4.0,
            )[..., None]
            # Mirror clean right hip onto handbag-occluded left hip
            g_rgb_mirror = g_rgb[:, ::-1, :]
            g_rgb = g_rgb * (1.0 - wc_alpha) + g_rgb_mirror * wc_alpha
        bad_pixels = (~g_alpha_eroded) | (g_lum > 185.0)
        if np.any(bad_pixels):
            g_rgb[bad_pixels] = fabric_med_bgr

        col_has_alpha = np.sum(g_alpha_eroded, axis=0) > 10
        col_top_y = np.where(col_has_alpha, np.argmax(g_alpha_eroded, axis=0), 0).astype(np.float32)
        # Kaynak bel bandı sağlam mı? (Ürün fotoğrafında üst giysi / el / çanta bel bandını kapatıyorsa üst kenar
        # çapraz-düzensiz olur; bu durumda "gerçek pelvis dokusu" harmanlanmaz, temiz kumaş dokusu kullanılır.)
        if np.count_nonzero(col_has_alpha) > 10:
            top_vals = col_top_y[col_has_alpha]
            top_spread = float(np.percentile(top_vals, 90) - np.percentile(top_vals, 10)) / max(float(gh), 1.0)
            top_cover = float(np.mean(g_alpha_eroded[: max(2, int(gh * 0.12)), :][:, col_has_alpha]))
            waistband_reliable = (top_spread < 0.06) and (top_cover > 0.70)
        else:
            waistband_reliable = False
        col_top_y = cv2.GaussianBlur(col_top_y.reshape(1, -1), (31, 1), 7.0).ravel()
        col_top_y = np.clip(col_top_y, 0.0, gh * 0.12)
        level_map_y = col_top_y[None, :] + (gy_grid / max(float(gh - 1), 1.0)) * (float(gh - 1) - col_top_y[None, :])
        g_rgb = cv2.remap(g_rgb, gx_grid, level_map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        g_alpha_leveled = cv2.remap(
            g_alpha_eroded.astype(np.uint8) * 255,
            gx_grid,
            level_map_y.astype(np.float32),
            cv2.INTER_NEAREST,
        ) > 127

        # Source horizontal bounds per row in leveled crop_g
        right_leg_mask = g_alpha_leveled.copy()
        right_leg_mask[:, : int(gw * 0.50)] = False
        row_has_rg = np.sum(right_leg_mask, axis=1) > 5
        g_rleg_l = np.where(row_has_rg, np.argmax(right_leg_mask, axis=1), int(gw * 0.54)).astype(np.float32)
        g_rleg_r = np.where(row_has_rg, gw - 1 - np.argmax(right_leg_mask[:, ::-1], axis=1), int(gw * 0.86)).astype(np.float32)
        g_rleg_l = cv2.GaussianBlur(g_rleg_l.reshape(-1, 1), (1, 31), 7.0).ravel() + 4.0
        g_rleg_r = cv2.GaussianBlur(g_rleg_r.reshape(-1, 1), (1, 31), 7.0).ravel() - 4.0
        g_rleg_w = np.maximum(g_rleg_r - g_rleg_l, 12.0)

        # Measure source garment's knee-to-thigh and hem-to-knee flare ratios (detects bootcut / flare trousers!)
        src_thigh_w = float(np.median(g_rleg_w[int(gh * 0.30) : int(gh * 0.45)])) if gh > 40 else 40.0
        src_knee_w = float(np.median(g_rleg_w[int(gh * 0.55) : int(gh * 0.68)])) if gh > 40 else 36.0
        src_hem_w = float(np.median(g_rleg_w[int(gh * 0.82) : int(gh * 0.94)])) if gh > 40 else 42.0
        src_flare_ratio = float(np.clip(src_hem_w / max(src_knee_w, 12.0), 0.88, 1.38))

        row_has_g = np.sum(g_alpha_leveled, axis=1) > 6
        g_row_l = np.where(row_has_g, np.argmax(g_alpha_leveled, axis=1), int(gw * 0.14)).astype(np.float32)
        g_row_r = np.where(row_has_g, gw - 1 - np.argmax(g_alpha_leveled[:, ::-1], axis=1), int(gw * 0.86)).astype(np.float32)
        g_row_l = cv2.GaussianBlur(g_row_l.reshape(-1, 1), (1, 31), 7.0).ravel() + 5.0
        g_row_r = cv2.GaussianBlur(g_row_r.reshape(-1, 1), (1, 31), 7.0).ravel() - 5.0

        # 3. Target Silhouette Grounded on Mannequin Waist/Hip Contour + XS..XXL & Fit Grading
        em = getattr(scale_spec, "extra_metrics", None) or {}
        size_step = float(em.get("size_step", 0.0))  # XS=-2 .. XXL=+3
        len_step_px = size_step * (torso_h * 0.026)

        sub = scale_spec.sub_category
        fit_p = self.pants_fit_profile(scale_spec, is_side_profile=(orientation == "side"))

        # Measure actual mannequin waist & hip contour half-width in photo
        lower_cloth_seg = np.isin(seg_map, [4, 5, 6, 7, 8, 12, 13])
        wb_y_chk = int(np.clip(anchor_y + 4.0, 0, h - 1))
        hip_y_chk = int(np.clip(anchor_23_24[1] + torso_h * 0.05, 0, h - 1))
        wb_band = lower_cloth_seg[max(0, wb_y_chk - 6) : min(h, wb_y_chk + 7), :]
        hip_band = lower_cloth_seg[max(0, hip_y_chk - 8) : min(h, hip_y_chk + 9), :]
        photo_waist_half = (
            0.5 * float(np.percentile(np.where(wb_band)[1], 97) - np.percentile(np.where(wb_band)[1], 3))
            if np.any(wb_band)
            else torso_h * 0.23
        )
        photo_hip_half = (
            0.5 * float(np.percentile(np.where(hip_band)[1], 97) - np.percentile(np.where(hip_band)[1], 3))
            if np.any(hip_band)
            else torso_h * 0.29
        )
        photo_hip_half = max(photo_hip_half, photo_waist_half * 1.12)

        # Physical Clamping Rule:
        # - In tight zones (ease_cm <= 1.0..1.5), garment contour locks 1:1 onto the body contour (cannot shrink inside body or float outside).
        # - Only in non-tight / loose zones (ease_cm > 1.5, or flare hem below the knee) does the garment stand away from the body contour.
        g_waist_3d = float(em.get("garment_waist_3d_cm", 74.0))
        g_hips_3d = float(em.get("garment_hips_3d_cm", 100.0))
        u_waist_3d = float(em.get("user_waist_3d_cm", 74.0))
        u_hips_3d = float(em.get("user_hips_3d_cm", 96.0))
        stretch_tol_cm = float(em.get("stretch_tolerance_cm", 0.0))
        waist_ease_cm = (g_waist_3d + stretch_tol_cm * 0.45) - u_waist_3d
        hips_ease_cm = (g_hips_3d + stretch_tol_cm * 0.45) - u_hips_3d

        # Doymayan fiziksel oran: görünen silüet = vücut × max(1, kıyafet_2B / vücut_2B).
        # Bel bandı vücuda oturur (lastik/kemer), bu yüzden bel bolluğunun yalnızca %35'i silüete yansır.
        r_waist = float(em.get("vis_ratio_waist", 1.0))
        r_hips = float(em.get("vis_ratio_hips", 1.0))
        r_thigh = float(em.get("vis_ratio_thigh", 1.0))
        r_knee = float(em.get("vis_ratio_knee", 1.0))
        r_hem = float(em.get("vis_ratio_hem", 1.3))
        waist_pos_mult = 1.0 + 0.35 * (r_waist - 1.0)
        hip_pos_mult = r_hips
        w_waist_half = photo_waist_half * waist_pos_mult
        w_hip_half = max(photo_hip_half * hip_pos_mult, w_waist_half * 1.08)

        if sub == "pants":
            leg_y0, leg_y1 = int(gh * 0.40), int(gh * 0.90)
            leg_x0 = int(np.median(g_rleg_l[leg_y0:leg_y1]))
            leg_x1 = int(np.median(g_rleg_r[leg_y0:leg_y1]))
            if leg_x1 > leg_x0 + 16 and leg_y1 > leg_y0 + 20:
                raw_swatch = g_rgb[leg_y0:leg_y1, leg_x0:leg_x1].copy()
            else:
                raw_swatch = g_rgb[int(gh * 0.35) : int(gh * 0.85), int(gw * 0.20) : int(gw * 0.80)].copy()
            swatch_full = cv2.resize(raw_swatch, (gw, gh), interpolation=cv2.INTER_LINEAR)
            swatch_low = cv2.GaussianBlur(swatch_full, (51, 51), 16.0)
            swatch_detail = np.clip(swatch_full - swatch_low, -18.0, 18.0)
            g_rgb_pants_clean = np.clip(fabric_med_bgr[None, None, :] + swatch_detail * 0.85, 0.0, 255.0).astype(np.float32)

            # Unclamped anatomical ankle hem (crucial when photo is cropped at thigh/knee like user2_arka/user2_yan!)
            y_hem_anatomical = max(
                anchor_y + torso_h * 1.70,
                float(max(ank_L[1], ank_R[1]) + scale_spec.gravity_drop_px * 0.14 + len_step_px),
            )
            if y_hem_anatomical >= float(h - 6):
                # Photo is cropped above the ankles: extend pant legs cleanly through bottom frame
                y_hem_draw = float(h + 8)
            else:
                y_hem_draw = min(float(h - 4), y_hem_anatomical)
        else:
            g_rgb_pants_clean = g_rgb
            y_hem_anatomical = max(
                float(knee_mid[1] + torso_h * 0.04 + scale_spec.gravity_drop_px * 0.28 + len_step_px * 1.4),
                anchor_y + torso_h * (1.05 + 0.055 * size_step),
            )
            y_hem_draw = min(float(h - 12), y_hem_anatomical)

        v_norm = np.clip((ys_vec - anchor_y) / max(y_hem_anatomical - anchor_y, 1.0), 0.0, 1.0)
        v_2d = v_norm[:, None]
        src_y_idx = np.clip((v_norm * (gh - 1)).astype(np.int32), 0, gh - 1)
        map_y = np.broadcast_to(v_2d * float(gh - 1), (h, w)).copy().astype(np.float32)

        # Fabric chromaticity unit vector for realistic studio specular highlights on dark navy/black woven fabrics
        fab_norm = max(float(np.linalg.norm(fabric_med_bgr)), 1.0)
        fab_tint = np.clip(fabric_med_bgr / fab_norm * 1.65, 0.65, 1.45)[None, None, :]

        if sub == "pants" and orientation != "side":
            # Front & Back Pants: C1-Continuous Pelvic-to-Leg Silhouette + True Bootcut/Flare Proportions
            legs = sorted(
                [(23, 25, 27), (24, 26, 28)],
                key=lambda t: float(coords[t[0], 0]) + float(coords[t[1], 0]),
            )
            hip_L_pt, knee_L_pt, ank_L_pt = coords[legs[0][0]], coords[legs[0][1]], coords[legs[0][2]]
            hip_R_pt, knee_R_pt, ank_R_pt = coords[legs[1][0]], coords[legs[1][1]], coords[legs[1][2]]

            y_hip = float(anchor_23_24[1] + torso_h * 0.04)
            y_crotch = float(anchor_23_24[1] + torso_h * 0.14)
            hip_v = float(np.clip((y_hip - anchor_y) / max(y_hem_anatomical - anchor_y, 1.0), 0.10, 0.22))
            crotch_v = float(np.clip((y_crotch - anchor_y) / max(y_hem_anatomical - anchor_y, 1.0), hip_v + 0.05, 0.30))
            knee_y_mid = 0.5 * float(knee_L_pt[1] + knee_R_pt[1])
            knee_v = float(np.clip((knee_y_mid - anchor_y) / max(y_hem_anatomical - anchor_y, 1.0), crotch_v + 0.18, 0.76))

            seam_L_x = anchor_x - 0.45 * w_hip_half
            seam_R_x = anchor_x + 0.45 * w_hip_half

            cx_0 = np.interp(v_norm, [0.0, crotch_v, knee_v, 1.0], [seam_L_x, float(hip_L_pt[0]), float(knee_L_pt[0]), float(ank_L_pt[0])])
            cx_1 = np.interp(v_norm, [0.0, crotch_v, knee_v, 1.0], [seam_R_x, float(hip_R_pt[0]), float(knee_R_pt[0]), float(ank_R_pt[0])])

            # Handle crossed legs (e.g. User 2 where ankles cross) without inverting cx_R - cx_L
            cx_L_raw = np.minimum(cx_0, cx_1)
            cx_R_raw = np.maximum(cx_0, cx_1)

            straight_blend = float(fit_p["straight"]) * 0.15
            cx_L = (1.0 - straight_blend) * cx_L_raw + straight_blend * seam_L_x
            cx_R = (1.0 - straight_blend) * cx_R_raw + straight_blend * seam_R_x
            mid_legs = 0.5 * (cx_L + cx_R)
            sep_now = np.maximum(cx_R - cx_L, 12.0)

            # Anatomical thigh & knee half-widths so bootcut/flare pants fit the thigh/knee and flare below the knee
            size_ease = 1.0 + 0.046 * size_step
            anat_thigh_hw = max(torso_h * 0.165, photo_hip_half * 0.44)
            anat_knee_hw = max(torso_h * 0.118, photo_hip_half * 0.31)
            # Uyluk / diz / paça: ANSUR bacak çevresi oranları + kalıp bolluğu → doymayan görünür genişlik oranı
            thigh_hw_out = anat_thigh_hw * r_thigh
            knee_hw_out = anat_knee_hw * r_knee
            anat_ankle_hw = anat_knee_hw * (0.235 / 0.385)
            hem_hw_out = max(anat_ankle_hw * r_hem, knee_hw_out * 0.78)
            if src_flare_ratio > 1.10:  # Ürün fotoğrafında İspanyol paça / bootcut algılandıysa koru
                hem_hw_out = max(hem_hw_out, knee_hw_out * src_flare_ratio)

            # Single C1-continuous outer contour from waist (v=0) -> hip (v=hip_v) -> thigh (v=crotch_v) -> knee -> flare hem
            crotch_row_i = int(np.clip(round(y_crotch), 0, h - 1))
            hip_row_i = int(np.clip(round(y_hip), 0, h - 1))
            hip_cx = float(mid_legs[hip_row_i])
            crotch_out_L = min(hip_cx - w_hip_half * 0.96, float(cx_L[crotch_row_i]) - thigh_hw_out)
            crotch_out_R = max(hip_cx + w_hip_half * 0.96, float(cx_R[crotch_row_i]) + thigh_hw_out)
            hip_out_L = min(hip_cx - w_hip_half, 0.35 * (anchor_x - w_waist_half) + 0.65 * crotch_out_L)
            hip_out_R = max(hip_cx + w_hip_half, 0.35 * (anchor_x + w_waist_half) + 0.65 * crotch_out_R)

            leg_hw_after_crotch = np.interp(
                v_norm,
                [crotch_v, (crotch_v + knee_v) * 0.5, knee_v, 1.0],
                [thigh_hw_out, 0.52 * thigh_hw_out + 0.48 * knee_hw_out, knee_hw_out, hem_hw_out],
            )
            # Smoothly anchor leg outer edge at crotch_v to crotch_out_L / crotch_out_R so zero step ever occurs!
            crotch_delta_L = crotch_out_L - (float(cx_L[crotch_row_i]) - thigh_hw_out)
            crotch_delta_R = crotch_out_R - (float(cx_R[crotch_row_i]) + thigh_hw_out)
            decay_after_crotch = np.clip(1.0 - (v_norm - crotch_v) / max(knee_v - crotch_v, 0.15), 0.0, 1.0)

            pelvis_L_curve = np.interp(v_norm, [0.0, hip_v, crotch_v], [anchor_x - w_waist_half, hip_out_L, crotch_out_L])
            pelvis_R_curve = np.interp(v_norm, [0.0, hip_v, crotch_v], [anchor_x + w_waist_half, hip_out_R, crotch_out_R])
            leg_L_curve = (cx_L - leg_hw_after_crotch) + crotch_delta_L * decay_after_crotch
            leg_R_curve = (cx_R + leg_hw_after_crotch) + crotch_delta_R * decay_after_crotch

            l_out = np.where(v_norm <= crotch_v, pelvis_L_curve, leg_L_curve)
            r_out = np.where(v_norm <= crotch_v, pelvis_R_curve, leg_R_curve)

            # Ensure outer bounds cover actual leg segmentation on crossed-leg poses (User 2)
            leg_seg_mask = np.isin(seg_map, [6, 12, 13])
            row_has_leg_seg = np.sum(leg_seg_mask, axis=1) > 8
            seg_lx = np.where(row_has_leg_seg, np.argmax(leg_seg_mask, axis=1), l_out).astype(np.float32)
            seg_rx = np.where(row_has_leg_seg, w - 1 - np.argmax(leg_seg_mask[:, ::-1], axis=1), r_out).astype(np.float32)
            below_crotch_rows = (ys_vec > y_crotch) & (ys_vec <= y_hem_draw) & row_has_leg_seg
            l_out = np.where(below_crotch_rows, np.minimum(l_out, seg_lx - 2.0), l_out)
            r_out = np.where(below_crotch_rows, np.maximum(r_out, seg_rx + 2.0), r_out)

            l_out = cv2.GaussianBlur(l_out.reshape(-1, 1), (1, 29), 6.0).ravel()
            r_out = cv2.GaussianBlur(r_out.reshape(-1, 1), (1, 29), 6.0).ravel()

            # Detect crossed legs (User 2) vs open-stance legs (User 1)
            raw_knee_sep = abs(float(knee_R_pt[0] - knee_L_pt[0]))
            raw_ank_sep = abs(float(ank_R_pt[0] - ank_L_pt[0]))
            is_crossed_or_closed = (
                (raw_knee_sep < torso_h * 0.19)
                or (float(ank_R_pt[0] - ank_L_pt[0]) * float(hip_R_pt[0] - hip_L_pt[0]) <= 0.0)
            )

            if is_crossed_or_closed:
                l_in = mid_legs + 1.5
                r_in = mid_legs - 1.5
            else:
                # Open-stance legs (e.g. User 1): anatomical inner leg half-width preserving the real thigh/knee daylight gap!
                leg_hw_in = np.interp(
                    v_norm,
                    [crotch_v, (crotch_v + knee_v) * 0.5, knee_v, 1.0],
                    [
                        thigh_hw_out * 0.96,
                        0.5 * (thigh_hw_out + knee_hw_out) * 0.97,
                        knee_hw_out * 0.96,
                        hem_hw_out * 0.92,
                    ],
                )
                arch_t = np.clip((v_norm - crotch_v) / max(1.0 - crotch_v, 0.1), 0.0, 1.0)
                ank_gap_target = max(4.0, min(14.0 - size_step * 1.2, raw_ank_sep * 0.30))
                knee_gap_target = max(9.0, min(22.0 - size_step * 1.2, raw_knee_sep * 0.28))
                min_gap_px = np.interp(arch_t, [0.0, 0.06, 0.45, 1.0], [0.0, 4.0, knee_gap_target, ank_gap_target])
                l_in = np.where(
                    v_norm <= crotch_v,
                    mid_legs + 1.5,
                    np.minimum(cx_L + leg_hw_in, mid_legs - 0.5 * min_gap_px),
                )
                r_in = np.where(
                    v_norm <= crotch_v,
                    mid_legs - 1.5,
                    np.maximum(cx_R - leg_hw_in, mid_legs + 0.5 * min_gap_px),
                )
                l_in = cv2.GaussianBlur(l_in.reshape(-1, 1), (1, 15), 3.0).ravel()
                r_in = cv2.GaussianBlur(r_in.reshape(-1, 1), (1, 15), 3.0).ravel()

            # Per-leg masks and cylindrical coordinates
            u_pelvis = np.clip((grid_x - l_out[:, None]) / np.maximum((r_out - l_out)[:, None], 12.0), 0.0, 1.0)
            u_left_leg = np.clip((grid_x - l_out[:, None]) / np.maximum((l_in - l_out)[:, None], 8.0), 0.0, 1.0)
            u_right_leg = np.clip((grid_x - r_in[:, None]) / np.maximum((r_out - r_in)[:, None], 8.0), 0.0, 1.0)

            wb_curve = anchor_y + 3.0 * np.sin(u_pelvis * math.pi)
            hem_curve_L = y_hem_draw + 4.0 * np.sin(u_left_leg * math.pi)
            hem_curve_R = y_hem_draw + 4.0 * np.sin(u_right_leg * math.pi)

            in_pelvis = (grid_y >= wb_curve) & (grid_y <= y_crotch) & (grid_x >= l_out[:, None]) & (grid_x <= r_out[:, None])
            in_left_leg = (grid_y > y_crotch) & (grid_y <= hem_curve_L) & (grid_x >= l_out[:, None]) & (grid_x <= l_in[:, None])
            in_right_leg = (grid_y > y_crotch) & (grid_y <= hem_curve_R) & (grid_x >= r_in[:, None]) & (grid_x <= r_out[:, None])
            valid_lower = in_pelvis | in_left_leg | in_right_leg

            map_x = (u_pelvis * float(gw - 1)).astype(np.float32)
            remapped_pants = cv2.remap(
                g_rgb_pants_clean, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )

            # Blend real leveled waistband & hip pocket detail from g_rgb in the upper pelvis (v_norm < 0.22)
            src_pelvis_l = g_row_l[src_y_idx][:, None]
            src_pelvis_r = g_row_r[src_y_idx][:, None]
            map_x_real = (src_pelvis_l + u_pelvis * np.maximum(src_pelvis_r - src_pelvis_l, 8.0)).astype(np.float32)
            real_pelvis_rgb = cv2.remap(
                g_rgb, map_x_real, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            pelvis_tex_w = np.clip(1.0 - (v_2d / max(crotch_v * 0.95, 0.18)), 0.0, 1.0) * (0.45 if waistband_reliable else 0.0)
            remapped_pants = remapped_pants * (1.0 - pelvis_tex_w[..., None]) + real_pelvis_rgb * pelvis_tex_w[..., None]

            # 3D Cylindrical Shading + Studio Specular Sheen + Pressed Trouser Crease
            w_leg_blend = (0.5 - 0.5 * np.cos(np.pi * np.clip((v_norm - (crotch_v - 0.08)) / 0.16, 0.0, 1.0)))[:, None]
            u_cyl_pelvis = np.clip((u_pelvis - 0.5) * 2.0, -1.2, 1.2)
            is_right_side = grid_x >= mid_legs[:, None]
            u_cyl_legs = np.clip(
                np.where(is_right_side, (u_right_leg - 0.5) * 2.0, (u_left_leg - 0.5) * 2.0),
                -1.2,
                1.2,
            )
            shade_pelvis = 1.06 - 0.18 * (u_cyl_pelvis ** 2)
            shade_legs = 1.08 - 0.22 * (u_cyl_legs ** 2)
            cyl_shade = (shade_pelvis * (1.0 - w_leg_blend) + shade_legs * w_leg_blend)[..., None]

            # Additive studio diffuse highlight along leg & hip cylinders + pressed center crease
            spec_pelvis = np.exp(-2.8 * (u_cyl_pelvis ** 2)) * 10.0 - (u_cyl_pelvis ** 2) * 8.0
            spec_legs = (
                np.exp(-3.2 * (u_cyl_legs ** 2)) * 15.0
                + np.exp(-0.5 * (u_cyl_legs / 0.12) ** 2) * 4.5
                - (u_cyl_legs ** 2) * 11.0
            )
            additive_sheen = ((spec_pelvis * (1.0 - w_leg_blend) + spec_legs * w_leg_blend)[..., None]) * fab_tint

            # Subtle knee & flare drape ripples
            flare_ripple = (
                np.sin(u_cyl_legs * math.pi * 2.0)
                * np.clip((v_2d - knee_v) / max(1.0 - knee_v, 0.15), 0.0, 1.0)
                * 4.0
            )[..., None] * fab_tint

            # Tailored waistband seam, fly seam, and inter-leg inseam / overlap shadow
            wb_seam = np.exp(-0.5 * ((grid_y - (wb_curve + 13.0)) / 2.0) ** 2) * (grid_y >= wb_curve)
            fly_seam = (
                np.exp(-0.5 * ((grid_x - mid_legs[:, None]) / 2.0) ** 2)
                * (grid_y >= wb_curve + 13.0)
                * (grid_y <= y_crotch - 3.0)
                * (1.0 if orientation == "front" else 0.4)
            )
            inseam_crease = (
                np.exp(-0.5 * ((grid_x - mid_legs[:, None]) / 3.2) ** 2)
                * (grid_y > y_crotch - 4.0)
                * (grid_y <= y_hem_draw)
            )
            seam_factor = (1.0 - 0.12 * wb_seam - 0.12 * fly_seam - 0.18 * inseam_crease)[..., None]
            shaded_lower = np.clip(
                (remapped_pants * cyl_shade + additive_sheen + flare_ripple) * seam_factor,
                0.0,
                255.0,
            )
            lower_mask_u8 = np.where(valid_lower, 255, 0).astype(np.uint8)

        else:
            # Side-View Pants OR Front/Back/Side Skirt
            if orientation == "side" and sub == "pants":
                y_hip = float(anchor_23_24[1] + torso_h * 0.05)
                y_knee = max(y_hip + 30.0, float(knee_mid[1]))
                hip_v = float(np.clip((y_hip - anchor_y) / max(y_hem_anatomical - anchor_y, 1.0), 0.12, 0.28))
                knee_v = float(np.clip((y_knee - anchor_y) / max(y_hem_anatomical - anchor_y, 1.0), hip_v + 0.20, 0.76))

                cx_spine = np.interp(
                    v_norm,
                    [0.0, hip_v, knee_v, 1.0],
                    [anchor_x, float(anchor_23_24[0]), float(knee_mid[0]), float(ank_mid[0])],
                )
                w_side_waist = w_waist_half * 0.76
                w_side_hip = w_hip_half * 0.80
                # Yan profil: vücut derinlik oranları (kalça=1, uyluk≈0.62, diz≈0.46, bilek≈0.33) × kalıp bolluğu
                body_side_hip = w_hip_half / max(r_hips, 1.0)
                w_side_thigh = body_side_hip * 0.62 * (1.0 + 0.75 * (r_thigh - 1.0))
                w_side_knee = body_side_hip * 0.46 * (1.0 + 0.75 * (r_knee - 1.0))
                w_side_hem = body_side_hip * 0.33 * (1.0 + 0.75 * (r_hem - 1.0))
                if src_flare_ratio > 1.10:
                    w_side_hem = max(w_side_hem, w_side_knee * src_flare_ratio * 0.94)

                drape_half_w = np.interp(
                    v_norm,
                    [0.0, hip_v, 0.38, knee_v, 1.0],
                    [w_side_waist, w_side_hip, w_side_thigh, w_side_knee, w_side_hem],
                )
            else:
                # Skirt (Front / Back / Side): Waist -> Hip -> Flared/Tapered Skirt Hem
                cx_spine = np.interp(v_norm, [0.0, 0.35, 1.0], [anchor_x, float(anchor_23_24[0]), float(knee_mid[0])])
                side_mult = 0.74 if orientation == "side" else 1.0
                w_hem_half = w_hip_half * (scale_spec.hem_flare_ratio / max(scale_spec.hip_flare_ratio, 0.5))
                drape_half_w = np.interp(
                    v_norm,
                    [0.0, 0.28, 1.0],
                    [w_waist_half * side_mult, w_hip_half * side_mult, w_hem_half * side_mult],
                )

            row_L = cx_spine - drape_half_w
            row_R = cx_spine + drape_half_w

            # Align side-view lower garment to cover actual side-profile leg/hip silhouette (eliminates front thigh cutout!)
            if orientation == "side":
                side_leg_seg = np.isin(seg_map, [5, 6, 7, 8, 12, 13])
                row_has_sl = np.sum(side_leg_seg, axis=1) > 5
                if np.any(row_has_sl):
                    sl_lx = np.where(row_has_sl, np.argmax(side_leg_seg, axis=1), row_L).astype(np.float32)
                    sl_rx = np.where(row_has_sl, w - 1 - np.argmax(side_leg_seg[:, ::-1], axis=1), row_R).astype(np.float32)
                    in_lower_rows = (ys_vec >= anchor_y) & (ys_vec <= y_hem_draw) & row_has_sl
                    side_pad = max(1.0, 2.5 + 1.5 * size_step)
                    row_L = np.where(in_lower_rows, np.minimum(row_L, sl_lx - side_pad), row_L)
                    row_R = np.where(in_lower_rows, np.maximum(row_R, sl_rx + side_pad), row_R)

            row_L = cv2.GaussianBlur(row_L.reshape(-1, 1), (1, 25), 5.0).ravel()
            row_R = cv2.GaussianBlur(row_R.reshape(-1, 1), (1, 25), 5.0).ravel()

            xl_2d = row_L[:, None]
            xr_2d = row_R[:, None]
            span_2d = np.maximum(xr_2d - xl_2d, 12.0)
            u_2d = np.clip((grid_x - xl_2d) / span_2d, 0.0, 1.0)
            wb_curve_y = anchor_y + 2.5 * np.sin(u_2d * math.pi)
            hem_curve_y = y_hem_draw + 4.5 * np.sin(u_2d * math.pi)

            valid_lower = (
                (grid_y >= wb_curve_y)
                & (grid_y <= hem_curve_y)
                & (grid_x >= xl_2d)
                & (grid_x <= xr_2d)
            )
            lower_mask_u8 = np.where(valid_lower, 255, 0).astype(np.uint8)

            if sub == "pants":
                map_x = (u_2d * float(gw - 1)).astype(np.float32)
                remapped_pants = cv2.remap(
                    g_rgb_pants_clean, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
            else:
                src_l_arr = g_row_l[src_y_idx][:, None]
                src_r_arr = g_row_r[src_y_idx][:, None]
                map_x = (src_l_arr + u_2d * np.maximum(src_r_arr - src_l_arr, 4.0)).astype(np.float32)
                remapped_pants = cv2.remap(
                    g_rgb, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
                )
            u_centered = np.clip((u_2d - 0.5) * 2.0, -1.2, 1.2)
            cyl_shade = (1.07 - 0.20 * (u_centered ** 2))[..., None]
            side_sheen = ((np.exp(-3.0 * (u_centered ** 2)) * 14.0 - (u_centered ** 2) * 10.0)[..., None]) * fab_tint
            side_seam = np.exp(-0.5 * (u_centered / 0.10) ** 2) * 0.05
            wb_seam_side = np.exp(-0.5 * ((grid_y - (wb_curve_y + 12.0)) / 2.0) ** 2) * 0.10
            shaded_lower = np.clip(
                (remapped_pants * cyl_shade + side_sheen) * (1.0 - side_seam - wb_seam_side)[..., None],
                0.0,
                255.0,
            )

        # 5. Anatomical Z-Index Subtraction (Preserve Foreground Hand, Fingers & Shoes!)
        z_masks = self.extract_anatomical_z_masks(user_bgr, seg_map, pose_landmarks, orientation=orientation)
        fg_cut = np.zeros((h, w), dtype=np.uint8)
        fg_cut = cv2.bitwise_or(fg_cut, z_masks["shoes"])
        if directives["carve_hands"]:
            fg_cut = cv2.bitwise_or(fg_cut, z_masks["hands"])
        if directives["carve_forearms"]:
            fg_cut = cv2.bitwise_or(fg_cut, z_masks["forearms"])
        if directives["carve_foreground_objects"]:
            fg_cut = cv2.bitwise_or(fg_cut, z_masks["foreground_objects"])

        carved_lower_u8 = cv2.subtract(lower_mask_u8, fg_cut)
        clean_bin, soft_alpha = self.refine_and_antialias_mask(
            carved_lower_u8, close_ksize=5, open_ksize=3, blur_ksize=5, blur_sigma=1.2
        )

        if np.any(fg_cut > 0):
            hand_on_pants = cv2.bitwise_and(fg_cut, lower_mask_u8)
            if np.any(hand_on_pants > 0):
                h_sh = cv2.GaussianBlur(hand_on_pants.astype(np.float32) / 255.0, (15, 15), 4.0)
                h_sh_shift = np.zeros_like(h_sh)
                h_sh_shift[4:, 3:] = h_sh[:-4, :-3]
                shaded_lower *= (
                    1.0 - np.clip(h_sh_shift - (fg_cut > 0).astype(np.float32), 0.0, 1.0) * 0.18
                )[..., None]

        # Subtle under-hem shadow onto bare legs / shoes
        hem_sh = cv2.GaussianBlur(soft_alpha, (13, 21), 5.0)
        hem_sh_shift = np.zeros_like(hem_sh)
        hem_sh_shift[6:, :] = hem_sh[:-6, :]
        under_sh = np.clip(hem_sh_shift - soft_alpha, 0.0, 1.0) * 0.18
        canvas = canvas * (1.0 - under_sh[..., None])

        final_bgr = np.clip(
            canvas * (1.0 - soft_alpha[..., None]) + shaded_lower * soft_alpha[..., None],
            0.0,
            255.0,
        ).astype(np.uint8)

        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[clean_bin > 0] = [255, 255, 255, 255]
        return final_bgr, mask_rgba

    # -----------------------------------------------------------------------
    # Vectorized Full-Body (Dress / Jumpsuit) Synthesis with De-Clothing & XS-XXL Grading
    # -----------------------------------------------------------------------
    def synthesize_vectorized_full_vton(
        self,
        user_bgr: np.ndarray,
        garment_bgra: np.ndarray,
        pose_landmarks,
        seg_map: np.ndarray,
        scale_spec: AsymmetricScaleSpec,
        is_side_profile: bool = False,
        is_back_view: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        100% vectorized full-body (dress/jumpsuit) synthesis:
        - Reconstructs a clean de-clothed plate (`reconstruct_clean_plate`) so any old black leggings (`User 1`)
          below the dress hem become natural 3D-shaded bare legs and any old bulky dress (`User 2`) outside
          the new silhouette becomes clean studio background.
        - Grades chest, waist, hip, skirt flare, and hem length across XS..XXL.
        """
        h, w = user_bgr.shape[:2]
        orientation = self.detect_pose_orientation(pose_landmarks, w, h)
        if is_side_profile:
            orientation = "side"
        elif is_back_view:
            orientation = "back"

        directives = CategoryRouter.get_masking_directives(orientation, "full_body")
        declothed_bgr, _ = self.reconstruct_clean_plate(
            user_bgr, seg_map, pose_landmarks, scale_spec, orientation=orientation
        )

        lm = self.router.extract_category_landmarks(pose_landmarks, w, h, "full_body")
        sh_L, sh_R = lm["left_shoulder"], lm["right_shoulder"]
        sh_mid = lm["shoulder_mid"]
        hip_mid = lm["hip_mid"]
        knee_mid = lm["knee_mid"]
        torso_h = max(abs(float(hip_mid[1] - sh_mid[1])), 80.0)
        sh_span = max(abs(float(sh_R[0] - sh_L[0])), torso_h * 0.34)

        em = getattr(scale_spec, "extra_metrics", None) or {}
        size_step = float(em.get("size_step", 0.0))
        step_scale_w = 1.0 + 0.068 * size_step
        len_step_px = size_step * (torso_h * 0.032)

        y_collar = max(6, int(sh_mid[1] - torso_h * 0.12))
        y_hem = int(
            min(
                float(h - 15),
                max(
                    float(knee_mid[1] + torso_h * 0.12 + scale_spec.gravity_drop_px * 0.28 + len_step_px),
                    float(y_collar) + torso_h * (1.85 + 0.05 * size_step),
                ),
            )
        )

        clean_garment, base_color_bgr = self.decontaminate_garment_edges(garment_bgra)
        gy, gx = np.where(clean_garment[:, :, 3] > 20)
        crop_g = (
            clean_garment[np.min(gy) : np.max(gy) + 1, np.min(gx) : np.max(gx) + 1]
            if len(gy) > 0
            else clean_garment
        )
        gh, gw = crop_g.shape[:2]
        g_rgb = crop_g[:, :, :3].astype(np.float32)
        g_alpha = crop_g[:, :, 3] > 120
        row_has_g = np.sum(g_alpha, axis=1) > 6
        g_row_l = np.where(row_has_g, np.argmax(g_alpha, axis=1), int(gw * 0.15)).astype(np.float32)
        g_row_r = np.where(row_has_g, gw - 1 - np.argmax(g_alpha[:, ::-1], axis=1), int(gw * 0.85)).astype(np.float32)
        g_row_l = cv2.GaussianBlur(g_row_l.reshape(-1, 1), (1, 31), 7.0).ravel() + 4.0
        g_row_r = cv2.GaussianBlur(g_row_r.reshape(-1, 1), (1, 31), 7.0).ravel() - 4.0

        ys_vec = np.arange(h, dtype=np.float32)
        v_norm = np.clip((ys_vec - float(y_collar)) / max(float(y_hem - y_collar), 1.0), 0.0, 1.0)
        cx_spine = np.interp(v_norm, [0.0, 0.45, 1.0], [float(sh_mid[0]), float(hip_mid[0]), float(knee_mid[0])])

        g_chest_half = 0.5 * float(em.get("garment_chest_px", torso_h * 0.54 * step_scale_w))
        g_waist_half = 0.5 * float(em.get("garment_waist_px", torso_h * 0.44 * step_scale_w))
        g_hips_half = 0.5 * float(em.get("garment_hips_px", torso_h * 0.62 * step_scale_w))
        w_hem_half = g_hips_half * (scale_spec.hem_flare_ratio / max(scale_spec.hip_flare_ratio, 0.5))

        side_mult = 0.72 if orientation == "side" else 1.0
        half_w_profile = (
            np.interp(
                v_norm,
                [0.0, 0.18, 0.36, 0.56, 1.0],
                [g_chest_half * 0.94, g_chest_half, g_waist_half, g_hips_half, w_hem_half],
            )
            * side_mult
        )

        row_L = cv2.GaussianBlur((cx_spine - half_w_profile).reshape(-1, 1), (1, 27), 6.0).ravel()
        row_R = cv2.GaussianBlur((cx_spine + half_w_profile).reshape(-1, 1), (1, 27), 6.0).ravel()

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        xl_2d = row_L[:, None]
        xr_2d = row_R[:, None]
        u_2d = np.clip((grid_x - xl_2d) / np.maximum(xr_2d - xl_2d, 12.0), 0.0, 1.0)
        v_2d = v_norm[:, None]

        neck_cx = float(sh_mid[0])
        collar_half_w = max(sh_span * 0.21, torso_h * 0.12)
        if orientation == "front":
            dx_neck_2d = (grid_x - neck_cx) / max(collar_half_w, 1.0)
            neck_curve_y_2d = float(y_collar) + torso_h * 0.10 * (1.0 - np.clip(dx_neck_2d, -1.0, 1.0) ** 2)
            collar_cond = (np.abs(dx_neck_2d) >= 1.0) | (grid_y >= neck_curve_y_2d)
        else:
            collar_cond = grid_y >= float(y_collar)

        hem_curve_y = float(y_hem) + 6.0 * np.sin(u_2d * math.pi)
        valid_dress = (
            (grid_y >= float(y_collar))
            & (grid_y <= hem_curve_y)
            & (grid_x >= xl_2d)
            & (grid_x <= xr_2d)
            & collar_cond
        )
        dress_mask_u8 = np.where(valid_dress, 255, 0).astype(np.uint8)

        src_y_idx = np.clip((v_norm * (gh - 1)).astype(np.int32), 0, gh - 1)
        src_l_for_row = g_row_l[src_y_idx][:, None]
        src_r_for_row = g_row_r[src_y_idx][:, None]
        map_x = (src_l_for_row + u_2d * np.maximum(src_r_for_row - src_l_for_row, 4.0)).astype(np.float32)
        map_y = np.broadcast_to(v_2d * float(gh - 1), (h, w)).copy().astype(np.float32)

        remapped_dress = cv2.remap(g_rgb, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        u_centered = np.clip((u_2d - 0.5) * 2.0, -1.2, 1.2)
        cyl_shade = (1.05 - 0.13 * (u_centered ** 2))[..., None]
        shaded_dress = np.clip(remapped_dress * cyl_shade, 0.0, 255.0)

        z_masks = self.extract_anatomical_z_masks(user_bgr, seg_map, pose_landmarks, orientation=orientation)
        fg_cut = np.zeros((h, w), dtype=np.uint8)
        fg_cut = cv2.bitwise_or(fg_cut, z_masks["shoes"])
        if directives["carve_hair"] and np.any(z_masks["hair"] > 0):
            fg_cut = cv2.bitwise_or(fg_cut, z_masks["hair"])
        if directives["carve_face_neck"] and np.any(z_masks["face_neck"] > 0):
            fg_cut = cv2.bitwise_or(fg_cut, z_masks["face_neck"])
        if directives["carve_hands"] and np.any(z_masks["hands"] > 0):
            fg_cut = cv2.bitwise_or(fg_cut, z_masks["hands"])
        if directives["carve_forearms"] and np.any(z_masks["forearms"] > 0):
            fg_cut = cv2.bitwise_or(fg_cut, z_masks["forearms"])

        carved_u8 = cv2.subtract(dress_mask_u8, fg_cut)
        clean_bin, soft_alpha = self.refine_and_antialias_mask(
            carved_u8, close_ksize=15, open_ksize=5, blur_ksize=9, blur_sigma=2.0
        )

        canvas = declothed_bgr.astype(np.float32).copy()
        hem_sh = cv2.GaussianBlur(soft_alpha, (13, 21), 5.0)
        hem_sh_shift = np.zeros_like(hem_sh)
        hem_sh_shift[6:, :] = hem_sh[:-6, :]
        under_sh = np.clip(hem_sh_shift - soft_alpha, 0.0, 1.0) * 0.18
        canvas = canvas * (1.0 - under_sh[..., None])

        final_bgr = np.clip(
            canvas * (1.0 - soft_alpha[..., None]) + shaded_dress * soft_alpha[..., None],
            0.0,
            255.0,
        ).astype(np.uint8)

        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[clean_bin > 0] = [255, 255, 255, 255]
        return final_bgr, mask_rgba

    # -----------------------------------------------------------------------
    # IDM-VTON Diffusion Inputs (mask + person-init) per category
    # -----------------------------------------------------------------------
    @staticmethod
    def _inpaint_ghost_region(img_bgr: np.ndarray, ghost_u8: np.ndarray, scale: float = 0.30) -> np.ndarray:
        """Remove leftover old-garment pixels (ghosting) by inpainting them from their surroundings."""
        if not np.any(ghost_u8 > 0):
            return img_bgr
        h, w = img_bgr.shape[:2]
        sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
        small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        gm = cv2.resize(ghost_u8, (sw, sh), interpolation=cv2.INTER_NEAREST)
        gm = cv2.dilate(gm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        filled = cv2.inpaint(small, gm, 5, cv2.INPAINT_TELEA)
        filled_full = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        alpha = cv2.GaussianBlur(ghost_u8.astype(np.float32) / 255.0, (9, 9), 2.5)[..., None]
        return np.clip(img_bgr.astype(np.float32) * (1.0 - alpha) + filled_full * alpha, 0, 255).astype(np.uint8)

    def build_diffusion_inputs(
        self,
        user_bgr: np.ndarray,
        garment_bgra: np.ndarray,
        pose_landmarks,
        scale_spec: AsymmetricScaleSpec,
        seg_map: Optional[np.ndarray] = None,
        is_side_profile: bool = False,
        is_back_view: bool = False,
    ) -> Dict[str, object]:
        """
        Prepare everything the diffusion stage (IDM-VTON `TryonPipeline`) and vectorized renderer need:
        - `declothed_bgr` & `declothing_stats`: clean plate with old conflicting clothes erased to skin/background.
        - `mask_u8`: single-channel 0/255 inpainting mask.
        - `init_bgr`: de-clothed person image with the category-routed anatomical synthesis composited on top.
        - `geometry_stats`: measured pixel spans of the synthesized garment for verification.
        """
        h, w = user_bgr.shape[:2]
        if seg_map is None:
            seg_map = get_global_models().get_semantic_map(user_bgr)

        orientation = self.detect_pose_orientation(pose_landmarks, w, h)
        if is_side_profile:
            orientation = "side"
        elif is_back_view:
            orientation = "back"

        declothed_bgr, decloth_stats = self.reconstruct_clean_plate(
            user_bgr, seg_map, pose_landmarks, scale_spec, orientation=orientation
        )

        clean_garment, _ = self.decontaminate_garment_edges(garment_bgra)
        warped_garment, _, _ = self.prewarp_garment_tps(
            garment_bgra=clean_garment,
            pose_landmarks=pose_landmarks,
            user_shape=(h, w),
            scale_spec=scale_spec,
            is_side_profile=(orientation == "side"),
            is_back_view=(orientation == "back"),
            seg_map=seg_map,
        )

        cat = scale_spec.canonical_category
        if cat == "upper_body":
            synth_bgr, synth_mask_rgba = self.synthesize_vectorized_upper_vton(
                user_bgr=user_bgr,
                garment_bgra=clean_garment,
                pose_landmarks=pose_landmarks,
                seg_map=seg_map,
                scale_spec=scale_spec,
                is_side_profile=(orientation == "side"),
                is_back_view=(orientation == "back"),
            )
        elif cat == "lower_body":
            synth_bgr, synth_mask_rgba = self.synthesize_vectorized_lower_vton(
                user_bgr=user_bgr,
                garment_bgra=clean_garment,
                pose_landmarks=pose_landmarks,
                seg_map=seg_map,
                scale_spec=scale_spec,
                is_side_profile=(orientation == "side"),
                is_back_view=(orientation == "back"),
            )
        else:
            synth_bgr, synth_mask_rgba = self.synthesize_vectorized_full_vton(
                user_bgr=user_bgr,
                garment_bgra=clean_garment,
                pose_landmarks=pose_landmarks,
                seg_map=seg_map,
                scale_spec=scale_spec,
                is_side_profile=(orientation == "side"),
                is_back_view=(orientation == "back"),
            )

        mask_u8 = np.ascontiguousarray(synth_mask_rgba[:, :, 0])
        init_bgr = synth_bgr

        # Measure actual synthesized garment bounding box & zone widths for diagnostic verification
        ys_m, xs_m = np.where(mask_u8 > 64)
        if len(ys_m) > 20:
            y_min_m, y_max_m = int(np.min(ys_m)), int(np.max(ys_m))
            x_min_m, x_max_m = int(np.min(xs_m)), int(np.max(xs_m))
            mh = max(1, y_max_m - y_min_m)
            r_top = int(y_min_m + mh * 0.25)
            r_mid = int(y_min_m + mh * 0.55)
            r_bot = int(y_min_m + mh * 0.85)

            def _row_span(r_idx: int) -> float:
                band = mask_u8[max(0, r_idx - 4) : min(h, r_idx + 5), :] > 64
                if not np.any(band):
                    return 0.0
                xs_b = np.where(band)[1]
                return float(np.max(xs_b) - np.min(xs_b))

            geometry_stats = {
                "bbox": f"[{x_min_m},{y_min_m} -> {x_max_m},{y_max_m}]",
                "height_px": float(mh),
                "top_span_px": _row_span(r_top),
                "mid_span_px": _row_span(r_mid),
                "bot_span_px": _row_span(r_bot),
                "active_mask_px": int(len(ys_m)),
            }
        else:
            geometry_stats = {
                "bbox": "[0,0 -> 0,0]",
                "height_px": 0.0,
                "top_span_px": 0.0,
                "mid_span_px": 0.0,
                "bot_span_px": 0.0,
                "active_mask_px": 0,
            }

        return {
            "mask_u8": mask_u8,
            "init_bgr": init_bgr,
            "declothed_bgr": declothed_bgr,
            "declothing_stats": decloth_stats,
            "geometry_stats": geometry_stats,
            "orientation": orientation,
            "warped_garment_bgra": warped_garment,
        }

    # -----------------------------------------------------------------------
    # Unified Multi-Category Synthesis (Upper, Lower Pants/Skirt, Full Dress)
    # -----------------------------------------------------------------------
    def synthesize_category_vton(
        self,
        user_bgr: np.ndarray,
        garment_bgra: np.ndarray,
        pose_landmarks,
        scale_spec: AsymmetricScaleSpec,
        seg_map: Optional[np.ndarray] = None,
        is_side_profile: bool = False,
        is_back_view: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_inputs = self.build_diffusion_inputs(
            user_bgr=user_bgr,
            garment_bgra=garment_bgra,
            pose_landmarks=pose_landmarks,
            scale_spec=scale_spec,
            seg_map=seg_map,
            is_side_profile=is_side_profile,
            is_back_view=is_back_view,
        )
        h, w = user_bgr.shape[:2]
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[d_inputs["mask_u8"] > 0] = [255, 255, 255, 255]
        return d_inputs["init_bgr"], mask_rgba


__all__ = [
    "AnatomicalMaskGenerator",
    "ThinPlateSplineWarper",
]
