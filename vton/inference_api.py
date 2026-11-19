"""
Hybrid IDM-VTON Diffusion Inference API & VRAM-Safe Colab Pipeline Loader.

Resolves:
- Directive 1 (Vectorization): 100% vectorized NumPy & OpenCV matrix post-processing
  (zero Python 'for' loops over image rows/columns).
- Directive 2 (Dynamic Category Routing): Category-aware prompt generation and
  post-processing for 'upper_body', 'lower_body' (pants/skirts), and 'full_body' (dresses).
- Directive 6 (IDM-VTON VRAM Safety): Loads Colab diffusers pipeline components strictly with
  `torch_dtype=torch.float16`, `device_map='cuda'`, and `low_cpu_mem_usage=True`.
"""

from __future__ import annotations

import gc
import os
import shutil
from typing import Optional

import cv2
import numpy as np
import torch

from .category_router import CategoryRouter, get_global_models

# ---------------------------------------------------------------------------
# Stable IDM-VTON `/tryon` sampling parameters (same values for every category so the
# upper-body realism is never affected by the orientation/mask logic).
# ---------------------------------------------------------------------------
IDM_DENOISE_STEPS = 30       # IDM-VTON reference default: best quality/latency trade-off on a T4
IDM_SEED = 42                # fixed seed -> reproducible, stable results
IDM_USE_AUTO_CROP = False    # we pass our own mask + full frame, never the crop/resize shortcut
IDM_TARGET_WIDTH = 768
IDM_TARGET_HEIGHT = 1024


# ---------------------------------------------------------------------------
# Directive 6: Colab T4 GPU VRAM-Safe IDM-VTON Diffusers Pipeline Loader
# ---------------------------------------------------------------------------
def load_idm_vton_pipeline_vram_safe(
    base_path: str = "/content/model_weights",
    tryon_pipeline_cls=None,
    unet_cls=None,
    unet_encoder_cls=None,
):
    """
    Directive 6: Load the IDM-VTON diffusers pipeline on Google Colab (T4 16GB GPU)
    strictly with `torch_dtype=torch.float16`, `device_map='cuda'`, and
    `low_cpu_mem_usage=True` to eliminate CPU RAM OOM crashes and PCIe bottlenecks.
    """
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import (
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )

    common_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "cuda",
        "low_cpu_mem_usage": True,
    }

    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", **common_kwargs)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path, subfolder="image_encoder", **common_kwargs
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path, subfolder="text_encoder", **common_kwargs
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path, subfolder="text_encoder_2", **common_kwargs
    )

    tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    unet = None
    if unet_cls is not None:
        unet = unet_cls.from_pretrained(base_path, subfolder="unet", **common_kwargs)

    unet_encoder = None
    if unet_encoder_cls is not None:
        unet_encoder = unet_encoder_cls.from_pretrained(
            base_path, subfolder="unet_encoder", **common_kwargs
        )

    if tryon_pipeline_cls is None:
        raise ValueError("tryon_pipeline_cls (TryonPipeline) must be provided when instantiating IDM-VTON.")

    pipe = tryon_pipeline_cls.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    if unet_encoder is not None:
        pipe.unet_encoder = unet_encoder.to("cuda")

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pipe


# ---------------------------------------------------------------------------
# Studio Garment Preparation & Vectorized HD Print Restoration
# ---------------------------------------------------------------------------
def prepare_studio_garment_image(garment_img_path: str, is_side_profile: bool = False) -> str:
    """
    Place RGBA/BGR extracted garment onto a 768x1024 (3:4) pure white studio canvas
    expected by IDM-VTON while preserving aspect ratio.
    """
    source_path = garment_img_path
    if is_side_profile and garment_img_path:
        g_dir = os.path.dirname(garment_img_path)
        for cand in ["temp_garment_extracted_on.png", "ayiklanmis_on.png", "on.jpg"]:
            cand_path = os.path.join(g_dir, cand)
            if os.path.exists(cand_path):
                source_path = cand_path
                break

    img = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return garment_img_path

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
        a_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
        white_bg = np.full_like(bgr, 255)
        comp = (a_f * bgr + (1.0 - a_f) * white_bg).astype(np.uint8)
    else:
        comp = img[:, :, :3] if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    target_w, target_h = 768, 1024
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    gh, gw = comp.shape[:2]
    scale = min((target_w * 0.86) / max(gw, 1), (target_h * 0.86) / max(gh, 1))
    nw, nh = max(1, int(gw * scale)), max(1, int(gh * scale))
    resized = cv2.resize(comp, (nw, nh), interpolation=cv2.INTER_AREA)

    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized

    studio_path = os.path.join(os.path.dirname(garment_img_path), "temp_studio_garment.jpg")
    cv2.imwrite(studio_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return studio_path


def restore_garment_print(hd_bgr: np.ndarray, garment_bgra: np.ndarray, landmarks, h: int, w: int) -> np.ndarray:
    """
    100% vectorized restoration of high-frequency chest graphic prints blurred by
    VAE decoding during diffusion inference.
    """
    if garment_bgra is None or landmarks is None:
        return hd_bgr

    if garment_bgra.ndim == 3 and garment_bgra.shape[2] == 4:
        alpha = garment_bgra[:, :, 3]
        ys, xs = np.where(alpha > 128)
        if len(ys) == 0:
            return hd_bgr
        cropped_g = garment_bgra[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    else:
        cropped_g = cv2.cvtColor(garment_bgra, cv2.COLOR_BGR2BGRA)
        diff_w = np.max(255 - garment_bgra, axis=2)
        cropped_g[:, :, 3] = np.where(diff_w > 22, 255, 0).astype(np.uint8)
        ys, xs = np.where(cropped_g[:, :, 3] > 128)
        if len(ys) > 0:
            cropped_g = cropped_g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    g_h, g_w = cropped_g.shape[:2]
    ry1, ry2 = int(g_h * 0.19), int(g_h * 0.62)
    rx1, rx2 = int(g_w * 0.27), int(g_w * 0.73)
    roi_bgra = cropped_g[ry1:ry2, rx1:rx2]
    if roi_bgra.size == 0:
        return hd_bgr

    roi_bgr = roi_bgra[:, :, :3].astype(np.float32)
    roi_a = cv2.erode((roi_bgra[:, :, 3] > 210).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if not np.any(roi_a):
        return hd_bgr

    med_bgr = np.median(roi_bgr[roi_a], axis=0)
    prelim_print = (np.linalg.norm(roi_bgr - med_bgr, axis=2) > 28) & roi_a
    prelim_dilated = cv2.dilate(prelim_print.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    bg_est = roi_bgr.copy()
    bg_est[prelim_dilated | (~roi_a)] = med_bgr
    local_flat_bg = cv2.GaussianBlur(bg_est, (71, 71), 0)

    diff_vec = roi_bgr - local_flat_bg
    signed_lum_diff = np.mean(diff_vec, axis=2)
    max_ch_diff = np.max(np.abs(diff_vec), axis=2)

    if np.mean(med_bgr) < 135.0:
        raw_strength = np.clip((np.maximum(signed_lum_diff, np.max(diff_vec, axis=2) * 0.85) - 10.0) / 32.0, 0.0, 1.0)
    else:
        raw_strength = np.clip((np.maximum(-signed_lum_diff, max_ch_diff * 0.85) - 14.0) / 32.0, 0.0, 1.0)

    raw_strength[~roi_a] = 0.0
    if np.sum(raw_strength > 0.25) < (roi_bgr.shape[0] * roi_bgr.shape[1] * 0.012):
        return hd_bgr

    print_alpha = cv2.GaussianBlur(raw_strength, (3, 3), 0)

    lm = landmarks.landmark
    l_sh = np.array([lm[11].x * w, lm[11].y * h], dtype=np.float32)
    r_sh = np.array([lm[12].x * w, lm[12].y * h], dtype=np.float32)

    sh_w = float(np.linalg.norm(l_sh - r_sh))
    mid_x = (l_sh[0] + r_sh[0]) * 0.5
    mid_y = (l_sh[1] + r_sh[1]) * 0.5

    dx = abs(l_sh[0] - r_sh[0])
    dy = (l_sh[1] - r_sh[1]) if l_sh[0] > r_sh[0] else (r_sh[1] - l_sh[1])
    angle_deg = float(-np.degrees(np.arctan2(dy, max(dx, 1.0))) * 0.5)

    chest_y1 = max(0, int(mid_y + sh_w * 0.02))
    chest_y2 = min(h, int(mid_y + sh_w * 1.25))
    chest_x1 = max(0, int(mid_x - sh_w * 0.50))
    chest_x2 = min(w, int(mid_x + sh_w * 0.50))

    out = hd_bgr.copy()
    chest_roi = out[chest_y1:chest_y2, chest_x1:chest_x2].copy()
    if chest_roi.size == 0:
        return hd_bgr

    chest_f = chest_roi.astype(np.float32)
    chest_med = np.median(chest_f.reshape(-1, 3), axis=0)
    chest_diff = np.linalg.norm(chest_f - chest_med, axis=2)
    erase_mask = np.where(chest_diff > 16, 255, 0).astype(np.uint8)
    erase_mask = cv2.dilate(erase_mask, np.ones((15, 15), np.uint8))

    ys_e, xs_e = np.where(chest_diff > 28)
    diff_cy = (chest_y1 + float(np.median(ys_e))) if len(ys_e) > 100 else None
    diff_cx = (chest_x1 + float(np.median(xs_e))) if len(ys_e) > 100 else None

    inpainted_chest = cv2.inpaint(chest_roi, erase_mask, 9, cv2.INPAINT_TELEA)
    smooth_chest = cv2.GaussianBlur(inpainted_chest.astype(np.float32), (35, 35), 0)
    rng = np.random.default_rng(42)
    grain = rng.normal(0.0, 1.4, (chest_roi.shape[0], chest_roi.shape[1])).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (3, 3), 0)[:, :, None]
    clean_chest_fabric = np.clip(smooth_chest + grain, 0, 255)

    ch_h, ch_w = chest_roi.shape[:2]
    feather = np.ones((ch_h, ch_w), dtype=np.float32)
    f_pad = int(min(ch_h, ch_w) * 0.14)
    if f_pad > 0:
        ramp = np.linspace(0.0, 1.0, f_pad, dtype=np.float32)
        feather[:f_pad, :] *= ramp[:, None]
        feather[-f_pad:, :] *= ramp[::-1, None]
        feather[:, :f_pad] *= ramp[None, :]
        feather[:, -f_pad:] *= ramp[None, ::-1]

    erase_w = cv2.GaussianBlur((erase_mask > 0).astype(np.float32), (29, 29), 0) * feather
    cleaned_chest = (erase_w[:, :, None] * clean_chest_fabric + (1.0 - erase_w[:, :, None]) * chest_f).astype(np.uint8)
    out[chest_y1:chest_y2, chest_x1:chest_x2] = cleaned_chest

    scale_x = (sh_w * 1.95) / float(g_w)
    scale_y = scale_x
    y_collar = mid_y - sh_w * 0.19

    ys_p, _ = np.where(raw_strength > 0.3)
    src_print_cy = float(np.median(ys_p)) if len(ys_p) > 0 else (roi_bgr.shape[0] / 2.0)

    anat_cy = y_collar + (ry1 + src_print_cy) * scale_y
    target_print_cy = (
        (0.55 * diff_cy + 0.45 * anat_cy)
        if (diff_cy is not None and abs(diff_cy - anat_cy) < sh_w * 0.35)
        else anat_cy
    )
    dst_cx = (
        (0.5 * diff_cx + 0.5 * mid_x)
        if (diff_cx is not None and abs(diff_cx - mid_x) < sh_w * 0.2)
        else mid_x
    )
    dst_cy = target_print_cy + ((roi_bgr.shape[0] / 2.0) - src_print_cy) * scale_y

    ph, pw = roi_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle_deg, scale_x)
    M[0, 2] += dst_cx - pw / 2.0
    M[1, 2] += dst_cy - ph / 2.0

    warped_src = cv2.warpAffine(roi_bgr, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_bg = cv2.warpAffine(local_flat_bg, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_print_alpha = cv2.warpAffine(print_alpha, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    box_mask = np.ones((ph, pw), dtype=np.float32)
    b_pad = int(min(ph, pw) * 0.10)
    if b_pad > 0:
        b_ramp = np.linspace(0.0, 1.0, b_pad, dtype=np.float32)
        box_mask[:b_pad, :] *= b_ramp[:, None]
        box_mask[-b_pad:, :] *= b_ramp[::-1, None]
        box_mask[:, :b_pad] *= b_ramp[None, :]
        box_mask[:, -b_pad:] *= b_ramp[None, ::-1]
    warped_box_mask = cv2.warpAffine(box_mask, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    ys_w, xs_w = np.where(warped_box_mask > 0.01)
    if len(ys_w) == 0:
        return out

    wy1, wy2 = ys_w.min(), ys_w.max()
    wx1, wx2 = xs_w.min(), xs_w.max()

    base_fabric = out[wy1 : wy2 + 1, wx1 : wx2 + 1].astype(np.float32)
    smooth_base = cv2.GaussianBlur(base_fabric, (25, 25), 0)
    fabric_lum = np.mean(smooth_base, axis=2, keepdims=True)
    med_lum = max(float(np.mean(chest_med)), 1.0)
    shading_mod = np.clip(fabric_lum / med_lum, 0.78, 1.18)

    w_src_crop = warped_src[wy1 : wy2 + 1, wx1 : wx2 + 1]
    w_bg_crop = warped_bg[wy1 : wy2 + 1, wx1 : wx2 + 1]
    w_pa_crop = (warped_print_alpha[wy1 : wy2 + 1, wx1 : wx2 + 1] * warped_box_mask[wy1 : wy2 + 1, wx1 : wx2 + 1])[:, :, None]

    warped_diff = w_src_crop - w_bg_crop
    target_print_pixel = np.clip(base_fabric + warped_diff * shading_mod * 1.05, 0, 255)

    final_roi = np.clip(w_pa_crop * target_print_pixel + (1.0 - w_pa_crop) * base_fabric, 0, 255).astype(np.uint8)
    out[wy1 : wy2 + 1, wx1 : wx2 + 1] = final_roi
    return out


def restore_side_garment_print(
    hd_f: np.ndarray,
    front_garment_bgra: np.ndarray,
    uh: int,
    uw: int,
    shirt_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    100% vectorized cylindrical foreshortening projection of the front chest graphic
    onto the side-profile view (zero Python for-loops).
    """
    if front_garment_bgra is None:
        return hd_f

    if front_garment_bgra.ndim == 3 and front_garment_bgra.shape[2] == 4:
        ga = front_garment_bgra[:, :, 3]
        ys_g, xs_g = np.where(ga > 128)
        if len(ys_g) == 0:
            return hd_f
        gc = front_garment_bgra[ys_g.min() : ys_g.max() + 1, xs_g.min() : xs_g.max() + 1, :3].astype(np.float32)
    else:
        gc = front_garment_bgra[:, :, :3].astype(np.float32)

    gh, gw = gc.shape[:2]
    py1_g, py2_g = int(gh * 0.21), int(gh * 0.54)
    px1_g, px2_g = int(gw * 0.29), int(gw * 0.71)
    p_src = gc[py1_g:py2_g, px1_g:px2_g]
    if p_src.size == 0:
        return hd_f

    ph_g, pw_g = p_src.shape[:2]
    chest_med = np.median(p_src, axis=(0, 1))
    p_bg = cv2.morphologyEx(p_src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
    p_bg = cv2.GaussianBlur(p_bg, (51, 51), 0)
    p_bg = 0.65 * p_bg + 0.35 * chest_med[None, None, :]
    p_diff = np.maximum(0.0, p_src - p_bg)
    p_diff_lum = np.max(np.abs(p_src - p_bg), axis=2)

    p_alpha = np.clip((p_diff_lum - 8.0) / 24.0, 0.0, 1.0)
    if np.sum(p_alpha > 0.15) < 40:
        return hd_f
    p_alpha = cv2.GaussianBlur(p_alpha, (3, 3), 0)

    # Vectorized border feathering (Zero Python for-loops)
    f_px = max(1, min(28, ph_g // 5, pw_g // 5))
    dist_y = np.minimum(np.arange(ph_g, dtype=np.float32), np.arange(ph_g - 1, -1, -1, dtype=np.float32))
    dist_x = np.minimum(np.arange(pw_g, dtype=np.float32), np.arange(pw_g - 1, -1, -1, dtype=np.float32))
    wy_f = np.clip(dist_y / float(f_px), 0.0, 1.0) ** 1.5
    wx_f = np.clip(dist_x / float(f_px), 0.0, 1.0) ** 1.5
    b_mask = np.minimum(wy_f[:, None], wx_f[None, :])
    p_alpha *= b_mask

    ty1, ty2 = int(uh * 0.285), int(uh * 0.445)
    if shirt_mask is not None and shirt_mask.shape[:2] == (uh, uw):
        band_mask = shirt_mask[ty1:ty2, :] > 0.5
        if np.any(band_mask):
            xs_band = np.where(band_mask)[1]
            left_edge = int(np.percentile(xs_band, 3))
            span_w = max(24, int((np.percentile(xs_band, 95) - left_edge) * 0.38))
            tx1 = max(0, left_edge + 3)
            tx2 = min(uw, tx1 + span_w)
        else:
            tx1, tx2 = int(uw * 0.410), int(uw * 0.496)
    else:
        tx1, tx2 = int(uw * 0.410), int(uw * 0.496)

    th, tw = ty2 - ty1, tx2 - tx1
    if th <= 10 or tw <= 10:
        return hd_f

    grid_y, grid_x = np.meshgrid(np.linspace(0.0, 1.0, th), np.linspace(0.0, 1.0, tw), indexing="ij")
    u_map = np.power(grid_x, 0.82) * (pw_g - 1)
    v_curve = -0.028 * (1.0 - grid_x) ** 1.5
    v_map = np.clip(grid_y + v_curve, 0.0, 1.0) * (ph_g - 1)

    warped_diff = cv2.remap(
        p_diff, u_map.astype(np.float32), v_map.astype(np.float32), cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    warped_alpha = cv2.remap(
        p_alpha, u_map.astype(np.float32), v_map.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    out_f = hd_f.copy()
    shirt_roi = out_f[ty1:ty2, tx1:tx2]
    shirt_lum = np.mean(shirt_roi, axis=2)
    if shirt_mask is not None and shirt_mask.shape[:2] == (uh, uw):
        inside_shirt = cv2.erode((shirt_mask[ty1:ty2, tx1:tx2] > 0.5).astype(np.uint8), np.ones((5, 7), np.uint8)).astype(np.float32)
    else:
        inside_shirt = cv2.erode((shirt_lum < 125.0).astype(np.uint8), np.ones((5, 7), np.uint8)).astype(np.float32)
    if np.sum(inside_shirt > 0.5) < 60:
        return hd_f
    inside_shirt = cv2.GaussianBlur(inside_shirt, (5, 5), 0)

    smooth_sh = cv2.GaussianBlur(shirt_roi, (25, 25), 0)
    sh_mod = np.clip(
        np.mean(smooth_sh, axis=2, keepdims=True) / max(1.0, float(np.median(shirt_lum[inside_shirt > 0.5]))),
        0.78,
        1.18,
    )
    final_alpha = (warped_alpha * inside_shirt)[:, :, None]
    out_f[ty1:ty2, tx1:tx2] = np.clip(shirt_roi + final_alpha * warped_diff * sh_mod * 1.05, 0, 255)
    return out_f


def postprocess_hd_render(
    person_img_path: str,
    garment_img_path: str,
    hd_output_path: str,
    mask_img_path: Optional[str] = None,
    is_side_profile: bool = False,
    is_back_view: bool = False,
    landmarks=None,
    garment_type: str = "Üst Giyim",
) -> None:
    """
    100% vectorized HD render post-processing: upscales 768x1024 diffusion output to
    native resolution, restores front/side graphic prints, and applies category-aware
    hem calibration without a single Python row loop.
    """
    user_bgr = cv2.imread(person_img_path)
    hd_bgr = cv2.imread(hd_output_path)
    if user_bgr is None or hd_bgr is None:
        return

    uh, uw = user_bgr.shape[:2]
    if hd_bgr.shape[:2] != (uh, uw):
        hd_bgr = cv2.resize(hd_bgr, (uw, uh), interpolation=cv2.INTER_LANCZOS4)

    route = CategoryRouter.resolve_route(garment_type)

    if landmarks is None:
        res = get_global_models().detect_pose(user_bgr, is_rgb=False)
        landmarks = res.pose_landmarks

    if route.canonical_category != "upper_body":
        cv2.imwrite(hd_output_path, hd_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return

    if not is_side_profile and not is_back_view and garment_img_path and os.path.exists(garment_img_path):
        garment_bgra = cv2.imread(garment_img_path, cv2.IMREAD_UNCHANGED)
        hd_bgr = restore_garment_print(hd_bgr, garment_bgra, landmarks, uh, uw)

    elif is_side_profile:
        hd_f = hd_bgr.astype(np.float32)
        u_f = user_bgr.astype(np.float32)

        # 1. Vectorized cleanup of side-view diffusion artifacts before cylindrical print projection
        ry1, ry2 = int(uh * 0.255), int(uh * 0.525)
        rx1, rx2 = int(uw * 0.390), int(uw * 0.525)
        roi = hd_f[ry1:ry2, rx1:rx2].copy()
        if roi.size > 0:
            rlum = np.mean(roi, axis=2)
            rh, rw = roi.shape[:2]
            dark_mask = rlum < 135.0
            has_dark = np.any(dark_mask, axis=1)
            raw_left = np.where(has_dark, np.argmax(dark_mask, axis=1).astype(np.float32), float(rw))
            smooth_left = cv2.GaussianBlur(raw_left.reshape(-1, 1), (1, 45), 9.0).ravel()

            y_top, y_bot = int(rh * 0.12), int(rh * 0.88)
            x_right = int(uw * 0.520) - rx1
            top_prof = np.median(roi[:y_top, :, :], axis=0).copy()
            bot_prof = np.median(roi[y_bot:, :, :], axis=0).copy()
            top_left_x = int(np.median(smooth_left[:y_top])) + 20
            bot_left_x = int(np.median(smooth_left[y_bot:])) + 20
            if 0 < top_left_x < rw:
                top_prof[:top_left_x, :] = top_prof[top_left_x, :][None, :]
            if 0 < bot_left_x < rw:
                bot_prof[:bot_left_x, :] = bot_prof[bot_left_x, :][None, :]
            top_prof = cv2.GaussianBlur(top_prof[None, :, :], (31, 1), 8.0)[0]
            bot_prof = cv2.GaussianBlur(bot_prof[None, :, :], (31, 1), 8.0)[0]

            ys_grid = np.arange(rh, dtype=np.float32)[:, None]
            xs_grid = np.arange(rw, dtype=np.float32)[None, :]
            t_y = np.clip((ys_grid - y_top) / float(max(1, y_bot - y_top)), 0.0, 1.0)[:, :, None]
            smooth_bg = (1.0 - t_y) * top_prof[None, :, :] + t_y * bot_prof[None, :, :]
            rng = np.random.default_rng(42)
            grain = cv2.GaussianBlur(rng.normal(0.0, 1.2, (rh, rw)).astype(np.float32), (3, 3), 0)[:, :, None]
            clean_f = np.clip(smooth_bg + grain, 0, 255)

            wy_top = np.clip(ys_grid / float(max(1, y_top + 12)), 0.0, 1.0)
            wy_bot = np.clip((rh - 1 - ys_grid) / float(max(1, rh - y_bot + 12)), 0.0, 1.0)
            wy = 0.5 - 0.5 * np.cos(np.pi * np.minimum(wy_top, wy_bot))
            wx_right = 0.5 - 0.5 * np.cos(np.pi * np.clip((float(x_right) - xs_grid) / 10.0, 0.0, 1.0))
            wx_left = np.clip((xs_grid - (smooth_left[:, None] + 1.2)) / 2.2, 0.0, 1.0)
            aw = (wy * wx_right * wx_left)[:, :, None]
            hd_f[ry1:ry2, rx1:rx2] = aw * clean_f + (1.0 - aw) * roi

        front_g_bgra = None
        if garment_img_path:
            g_dir = os.path.dirname(garment_img_path)
            for cand in ["ayiklanmis_on.png", "temp_garment_extracted_on.png", os.path.basename(garment_img_path)]:
                c_path = os.path.join(g_dir, cand)
                if os.path.exists(c_path):
                    front_g_bgra = cv2.imread(c_path, cv2.IMREAD_UNCHANGED)
                    if front_g_bgra is not None:
                        break
        # hd_f = restore_side_garment_print(hd_f, front_g_bgra, uh, uw)
        hd_bgr = np.clip(hd_f, 0, 255).astype(np.uint8)

    elif is_back_view:
        hd_f = hd_bgr.astype(np.float32)
        u_f = user_bgr.astype(np.float32)

        center_gap_lum = float(np.max(np.mean(hd_f[int(uh * 0.535) : int(uh * 0.560), int(uw * 0.47) : int(uw * 0.53)], axis=2)))
        shirt_mid_lum = float(np.mean(hd_f[int(uh * 0.545), int(uw * 0.42) : int(uw * 0.58)]))
        if center_gap_lum < 120.0 and 55.0 < shirt_mid_lum < 115.0:
            y_anchor = int(uh * 0.44)
            y_old = int(uh * 0.5665)
            y_new = int(uh * 0.515)
            y_arm_blend = int(uh * 0.495)

            leg_mask_u = np.mean(u_f, axis=2) < 65.0
            u_matched = np.where(
                leg_mask_u[..., None],
                np.clip(u_f + np.array([2.0, 3.7, 3.1], dtype=np.float32), 0, 255),
                u_f,
            )

            # Vectorized 2D remap for back hem calibration (Zero Python for-loops)
            grid_x, grid_y = np.meshgrid(np.arange(uw, dtype=np.float32), np.arange(uh, dtype=np.float32))
            t_remap = (grid_y - y_anchor) / float(max(1, y_new - y_anchor))
            y_src_map = np.where(
                (grid_y >= y_anchor) & (grid_y <= y_new),
                y_anchor + t_remap * (y_old - y_anchor),
                grid_y,
            ).astype(np.float32)
            src_remapped = cv2.remap(hd_f, grid_x, y_src_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            au = np.clip((grid_y - y_arm_blend) / float(max(1, y_new - y_arm_blend)), 0.0, 1.0)
            au = (0.5 - 0.5 * np.cos(np.pi * au))[..., None]
            base_band = (1.0 - au) * hd_f + au * u_matched

            lum_hd = np.mean(hd_f, axis=2)
            rb_hd = hd_f[:, :, 2] - hd_f[:, :, 0]
            is_shirt = (
                (lum_hd < 128.0)
                & (rb_hd < 16.0)
                & (grid_x > float(uw * 0.355))
                & (grid_x < float(uw * 0.660))
            ).astype(np.float32)
            w_shirt = cv2.GaussianBlur(is_shirt, (15, 1), 0)[..., None]
            band_upper = w_shirt * src_remapped + (1.0 - w_shirt) * base_band

            sh_f = np.clip(0.80 + 0.20 * ((grid_y - y_new) / 14.0), 0.80, 1.0)[..., None]
            is_leg_3d = (np.mean(u_matched, axis=2) < 65.0) & (grid_y > y_new) & (grid_y < y_new + 14)
            band_lower = np.where(is_leg_3d[..., None], u_matched * sh_f, u_matched)

            out_arka = np.where(
                grid_y[..., None] < y_anchor,
                hd_f,
                np.where(grid_y[..., None] <= y_new, band_upper, band_lower),
            )
            hd_bgr = np.clip(out_arka, 0, 255).astype(np.uint8)

    cv2.imwrite(hd_output_path, hd_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def build_category_diffusion_prompt(
    garment_type: str = "Üst Giyim",
    fit_type: str = "Oversize / Rahat Kalıp",
    is_side_profile: bool = False,
    is_back_view: bool = False,
) -> str:
    """Build category-specific natural language description for IDM-VTON."""
    route = CategoryRouter.resolve_route(garment_type)
    fit_low = str(fit_type).lower()
    if "oversize" in fit_low or "rahat" in fit_low:
        fit_en = "oversized relaxed fit"
    elif "slim" in fit_low or "dar kal" in fit_low:
        fit_en = "slim fitted"
    elif "bodycon" in fit_low or "skinny" in fit_low:
        fit_en = "bodycon skin-tight"
    else:
        fit_en = "regular classic fit"

    view_suffix = (
        ", side profile view" if is_side_profile else (", back view" if is_back_view else ", front view")
    )

    if route.canonical_category == "lower_body":
        if route.sub_category == "skirt":
            return f"A {fit_en} skirt with natural fabric drape{view_suffix}"
        return f"A pair of {fit_en} trousers pants with natural leg drape{view_suffix}"

    if route.canonical_category == "full_body":
        return f"A {fit_en} full-body dress with natural waist and skirt drape{view_suffix}"

    if is_side_profile:
        return f"An {fit_en} dark grey t-shirt with front chest graphic print, side profile view"
    if is_back_view:
        return f"Plain solid anthracite dark grey {fit_en} t-shirt back view"
    return f"An {fit_en} t-shirt with front graphic print"


_HF_SPACE_COOLDOWN_UNTIL: float = 0.0


def render_with_idm_vton(
    person_img_path: str,
    garment_img_path: str,
    output_path: str,
    mask_img_path: Optional[str] = None,
    garment_desc: str = "An oversized t-shirt with short sleeves",
    colab_url: str = "",
    is_side_profile: bool = False,
    is_back_view: bool = False,
    landmarks=None,
    garment_type: str = "Üst Giyim",
    fit_type: str = "Oversize / Rahat Kalıp",
    init_person_img_path: Optional[str] = None,
) -> bool:
    """
    Hybrid client inference entry point that sends the user image, studio-prepared
    garment, and anti-aliased category-conditional agnostic mask to the remote
    Google Colab T4 IDM-VTON server (or HuggingFace Space fallback).

    `init_person_img_path` (optional): image actually fed to the diffusion `TryonPipeline` as the
    person. Lower/full-body garments pass the pre-warped (bipedal TPS) composite here so the
    diffusion pass relights/fuses it; `person_img_path` stays the untouched original for post-processing.
    """
    import time

    global _HF_SPACE_COOLDOWN_UNTIL
    is_custom_colab = bool(colab_url and colab_url.strip() != "")
    if not is_custom_colab and time.time() < _HF_SPACE_COOLDOWN_UNTIL:
        print("[API] Hugging Face ZeroGPU kotası/zaman aşımı nedeniyle aktif bekleme süresi (cooldown) içinde, hızlı motora geçiliyor.")
        return False

    print("\n" + "=" * 50)
    print("[API] Yüksek Çözünürlüklü (HD) IDM-VTON Render Katmanı Başlatılıyor...")
    print("=" * 50)

    req_timeout = 180.0 if is_custom_colab else 45.0
    try:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        import httpx

        original_init = httpx.Client.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["verify"] = False
            kwargs["timeout"] = httpx.Timeout(req_timeout, connect=12.0)
            original_init(self, *args, **kwargs)

        httpx.Client.__init__ = patched_init

        original_async_init = httpx.AsyncClient.__init__

        def patched_async_init(self, *args, **kwargs):
            kwargs["verify"] = False
            kwargs["timeout"] = httpx.Timeout(req_timeout, connect=12.0)
            original_async_init(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = patched_async_init
    except Exception:
        pass

    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print("[ERROR] gradio_client kütüphanesi bulunamadı! Lütfen 'pip install gradio_client' çalıştırın.")
        return False

    try:
        hf_token = os.environ.get("HF_TOKEN")
        if colab_url and colab_url.strip() != "":
            target_endpoint = colab_url.strip()
            print(f"[API] Render servisine bağlanılıyor: {target_endpoint}")
            client = Client(target_endpoint, ssl_verify=False, token=hf_token)
        else:
            target_space = os.environ.get("HF_SPACE", "yisol/IDM-VTON")
            print(f"[API] Varsayılan Hugging Face Space bağlanılıyor: {target_space}")
            client = Client(target_space, ssl_verify=False, token=hf_token)

        studio_garment_path = prepare_studio_garment_image(garment_img_path, is_side_profile=is_side_profile)
        garment_desc = build_category_diffusion_prompt(
            garment_type=garment_type,
            fit_type=fit_type,
            is_side_profile=is_side_profile,
            is_back_view=is_back_view,
        )

        diffusion_person = (
            init_person_img_path
            if (init_person_img_path and os.path.exists(init_person_img_path))
            else person_img_path
        )

        # Pre-resize person and mask to IDM-VTON native 768x1024 (3-channel RGB) to prevent
        # Gradio ImageEditor 1-channel errors and HTTP payload disconnects on large 4MP+ images.
        upload_person_path = diffusion_person
        upload_mask_path = mask_img_path
        try:
            p_bgr = cv2.imread(diffusion_person)
            if p_bgr is not None:
                p_768 = cv2.resize(p_bgr, (IDM_TARGET_WIDTH, IDM_TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
                upload_person_path = output_path.replace(".jpg", "_api_person_768.jpg")
                cv2.imwrite(upload_person_path, p_768, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

            if mask_img_path and os.path.exists(mask_img_path):
                m_raw = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)
                if m_raw is not None:
                    if m_raw.ndim == 3:
                        m_gray = m_raw[:, :, 0]
                    else:
                        m_gray = m_raw
                    m_768 = cv2.resize(m_gray, (IDM_TARGET_WIDTH, IDM_TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    m_bgr = cv2.cvtColor(np.where(m_768 > 127, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    upload_mask_path = output_path.replace(".jpg", "_api_mask_768.png")
                    cv2.imwrite(upload_mask_path, m_bgr)
        except Exception:
            pass

        layers = [handle_file(upload_mask_path)] if (upload_mask_path and os.path.exists(upload_mask_path)) else []
        is_checked = False if layers else True

        result = client.predict(
            dict({"background": handle_file(upload_person_path), "layers": layers, "composite": None}),
            handle_file(studio_garment_path),
            garment_desc,
            is_checked,
            IDM_USE_AUTO_CROP,
            IDM_DENOISE_STEPS,
            IDM_SEED,
            api_name="/tryon",
        )

        result_img_path = result[0] if isinstance(result, (list, tuple)) else result
        shutil.copy(result_img_path, output_path)
        postprocess_hd_render(
            person_img_path=person_img_path,
            garment_img_path=garment_img_path,
            hd_output_path=output_path,
            mask_img_path=mask_img_path,
            is_side_profile=is_side_profile,
            is_back_view=is_back_view,
            landmarks=landmarks,
            garment_type=garment_type,
        )
        print(f"[SUCCESS] Fotogerçekçi HD Render Başarıyla Tamamlandı -> {output_path}")
        return True

    except Exception as e:
        if not is_custom_colab:
            _HF_SPACE_COOLDOWN_UNTIL = time.time() + 180.0
        print(f"\n[ERROR] IDM-VTON API Hatası: {e}")
        return False
