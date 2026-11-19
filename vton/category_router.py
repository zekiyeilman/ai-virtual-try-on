"""
Dynamic Multi-Category Router, Orientation Directives & Global Model Singleton Registry.

Resolves:
- Directive 1 (Latency Optimization): Global one-time startup initialization of
  SegFormer, U2Net (rembg), and MediaPipe Pose, mapped to CUDA where applicable,
  with SHA-1 semantic segmentation caching.
- Directive 2 (Dynamic Category Routing & Lower-Body Anchor): Routes 'upper_body',
  'lower_body', and 'full_body' to SegFormer labels and MediaPipe landmarks, locking
  the 'lower_body' top-center anchor strictly to the midpoint of landmarks 23 & 24
  and stretching downward toward knees (25, 26) and ankles (27, 28).
- Orientation-Aware Masking Directives (`get_masking_directives`): Controls Z-index
  carving rules for 'front', 'side', and 'back' poses (enforcing solid-block masking
  in 'back' pose without skin/background intersection).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import rembg
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

from .models.ansur import Ansur3DTo2DConverter

# ---------------------------------------------------------------------------
# SegFormer (mattmdjaga/segformer_b2_clothes) Semantic Label Constants
# ---------------------------------------------------------------------------
SEG_LABELS: Dict[str, int] = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "face": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

MP_LANDMARKS: Dict[str, List[int]] = {
    "upper_body": [11, 12, 23, 24],
    "upper_body_arms": [11, 12, 13, 14, 15, 16],
    "lower_body": [23, 24, 25, 26, 27, 28],
    "full_body": [11, 12, 23, 24, 25, 26, 27, 28],
    "hands": [15, 16, 17, 18, 19, 20, 21, 22],
}


@dataclass(frozen=True)
class CategoryRouteConfig:
    """Immutable routing configuration for a canonical garment category."""

    canonical_category: str          # 'upper_body' | 'lower_body' | 'full_body'
    sub_category: str                # 'top' | 'pants' | 'skirt' | 'dress'
    segformer_garment_labels: Tuple[int, ...]
    segformer_fallback_labels: Tuple[int, ...]
    user_clothes_to_replace: Tuple[int, ...]
    primary_landmarks: Tuple[int, ...]
    auxiliary_landmarks: Tuple[int, ...]
    ansur_primary_part: str
    ansur_secondary_part: str
    anchor_mode: str                 # 'shoulders' | 'hips_midpoint_23_24' | 'shoulders_to_ankles'
    gravity_drop_factor: float
    default_base_width_cm_female: float
    default_base_width_cm_male: float
    default_base_length_cm_female: float
    default_base_length_cm_male: float

    @property
    def key(self) -> str:
        if self.canonical_category == "lower_body":
            return f"lower_body_{self.sub_category}"
        return self.canonical_category

    @property
    def category(self) -> str:
        return self.canonical_category


@dataclass
class AsymmetricScaleSpec:
    """Output of ANSUR + gravity asymmetric scaling computation."""

    canonical_category: str
    sub_category: str
    scale_x_ratio: float
    scale_y_ratio: float
    target_width_px: int
    target_height_px: int
    torso_width_px: float
    gravity_drop_px: float
    waist_taper_ratio: float
    hip_flare_ratio: float
    hem_flare_ratio: float
    anchor_y_offset_ratio: float
    ansur_2d_width_cm: float
    ansur_2d_depth_cm: float
    projected_garment_2d_cm: float
    effective_ease_cm: float
    size_modifier_cm: float = 0.0
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    fit_type: str = ""


# ---------------------------------------------------------------------------
# Global Model Singleton Registry (Initialized Once at Startup)
# ---------------------------------------------------------------------------
class GlobalModelRegistry:
    _instance: Optional["GlobalModelRegistry"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "GlobalModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, segformer_model_id: str = "mattmdjaga/segformer_b2_clothes") -> None:
        if self.__class__._initialized:
            return

        self.segformer_model_id = segformer_model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        print(f"[GlobalModelRegistry] Initializing CV & DL models globally on {self.device}...")

        try:
            self.segformer_processor = SegformerImageProcessor.from_pretrained(
                segformer_model_id, local_files_only=True
            )
            self.segformer_model = AutoModelForSemanticSegmentation.from_pretrained(
                segformer_model_id, local_files_only=True
            )
        except Exception:
            self.segformer_processor = SegformerImageProcessor.from_pretrained(segformer_model_id)
            self.segformer_model = AutoModelForSemanticSegmentation.from_pretrained(segformer_model_id)

        self.segformer_model.to(self.device)
        self.segformer_model.eval()

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        try:
            self.u2net_session = rembg.new_session("u2net", providers=providers)
        except Exception:
            self.u2net_session = rembg.new_session("u2net")

        try:
            self.u2net_cloth_session = rembg.new_session("u2net_cloth_seg", providers=providers)
        except Exception:
            self.u2net_cloth_session = self.u2net_session

        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

        self._seg_cache: Dict[str, np.ndarray] = {}
        self._pose_cache: Dict[str, object] = {}
        self.__class__._initialized = True
        print(f"[GlobalModelRegistry] All models initialized and ready on {self.device}.")

    @staticmethod
    def _hash_image(img: np.ndarray) -> str:
        h, w = img.shape[:2]
        stride = max(1, (h * w) // 16384)
        flat_sample = img.reshape(-1)[::stride]
        digest = hashlib.sha1(flat_sample.tobytes()).hexdigest()
        return f"{h}x{w}_{digest}"

    def detect_pose(self, bgr_or_rgb: np.ndarray, is_rgb: bool = False):
        rgb = bgr_or_rgb if is_rgb else cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2RGB)
        key = self._hash_image(rgb)
        if key in self._pose_cache:
            return self._pose_cache[key]
        res = self.pose_detector.process(rgb)
        if len(self._pose_cache) >= 32:
            self._pose_cache.pop(next(iter(self._pose_cache)))
        self._pose_cache[key] = res
        return res

    def get_semantic_map(self, bgr_image: np.ndarray) -> np.ndarray:
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input bgr_image is empty or None.")

        key = self._hash_image(bgr_image)
        if key in self._seg_cache:
            return self._seg_cache[key]

        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self.segformer_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = self.segformer_model(**inputs).logits
            upsampled = F.interpolate(
                logits.float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            seg_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        if len(self._seg_cache) >= 32:
            self._seg_cache.pop(next(iter(self._seg_cache)))
        self._seg_cache[key] = seg_map
        return seg_map


GLOBAL_MODELS: Optional[GlobalModelRegistry] = None


def get_global_models() -> GlobalModelRegistry:
    global GLOBAL_MODELS
    if GLOBAL_MODELS is None:
        GLOBAL_MODELS = GlobalModelRegistry()
    return GLOBAL_MODELS


# ---------------------------------------------------------------------------
# Dynamic Category Router & Orientation Directives
# ---------------------------------------------------------------------------
class CategoryRouter:
    # Standardized ISO / Retail Anthropometric Reference Body Circumferences (cm) per Size
    STANDARD_SIZE_TABLES: Dict[str, Dict[str, Dict[str, float]]] = {
        "female": {
            "XS":  {"chest": 81.0,  "waist": 64.0, "hips": 90.0,  "step": -2.0},
            "S":   {"chest": 85.0,  "waist": 68.0, "hips": 94.0,  "step": -1.0},
            "M":   {"chest": 89.5,  "waist": 72.5, "hips": 98.5,  "step":  0.0},
            "L":   {"chest": 95.0,  "waist": 78.0, "hips": 104.0, "step": +1.0},
            "XL":  {"chest": 101.0, "waist": 84.0, "hips": 110.0, "step": +2.0},
            "XXL": {"chest": 107.5, "waist": 90.5, "hips": 116.5, "step": +3.0},
        },
        "male": {
            "XS":  {"chest": 88.0,  "waist": 75.0, "hips": 91.0,  "step": -2.0},
            "S":   {"chest": 93.0,  "waist": 80.0, "hips": 96.0,  "step": -1.0},
            "M":   {"chest": 98.5,  "waist": 85.5, "hips": 101.5, "step":  0.0},
            "L":   {"chest": 104.5, "waist": 91.5, "hips": 107.0, "step": +1.0},
            "XL":  {"chest": 111.0, "waist": 98.0, "hips": 113.0, "step": +2.0},
            "XXL": {"chest": 118.0, "waist": 105.0, "hips": 119.5, "step": +3.0},
        },
    }

    SIZE_ALIASES: Dict[str, str] = {
        "XXS": "XS", "XS": "XS", "32": "XS", "34": "XS", "XS/34": "XS", "24": "XS", "25": "XS",
        "S": "S", "36": "S", "S/36": "S", "26": "S", "27": "S",
        "M": "M", "38": "M", "M/38": "M", "28": "M", "29": "M",
        "L": "L", "40": "L", "L/40": "L", "30": "L", "31": "L",
        "XL": "XL", "42": "XL", "XL/42": "XL",
        "XXL": "XXL", "2XL": "XXL", "3XL": "XXL", "44": "XXL", "46": "XXL", "2XL/44": "XXL",
    }

    # Kumaş, gövde etrafında serbest asılı olduğunda kesiti vücuttan biraz daha basıktır (önden daha geniş görünür).
    # Bu oran BEDENDEN BAĞIMSIZDIR; beden farkı yalnızca 3B çevre tablosundan gelir (çift sayım engellenir).
    GARMENT_DRAPE_FLATTEN: float = 1.04

    # ANSUR II kadın/erkek ortalamalarından türetilmiş bacak çevresi / basen çevresi oranları
    LEG_BODY_RATIOS: Dict[str, float] = {"thigh": 0.60, "knee": 0.385, "ankle": 0.235}

    # Pantolon kalıbına göre uyluk / diz / paça ağzı tasarım bolluğu (cm, çevre). Paça = paça ağzı çevresi.
    PANTS_LEG_EASE: Dict[str, Dict[str, float]] = {
        "skinny":   {"thigh": -1.5, "knee": -1.0, "hem_circ": 25.0},
        "slim":     {"thigh": 3.0,  "knee": 3.0,  "hem_circ": 33.0},
        "regular":  {"thigh": 8.0,  "knee": 9.0,  "hem_circ": 42.0},
        "oversize": {"thigh": 18.0, "knee": 22.0, "hem_circ": 56.0},
    }

    @staticmethod
    def fit_key_of(fit_type: Optional[str]) -> str:
        ft = str(fit_type or "").lower()
        if "skinny" in ft or "bodycon" in ft or "saran" in ft or "tayt" in ft:
            return "skinny"
        if "slim" in ft or "dar kal" in ft:
            return "slim"
        if "oversize" in ft or "rahat" in ft or "wide" in ft or "baggy" in ft or "geniş" in ft:
            return "oversize"
        return "regular"

    @staticmethod
    def stretch_tolerance_cm(fabric_stretch: Optional[str]) -> float:
        fs = str(fabric_stretch or "").lower()
        if "yüksek" in fs or "10" in fs or "tayt" in fs:
            return 10.0
        if "orta" in fs or "5" in fs or "8" in fs or "triko" in fs:
            return 6.0
        if "az" in fs or "2" in fs or "3" in fs or "süprem" in fs:
            return 3.0
        return 0.0

    @staticmethod
    def visible_width_ratio(garment_2d_cm: float, body_2d_cm: float, cap: float = 1.95) -> float:
        """
        Önden görünen silüet = max(vücut, kıyafet). Kıyafet vücuttan dar ise kumaş gerilir ama silüet vücuttur (oran 1.0);
        bol ise silüet kıyafetin izdüşüm genişliğidir. Doymayan (saturating olmayan) fiziksel oran döndürür.
        """
        return float(np.clip(float(garment_2d_cm) / max(float(body_2d_cm), 1.0), 1.0, cap))

    def recommend_size(
        self,
        user_measurements: Optional[Dict[str, float]],
        gender: str,
        category_or_garment_type: str,
        fabric_stretch: str = "",
    ) -> Tuple[str, Dict[str, float]]:
        """
        Perakende beden önerisi: kullanıcının vücut ölçüleri, her bedenin REFERANS VÜCUT ölçüleriyle karşılaştırılır
        (kalıp bolluğu zaten kıyafetin tasarımında olduğu için öneri kalıptan bağımsızdır).
        Darlık (kullanıcı > beden vücudu) bolluğa göre 3.3 kat daha ağır cezalandırılır; kumaş esnekliği darlığı tolere eder.
        """
        route = self.resolve_route(category_or_garment_type)
        is_male = "erkek" in str(gender).lower()
        table = self.STANDARD_SIZE_TABLES["male" if is_male else "female"]
        m = user_measurements or {}
        u = {
            "chest": float(m.get("chest", 98.5 if is_male else 89.5)),
            "waist": float(m.get("waist", 85.5 if is_male else 72.5)),
            "hips": float(m.get("hips", 101.5 if is_male else 98.5)),
        }
        if route.canonical_category == "lower_body":
            weights = {"chest": 0.0, "waist": 1.0, "hips": 1.0}
        elif route.canonical_category == "full_body":
            weights = {"chest": 1.0, "waist": 0.7, "hips": 1.0}
        else:
            weights = {"chest": 1.0, "waist": 0.25, "hips": 0.35}
        tol = self.stretch_tolerance_cm(fabric_stretch) * 0.5
        costs: Dict[str, float] = {}
        for sz, ref in table.items():
            c = 0.0
            for part, wgt in weights.items():
                d = u[part] - float(ref[part])
                tight = max(0.0, d - tol)
                loose = max(0.0, -d)
                c += wgt * (2.0 * tight + 0.6 * loose)
            costs[sz] = round(c, 2)
        best = min(costs, key=costs.get)
        return best, costs

    def lower_leg_metrics(self, ref_hips_cm: float, user_hips_cm: float, fit_type: str, size_step: float) -> Dict[str, float]:
        """Uyluk / diz / paça için vücut ve kıyafet 3B çevreleri (cm)."""
        fk = self.fit_key_of(fit_type)
        ease = self.PANTS_LEG_EASE[fk]
        r = self.LEG_BODY_RATIOS
        return {
            "user_thigh_3d_cm": user_hips_cm * r["thigh"],
            "user_knee_3d_cm": user_hips_cm * r["knee"],
            "user_ankle_3d_cm": user_hips_cm * r["ankle"],
            "garment_thigh_3d_cm": ref_hips_cm * r["thigh"] + ease["thigh"],
            "garment_knee_3d_cm": ref_hips_cm * r["knee"] + ease["knee"],
            "garment_hem_3d_cm": ease["hem_circ"] + 1.2 * size_step,
        }

    ROUTES: Dict[str, CategoryRouteConfig] = {
        "upper_body": CategoryRouteConfig(
            canonical_category="upper_body",
            sub_category="top",
            segformer_garment_labels=(4,),
            segformer_fallback_labels=(4, 7),
            user_clothes_to_replace=(4, 7),
            primary_landmarks=(11, 12, 23, 24),
            auxiliary_landmarks=(13, 14, 15, 16),
            ansur_primary_part="chest",
            ansur_secondary_part="waist",
            anchor_mode="shoulders",
            gravity_drop_factor=1.45,
            default_base_width_cm_female=53.5,
            default_base_width_cm_male=57.0,
            default_base_length_cm_female=66.0,
            default_base_length_cm_male=71.0,
        ),
        "lower_body_pants": CategoryRouteConfig(
            canonical_category="lower_body",
            sub_category="pants",
            segformer_garment_labels=(6, 5),
            segformer_fallback_labels=(6, 5, 8),
            user_clothes_to_replace=(5, 6, 7, 8),
            primary_landmarks=(23, 24, 25, 26, 27, 28),
            auxiliary_landmarks=(11, 12, 15, 16),
            ansur_primary_part="hips",
            ansur_secondary_part="thigh",
            anchor_mode="hips_midpoint_23_24",
            gravity_drop_factor=1.85,
            default_base_width_cm_female=49.5,
            default_base_width_cm_male=52.0,
            default_base_length_cm_female=102.0,
            default_base_length_cm_male=106.0,
        ),
        "lower_body_skirt": CategoryRouteConfig(
            canonical_category="lower_body",
            sub_category="skirt",
            segformer_garment_labels=(5, 6),
            segformer_fallback_labels=(5, 6, 7),
            user_clothes_to_replace=(5, 6, 7, 8),
            primary_landmarks=(23, 24, 25, 26, 27, 28),
            auxiliary_landmarks=(11, 12, 15, 16),
            ansur_primary_part="hips",
            ansur_secondary_part="waist",
            anchor_mode="hips_midpoint_23_24",
            gravity_drop_factor=1.60,
            default_base_width_cm_female=49.0,
            default_base_width_cm_male=51.0,
            default_base_length_cm_female=48.0,
            default_base_length_cm_male=52.0,
        ),
        "full_body": CategoryRouteConfig(
            canonical_category="full_body",
            sub_category="dress",
            segformer_garment_labels=(7,),
            segformer_fallback_labels=(7, 4, 5, 6),
            user_clothes_to_replace=(4, 5, 6, 7, 8),
            primary_landmarks=(11, 12, 23, 24, 25, 26, 27, 28),
            auxiliary_landmarks=(13, 14, 15, 16),
            ansur_primary_part="chest",
            ansur_secondary_part="hips",
            anchor_mode="shoulders_to_ankles",
            gravity_drop_factor=2.10,
            default_base_width_cm_female=49.5,
            default_base_width_cm_male=54.0,
            default_base_length_cm_female=114.0,
            default_base_length_cm_male=118.0,
        ),
    }

    @classmethod
    def normalize_size_label(cls, size_str: Optional[str]) -> str:
        s = str(size_str or "M").strip().upper()
        if s in cls.SIZE_ALIASES:
            return cls.SIZE_ALIASES[s]
        for k, v in sorted(cls.SIZE_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
            if k in s:
                return v
        return "M"

    @classmethod
    def get_fit_ease_spec(cls, canonical_category: str, sub_category: str, fit_type: str) -> Dict[str, float]:
        ft = str(fit_type or "").lower()
        if "skinny" in ft or "bodycon" in ft or "saran" in ft:
            fit_key = "skinny"
        elif "slim" in ft or "dar kal" in ft:
            fit_key = "slim"
        elif "oversize" in ft or "rahat" in ft or "wide" in ft or "baggy" in ft:
            fit_key = "oversize"
        else:
            fit_key = "regular"

        if canonical_category == "upper_body":
            table = {
                "skinny":   {"chest_ease": -2.0, "waist_ease": -2.0, "hips_ease": -1.0, "len_mod": -4.0, "visual_mult": 0.90},
                "slim":     {"chest_ease": +4.0, "waist_ease": +3.0, "hips_ease": +3.0, "len_mod": -1.5, "visual_mult": 0.96},
                "regular":  {"chest_ease": +10.0, "waist_ease": +10.0, "hips_ease": +8.0, "len_mod":  0.0, "visual_mult": 1.03},
                "oversize": {"chest_ease": +20.0, "waist_ease": +22.0, "hips_ease": +18.0, "len_mod": +4.0, "visual_mult": 1.14},
            }
            return table[fit_key]
        if canonical_category == "lower_body" and sub_category == "pants":
            table = {
                "skinny":   {"chest_ease": 0.0, "waist_ease": -1.0, "hips_ease": -1.0, "len_mod": -2.0, "visual_mult": 0.88, "knee_mult": 0.88, "hem_mult": 0.80},
                "slim":     {"chest_ease": 0.0, "waist_ease": +1.5, "hips_ease": +3.0, "len_mod":  0.0, "visual_mult": 0.96, "knee_mult": 1.00, "hem_mult": 0.95},
                "regular":  {"chest_ease": 0.0, "waist_ease": +3.0, "hips_ease": +6.5, "len_mod": +1.0, "visual_mult": 1.05, "knee_mult": 1.14, "hem_mult": 1.16},
                "oversize": {"chest_ease": 0.0, "waist_ease": +5.5, "hips_ease": +15.0, "len_mod": +3.5, "visual_mult": 1.22, "knee_mult": 1.38, "hem_mult": 1.48},
            }
            return table[fit_key]
        if canonical_category == "lower_body":
            table = {
                "skinny":   {"chest_ease": 0.0, "waist_ease": -1.0, "hips_ease":  0.0, "len_mod": -3.0, "visual_mult": 0.90, "hem_mult": 0.90},
                "slim":     {"chest_ease": 0.0, "waist_ease": +1.5, "hips_ease": +3.5, "len_mod": -1.0, "visual_mult": 0.97, "hem_mult": 1.04},
                "regular":  {"chest_ease": 0.0, "waist_ease": +2.5, "hips_ease": +7.5, "len_mod":  0.0, "visual_mult": 1.06, "hem_mult": 1.24},
                "oversize": {"chest_ease": 0.0, "waist_ease": +4.5, "hips_ease": +14.0, "len_mod": +3.0, "visual_mult": 1.18, "hem_mult": 1.44},
            }
            return table[fit_key]
        # full_body (dress)
        table = {
            "skinny":   {"chest_ease": -2.0, "waist_ease": -2.0, "hips_ease":  0.0, "len_mod": -4.0, "visual_mult": 0.90, "hem_mult": 0.90},
            "slim":     {"chest_ease": +3.0, "waist_ease": +2.5, "hips_ease": +4.0, "len_mod": -1.5, "visual_mult": 0.97, "hem_mult": 1.06},
            "regular":  {"chest_ease": +8.0, "waist_ease": +6.5, "hips_ease": +9.5, "len_mod":  0.0, "visual_mult": 1.05, "hem_mult": 1.24},
            "oversize": {"chest_ease": +16.0, "waist_ease": +16.0, "hips_ease": +18.0, "len_mod": +4.0, "visual_mult": 1.16, "hem_mult": 1.42},
        }
        return table[fit_key]

    def __init__(self, gender: str = "Kadın") -> None:
        self.ansur = Ansur3DTo2DConverter(gender=gender)

    @classmethod
    def resolve_route(cls, category_or_garment_type: Optional[str]) -> CategoryRouteConfig:
        raw = (category_or_garment_type or "upper_body").strip().lower()

        if raw in ("full_body", "dress", "dresses", "jumpsuit") or any(
            k in raw for k in ("elbise", "tulum", "salopet", "full_body", "full body")
        ):
            return cls.ROUTES["full_body"]

        if any(k in raw for k in ("etek", "şort", "sort", "skirt", "shorts", "bermuda")):
            return cls.ROUTES["lower_body_skirt"]

        if raw in ("lower_body", "pants", "trousers") or any(
            k in raw
            for k in (
                "alt giyim",
                "pantolon",
                "jean",
                "kot",
                "eşofman",
                "esofman",
                "tayt",
                "lower_body",
                "lower body",
            )
        ):
            return cls.ROUTES["lower_body_pants"]

        return cls.ROUTES["upper_body"]

    get_route = resolve_route

    @classmethod
    def get_masking_directives(
        cls,
        orientation: str,
        category_or_garment_type: Optional[str] = "upper_body",
    ) -> Dict[str, bool]:
        """
        Return explicit orientation-conditional masking & Z-index carving directives:
        - When `orientation == "front"`: Carve out `face_neck`, `chest` (front neckline),
          `hands`, and `foreground_objects`.
        - When `orientation == "back"`: Disable all SegFormer skin/neck/chest/background
          intersections (`solid_block_mode = True`), keeping the garment mask as a solid,
          unbroken block so exposed back skin (e.g. racerback sports bra cutouts) never
          punches holes or dark imprints into the garment.
        - When `orientation == "side"`: Do not carve front `chest`/`face_neck` neckline,
          but preserve the near foreground arm/hand (`preserve_side_arm_hand = True`)
          hanging in front of the torso/hip.
        """
        route = cls.resolve_route(category_or_garment_type)
        orient = (orientation or "front").strip().lower()

        if orient == "back":
            return {
                "solid_block_mode": True,
                "carve_face_neck": False,
                "carve_chest_neckline": False,
                "carve_hands": False,
                "carve_forearms": False,
                "carve_foreground_objects": False,
                "carve_hair": True,
                "disable_skin_bg_intersection": True,
                "render_front_print": False,
                "render_side_print": False,
                "preserve_bare_lower_legs": route.sub_category in ("skirt", "dress"),
            }

        if orient == "side":
            return {
                "solid_block_mode": False,
                "carve_face_neck": False,
                "carve_chest_neckline": False,
                "carve_hands": True,
                "carve_forearms": True,
                "carve_foreground_objects": True,
                "carve_hair": True,
                "disable_skin_bg_intersection": True,
                "render_front_print": False,
                "render_side_print": False,
                "preserve_bare_lower_legs": route.sub_category in ("skirt", "dress"),
            }

        # Default: 'front'
        return {
            "solid_block_mode": False,
            "carve_face_neck": True,
            "carve_chest_neckline": True,
            "carve_hands": True,
            "carve_forearms": True,
            "carve_foreground_objects": True,
            "carve_hair": True,
            "disable_skin_bg_intersection": False,
            "render_front_print": route.canonical_category == "upper_body",
            "render_side_print": False,
            "preserve_bare_lower_legs": route.sub_category in ("skirt", "dress"),
        }

    @classmethod
    def get_segformer_target_labels(
        cls, category_or_garment_type: Optional[str], include_fallback: bool = False
    ) -> List[int]:
        route = cls.resolve_route(category_or_garment_type)
        labels = route.segformer_fallback_labels if include_fallback else route.segformer_garment_labels
        return list(labels)

    @classmethod
    def extract_category_landmarks(
        cls,
        pose_landmarks,
        image_w: int,
        image_h: int,
        category_or_garment_type: Optional[str] = "upper_body",
    ) -> Dict[str, np.ndarray]:
        """
        Extract MediaPipe skeletal landmarks in pixel coordinates and compute explicit
        category anchor points:
        - For `lower_body`: `lower_body_anchor` is locked strictly to the exact midpoint
          `0.5 * (LM_23 + LM_24)` in (X, Y) pixel space, and stretching proceeds downward
          from `lower_body_anchor` to knees `(25, 26)` and ankles `(27, 28)`.
        """
        route = cls.resolve_route(category_or_garment_type)
        lm = pose_landmarks.landmark

        coords = np.array(
            [[p.x * image_w, p.y * image_h, p.z * image_w, p.visibility] for p in lm],
            dtype=np.float32,
        )

        primary_idx = np.array(route.primary_landmarks, dtype=np.int32)
        aux_idx = np.array(route.auxiliary_landmarks, dtype=np.int32)

        sh_mid = 0.5 * (coords[11, :2] + coords[12, :2])
        hip_mid_23_24 = 0.5 * (coords[23, :2] + coords[24, :2])
        knee_mid = 0.5 * (coords[25, :2] + coords[26, :2])
        ankle_mid = 0.5 * (coords[27, :2] + coords[28, :2])

        result: Dict[str, np.ndarray] = {
            "all_coords": coords,
            "primary_coords": coords[primary_idx],
            "auxiliary_coords": coords[aux_idx],
            "left_shoulder": coords[11, :2],
            "right_shoulder": coords[12, :2],
            "left_elbow": coords[13, :2],
            "right_elbow": coords[14, :2],
            "left_wrist": coords[15, :2],
            "right_wrist": coords[16, :2],
            "left_hip": coords[23, :2],
            "right_hip": coords[24, :2],
            "left_knee": coords[25, :2],
            "right_knee": coords[26, :2],
            "left_ankle": coords[27, :2],
            "right_ankle": coords[28, :2],
            "shoulder_mid": sh_mid,
            "hip_mid": hip_mid_23_24,
            "lower_body_anchor": hip_mid_23_24.copy(),  # Exact (X, Y) midpoint of Landmarks 23 & 24
            "knee_mid": knee_mid,
            "ankle_mid": ankle_mid,
        }

        if route.canonical_category == "upper_body":
            result["anchor_top"] = sh_mid
            result["anchor_bottom"] = hip_mid_23_24
        elif route.canonical_category == "lower_body":
            # Strictly anchored at the midpoint of Landmarks 23 & 24
            result["anchor_top"] = hip_mid_23_24.copy()
            result["anchor_bottom"] = ankle_mid if route.sub_category == "pants" else knee_mid
        else:
            result["anchor_top"] = sh_mid
            result["anchor_bottom"] = knee_mid if route.sub_category == "dress" else ankle_mid

        return result

    @staticmethod
    def measure_garment_torso_ratio_vectorized(cropped_garment: np.ndarray) -> float:
        if cropped_garment.ndim == 3 and cropped_garment.shape[2] == 4:
            alpha = cropped_garment[:, :, 3]
        else:
            alpha = np.full(cropped_garment.shape[:2], 255, dtype=np.uint8)

        gh, gw = alpha.shape[:2]
        if gh < 10 or gw < 10:
            return 0.70

        y0, y1 = int(gh * 0.55), int(gh * 0.85)
        band = alpha[y0:y1, :] > 128
        row_counts = np.sum(band, axis=1)
        valid_rows = row_counts > 10
        if not np.any(valid_rows):
            return 0.70

        valid_band = band[valid_rows]
        first_x = np.argmax(valid_band, axis=1)
        last_x = gw - 1 - np.argmax(valid_band[:, ::-1], axis=1)
        widths = (last_x - first_x).astype(np.float32)

        torso_w_px = float(np.median(widths))
        ratio = torso_w_px / max(float(gw), 1.0)
        return float(np.clip(ratio, 0.55, 0.92))

    def compute_size_chart_comparison(
        self,
        category_or_garment_type: str,
        fit_type: str,
        fabric_stretch: str,
        gender: str,
        user_measurements: Optional[Dict[str, float]],
        user_cm_per_pixel: float,
        base_length_cm: float = 66.0,
        model_worn_size: str = "M",
        selected_size: str = "M",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute the complete XS -> XXL physical & pixel grading table for the given
        category, fit, gender, and user measurements so the user can verify every size step.
        """
        route = self.resolve_route(category_or_garment_type)
        is_male = "erkek" in str(gender).lower()
        g_key = "male" if is_male else "female"
        size_table = self.STANDARD_SIZE_TABLES[g_key]
        fit_spec = self.get_fit_ease_spec(route.canonical_category, route.sub_category, fit_type)

        fs_low = (fabric_stretch or "").lower()
        if "yüksek" in fs_low or "10" in fs_low:
            stretch_tol_cm = 10.0
        elif "orta" in fs_low or "5" in fs_low or "triko" in fs_low:
            stretch_tol_cm = 6.0
        elif "az" in fs_low or "2" in fs_low or "süprem" in fs_low:
            stretch_tol_cm = 3.0
        else:
            stretch_tol_cm = 0.0

        m = user_measurements or {}
        u_chest = float(m.get("chest", 98.5 if is_male else 92.0))
        u_waist = float(m.get("waist", 85.5 if is_male else 74.0))
        u_hips = float(m.get("hips", 101.5 if is_male else 89.0))
        cm_per_px = max(float(user_cm_per_pixel), 1e-4)

        self.ansur.set_gender(gender)
        u_chest_2d = self.ansur.circumference_to_width(u_chest, "chest")
        u_waist_2d = self.ansur.circumference_to_width(u_waist, "waist")
        u_hips_2d = self.ansur.circumference_to_width(u_hips, "hips")

        chart: Dict[str, Dict[str, float]] = {}
        length_step_cm = 2.5 if route.canonical_category == "lower_body" else (3.0 if route.canonical_category == "full_body" else 2.2)

        for sz in ("XS", "S", "M", "L", "XL", "XXL"):
            ref = size_table[sz]
            step = float(ref["step"])
            g_chest_3d = float(ref["chest"] + fit_spec["chest_ease"])
            g_waist_3d = float(ref["waist"] + fit_spec["waist_ease"])
            g_hips_3d = float(ref["hips"] + fit_spec["hips_ease"])
            g_len_cm = max(28.0, float(base_length_cm + fit_spec["len_mod"] + step * length_step_cm))

            # Project 3D garment circumferences to 2D front flat widths using ANSUR body-to-drape ratios
            drape_step_factor = self.GARMENT_DRAPE_FLATTEN  # sabit: beden farkı zaten 3B çevrede (çift sayım yok)
            g_chest_2d = self.ansur.circumference_to_width(g_chest_3d, "chest") * drape_step_factor
            g_waist_2d = self.ansur.circumference_to_width(g_waist_3d, "waist") * drape_step_factor
            g_hips_2d = self.ansur.circumference_to_width(g_hips_3d, "hips") * drape_step_factor

            if route.canonical_category == "lower_body":
                primary_g_3d = g_hips_3d
                primary_u_3d = u_hips
                primary_2d_cm = g_hips_2d
                secondary_diff = g_waist_3d - u_waist
            elif route.canonical_category == "full_body":
                primary_g_3d = g_chest_3d
                primary_u_3d = u_chest
                primary_2d_cm = max(g_chest_2d, g_hips_2d)
                secondary_diff = g_hips_3d - u_hips
            else:
                primary_g_3d = g_chest_3d
                primary_u_3d = u_chest
                primary_2d_cm = g_chest_2d
                secondary_diff = g_waist_3d - u_waist

            raw_ease_3d = primary_g_3d - primary_u_3d
            eff_ease_3d = raw_ease_3d + stretch_tol_cm * 0.65

            chart[sz] = {
                "step": step,
                "chest_3d_cm": round(g_chest_3d, 1),
                "waist_3d_cm": round(g_waist_3d, 1),
                "hips_3d_cm": round(g_hips_3d, 1),
                "length_cm": round(g_len_cm, 1),
                "chest_2d_cm": round(g_chest_2d, 1),
                "waist_2d_cm": round(g_waist_2d, 1),
                "hips_2d_cm": round(g_hips_2d, 1),
                "chest_px": round(g_chest_2d / cm_per_px, 1),
                "waist_px": round(g_waist_2d / cm_per_px, 1),
                "hips_px": round(g_hips_2d / cm_per_px, 1),
                "length_px": round(g_len_cm / cm_per_px, 1),
                "primary_diff_3d_cm": round(raw_ease_3d, 1),
                "secondary_diff_3d_cm": round(secondary_diff, 1),
                "effective_ease_3d_cm": round(eff_ease_3d, 1),
                "flat_half_width_cm": round(primary_g_3d * 0.5, 1),
                "user_chest_2d_cm": round(u_chest_2d, 1),
                "user_waist_2d_cm": round(u_waist_2d, 1),
                "user_hips_2d_cm": round(u_hips_2d, 1),
            }
        return chart

    def compute_asymmetric_drape_scaling(
        self,
        garment_bgra: np.ndarray,
        garment_cm_width: float,
        garment_length_cm: float,
        user_cm_per_pixel: float,
        user_measurements: Optional[Dict[str, float]] = None,
        category_or_garment_type: str = "upper_body",
        fit_type: str = "Oversize / Rahat Kalıp",
        fabric_stretch: str = "Az Esnek (%2-3 Likra/Süprem)",
        size_modifier_cm: float = 0.0,
        is_side_profile: bool = False,
        gender: str = "Kadın",
        desired_size: Optional[str] = None,
    ) -> AsymmetricScaleSpec:
        route = self.resolve_route(category_or_garment_type)
        self.ansur.set_gender(gender)

        is_male = "erkek" in str(gender).lower()
        measurements = user_measurements or {}
        u_chest = float(measurements.get("chest", 102.0 if is_male else 92.0))
        u_waist = float(measurements.get("waist", 84.0 if is_male else 74.0))
        u_hips = float(measurements.get("hips", 98.0 if is_male else 89.0))

        cm_per_px = max(float(user_cm_per_pixel), 1e-4)
        ft_low = (fit_type or "").lower()
        is_oversize = "oversize" in ft_low or "rahat" in ft_low or "wide" in ft_low or "baggy" in ft_low
        is_slim = "slim" in ft_low or "dar kal" in ft_low
        is_bodycon = "bodycon" in ft_low or "skinny" in ft_low or "saran" in ft_low

        fs_low = (fabric_stretch or "").lower()
        if "yüksek" in fs_low or "10" in fs_low:
            stretch_gravity_boost = 1.18
            stretch_tol_cm = 10.0
        elif "orta" in fs_low or "5" in fs_low or "triko" in fs_low:
            stretch_gravity_boost = 1.12
            stretch_tol_cm = 6.0
        elif "az" in fs_low or "2" in fs_low or "süprem" in fs_low:
            stretch_gravity_boost = 1.05
            stretch_tol_cm = 3.0
        else:
            stretch_gravity_boost = 1.00
            stretch_tol_cm = 0.0

        # Resolve size label & step (-2.0 for XS .. +3.0 for XXL)
        if desired_size:
            norm_sz = self.normalize_size_label(desired_size)
        else:
            # Infer size step from size_modifier_cm when desired_size is not explicitly passed
            if size_modifier_cm <= -4.5:
                norm_sz = "XS"
            elif size_modifier_cm <= -1.5:
                norm_sz = "S"
            elif size_modifier_cm <= 1.5:
                norm_sz = "M"
            elif size_modifier_cm <= 4.5:
                norm_sz = "L"
            elif size_modifier_cm <= 7.5:
                norm_sz = "XL"
            else:
                norm_sz = "XXL"

        g_key = "male" if is_male else "female"
        sz_ref = self.STANDARD_SIZE_TABLES[g_key][norm_sz]
        size_step = float(sz_ref["step"])
        fit_spec = self.get_fit_ease_spec(route.canonical_category, route.sub_category, fit_type)

        # Multi-zone 3D garment circumferences & 2D ANSUR projections
        g_chest_3d = float(sz_ref["chest"] + fit_spec["chest_ease"])
        g_waist_3d = float(sz_ref["waist"] + fit_spec["waist_ease"])
        g_hips_3d = float(sz_ref["hips"] + fit_spec["hips_ease"])

        # Incorporate explicit garment_cm_width delta only when desired_size is not explicitly set
        expected_half = (
            g_hips_3d * 0.5 if route.canonical_category == "lower_body" else g_chest_3d * 0.5
        )
        if not desired_size and garment_cm_width > 25.0:
            blend_half = 0.75 * expected_half + 0.25 * float(garment_cm_width)
            delta_circ = (blend_half - expected_half) * 2.0
            g_chest_3d += delta_circ * 0.5
            g_waist_3d += delta_circ * 0.5
            g_hips_3d += delta_circ * 0.5

        u_chest_w = self.ansur.circumference_to_width(u_chest, "chest")
        u_chest_d = self.ansur.circumference_to_depth(u_chest, "chest")
        u_waist_w = self.ansur.circumference_to_width(u_waist, "waist")
        u_waist_d = self.ansur.circumference_to_depth(u_waist, "waist")
        u_hips_w = self.ansur.circumference_to_width(u_hips, "hips")
        u_hips_d = self.ansur.circumference_to_depth(u_hips, "hips")

        drape_step_factor = self.GARMENT_DRAPE_FLATTEN  # sabit: beden farkı zaten 3B çevrede (çift sayım yok)
        g_chest_2d = self.ansur.circumference_to_width(g_chest_3d, "chest") * drape_step_factor
        g_waist_2d = self.ansur.circumference_to_width(g_waist_3d, "waist") * drape_step_factor
        g_hips_2d = self.ansur.circumference_to_width(g_hips_3d, "hips") * drape_step_factor

        if garment_bgra.ndim == 3 and garment_bgra.shape[2] == 4:
            ys, xs = np.where(garment_bgra[:, :, 3] > 10)
            if len(ys) > 0:
                cropped = garment_bgra[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            else:
                cropped = garment_bgra
        else:
            cropped = garment_bgra

        gh, gw = cropped.shape[:2]

        if route.canonical_category == "upper_body":
            ansur_w = u_chest_w
            ansur_d = u_chest_d
            ansur_waist_w = u_waist_w

            torso_ratio = self.measure_garment_torso_ratio_vectorized(cropped)
            projected_2d_cm = g_chest_2d
            torso_w_px = projected_2d_cm / cm_per_px
            full_w_px = torso_w_px / max(torso_ratio, 0.55)

            ease_cm = g_chest_3d - u_chest + stretch_tol_cm
            gravity_drop_cm = (
                max(-2.5, size_step * 1.6)
                + (1.6 if is_oversize else (0.4 if not is_bodycon else -0.8)) * stretch_gravity_boost
            )
            gravity_drop_px = (gravity_drop_cm * route.gravity_drop_factor) / cm_per_px

            waist_taper_ratio = float(
                np.clip(
                    (g_waist_2d / max(g_chest_2d, 1.0)) * (1.03 if is_oversize else (0.92 if is_bodycon else 0.97)),
                    0.76,
                    1.04,
                )
            )
            hip_flare_ratio = float(
                np.clip(
                    (g_hips_2d / max(g_chest_2d, 1.0)) * (1.02 if is_oversize else 0.98),
                    0.88,
                    1.14,
                )
            )
            hem_flare_ratio = float(hip_flare_ratio * (1.03 if is_oversize else 0.97))
            anchor_y_offset_ratio = -0.14

        elif route.canonical_category == "lower_body":
            ansur_w = u_hips_w
            ansur_d = u_hips_d
            ansur_waist_w = u_waist_w

            projected_2d_cm = g_hips_2d
            torso_w_px = projected_2d_cm / cm_per_px
            full_w_px = torso_w_px * (1.24 if route.sub_category == "skirt" else (1.22 if is_oversize else 1.12))

            ease_cm = g_hips_3d - u_hips + stretch_tol_cm
            gravity_drop_cm = (
                max(-2.0, size_step * 1.8)
                + (2.0 if route.sub_category == "pants" else 1.3) * stretch_gravity_boost
            )
            gravity_drop_px = (gravity_drop_cm * route.gravity_drop_factor) / cm_per_px

            waist_taper_ratio = float(np.clip(g_waist_2d / max(g_hips_2d, 1.0), 0.72, 0.96))
            hip_flare_ratio = float(
                (1.16 if is_oversize else (0.94 if is_bodycon else 1.04)) * (1.0 + 0.045 * size_step)
            )
            hem_flare_ratio = float(
                (
                    (1.32 if not is_bodycon else 0.88)
                    if route.sub_category == "skirt"
                    else (1.42 if is_oversize else (0.76 if is_bodycon else (0.92 if is_slim else 1.08)))
                )
                * (1.0 + 0.055 * size_step)
            )
            anchor_y_offset_ratio = 0.0

        else:  # full_body
            ansur_chest_w = u_chest_w
            ansur_hips_w = u_hips_w
            ansur_waist_w = u_waist_w
            ansur_w = max(ansur_chest_w, ansur_hips_w)
            ansur_d = max(u_chest_d, u_hips_d)

            torso_ratio = self.measure_garment_torso_ratio_vectorized(cropped)
            projected_2d_cm = max(g_chest_2d, g_hips_2d * 0.96)
            torso_w_px = projected_2d_cm / cm_per_px
            full_w_px = torso_w_px / max(torso_ratio, 0.62)

            ease_cm = max(g_chest_3d - u_chest, g_hips_3d - u_hips) + stretch_tol_cm
            gravity_drop_cm = max(-2.5, size_step * 2.2) + 1.8 * stretch_gravity_boost
            gravity_drop_px = (gravity_drop_cm * route.gravity_drop_factor) / cm_per_px

            waist_taper_ratio = float(
                0.95 if is_oversize else np.clip(g_waist_2d / max(g_chest_2d, 1.0), 0.72, 0.92)
            )
            hip_flare_ratio = float(
                np.clip((g_hips_2d / max(g_chest_2d, 1.0)) * (1.0 + 0.04 * size_step), 0.94, 1.30)
            )
            hem_flare_ratio = float(
                (0.90 if is_bodycon else (1.32 if not is_slim else 1.12)) * (1.0 + 0.05 * size_step)
            )
            anchor_y_offset_ratio = -0.15

        target_w_px = int(torso_w_px * (ansur_d / max(ansur_w, 1.0)) * 1.12) if is_side_profile else int(full_w_px)
        target_h_px = int((garment_length_cm / cm_per_px) + gravity_drop_px)

        target_w_px = max(16, target_w_px)
        target_h_px = max(16, target_h_px)

        scale_x_ratio = float(target_w_px) / max(float(gw), 1.0)
        scale_y_ratio = float(target_h_px) / max(float(gh), 1.0)

        effective_size_mod_cm = (
            float(size_modifier_cm) if abs(size_modifier_cm) > 0.01 else float(size_step * 3.0)
        )

        leg_metrics = self.lower_leg_metrics(float(sz_ref["hips"]), u_hips, fit_type, size_step)
        vis_ratio_chest = self.visible_width_ratio(g_chest_2d, u_chest_w)
        vis_ratio_waist = self.visible_width_ratio(g_waist_2d, u_waist_w)
        vis_ratio_hips = self.visible_width_ratio(g_hips_2d, u_hips_w)
        vis_ratio_thigh = self.visible_width_ratio(
            leg_metrics["garment_thigh_3d_cm"] * self.GARMENT_DRAPE_FLATTEN, leg_metrics["user_thigh_3d_cm"]
        )
        vis_ratio_knee = self.visible_width_ratio(
            leg_metrics["garment_knee_3d_cm"] * self.GARMENT_DRAPE_FLATTEN, leg_metrics["user_knee_3d_cm"], cap=2.6
        )
        vis_ratio_hem = self.visible_width_ratio(
            leg_metrics["garment_hem_3d_cm"] * self.GARMENT_DRAPE_FLATTEN, leg_metrics["user_ankle_3d_cm"], cap=3.2
        )

        return AsymmetricScaleSpec(
            canonical_category=route.canonical_category,
            sub_category=route.sub_category,
            scale_x_ratio=scale_x_ratio,
            scale_y_ratio=scale_y_ratio,
            target_width_px=target_w_px,
            target_height_px=target_h_px,
            torso_width_px=float(torso_w_px),
            gravity_drop_px=float(gravity_drop_px),
            waist_taper_ratio=float(waist_taper_ratio),
            hip_flare_ratio=float(hip_flare_ratio),
            hem_flare_ratio=float(hem_flare_ratio),
            anchor_y_offset_ratio=float(anchor_y_offset_ratio),
            ansur_2d_width_cm=float(ansur_w),
            ansur_2d_depth_cm=float(ansur_d),
            projected_garment_2d_cm=float(projected_2d_cm),
            effective_ease_cm=float(ease_cm),
            size_modifier_cm=effective_size_mod_cm,
            extra_metrics={
                "stretch_gravity_boost": float(stretch_gravity_boost),
                "stretch_tolerance_cm": float(stretch_tol_cm),
                "size_step": float(size_step),
                "visual_mult": float(fit_spec["visual_mult"]),
                "garment_chest_3d_cm": float(g_chest_3d),
                "garment_waist_3d_cm": float(g_waist_3d),
                "garment_hips_3d_cm": float(g_hips_3d),
                "garment_chest_2d_cm": float(g_chest_2d),
                "garment_waist_2d_cm": float(g_waist_2d),
                "garment_hips_2d_cm": float(g_hips_2d),
                "user_chest_3d_cm": float(u_chest),
                "user_waist_3d_cm": float(u_waist),
                "user_hips_3d_cm": float(u_hips),
                "user_chest_2d_cm": float(u_chest_w),
                "user_waist_2d_cm": float(u_waist_w),
                "user_hips_2d_cm": float(u_hips_w),
                "user_chest_depth_cm": float(u_chest_d),
                "user_waist_depth_cm": float(u_waist_d),
                "user_hips_depth_cm": float(u_hips_d),
                "body_chest_px": float(u_chest_w / cm_per_px),
                "body_waist_px": float(u_waist_w / cm_per_px),
                "body_hips_px": float(u_hips_w / cm_per_px),
                "garment_chest_px": float(g_chest_2d / cm_per_px),
                "garment_waist_px": float(g_waist_2d / cm_per_px),
                "garment_hips_px": float(g_hips_2d / cm_per_px),
                "user_cm_per_pixel": float(cm_per_px),
                "vis_ratio_chest": vis_ratio_chest,
                "vis_ratio_waist": vis_ratio_waist,
                "vis_ratio_hips": vis_ratio_hips,
                "vis_ratio_thigh": vis_ratio_thigh,
                "vis_ratio_knee": vis_ratio_knee,
                "vis_ratio_hem": vis_ratio_hem,
                **{k: float(v) for k, v in leg_metrics.items()},
            },
            fit_type=str(fit_type or ""),
        )
