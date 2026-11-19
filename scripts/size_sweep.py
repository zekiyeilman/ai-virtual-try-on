"""Beden taraması: aynı kıyafeti birden fazla bedende giydirip ölçer ve montaj üretir.

Beden farkı / ısı haritası / alt giyim regresyonlarını gözle ve sayıyla kontrol etmek için.

Örnekler:
    python scripts/size_sweep.py                          # üst giyim, User 1, ön, XS/M/XXL
    python scripts/size_sweep.py --preset lower --user u2 # pantolon, User 2
    python scripts/size_sweep.py --angle arka --sizes S L XL --ai

Çıktı: out/size_sweep/<etiket>.jpg (+ _heatmap, _mask) ve montage_result.jpg / montage_heatmap.jpg.
Konsolda her beden için kıyafet maskesinin %30/%60/%90 yükseklikteki genişliği (px) ve alanı basılır;
beden büyüdükçe genişlik ve alan artmalı, ısı haritasındaki vücut (mavi) genişliği sabit kalmalıdır.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vton.engine import VTONEngine  # noqa: E402
from vton.paths import goruntu  # noqa: E402

USERS = {
    "u1": {"image": "user_{a}.jpg", "meas": {"height": 164.0, "chest": 92.0, "waist": 60.0, "hips": 89.0}},
    "u2": {"image": "user2_{a}.png", "meas": {"height": 170.0, "chest": 100.0, "waist": 84.0, "hips": 114.0}},
}
PRESETS = {
    "upper": {"garment": "temp_garment_extracted_{a}.png", "cat": "Üst Giyim (Tişört / Gömlek / Kazak / Ceket)"},
    "lower": {"garment": "test_pants_garment.png", "cat": "Alt Giyim - Pantolon / Jean / Eşofman"},
}
FITS = {
    "oversize": "Oversize / Rahat Kalıp",
    "regular": "Regular / Standart Kalıp",
    "slim": "Slim Fit / Dar Kalıp",
}


def mask_info(mask_path: str) -> str:
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return "maske yok"
    mm = (m[..., 0] if m.ndim == 3 else m) > 0
    ys = np.where(mm.any(1))[0]
    if not len(ys):
        return "boş maske"
    y0, y1 = int(ys.min()), int(ys.max())
    rows = [int(y0 + (y1 - y0) * f) for f in (0.3, 0.6, 0.9)]
    return f"y=({y0},{y1}) genişlik@30/60/90%={[int(mm[r].sum()) for r in rows]} alan={int(mm.sum())}"


def montage(out_path: str, files: list[str], h: int = 600) -> None:
    ims = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        cv2.putText(im, os.path.basename(f)[:28], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        ims.append(im)
    if ims:
        cv2.imwrite(out_path, np.hstack(ims))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="upper")
    ap.add_argument("--user", choices=USERS, default="u1")
    ap.add_argument("--angle", choices=["on", "arka", "yan"], default="on")
    ap.add_argument("--sizes", nargs="+", default=["XS", "M", "XXL"])
    ap.add_argument("--fit", choices=FITS, default="oversize")
    ap.add_argument("--stretch", default="Az Esnek (%2-3 Likra/Süprem)")
    ap.add_argument("--ai", action="store_true", help="HD (önbellekli HF master / Colab) yolunu kullan")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "size_sweep"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    user, preset = USERS[args.user], PRESETS[args.preset]
    eng = VTONEngine()
    results, heatmaps = [], []
    for size in args.sizes:
        tag = f"{args.preset}_{args.user}_{args.angle}_{args.fit}_{size}"
        out = os.path.join(args.out, f"{tag}.jpg")
        t = time.time()
        msg = eng.run(
            user_image_path=goruntu(user["image"].format(a=args.angle)),
            garment_image_path=goruntu(preset["garment"].format(a=args.angle)),
            original_on_jpg_path=goruntu("on.jpg"),
            output_name=out,
            user_measurements=user["meas"],
            desired_size=size,
            model_worn_size="M",
            garment_type=preset["cat"],
            use_ai_render=args.ai,
            colab_url="",
            fit_type=FITS[args.fit],
            fabric_stretch=args.stretch,
            gender="Kadın",
        )
        print(f"== {tag} {time.time() - t:.1f}s {mask_info(out.replace('.jpg', '_mask.png'))}")
        for line in (msg or "").splitlines():
            if "Render Durumu" in line or "Uygulanan" in line or line.startswith("  •"):
                print("   ", line[:300])
        results.append(out)
        heatmaps.append(out.replace(".jpg", "_heatmap.jpg"))
    montage(os.path.join(args.out, "montage_result.jpg"), results)
    montage(os.path.join(args.out, "montage_heatmap.jpg"), heatmaps)
    print(f"Montajlar: {args.out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
