"""
Garment Extractor Module (Global Singleton U2Net & MediaPipe Pose).
"""

import os
import cv2
import rembg
from .category_router import get_global_models


def ayikla_ust_kiyafet(input_image_path: str, output_image_path: str) -> None:
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"[ERROR] Dosya okunamadı: {input_image_path}")
        return

    models = get_global_models()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = models.detect_pose(img_rgb, is_rgb=True)

    session = models.u2net_cloth_session if results.pose_landmarks else models.u2net_session
    rgba_cloth = rembg.remove(img_rgb, session=session)
    rgba_cloth = cv2.cvtColor(rgba_cloth, cv2.COLOR_RGBA2BGRA)

    if results.pose_landmarks:
        h, w, _ = img.shape
        lm = results.pose_landmarks.landmark
        l_hip = lm[models.mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[models.mp_pose.PoseLandmark.RIGHT_HIP]
        l_sh = lm[models.mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[models.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if l_hip.visibility > 0.5 or r_hip.visibility > 0.5:
            hip_y = int(((l_hip.y + r_hip.y) / 2.0) * h)
            shoulder_y = int(((l_sh.y + r_sh.y) / 2.0) * h)
            torso_length = hip_y - shoulder_y
            kesme_hizasi_y = hip_y + int(torso_length * 0.25)
            if kesme_hizasi_y < h:
                rgba_cloth[kesme_hizasi_y:, :, 3] = 0

    cv2.imwrite(output_image_path, rgba_cloth)


if __name__ == "__main__":
    from vton.paths import goruntu

    hedef_dosyalar = {
        goruntu("on.jpg"): goruntu("ayiklanmis_on.png"),
        goruntu("arka.jpg"): goruntu("ayiklanmis_arka.png"),
    }
    for input_path, output_path in hedef_dosyalar.items():
        if os.path.exists(input_path):
            ayikla_ust_kiyafet(input_path, output_path)
