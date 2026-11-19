"""
HD Render & Print Restoration Module (delegates to vectorized vton.inference_api).
"""

import os
from .inference_api import (
    build_category_diffusion_prompt,
    load_idm_vton_pipeline_vram_safe,
    postprocess_hd_render,
    prepare_studio_garment_image,
    render_with_idm_vton,
    restore_garment_print,
    restore_side_garment_print,
)

__all__ = [
    "prepare_studio_garment_image",
    "restore_garment_print",
    "restore_side_garment_print",
    "postprocess_hd_render",
    "build_category_diffusion_prompt",
    "render_with_idm_vton",
    "load_idm_vton_pipeline_vram_safe",
]

if __name__ == "__main__":
    from vton.paths import goruntu

    if os.path.exists(goruntu("user_on.jpg")) and os.path.exists(goruntu("ayiklanmis_on.png")):
        render_with_idm_vton(
            goruntu("user_on.jpg"),
            goruntu("ayiklanmis_on.png"),
            goruntu("vton_hd_sonuc_on.jpg"),
        )
