"""
VTON - Size-Aware & Physics-Aware 360° Virtual Try-On Package.
"""

import importlib

__version__ = "0.3.0"

_LAZY = {
    "VTONEngine": "vton.engine",
    "run_vton_pipeline": "vton.engine",
    "SegFormerNode": "vton.nodes.segmentation",
    "BodyMeshNode": "vton.nodes.body_mesh",
    "FitHeatmapNode": "vton.nodes.heatmap",
    "Ansur3DTo2DConverter": "vton.models.ansur",
    "CategoryRouter": "vton.category_router",
    "CategoryRouteConfig": "vton.category_router",
    "AsymmetricScaleSpec": "vton.category_router",
    "GlobalModelRegistry": "vton.category_router",
    "get_global_models": "vton.category_router",
    "AnatomicalMaskGenerator": "vton.mask_generator",
    "ThinPlateSplineWarper": "vton.mask_generator",
    "load_idm_vton_pipeline_vram_safe": "vton.inference_api",
    "render_with_idm_vton": "vton.inference_api",
    "ayikla_ust_kiyafet": "vton.extractor",
    "vton_otonom_boru_hatti": "vton.scraper.pipeline",
}

__all__ = list(_LAZY) + ["paths", "__version__"]


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    if name == "paths":
        return importlib.import_module("vton.paths")
    raise AttributeError(f"module 'vton' has no attribute {name!r}")
