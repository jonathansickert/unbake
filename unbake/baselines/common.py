"""
Shared by the baselines: helpers for the 2D baselines, which work on trimesh parts (load_parts,
...), and the writer of their textured.glb. The prediction layout is in unbake/predictions.py.
"""
import hashlib

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from ..uv_bake import TrellisFrame, _rasterize_uv, to_trellis


def load_parts(path):
    """Load an asset as a list of parts (scene transforms applied), each with its own
    UVs (trimesh convention: v up) and base-color texture."""
    scene = trimesh.load(path, force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, gname = scene.graph[node]
        geom = scene.geometry[gname]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        geom = geom.copy()
        geom.apply_transform(transform)
        uv = getattr(geom.visual, "uv", None)
        if uv is None or len(uv) != len(geom.vertices):
            raise ValueError(f"{path}: part {gname} has no per-vertex UVs")
        tex = base_color_texture(geom.visual)
        parts.append({"name": gname, "mesh": geom, "uv": np.asarray(uv, dtype=np.float32), "texture": tex,
                      "tex_key": hashlib.md5(tex.tobytes()).hexdigest() + str(tex.shape)})
    if not parts:
        raise ValueError(f"{path}: no triangle meshes found")
    return parts


def base_color_texture(visual):
    """Baked texture as float RGB in [0, 1] (texture space, i.e. sRGB-encoded); a 1x1
    image if the material only has a constant color."""
    mat = getattr(visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    factor = getattr(mat, "baseColorFactor", None)
    factor = np.ones(3, dtype=np.float32) if factor is None else np.asarray(factor[:3], dtype=np.float32) / 255
    if img is None:
        return factor[None, None].astype(np.float32)
    return (np.asarray(img.convert("RGB"), dtype=np.float32) / 255 * factor).astype(np.float32)


def normalization(vertices):
    """(center, scale) of the TRELLIS.2 space, as Trellis2TexturingPipeline.preprocess_mesh."""
    frame = TrellisFrame(vertices)
    return frame.center, frame.scale


def normalize(vertices, center, scale):
    """Into TRELLIS space: unit cube around the origin, Z up (glTF is Y up)."""
    return to_trellis(vertices, center, scale).astype(np.float32)


@torch.no_grad()
def uv_coverage(uv, faces, h, w):
    """(h, w) bool: texels covered by the UV layout (trimesh UVs). Each triangle is shifted into
    the unit tile by the floor of its centroid (glTF repeat wrapping)."""
    uv_tri = uv[faces].astype(np.float32)
    uv_tri = uv_tri - np.floor(uv_tri.mean(1, keepdims=True))
    uv_gltf = np.stack([uv_tri[..., 0], 1 - uv_tri[..., 1]], -1).reshape(-1, 2)  # glTF: v down
    rast, _ = _rasterize_uv(uv_gltf, np.arange(len(uv_gltf)).reshape(-1, 3), (w, h))
    return (rast[0, ..., 3] > 0).cpu().numpy()


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def dilate(img_u8, mask):
    """Fill texels outside the UV islands so that the GLB has no dark seams."""
    return cv2.inpaint(img_u8, (~mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)


def pbr_material(base_color_u8, roughness_u8, metallic_u8, known=None):
    """glTF PBR material from uint8 maps; texels outside `known` (bool mask) are filled by dilation."""
    fill = (lambda img: dilate(img, known)) if known is not None and known.any() else (lambda img: img)
    rough, metal = fill(roughness_u8), fill(metallic_u8)
    return trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(fill(base_color_u8)),
        metallicRoughnessTexture=Image.fromarray(np.stack([np.zeros_like(rough), rough, metal], -1)),
        metallicFactor=1.0, roughnessFactor=1.0, doubleSided=True)


def export_textured_glb(parts, material_of, path):
    """Input geometry and UVs with new materials; material_of(part) -> PBRMaterial."""
    meshes = []
    for p in parts:
        mesh = p["mesh"].copy()
        mesh.visual = trimesh.visual.TextureVisuals(uv=p["uv"], material=material_of(p))
        meshes.append(mesh)
    trimesh.Scene(meshes).export(path)
