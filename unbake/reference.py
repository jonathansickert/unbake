"""
Reference (conditioning) image of a baked-lighting asset: the asset rendered unlit, i.e. its
baked texture exactly as a viewer shows it. This is the only image a user with just the baked
mesh has; the dataset's PNG renderings show the ground-truth PBR asset and must not be used.
"""
import numpy as np
import torch
from PIL import Image

from trellis2.renderers import EnvMap, PbrMeshRenderer
from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

from .evaluation.render import render_mesh
from .gltf_io import Glb
from .uv_bake import TrellisFrame

# Three-quarter view from above, similar to the dataset renderings. TRELLIS.2 crops the
# reference image to the object, so only the direction matters, not the distance.
YAW_DEG = 45.0
PITCH_DEG = 25.0


@torch.no_grad()
def render_unlit(glb: Glb, frame: TrellisFrame, yaw_deg: float = YAW_DEG, pitch_deg: float = PITCH_DEG,
                 resolution: int = 1024, ssaa: int = 2) -> Image.Image:
    """RGBA image of the asset's base color texture without any shading (alpha = coverage)."""
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics([np.deg2rad(yaw_deg)], [np.deg2rad(pitch_deg)], 2.0, 40.0)
    renderer = PbrMeshRenderer({"resolution": resolution, "near": 1, "far": 100, "ssaa": ssaa, "peel_layers": 1})
    unused_light = EnvMap(torch.ones(64, 128, 3, device="cuda"))  # required by the renderer; base_color ignores it
    res = renderer.render(render_mesh(glb, frame), extr[0], intr[0], envmap=unused_light)
    rgb = res.base_color.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    alpha = res.mask.reshape(rgb.shape[:2]).clamp(0, 1).cpu().numpy()
    return Image.fromarray((np.dstack([rgb, alpha]) * 255 + 0.5).astype(np.uint8), "RGBA")
