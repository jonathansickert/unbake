"""
Rendering for the relighting evaluation: assets rendered with the TRELLIS.2 PBR renderer under
HDR environment lighting, with base color / roughness / metallic only (no normal, emissive or
occlusion maps), so differences come from the estimated channels alone.
"""
import numpy as np
import OpenEXR
import torch

from trellis2.renderers import EnvMap, PbrMeshRenderer
from trellis2.representations.mesh import MeshWithPbrMaterial, PbrMaterial, Texture
from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

from ..gltf_io import Glb
from ..uv_bake import TrellisFrame
from .metrics import material_maps

def render_mesh(glb: Glb, frame: TrellisFrame, constants: dict[str, float] | None = None,
                device="cuda") -> MeshWithPbrMaterial:
    """GLB -> MeshWithPbrMaterial (base color / roughness / metallic) in TRELLIS space."""
    prims = glb.world_primitives()
    materials = []
    for mi in range(len(glb.json.get("materials", []))):
        img = glb.material_base_color_image(mi)
        size = glb.image(img).size if img is not None else (1024, 1024)
        maps = material_maps(glb, mi, size, constants)
        to_tex = lambda a: Texture(torch.from_numpy(np.ascontiguousarray(a)).float().reshape(size[1], size[0], -1).to(device))
        materials.append(PbrMaterial(
            base_color_texture=to_tex(maps["base_color"]),
            roughness_texture=to_tex(maps["roughness"]),
            metallic_texture=to_tex(maps["metallic"]),
        ).to(device))
    default = len(materials)
    materials.append(PbrMaterial(base_color_factor=[0.8, 0.8, 0.8], metallic_factor=0.0, roughness_factor=0.5).to(device))

    vertices, faces, material_ids, uv_coords, offset = [], [], [], [], 0
    for p in prims:
        vertices.append(frame(p.positions))
        faces.append(p.faces + offset)
        material_ids.append(np.full(len(p.faces), default if p.material is None else p.material))
        uv_coords.append(p.uvs[p.faces] if p.uvs is not None else np.zeros((len(p.faces), 3, 2)))
        offset += len(p.positions)
    t = lambda a, dt: torch.from_numpy(np.concatenate(a)).to(device=device, dtype=dt)
    return MeshWithPbrMaterial(
        vertices=t(vertices, torch.float32),
        faces=t(faces, torch.int32),
        material_ids=t(material_ids, torch.long),
        uv_coords=t(uv_coords, torch.float32),
        materials=materials,
    )


def load_envmap(path: str, mean_luminance: float | None = None, device="cuda") -> EnvMap:
    """
    Equirectangular HDR env map. With `mean_luminance`, the map is scaled so its solid-angle
    weighted mean luminance equals that value (same exposure for bright and dim maps).
    """
    with OpenEXR.File(path) as f:
        channels = f.channels()
        if "RGBA" in channels or "RGB" in channels:
            px = channels["RGBA" if "RGBA" in channels else "RGB"].pixels[..., :3]
        else:
            px = np.stack([channels[c].pixels for c in "RGB"], -1)
    px = np.ascontiguousarray(px, dtype=np.float32)
    if mean_luminance is not None:
        lum = px.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
        rows = px.shape[0]
        solid_angle = np.cos((np.arange(rows) + 0.5) / rows * np.pi - np.pi / 2)[:, None]  # per-row weight
        px *= mean_luminance / ((lum * solid_angle).sum() / (solid_angle.sum() * px.shape[1]))
    return EnvMap(torch.from_numpy(px).to(device))


def held_out_cameras(num_views: int = 8, radius: float = 2.0, fov: float = 40.0):
    """Evenly spaced yaws (offset half a step from 0) at alternating 15/35 degree elevation."""
    yaws = [(i + 0.5) * 2 * np.pi / num_views for i in range(num_views)]
    pitches = [np.deg2rad(15 if i % 2 == 0 else 35) for i in range(num_views)]
    return yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches, radius, fov)


@torch.no_grad()
def render_views(mesh, envmaps: dict[str, EnvMap], cameras, resolution=512, ssaa=2, seed=0):
    """
    Returns {env name: (N, 3, H, W) tone-mapped renders in [0, 1]}, and (N, H, W) object masks.
    The renderer's SSAO draws random samples; they are seeded per view, so renders are
    reproducible and every mesh rendered from a view gets the same SSAO noise.
    """
    renderer = PbrMeshRenderer({"resolution": resolution, "near": 1, "far": 100, "ssaa": ssaa, "peel_layers": 1})
    out = {k: [] for k in envmaps}
    masks = []
    for view, (extr, intr) in enumerate(zip(*cameras)):
        with torch.random.fork_rng(devices=[extr.device] if extr.is_cuda else []):
            torch.manual_seed(seed + view)
            res = renderer.render(mesh, extr, intr, envmap=envmaps)
        for k in envmaps:
            out[k].append(res[f"shaded_{k}"])
        masks.append(res["mask"])
    return {k: torch.stack(v) for k, v in out.items()}, torch.stack(masks)
