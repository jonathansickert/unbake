"""
Output stage: bake per-voxel PBR materials into a mesh's original UV layout by sampling
the voxel grid at the 3D surface point of every texel.
"""
import cv2
import flex_gemm
import numpy as np
import nvdiffrast.torch as dr
import torch
from PIL import Image

from .gltf_io import Primitive

# Channel layout of the decoded TRELLIS.2 texture voxels
PBR_LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}


class TrellisFrame:
    """
    World space (glTF, Y-up) -> TRELLIS.2 space: centered, scaled into [-0.5, 0.5]^3, Z-up.
    Same transform as Trellis2TexturingPipeline.preprocess_mesh, but reusable for texel positions.
    """

    def __init__(self, positions: np.ndarray):
        lo, hi = positions.min(0), positions.max(0)
        self.center = (lo + hi) / 2
        self.scale = 0.99999 / (hi - lo).max()

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        return to_trellis(positions, self.center, self.scale)


def to_trellis(positions: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    p = (positions - center) * scale
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)  # (x, y, z) -> (x, -z, y)


def merge(prims: list[Primitive], frame: TrellisFrame):
    """Concatenate primitives into one (positions in TRELLIS space, uvs, faces) triangle soup."""
    positions, uvs, faces, offset = [], [], [], 0
    for p in prims:
        positions.append(frame(p.positions))
        if p.uvs is not None:
            uvs.append(p.uvs)
        faces.append(p.faces + offset)
        offset += len(p.positions)
    return (np.concatenate(positions).astype(np.float32),
            np.concatenate(uvs).astype(np.float32) if len(uvs) == len(prims) else None,
            np.concatenate(faces).astype(np.int32))


@torch.no_grad()
def bake_voxels_to_uv(
    prims: list[Primitive],
    frame: TrellisFrame,
    pbr_voxel,                 # SparseTensor of decoded materials, values in [0, 1]
    resolution: int,           # voxel grid resolution the materials were decoded at
    texture_size: tuple[int, int],  # (width, height) of the output textures
    pad_pixels: int = 8,
) -> dict[str, np.ndarray]:
    """
    Returns float32 maps in [0, 1] laid out in the primitives' own UVs:
    'base_color' (H, W, 3), 'metallic' (H, W), 'roughness' (H, W), 'alpha' (H, W), 'mask' (H, W) bool.
    Texels outside the UV islands are filled by dilating the islands by `pad_pixels`
    (avoids seams from bilinear/mip filtering) and left at the island-average elsewhere.
    """
    positions, uvs, faces = merge(prims, frame)
    assert uvs is not None, "every primitive needs TEXCOORD_0 to bake into its UV layout"
    width, height = texture_size
    device = "cuda"
    rast, faces_t = _rasterize_uv(uvs, faces, texture_size)
    mask = rast[0, ..., 3] > 0
    pos_t = torch.from_numpy(positions).to(device)
    texel_pos = dr.interpolate(pos_t[None], rast, faces_t)[0][0]  # (H, W, 3) surface point per texel

    # Sample the sparse voxel grid at every covered texel's surface point. Missing (inactive)
    # voxels count as 0 in trilinear interpolation, so a constant-1 channel is sampled too
    # and used to renormalize; texels with no active voxel nearby become holes to fill.
    feats = pbr_voxel.feats.float()
    feats = torch.cat([feats, torch.ones_like(feats[:, :1])], dim=1)
    sampled = flex_gemm.ops.grid_sample.grid_sample_3d(
        feats,
        pbr_voxel.coords,
        shape=torch.Size([pbr_voxel.shape[0], feats.shape[1], *pbr_voxel.spatial_shape]),
        grid=((texel_pos[mask] + 0.5) * resolution).reshape(1, -1, 3),
        mode="trilinear",
    ).reshape(-1, feats.shape[1])
    weight = sampled[:, -1:]
    valid = weight[:, 0] > 1e-3
    attrs = torch.zeros(height, width, feats.shape[1] - 1, device=device)
    attrs[mask] = (sampled[:, :-1] / weight.clamp(min=1e-3)).clamp(0, 1)
    sampled_ok = torch.zeros_like(mask)
    sampled_ok[mask] = valid
    attrs = attrs.cpu().numpy()
    mask_np = mask.cpu().numpy()

    attrs = _pad_islands(attrs, sampled_ok.cpu().numpy(), pad_pixels)
    return {
        "base_color": attrs[..., PBR_LAYOUT["base_color"]],
        "metallic": attrs[..., PBR_LAYOUT["metallic"]][..., 0],
        "roughness": attrs[..., PBR_LAYOUT["roughness"]][..., 0],
        "alpha": attrs[..., PBR_LAYOUT["alpha"]][..., 0],
        "mask": mask_np,
    }


def _rasterize_uv(uvs: np.ndarray, faces: np.ndarray, texture_size: tuple[int, int]):
    """
    Rasterize triangles in UV space. glTF UV (0,0) is the top-left texel, which nvdiffrast's
    row 0 (clip y = -1) also maps to, so the result needs no flip.
    """
    width, height = texture_size
    uv_t = torch.from_numpy(np.ascontiguousarray(uvs, dtype=np.float32)).cuda()
    faces_t = torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int32)).cuda()
    clip = torch.cat([uv_t * 2 - 1, torch.zeros_like(uv_t[:, :1]), torch.ones_like(uv_t[:, :1])], dim=-1)[None]
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(), clip, faces_t, resolution=[height, width])
    return rast, faces_t


@torch.no_grad()
def uv_surface_weights(prims: list[Primitive], texture_size: tuple[int, int]) -> np.ndarray:
    """
    (H, W) float: surface area each texel represents (0 outside the UV islands).
    Each texel is weighted by its triangle's 3D-area / UV-area ratio, so averages over
    texels become averages over the surface, independent of how densely each island is packed.
    """
    uvs, faces, positions, offset = [], [], [], 0
    for p in prims:
        uvs.append(p.uvs)
        positions.append(p.positions)
        faces.append(p.faces + offset)
        offset += len(p.uvs)
    uvs, faces, positions = np.concatenate(uvs), np.concatenate(faces), np.concatenate(positions)
    rast, _ = _rasterize_uv(uvs, faces, texture_size)
    tri = rast[0, ..., 3].long().cpu().numpy() - 1  # -1 = no triangle

    p0, p1, p2 = positions[faces[:, 0]], positions[faces[:, 1]], positions[faces[:, 2]]
    area_3d = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    t0, t1, t2 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]
    e1, e2 = t1 - t0, t2 - t0
    area_uv = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    density = np.where(area_uv > 0, area_3d / np.maximum(area_uv, 1e-20), 0.0)

    weights = np.zeros(tri.shape, dtype=np.float64)
    covered = tri >= 0
    weights[covered] = density[tri[covered]]
    return weights


def _pad_islands(attrs: np.ndarray, mask: np.ndarray, pad_pixels: int) -> np.ndarray:
    """Grow UV islands outward by repeatedly averaging covered neighbours (edge padding)."""
    out = attrs.copy()
    filled = mask.astype(np.float32)
    kernel = np.ones((3, 3), np.float32)
    for _ in range(pad_pixels):
        weight = cv2.filter2D(filled, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        summed = np.stack([cv2.filter2D(out[..., c] * filled, -1, kernel, borderType=cv2.BORDER_CONSTANT)
                           for c in range(out.shape[-1])], axis=-1)
        grow = (filled == 0) & (weight > 0)
        out[grow] = summed[grow] / weight[grow][:, None]
        filled[grow] = 1
    if mask.any():  # everything else: island average, so mip levels don't bleed in black
        out[filled == 0] = attrs[mask].mean(0)
    return out


def to_gltf_images(maps: dict[str, np.ndarray]) -> tuple[Image.Image, Image.Image]:
    """
    glTF textures: baseColor (RGB), metallicRoughness (G = roughness, B = metallic, R unused = 255).
    """
    u8 = lambda x: (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    base_color = Image.fromarray(u8(maps["base_color"]), "RGB")
    ones = np.ones_like(maps["roughness"])
    metallic_roughness = Image.fromarray(u8(np.stack([ones, maps["roughness"], maps["metallic"]], -1)), "RGB")
    return base_color, metallic_roughness
