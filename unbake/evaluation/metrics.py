"""
Metrics of estimated PBR materials against the ground truth.

  * UV space: per-channel MAE / PSNR (base color, roughness, metallic) between textures that
    share the same UV layout. Texels are weighted by the surface area they cover, so results are
    averages over the surface rather than over the (unevenly packed) atlas.
  * Images: LPIPS (perceptual), PSNR, SSIM and MAE of relit renders (see render.py).
"""
import numpy as np
import torch
from PIL import Image

from ..gltf_io import Glb, Primitive
from ..uv_bake import uv_surface_weights

CHANNELS = ("base_color", "roughness", "metallic")

# ------------------------------------------------------------------ material maps
def material_maps(glb: Glb, material: int | None, size: tuple[int, int],
                  constants: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """
    Base color (H, W, 3) and roughness / metallic (H, W) in [0, 1], glTF factors applied.
    `constants` (e.g. {"roughness": 0.5, "metallic": 0.0}) replaces those channels with a
    fixed value, ignoring whatever the file stores (used for the input-as-albedo baseline).
    """
    maps = _file_material_maps(glb, material, size)
    for channel, value in (constants or {}).items():
        assert channel in ("roughness", "metallic"), f"can only override roughness/metallic, not {channel}"
        maps[channel] = np.full_like(maps[channel], value)
    return maps


def _file_material_maps(glb: Glb, material: int | None, size: tuple[int, int]) -> dict[str, np.ndarray]:
    width, height = size
    pbr = {} if material is None else glb.json["materials"][material].get("pbrMetallicRoughness", {})

    def texture(key) -> np.ndarray | None:
        ref = pbr.get(key)
        if ref is None:
            return None
        img = glb.image(glb.json["textures"][ref["index"]]["source"]).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255

    bc_factor = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], dtype=np.float32)
    bc = texture("baseColorTexture")
    bc = (bc if bc is not None else np.ones((height, width, 3), np.float32)) * bc_factor
    mr = texture("metallicRoughnessTexture")  # glTF: G = roughness, B = metallic
    rough = mr[..., 1] if mr is not None else np.ones((height, width), np.float32)
    metal = mr[..., 2] if mr is not None else np.ones((height, width), np.float32)
    return {
        "base_color": bc,
        "roughness": rough * pbr.get("roughnessFactor", 1.0),
        "metallic": metal * pbr.get("metallicFactor", 1.0),
    }


def _face_corners(prims: list[Primitive]):
    """Per face: corner positions (F, 3, 3), corner UVs (F, 3, 2; NaN without UVs), material (F,; -1 = none)."""
    pos, uvs, mats = [], [], []
    for p in prims:
        pos.append(p.positions[p.faces])
        uvs.append(p.uvs[p.faces] if p.uvs is not None else np.full((len(p.faces), 3, 2), np.nan))
        mats.append(np.full(len(p.faces), -1 if p.material is None else p.material))
    return np.concatenate(pos), np.concatenate(uvs), np.concatenate(mats)


def match_materials(gt: Glb, pred: Glb, tol: float = 1e-5, uv_tol: float = 1e-4) -> dict[int, int]:
    """
    Ground-truth material -> predicted material, for two assets with the same geometry and UV
    layout but possibly different mesh names, mesh order, vertex indexing and material numbering
    (as for the dataset's baked assets and predictions written into them). Faces are matched by
    position (nearest centroid, corners within `tol` x bounding-box diagonal); their per-corner
    UVs must agree. Raises if the layouts differ, so a UV-space comparison would be invalid.
    """
    from scipy.spatial import cKDTree

    g_pos, g_uv, g_mat = _face_corners(gt.world_primitives())
    p_pos, p_uv, p_mat = _face_corners(pred.world_primitives())
    assert len(g_pos) == len(p_pos), f"face count differs (ground truth {len(g_pos)}, prediction {len(p_pos)})"
    diag = np.linalg.norm(g_pos.reshape(-1, 3).max(0) - g_pos.reshape(-1, 3).min(0))
    _, j = cKDTree(p_pos.mean(1)).query(g_pos.mean(1))
    dist = np.linalg.norm(g_pos[:, :, None] - p_pos[j][:, None], axis=-1)  # (F, 3 gt corners, 3 pred corners)
    assert dist.min(-1).max() < tol * diag, "geometry differs from the ground truth: UV-space comparison is not valid"
    corner = dist.argmin(-1)
    p_uv_matched = np.take_along_axis(p_uv[j], corner[..., None], 1)
    has_uv = ~np.isnan(g_uv).any(axis=(1, 2))
    assert np.abs(g_uv[has_uv] - p_uv_matched[has_uv]).max(initial=0) < uv_tol, \
        "UV layout differs from the ground truth: UV-space comparison is not valid"

    mapping = {}
    for m in np.unique(g_mat[g_mat >= 0]):
        targets = np.unique(p_mat[j[g_mat == m]])
        assert len(targets) == 1 and targets[0] >= 0, \
            f"ground-truth material {m} maps to predicted materials {targets.tolist()} (expected exactly one)"
        mapping[int(m)] = int(targets[0])
    return mapping


# ------------------------------------------------------------------ UV-space metrics
def uv_metrics(gt: Glb, pred: Glb, pred_constants: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    """
    Surface-area-weighted MAE, MSE and PSNR per channel. Requires both assets to use the same
    geometry and UV layout (checked by match_materials); mesh names and order may differ.
    Primitives without a material are skipped.
    """
    gt_prims = gt.world_primitives()
    material_map = match_materials(gt, pred)
    err_sum = {c: 0.0 for c in CHANNELS}
    sq_sum = {c: 0.0 for c in CHANNELS}
    weight_sum = 0.0

    for material, pred_material in material_map.items():
        group = [p for p in gt_prims if p.material == material]
        img = gt.material_base_color_image(material)
        size = gt.image(img).size if img is not None else (1024, 1024)
        weights = uv_surface_weights(group, size)
        gt_maps = material_maps(gt, material, size)
        pred_maps = material_maps(pred, pred_material, size, pred_constants)  # resampled to the gt texture size
        for c in CHANNELS:
            diff = pred_maps[c] - gt_maps[c]
            if diff.ndim == 3:  # base color: average over RGB
                err, sq = np.abs(diff).mean(-1), (diff ** 2).mean(-1)
            else:
                err, sq = np.abs(diff), diff ** 2
            err_sum[c] += float((err * weights).sum())
            sq_sum[c] += float((sq * weights).sum())
        weight_sum += float(weights.sum())

    return {c: {"MAE": err_sum[c] / weight_sum, "MSE": sq_sum[c] / weight_sum, "PSNR": _psnr(sq_sum[c] / weight_sum)}
            for c in CHANNELS}


def _psnr(mse: float) -> float:
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


# ------------------------------------------------------------------ image metrics
class RenderMetrics:
    def __init__(self, device="cuda"):
        import lpips
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        """pred, gt: (N, 3, H, W) in [0, 1]; mask: (N, H, W). Pixel metrics use object pixels only."""
        import kornia
        m = (mask > 0.5).float()[:, None]
        n = m.sum() * 3
        mae = ((pred - gt).abs() * m).sum() / n
        mse = (((pred - gt) ** 2) * m).sum() / n
        ssim_map = kornia.metrics.ssim(pred, gt, window_size=11)
        ssim = (ssim_map * m).sum() / n
        lp = self.lpips(pred * 2 - 1, gt * 2 - 1).mean()  # full frame (identical black background)
        return {"LPIPS": float(lp), "PSNR": _psnr(float(mse)), "SSIM": float(ssim), "MAE": float(mae)}
