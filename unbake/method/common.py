"""Shared by latent_opt.py, guided_sampling.py and the experiments: the TRELLIS.2 pipeline, the conditioning
and shape latent exactly as in the unguided baseline (so that zero guidance reproduces it), decoder
checkpointing, voxel lookup and writing a prediction in the layout of unbake/predictions.py."""
import hashlib
import json
import os
import time

import numpy as np
import torch
import trimesh

from unbake.reference import render_unlit
from unbake.uv_bake import TrellisFrame, bake_voxels_to_uv, merge, to_gltf_images

REPO = "microsoft/TRELLIS.2-4B"
SEED = 42  # default seed of the unguided baseline


def load_pipeline():
    """TRELLIS.2 texturing pipeline with all models on the GPU (reference images have alpha, so no RMBG)."""
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    from xformers.ops.fmha import dispatch
    dispatch._set_use_fa3(False)  # FlashAttention-3 crashes on Blackwell, see the baseline
    from trellis2.pipelines import Trellis2TexturingPipeline, rembg
    rembg.BiRefNet = lambda **kwargs: None
    pipe = Trellis2TexturingPipeline.from_pretrained(REPO, config_file="texturing_pipeline.json")
    pipe.cuda()
    return pipe


@torch.no_grad()
def prepare(pipe, lighting, resolution):
    """Conditioning, shape latent and TRELLIS frame of one dataset input, as in the unguided baseline."""
    glb = lighting.baked()
    prims = glb.world_primitives()
    frame = TrellisFrame(np.concatenate([p.positions for p in prims]))
    image = render_unlit(glb, frame)
    cond = pipe.get_cond([pipe.preprocess_image(image)], resolution, include_neg_cond=False)["cond"]
    positions, _, faces = merge(prims, frame)
    shape_slat = pipe.encode_shape_slat(trimesh.Trimesh(vertices=positions, faces=faces, process=False), resolution)
    return {"cond": cond, "neg_cond": torch.zeros_like(cond)}, shape_slat, frame, image


def write_prediction(lighting, frame, pbr_voxel, resolution, out_dir, meta, image=None):
    """Bake per-voxel materials (values in [0, 1]) into the input's UV layout and save textured.glb, cond.png
    and meta.json (written last: marks the input as done), as the unguided baseline does."""
    os.makedirs(out_dir, exist_ok=True)
    glb = lighting.baked()
    prims = glb.world_primitives()
    for material in sorted({p.material for p in prims if p.material is not None}):
        img_index = glb.material_base_color_image(material)
        size = glb.image(img_index).size if img_index is not None else (1024, 1024)
        maps = bake_voxels_to_uv([p for p in prims if p.material == material], frame, pbr_voxel, resolution, size)
        glb.set_pbr_textures(material, *to_gltf_images(maps))
    glb.save(os.path.join(out_dir, "textured.glb"))
    if image is not None:
        image.save(os.path.join(out_dir, "cond.png"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"input": lighting.baked_path, "resolution": resolution, **meta}, f, indent=2)


def set_block_checkpointing(dec, on):
    """Recompute every decoder block in the backward pass instead of storing its activations."""
    from torch.utils.checkpoint import checkpoint
    for res in dec.blocks:
        for block in res:
            if not hasattr(block, "_orig_forward"):
                block._orig_forward = block.forward
            if on:
                block.forward = (lambda f: lambda *a, **k: checkpoint(f, *a, use_reentrant=False, **k))(block._orig_forward)
            else:
                block.forward = block._orig_forward


def voxel_index(coords, pos, res):
    """Index of the decoded voxel containing each point (-1 if none). coords: [M, 4] (batch, x, y, z)."""
    c = coords[:, 1:].long()
    key = (c[:, 0] * res + c[:, 1]) * res + c[:, 2]
    order = torch.argsort(key)
    skey = key[order]
    q = ((pos + 0.5) * res).floor().long().clamp(0, res - 1)
    qk = (q[:, 0] * res + q[:, 1]) * res + q[:, 2]
    i = torch.searchsorted(skey, qk).clamp(max=len(skey) - 1)
    return torch.where(skey[i] == qk, order[i], torch.full_like(i, -1))


def select_inputs(dataset, only=None, lightings=None, max_items=None):
    """(item, lighting) pairs of the dataset; `only`: comma-separated item id prefixes, `lightings`: names."""
    items = [dataset[i] for i in range(len(dataset))[:max_items]]
    if only:
        items = [it for it in items if any(it.id.startswith(o) for o in only.split(","))]
    names = set(lightings.split(",")) if lightings else None
    return [(it, l) for it in items for l in it.lightings if names is None or l.name in names]


def elapsed(t0):
    return f"{time.time() - t0:.0f} s"


def material_maps(glb, material):
    """Base color (H, W, 3), roughness and metallic (H, W) of a PBR material of a Glb, float in [0, 1]."""
    pbr = glb.json["materials"][material].get("pbrMetallicRoughness", {})
    bc = np.asarray(glb.image(glb.material_base_color_image(material)).convert("RGB"), np.float32) / 255
    h, w = bc.shape[:2]
    mr = pbr.get("metallicRoughnessTexture")
    if mr is not None:
        img = glb.image(glb.json["textures"][mr["index"]]["source"]).convert("RGB").resize((w, h))
        img = np.asarray(img, np.float32) / 255
        rough, metal = img[..., 1] * pbr.get("roughnessFactor", 1.0), img[..., 2] * pbr.get("metallicFactor", 1.0)
    else:
        rough = np.full((h, w), pbr.get("roughnessFactor", 1.0), np.float32)
        metal = np.full((h, w), pbr.get("metallicFactor", 1.0), np.float32)
    return bc, rough, metal


def write_atlas_maps(baked_path, asset, maps, out_dir, meta):
    """Write per-atlas maps of an Asset ({atlas index: (base color (H, W, 3), roughness, metallic, covered mask)},
    float in [0, 1]) into the baked input's own materials (matched by their baked texture) and save
    textured.glb and meta.json in the prediction layout."""
    from unbake.baselines.common import dilate, to_u8
    from unbake.gltf_io import Glb
    key = lambda tex: hashlib.md5(np.ascontiguousarray((np.asarray(tex) * 255).round().astype(np.uint8)).tobytes()).hexdigest()
    by_texture = {key(asset.parts[idx[0]]["texture"]): a for a, (_, idx) in enumerate(asset.atlases.items())}
    glb = Glb(baked_path)
    for material in sorted({p.material for p in glb.world_primitives() if p.material is not None}):
        img = glb.material_base_color_image(material)
        if img is None:
            continue
        a = by_texture[key(np.asarray(glb.image(img).convert("RGB"), np.float32) / 255)]
        bc, rough, metal, mask = maps[a]
        fill = lambda x: dilate(to_u8(x), mask).astype(np.float32) / 255
        glb.set_pbr_textures(material, *to_gltf_images({"base_color": fill(bc), "roughness": fill(rough),
                                                         "metallic": fill(metal)}))
    os.makedirs(out_dir, exist_ok=True)
    glb.save(os.path.join(out_dir, "textured.glb"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"input": str(baked_path), **meta}, f, indent=2)
