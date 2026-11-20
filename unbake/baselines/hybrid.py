"""Hybrid baseline: TRELLIS.2 base color + SuperMat roughness / metallic, in the input's UV layout.

For each input, the TRELLIS.2 prediction (input GLB with new PBR textures per material) keeps its base
color, and its metallicRoughness texture is replaced by SuperMat's maps of the same atlas. Materials are
matched to SuperMat atlases through their baked input texture.

    python -m unbake.baselines.hybrid --trellis trellis_unguided_1024   # after unguided_trellis2 and supermat
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from unbake.baselines.common import dilate, load_parts
from unbake.gltf_io import Glb
from unbake.predictions import PREDICTIONS, PREDICTION_FILE


def baked_texture(glb, material):
    """Base color texture * factor in [0, 1], as common.base_color_texture."""
    pbr = glb.json["materials"][material].get("pbrMetallicRoughness", {})
    factor = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], np.float32)
    img = glb.material_base_color_image(material)
    if img is None:
        return factor[None, None]
    return np.asarray(glb.image(img).convert("RGB"), np.float32) / 255 * factor


def supermat_atlases(input_path, sm_dir):
    """[(baked texture, metallicRoughness PIL image)] in SuperMat's atlas order."""
    parts = load_parts(input_path)
    order = {}
    for p in parts:
        order.setdefault(p["tex_key"], p["texture"])
    meta = json.loads((sm_dir / "meta.json").read_text())
    assert len(meta["atlases"]) == len(order), f"{sm_dir}: atlas count differs"
    out = []
    for a, tex in enumerate(order.values()):
        cover = np.asarray(Image.open(sm_dir / f"tex{a}_mask.png")) > 127
        r = dilate(np.asarray(Image.open(sm_dir / f"tex{a}_roughness.png").convert("L")), cover)
        m = dilate(np.asarray(Image.open(sm_dir / f"tex{a}_metallic.png").convert("L")), cover)
        out.append((tex, Image.fromarray(np.stack([np.full_like(r, 255), r, m], -1), "RGB")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=PREDICTIONS)
    ap.add_argument("--trellis", default="trellis_unguided_1024")
    ap.add_argument("--supermat", default="supermat")
    ap.add_argument("--out", default=os.path.join(PREDICTIONS, "trellis1024_bc+supermat_rm"))
    args = ap.parse_args()
    root = Path(args.predictions)
    names = sorted(p.name for p in (root / args.trellis).iterdir() if (p / PREDICTION_FILE).exists())
    for name in names:
        sm_dir = root / args.supermat / name
        if not (sm_dir / PREDICTION_FILE).exists():
            print(f"skip {name}: no SuperMat prediction"); continue
        input_path = json.loads((sm_dir / "meta.json").read_text())["input"]
        inp, glb = Glb(input_path), Glb(str(root / args.trellis / name / PREDICTION_FILE))
        atlases = supermat_atlases(input_path, sm_dir)
        worst = 0.0
        for mi in range(len(glb.json.get("materials", []))):
            if inp.material_base_color_image(mi) is None:
                continue
            baked = baked_texture(inp, mi)
            diffs = [np.abs(baked - t).mean() if t.shape == baked.shape else np.inf for t, _ in atlases]
            k = int(np.argmin(diffs)); worst = max(worst, diffs[k])
            pbr = glb.json["materials"][mi]["pbrMetallicRoughness"]
            bc = glb.image(glb.json["textures"][pbr["baseColorTexture"]["index"]]["source"]).convert("RGB")
            mr = atlases[k][1]
            if mr.size != bc.size:
                mr = mr.resize(bc.size, Image.BILINEAR)
            glb.set_pbr_textures(mi, bc, mr)
        assert worst < 1e-3, f"{name}: no SuperMat atlas matches a material (diff {worst:.4f})"
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        glb.save(str(out / PREDICTION_FILE))
        (out / "meta.json").write_text(json.dumps({"method": "hybrid", "base_color": args.trellis,
                                                   "roughness_metallic": args.supermat}, indent=2))
    print(f"wrote {len(list(Path(args.out).iterdir()))} predictions to {args.out}")


if __name__ == "__main__":
    main()
