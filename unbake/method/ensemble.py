"""Combine the material maps of several result folders per asset (e.g. a seed ensemble).

    python method/ensemble.py --dirs results/s0 results/s1 results/s2 --out results/ens [--const-rm]

Base color (and roughness / metallic unless --const-rm) is the mean over --dirs. --const-rm sets roughness /
metallic to the constants of the input-as-albedo reference (0.5 / 0). Writes the baseline output format
including textured.glb, so evaluate_uv.py and evaluate_relit.py apply.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))

import numpy as np
import trimesh
from PIL import Image

from unguided_trellis2 import load_parts, dilate


def write_glb(meta, out):
    parts = load_parts(meta["input"])
    mats = {}
    for at in meta["atlases"]:
        a = at["index"]
        cover = np.asarray(Image.open(out / f"tex{a}_mask.png")) > 127
        bc = np.asarray(Image.open(out / f"tex{a}_basecolor.png").convert("RGB"))
        ro = np.asarray(Image.open(out / f"tex{a}_roughness.png").convert("L"))
        me = np.asarray(Image.open(out / f"tex{a}_metallic.png").convert("L"))
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(dilate(bc, cover)),
            metallicRoughnessTexture=Image.fromarray(np.stack([np.zeros_like(ro), dilate(ro, cover), dilate(me, cover)], -1)),
            metallicFactor=1.0, roughnessFactor=1.0, doubleSided=True)
        for n in at["parts"]:
            mats[n] = mat
    meshes = []
    for p in parts:
        mesh = p["mesh"].copy()
        mesh.visual = trimesh.visual.TextureVisuals(uv=p["uv"], material=mats[p["name"]])
        meshes.append(mesh)
    trimesh.Scene(meshes).export(out / "textured.glb")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--const-rm", action="store_true")
    ap.add_argument("--no-glb", action="store_true")
    args = ap.parse_args()
    dirs = [Path(d) for d in args.dirs]
    names = set.intersection(*[{p.parent.name for p in d.glob("*/meta.json")} for d in dirs])
    n = 0
    for name in sorted(names):
        src = [d / name for d in dirs]
        meta = json.loads((src[0] / "meta.json").read_text())
        chans = ["basecolor"] + ([] if args.const_rm else ["roughness", "metallic"])
        if not all((s / f"tex{a['index']}_{c}.png").exists() for s in src for a in meta["atlases"] for c in chans):
            continue
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        for at in meta["atlases"]:
            a = at["index"]
            bc = np.mean([np.asarray(Image.open(s / f"tex{a}_basecolor.png").convert("RGB"), np.float32) for s in src], 0)
            Image.fromarray(bc.round().astype(np.uint8)).save(out / f"tex{a}_basecolor.png")
            h, w = bc.shape[:2]
            for ch, const in [("roughness", 128), ("metallic", 0)]:
                if args.const_rm:
                    x = np.full((h, w), const, np.float32)
                else:
                    x = np.mean([np.asarray(Image.open(s / f"tex{a}_{ch}.png").convert("L"), np.float32) for s in src], 0)
                Image.fromarray(x.round().astype(np.uint8)).save(out / f"tex{a}_{ch}.png")
            shutil.copy(src[0] / f"tex{a}_mask.png", out / f"tex{a}_mask.png")
        meta.update({"method": "ensemble" + ("_const_rm" if args.const_rm else ""), "members": [str(s) for s in src]})
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        if not args.no_glb:
            write_glb(meta, out)
        n += 1
    print(f"{n} assets -> {args.out}")


if __name__ == "__main__":
    main()
