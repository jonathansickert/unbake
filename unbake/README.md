# UnBake: dataset, baselines and evaluation

Install the repo once, editable, into the environment that has the TRELLIS.2 dependencies:

```sh
pip install --no-deps --no-build-isolation -e .     # unbake + trellis2, from the repo root
```

Then every script runs from any directory, by path (`python unbake/evaluation/compare.py`) or as a
module (`python -m unbake.evaluation.compare`). Default paths (`data/`, `output/`, `env_maps/`)
always point into the repository (`unbake/paths.py`).

```
unbake/
  dataset.py            dataset access: items, their lighting conditions, baked assets, ground truth
  download_dataset.py   fetch the dataset from Hugging Face
  gltf_io.py            GLB reading / writing that keeps the input's structure (Glb)
  uv_bake.py            TRELLIS.2 space (TrellisFrame), per-voxel materials -> original UV layout
  reference.py          unlit render of a baked asset (conditioning image)
  predictions.py        prediction layout and method registry (see below)
  paths.py              default locations (data/, output/, env_maps/)
  guided_trellis2.py    UnBake on one asset: rebake-guided TRELLIS.2 sampling (docs/Method.md)
  guided_dataset.py     UnBake on a whole dataset
  prior2d.py            2D roughness / metallic prior: SuperMat / RGB<->X on N views, fused per voxel
  detail.py             detail recovery: high frequencies of the shading-normalized input into the base color
  hybrid_prior.py       base color of one prediction + 16-view SuperMat roughness / metallic fused per texel
  baselines/            one script per baseline; all write the prediction layout
    common.py           shared by the baselines: trimesh helpers, textured.glb writer
    unguided_trellis2.py, supermat.py, material_anything.py, rgbx.py
    hybrid.py           hybrid baseline: TRELLIS.2 base color + SuperMat roughness / metallic
    patches/            patch for the Material Anything repo
  evaluation/
    metrics.py          UV-space material metrics, image metrics
    render.py           relighting renders (TRELLIS.2 PBR renderer, env maps, held-out cameras)
    summary.py          shared by the evaluation scripts: arguments, summaries, printed tables
    evaluate_uv.py, evaluate_relit.py, compare.py, seeds.py, figure.py, uv_figure.py
    material_figure.py  material maps of every method in image space
    relit_glbs.py       relighting metrics and figure for explicit GLBs (e.g. the examples/ cauldron)
    part_metrics.py     SAM part segmentation of a baked asset, part-level roughness / metallic metrics
    part_metrics_dataset.py   the same over a dataset
  run_eval.sh           unguided TRELLIS.2 at 512 and 1024, both evaluations, tables
```

## Prediction layout

Every method writes one folder per dataset input (an item's asset baked under one lighting),
with the input's geometry and UV layout, so both evaluations score any method the same way:

```
output/predictions/<method>/<item id>__<lighting name>/textured.glb   (+ cond.png, meta.json, tex maps)
```

`predictions.py` maps the folders to display names (`METHOD_NAMES`); other folders are evaluated
under their folder name (e.g. extra seeds). The input-as-albedo reference (baked texture as base
color, roughness 0.5, metallic 0) is always included.

## Baselines

| Module | Method |
|---|---|
| `baselines.unguided_trellis2` | TRELLIS.2 material stage conditioned on the geometry and an unlit render of the baked asset, voxel materials baked into the original UV layout. `--resolution 512/1024`, `--seed`. |
| `baselines.supermat` | SuperMat: 6 orthographic renders of the baked asset, multi-view prediction, back-projection into the UV layout and SuperMat's UV refinement. |
| `baselines.material_anything` | Material Anything in its mode for textured meshes (36 views, UV refinement); needs `patches/material_anything_skip_controlnet.patch` in the MA repo. |
| `baselines.rgbx` | RGB↔X: per-view RGB→X predictions on the SuperMat views, back-projected into the UV layout, with their spread across views. |

## Evaluation

| Module | Purpose |
|---|---|
| `evaluation.evaluate_uv` | Material accuracy in UV space: surface-area-weighted MAE / PSNR of base color, roughness and metallic against `pbr.glb`. Predictions are matched to the ground truth face by face (mesh names, order and material numbering may differ; a different UV layout is an error). |
| `evaluation.evaluate_relit` | Relighting: prediction and ground truth rendered under the held-out env maps in `env_maps/eval/` (not used to create the dataset, scaled to the same mean luminance), 8 held-out views; LPIPS on the full frame, PSNR / SSIM / MAE on object pixels. Saves all renders. |
| `evaluation.compare` | Markdown table and the LaTeX tables, on the inputs every method has (`--methods` restricts the methods, `--label-suffix` changes the LaTeX labels). |
| `evaluation.seeds` | Mean ± std over runs of one method (e.g. seeds). |
| `evaluation.figure` | Qualitative figure from the renders of `evaluate_relit`. |
| `evaluation.uv_figure` | Material maps of one asset side by side, in the ground truth's UV layout. |

Both evaluations write one record per (item, lighting, method) and a summary to
`output/eval/metrics_uv.json` / `metrics_relit.json`; missing predictions are skipped and reported.
The renderer's SSAO samples are seeded per view, so relit renders and metrics are reproducible.

## Usage

```sh
unbake/run_eval.sh                                                   # TRELLIS.2 512 + 1024, evaluation, tables
python -m unbake.baselines.supermat --sm-dir ... --ckpt-dir ...      # 2D baselines -> output/predictions/<method>
python -m unbake.baselines.material_anything --ma-dir ... --ma-python ...
python -m unbake.baselines.rgbx --weights /tmp/rgb-to-x            # code: rgbx/ submodule
python -m unbake.evaluation.evaluate_uv && python -m unbake.evaluation.evaluate_relit
python -m unbake.evaluation.compare --all --latex output/tables.tex   # every method on all of its inputs
python -m unbake.evaluation.compare --label-suffix=-subset --latex output/tables_subset.tex   # 20 inputs of Material Anything
python -m unbake.evaluation.figure && python -m unbake.evaluation.uv_figure
```

Seeds: `python -m unbake.baselines.unguided_trellis2 --seed 0` writes `trellis_unguided_1024_s0`;
after the evaluation, `python -m unbake.evaluation.seeds "TRELLIS.2 (unguided, 1024)" trellis_unguided_1024_s0 ...`.

## Notes

- Resolution 1024 is the default of `unguided_trellis2`. At 512 the metallic channel collapses
  to 1 everywhere for some seeds.
- Normal maps are not used in the relighting evaluation (face normals for all methods).

## Environment

- `transformers` 4.x (tested 4.57.1). 5.x renamed DINOv3's layers.
- `opencv-python-headless` 4.x (tested 4.11.0.86).
- xformers 0.0.31 on Blackwell (sm_120) dispatches to its Hopper-only FlashAttention-3 kernel;
  `unguided_trellis2` turns it off.
- Build the CUDA extensions for your GPU: set `TORCH_CUDA_ARCH_LIST` to its compute capability
  (`nvidia-smi --query-gpu=compute_cap --format=csv`).
- Gated weights: `facebook/dinov3-vitl16-pretrain-lvd1689m` and `briaai/RMBG-2.0`.

## Speed

Per input after the first one (the first run after a GPU change spends about a minute on Triton autotuning):

| GPU | 512 | 1024 |
|---|---|---|
| A30 24 GB | ~22 s, 3.2 GB | ~55 s, 8.2 GB |
| RTX PRO 6000 Blackwell (MIG 1g.24gb) | ~30 s, 3.1 GB | ~50–60 s, 6.9 GB |
| RTX A5000 24 GB (stage-wise over the dataset) | ~6 s | ~18 s |
