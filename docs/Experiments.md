# UnBake: Experiments (2026-11-14)

All experiments run in this session, in order. The method is described in `Method.md`.

**Setup common to all experiments.**
- **Dataset:** 10 Objaverse-XL assets × 8 environment maps = 80 inputs (`data/`).
- **Example asset:** `examples/cauldron_baked.glb` / `cauldron_pbr.glb`, which is not part of the dataset.
- **UV metrics:** MAE and PSNR against the ground-truth maps, weighted by surface area (`unbake.evaluation.evaluate_uv`).
- **Relighting metrics:** 3 held-out environment maps × 8 views; LPIPS on the full frame, PSNR / SSIM / MAE on object pixels (`unbake.evaluation.evaluate_relit`).
- **Input as albedo:** the baked texture as base color, with constants roughness 0.5 and metallic 0.
- **Seed:** 42 for all TRELLIS.2 samples.

**Caveat.** Every guided result is on a single asset (the cauldron) with a single seed, unless marked as a
dataset result. A repeated rebake-only run with the same seed differed in the third decimal (0.1010 vs. 0.1013),
apparently from non-determinism in the sparse GPU ops, so differences at that level are noise.

---

## 1. RGB↔X installed and checked in image space (cauldron)

- **Setup:** RGB↔X (`rgbx/` submodule, weights `zheng95z/rgb-to-x`) on 6 unlit orthographic renders (512²) of the
  baked cauldron, compared with unlit renders of the ground-truth maps on object pixels.
- **Files:** script `output/rgbx/check_example.py`; the folder `output/rgbx/` no longer exists, so the numbers below
  are as reported at the time.

| channel | MAE | PSNR |
|---|---|---|
| albedo | 0.264 (0.189 after a scale fit per channel; best scale ≈ 0.6) | 9.9 dB |
| roughness | 0.221 | 10.9 dB |
| metallic | 0.349 | 7.6 dB |

- **Albedo:** about 1.6× too bright; the metal pot is predicted near white.
- **Roughness:** follows the texture pattern rather than the material.
- **Metallic:** a near-uniform ≈ 0.5, with no separation between metal and non-metal.

## 2. RGB↔X baseline on the dataset

- **Setup:** the `unbake.baselines.rgbx` baseline on all 80 inputs (6 SuperMat views, written back into the UV
  layout), about 40–58 s per input, then both evaluations.
- **Files:** `output/eval/`, `output/tables*.tex`. The prediction folder `output/predictions/rgbx/` no longer exists.

All 80 inputs:

| method | n | base color MAE | roughness MAE | metallic MAE | relit LPIPS | relit PSNR | relit SSIM | relit MAE |
|---|---|---|---|---|---|---|---|---|
| input as albedo | 80 | 0.151 | **0.204** | **0.238** | 0.070 | 14.42 | **0.750** | 0.165 |
| TRELLIS.2 unguided 512 | 80 | 0.123 | 0.349 | 0.349 | 0.087 | 14.49 | 0.620 | 0.155 |
| TRELLIS.2 unguided 1024 | 80 | **0.115** | 0.418 | 0.435 | 0.085 | 14.09 | 0.585 | 0.172 |
| SuperMat | 80 | 0.157 | 0.219 | 0.268 | **0.068** | **15.62** | 0.740 | **0.149** |
| Material Anything | 20 | 0.262 | 0.238 | 0.407 | 0.094 | 12.49 | 0.607 | 0.204 |
| RGB↔X | 80 | 0.321 | 0.218 | 0.443 | 0.085 | 11.46 | 0.620 | 0.242 |

On the 20 inputs every method has, RGB↔X scores base color 0.329, roughness 0.220, metallic 0.444 and relit PSNR
11.35.

- RGB↔X has the worst base color and relighting of all methods.
- Its roughness is about level with SuperMat. Used only for roughness and metallic, it is still worse than the
  constants (0.218 vs. 0.204 and 0.443 vs. 0.238).

Re-scoring the other methods reproduced their earlier numbers exactly.

## 3. Hybrid baseline: TRELLIS.2 base color + SuperMat roughness / metallic

- **Setup:** TRELLIS.2 (1024) predictions with their metallic-roughness texture replaced by SuperMat's maps of the
  same UV atlas. Materials are matched through the baked input texture, and all matched exactly.
- **Files:** `output/hybrid_test/`, with the script `make_hybrid.py`.

All 80 inputs:

| method | base color | roughness | metallic | relit LPIPS | relit PSNR | relit SSIM | relit MAE |
|---|---|---|---|---|---|---|---|
| input as albedo | 0.151 | **0.204** | **0.238** | 0.070 | 14.42 | **0.750** | 0.165 |
| TRELLIS.2 1024 | **0.115** | 0.418 | 0.435 | 0.085 | 14.09 | 0.585 | 0.172 |
| SuperMat | 0.157 | 0.219 | 0.268 | 0.068 | 15.62 | 0.740 | 0.149 |
| **hybrid** | **0.115** | 0.219 | 0.268 | **0.067** | **16.42** | 0.702 | **0.123** |

- The hybrid has the best relighting so far on LPIPS, PSNR and MAE: +2.3 dB over TRELLIS.2 and +0.8 dB over
  SuperMat.
- Its SSIM is below the input-as-albedo baseline and SuperMat, likely because TRELLIS.2's voxel resolution loses
  fine texture detail.
- TRELLIS.2's relighting was held back by its roughness and metallic, not its base color. The hybrid is the
  baseline the guided method has to beat.

## 4. Guided sampling, basic version: SH rebake loss only (cauldron, 512³)

- **Setup:** rebake loss with order-2 SH and no visibility. Guidance at steps with t ≤ 0.8, 3 normalized gradient
  steps per guided step, lr 0.05.
- **Files:** `output/guided/cauldron/`.

The first run had no effect: the gradient underflowed to zero in the decoder's fp16 layers. Scaling the loss by
1e4 before backpropagation fixed it. Separately, flex_gemm's sparse-convolution backward crashes when the decoder
weights are frozen, so they keep `requires_grad=True`.

| | base color MAE | base color MAE after scale fit | rebake loss |
|---|---|---|---|
| input as albedo | 0.1088 | 0.1050 | – |
| unguided | **0.0951** | **0.0947** | 0.00245 |
| rebake | 0.1010 | 0.1027 | **0.00087** |

The rebake fit improves almost 3×, but the base color gets worse. The error after fitting the scale also rises,
towards the input's: without visibility, baked shadows go into the albedo.

## 5. 2D prior from RGB↔X: albedo + roughness, one combined gradient (cauldron, 512³)

- **Setup:** 16 Fibonacci views, RGB↔X albedo and roughness, cos²-weighted fusion with confidence weights. 70% of
  observed voxels are seen in at least 2 views; mean weights 0.36 / 0.37. Total loss rebake + β · L2D with one
  combined, normalized gradient.
- **Files:** `output/guided/cauldron_rgbx/`.

| | base color | roughness | metallic | rebake loss |
|---|---|---|---|---|
| unguided | **0.0951** | 0.293 | 0.434 | 0.00245 |
| rebake | 0.1011 | 0.292 | 0.431 | 0.00088 |
| + RGB↔X, β = 0.01 | 0.1094 | 0.287 | 0.428 | **0.00086** |
| + RGB↔X, β = 0.1 | 0.1434 | **0.228** | 0.431 | 0.00198 |
| input as albedo | 0.1088 | **0.147** | 0.566 | – |

- The albedo prior pulls the base color towards RGB↔X's too-bright albedo.
- Roughness improves, but stays far from the constant 0.5.

## 6. 2D prior on roughness + metallic only, one combined gradient (cauldron, 512³)

- **Setup:** as experiment 5, with RGB↔X roughness and metallic and no albedo term.
- **Files:** `output/guided/cauldron_rgbx_rm/`.

| | base color | roughness | metallic | rebake loss |
|---|---|---|---|---|
| unguided | 0.0951 | 0.293 | 0.434 | 0.00245 |
| rebake | 0.1010 | 0.292 | 0.431 | **0.00087** |
| + RGB↔X, β = 0.01 | 0.1017 | 0.294 | **0.261** | 0.00088 |
| + RGB↔X, β = 0.1 | **0.0950** | 0.233 | 0.349 | 0.00186 |
| + RGB↔X, β = 1 | **0.0950** | **0.201** | 0.347 | 0.00237 |

- Metallic and roughness improve.
- At β ≥ 0.1 the rebake loss climbs back to the unguided level (0.00237), so the base color equals the unguided
  result. The 2D gradient (about 100× larger) dominates the combined, normalized step and undoes the rebake
  correction through the shared latent.

## 7. Gradients normalized separately (cauldron, 512³)

- **Setup:** each loss term's gradient normalized to unit RMS, combined as ĝ_rebake + β ĝ_2D, and rescaled to step
  size lr. Raw gradient RMS: rebake 7.8e-4, L2D 7.4e-2.
- **Files:** `output/guided/cauldron_rgbx_rm_sepnorm/`.

| | base color | roughness | metallic | rebake loss |
|---|---|---|---|---|
| unguided | **0.0951** | 0.293 | 0.434 | 0.00245 |
| rebake | 0.1013 | 0.292 | 0.432 | 0.00088 |
| + RGB↔X, β = 0.25 | 0.1011 | 0.233 | 0.394 | **0.00086** |
| + RGB↔X, β = 0.5 | 0.1014 | 0.229 | 0.334 | **0.00086** |
| + RGB↔X, β = 1 | 0.1003 | 0.218 | **0.314** | 0.00087 |
| + RGB↔X, β = 2 | 0.0995 | **0.206** | 0.337 | 0.00093 |

The rebake fit is now kept at every β (0.00086–0.00093), while the prior improves roughness and metallic. β = 1 is
used as the default from here on.

## 8. Choice of 2D prior model (cauldron, 512³, separately normalized gradients)

- **Setup:** three sources with the same fusion and loss: SuperMat multi-view on its 6 axis-aligned views
  (`--prior supermat`), SuperMat single-view on the 16 Fibonacci views (`--prior supermat_sv`), and RGB↔X on 16
  views (experiment 7).
- **Files:** `output/guided/cauldron_supermat_rm/`, `output/guided/cauldron_supermat_sv16/`.

| prior | voxels seen in ≥ 2 views | mean weight | β | base color | roughness | metallic |
|---|---|---|---|---|---|---|
| SuperMat multi-view, 6 views | 35% | 0.18 | 0.25 | 0.1006 | 0.197 | 0.424 |
| | | | 0.5 | 0.1009 | **0.143** | 0.460 |
| | | | 1 | 0.0997 | 0.163 | 0.550 |
| | | | 2 | 0.0992 | 0.175 | 0.554 |
| SuperMat single-view, 16 views | 70% | 0.37 | 0.25 | 0.1010 | 0.230 | 0.378 |
| | | | 0.5 | 0.1013 | 0.215 | 0.294 |
| | | | 1 | 0.0999 | 0.188 | **0.275** |
| | | | 2 | 0.0994 | 0.179 | 0.291 |
| RGB↔X, 16 views | 70% | 0.37 | 1 | 0.1003 | 0.218 | 0.314 |
| *unguided* | | | | *0.0951* | *0.293* | *0.434* |
| *input as albedo* | | | | *0.1088* | *0.147* | *0.566* |

- **Multi-view SuperMat:** the best roughness, and the first result to beat the constant (0.143 vs. 0.147), but it
  predicts the metal pot as non-metal and covers half as many voxels.
- **Single-view SuperMat on 16 views:** the best metallic and the best all-round prior. Selected.

## 9. Guided sampling at 1024³ (cauldron)

- **Setup:** rebake + SuperMat single-view 16 views, β = 1.
- **Memory:** the first attempt ran out of memory, since the GPU is a 24 GB MIG slice, not the full card. Gradient
  checkpointing in the decoder fixed it.
- **Files:** `output/guided/cauldron_supermat_sv16_1024/`.

| | base color | roughness | metallic | rebake loss | time |
|---|---|---|---|---|---|
| unguided | **0.0967** | 0.335 | **0.302** | 0.00179 | 31 s |
| rebake | 0.1032 | 0.328 | 0.303 | **0.00064** | 246 s (first run) |
| + SuperMat single-view, β = 1 | 0.1005 | **0.229** | 0.356 | 0.00065 | 146 s |

- Only 13% of decoded voxels have an observation at 1024³ (39% at 512³), because the 1024² textures are too sparse
  for the grid.
- Metallic gets worse with the prior here, unlike at 512³.

## 10. Dataset run at 1024³ (rebake without visibility + SuperMat single-view 16 views, β = 1)

- **Setup:** `unbake.guided_dataset`, staged so each model loads once: conditioning, then 80 SuperMat priors (about
  20 s each), then guided sampling (about 125–175 s each).
- **Run history:** a first run at 512³ was stopped for 1024³. The 1024³ run was interrupted after 45 of 80 inputs
  by a container restart and resumed; it skips finished inputs and finished priors.
- **Files:** predictions in `output/predictions/guided_supermat_sv_b1_1024/`, evaluation in `output/guided_eval/`.

The evaluation could not include SuperMat and TRELLIS.2 512: `output/predictions/supermat`, `trellis_unguided_512`,
`material_anything` and `rgbx` are no longer present. Their earlier scores are in experiments 2 and 3.

All 80 inputs:

| method | base color | roughness | metallic | relit LPIPS | relit PSNR | relit SSIM | relit MAE |
|---|---|---|---|---|---|---|---|
| input as albedo | 0.151 | **0.204** | **0.238** | 0.070 | 14.42 | **0.750** | 0.165 |
| TRELLIS.2 unguided 1024 | **0.115** | 0.418 | 0.435 | 0.085 | 14.09 | 0.585 | 0.172 |
| guided (no visibility, 1× sampling) | 0.117 | 0.302 | 0.316 | 0.077 | 14.97 | 0.681 | 0.150 |
| hybrid (experiment 3) | **0.115** | 0.219 | 0.268 | **0.067** | **16.42** | 0.702 | **0.123** |

- Against unguided TRELLIS.2, the guided method improves roughness, metallic and every relighting metric
  (+0.9 dB PSNR, +0.10 SSIM).
- Base color is unchanged (0.117 vs. 0.115).
- It loses to the hybrid on every metric.

Likely causes, both addressed in experiment 11:
- no visibility in the rebake loss;
- only about 13% of decoded voxels receive guidance at 1024³.

## 11. Visibility × observation density (cauldron, 1024³)

- **Setup:** visibility off/on × texture sampling 1×/2×/4×. Each combination runs rebake only and rebake +
  SuperMat single-view 16 views (β = 1). The prior is re-fused per voxel set from the saved per-view predictions.
- **Visibility:** 256 directions, 2048² shadow maps.
  - On unoccluded voxels the Monte Carlo transfer matches the analytic one to 0.2%.
  - Mean unoccluded hemisphere is 0.44–0.51. Normals point outward (flipped: 0.19), and a 4× normal offset barely
    changes it (0.45), so this is not self-shadowing.
  - Visibility correlates with baked luminance (r = 0.38).
- **Files:** `output/guided/cauldron_vis_ss_1024/`.

| visibility | sampling | observed voxels | decoded voxels with observation | guidance | base color | roughness | metallic |
|---|---|---|---|---|---|---|---|
| – | – | – | – | unguided | 0.0967 | 0.335 | **0.302** |
| off | 1× | 0.70 M | 13% | rebake | 0.1030 | 0.328 | 0.303 |
| off | 1× | | | + SuperMat | 0.1012 | 0.229 | 0.359 |
| off | 2× | 2.08 M | 39% | rebake | 0.1037 | 0.328 | 0.303 |
| off | 2× | | | + SuperMat | 0.1015 | 0.198 | 0.387 |
| off | 4× | 3.89 M | 74% | rebake | 0.1040 | 0.331 | 0.302 |
| off | 4× | | | + SuperMat | 0.1023 | 0.157 | 0.384 |
| on | 1× | 0.70 M | 13% | rebake | 0.0893 | 0.332 | 0.303 |
| on | 1× | | | + SuperMat | 0.0887 | 0.228 | 0.356 |
| on | 2× | 2.08 M | 39% | rebake | 0.0885 | 0.333 | 0.303 |
| on | 2× | | | + SuperMat | **0.0882** | 0.202 | 0.389 |
| on | 4× | 3.89 M | 74% | rebake | 0.0885 | 0.333 | 0.303 |
| on | **4×** | | | **+ SuperMat** | **0.0882** | **0.156** | 0.386 |
| | | | | input as albedo | 0.1088 | 0.147 | 0.566 |

- **Visibility:** base color MAE drops from about 0.103 to about 0.088, 8–9% better than unguided (0.0967), with
  PSNR 16.7–16.9 dB vs. 16.5 dB. Without visibility, the rebake loss made base color worse in every combination.
- **Denser sampling:** little effect on base color. Through the prior, it strongly improves roughness
  (0.229 → 0.156, close to the constant 0.147). Time per run is unchanged (about 145 s); visibility adds 1–2 s.
- **Metallic:** worse with the prior at every setting at 1024³ (0.30 → 0.36–0.39).

The new default is visibility on, 4× sampling, SuperMat single-view 16 views, β = 1.

## 12. Smooth vs. face normals in the rebake loss (cauldron, 1024³)

- **Setup:** visibility on, 4× sampling, SuperMat single-view 16 views, β = 1. The observations (and the visibility
  and the view weighting of the prior) use either flat triangle normals or the GLB's smooth vertex normals
  (`NORMAL`, interpolated), which Blender used to shade the bake.
- **Motivation:** the relighting renderer and the rebake loss used flat triangle normals, so the shading model
  could not reproduce the smooth baked shading.
- **Files:** `output/guided/cauldron_normals_1024/`, with the image-space material figure `materials.pdf`
  (`unbake/evaluation/material_figure.py`).

UV-space MAE:

| guidance | normals | base color | roughness | metallic |
|---|---|---|---|---|
| unguided | – | 0.0967 | 0.335 | **0.302** |
| rebake | face | 0.0885 | 0.333 | 0.303 |
| rebake | smooth | **0.0879** | 0.333 | 0.303 |
| + SuperMat | face | 0.0881 | **0.155** | 0.386 |
| + SuperMat | smooth | 0.0890 | 0.156 | 0.388 |
| input as albedo | | 0.1088 | 0.147 | 0.566 |

- No measurable difference: every change is within the run-to-run noise. Each voxel's normal is already averaged
  over the surface it covers, and visibility matters far more than the exact normal.
- Smooth normals are kept as the default because they match how the bake was shaded.
- In image space, the guided base color beats both unguided and input as albedo (0.094–0.098 vs. 0.107 and
  0.104). With the prior, roughness reaches about the constant (0.127–0.131 vs. 0.122).
- Metallic stays at about 0.447 in image space for every method: the ground truth's whole pot body is metal, and
  every method predicts it as non-metal.

## 13. Dataset run with the current method (1024³)

- **Setup:** `unbake.guided_dataset` with the defaults from `docs/Method.md`: visibility on, 4× sampling, smooth
  normals, SuperMat single-view 16 views, β = 1, λ_r = λ_m = 1. The priors are fused in memory from the per-view
  SuperMat predictions saved by experiment 10, so SuperMat was not run again.
- **Run time:** 2h17 for 80 inputs, 26–180 s per input depending on the asset.
- **Files:** predictions in `output/predictions/guided_supermat_sv_b1_1024_vis_ss4/`, evaluation in
  `output/guided_eval/` (`compare.md`, `tables.tex`).
- **Run history:** the first attempts failed because the home directory's disk quota filled up while writing
  per-input prior files (up to 90 MB each at 4×). A failed write also truncated `unbake/guided_dataset.py`, which
  was rebuilt. The driver now writes only the predictions.

All 80 inputs:

| method | base color MAE | base color PSNR | roughness MAE | metallic MAE | relit LPIPS ↓ | relit PSNR ↑ | relit SSIM ↑ | relit MAE ↓ |
|---|---|---|---|---|---|---|---|---|
| input as albedo | 0.151 | 13.30 | **0.204** | **0.238** | 0.070 | 14.42 | **0.750** | 0.165 |
| TRELLIS.2 unguided 1024 | 0.115 | 15.87 | 0.418 | 0.435 | 0.085 | 14.09 | 0.585 | 0.172 |
| guided, experiment 10 (no visibility, 1× sampling) | 0.117 | 15.42 | 0.302 | 0.316 | 0.077 | 14.97 | 0.681 | 0.150 |
| **guided, current method** (visibility, 4×, smooth normals) | **0.114** | **15.99** | 0.279 | 0.321 | 0.071 | 15.55 | 0.700 | 0.141 |
| hybrid (experiment 3) | 0.115 | 15.87 | 0.219 | 0.268 | **0.067** | **16.42** | 0.702 | **0.123** |

- **Against experiment 10:** visibility and 4× sampling improve every metric except metallic.
  - Base color 0.117 → 0.114, PSNR 15.42 → 15.99.
  - Roughness 0.302 → 0.279.
  - Relit PSNR 14.97 → 15.55, LPIPS 0.077 → 0.071, SSIM 0.681 → 0.700.

  Metallic stays the same (0.316 → 0.321).
- **Against unguided TRELLIS.2:** better or equal everywhere. Base color is now slightly better (0.115 → 0.114,
  +0.12 dB), but the gain is much smaller than on the cauldron (8%), so visibility helps that asset more than the
  average one.
- **Against the hybrid:** its base color is as good as the hybrid's, but it trails on roughness (0.279 vs. 0.219),
  metallic (0.321 vs. 0.268) and relighting (−0.9 dB PSNR). Steering the latent towards SuperMat's roughness and
  metallic gets only part of the way to using SuperMat's maps directly.

## 14. Ground-truth metallic statistics (dataset)

Surface-area-weighted, ground-truth maps of the 10 assets:

- **Metal area:** 21.6% of the surface overall (m > 0.5), very unevenly spread. Five assets are almost metal-free
  (0–5%), and five have a lot (19–61%; the cauldron 61%).
- **Binary values:** 76% of the surface is below 0.1 or above 0.9. Two assets (796f, a7da) have many in-between
  artist values.
- **MAE of simple predictions:** constant 0: 0.238; constant 0.5: 0.409; the ground truth thresholded at 0.5 (a
  perfect metal / non-metal decision): 0.091.

Metallic is mainly a metal / non-metal decision. Grey predictions are never optimal under MAE when the true value
is 0 or 1; the more likely of 0 and 1 is. Getting the decision right per part would cut the error by about 60%.

## 15. The current method on a specular bake (cauldron, 1024³)

- **Data:** `examples/cauldron_specular.glb`, baked with `data_toolkit/bake_multiview.py`: 32 path-traced views
  (128 samples) blended per texel with weights cos⁴(view, normal), monkstown_castle at 267.5°. Texels no camera saw
  (15.4%) fall back to the diffuse bake. The specular bake changes the texture by 66% of the diffuse brightness
  (linear color); the metal parts (61% of the surface) are 0.59× as bright as in the diffuse bake, the non-metal parts
  unchanged (0.99×).
- **Setup:** current method (visibility, 4× sampling, smooth normals, SuperMat 16 views, β = 1).
- **Files:** `output/guided/cauldron_specular_1024/`.

| | base color MAE | roughness MAE | metallic MAE |
|---|---|---|---|
| input as albedo | 0.1072 | 0.147 | 0.566 |
| unguided | **0.0953** | 0.267 | **0.344** |
| rebake (diffuse model) | 0.1098 | 0.270 | 0.345 |
| + SuperMat (roughness + metallic) | 0.1107 | **0.141** | 0.425 |

- **Base color:** the diffuse rebake model now makes it worse (on the diffuse bake it made it better). The metal parts
  follow a different image formation, which biases the shared lighting fit; the error grows most on the non-metal
  parts (0.147 → 0.179).
- **SuperMat on specular renders:** roughness beats the constant for the first time (0.141 vs. 0.147), but metallic gets
  much worse: SuperMat reads the dark, reflective metal as non-metal (predicted metallic on metal parts 0.41 → 0.24).

## 16. Specular rebake term (Cook-Torrance GGX) on the specular cauldron

- **Model** (`unbake/specular.py`, `--specular 1`): metallic-roughness Cook-Torrance (GGX, Smith, Schlick split into an
  F0 part and a white part) under SH lighting, the viewing direction averaged over p(ω_o) ∝ cos⁴ (the bake's view
  blending), per-band lobe gains (Funk–Hecke) tabulated over roughness; diffuse transfer with visibility, scalar
  specular occlusion; clipped texels dropped. Lighting still linear, fitted by least squares on the detached materials.
- **Setup:** order-4 SH, view-blended, SuperMat prior on roughness only (λ_m = 0), β = 1.
- **Files:** `output/guided/cauldron_specular_model_1024/` (with `relit/` from `unbake/evaluation/relit_glbs.py`),
  `output/guided/cauldron_specular_sh8/`.

| rebake model | guidance | base color | roughness | metallic | relit LPIPS ↓ | relit PSNR ↑ | relit SSIM ↑ |
|---|---|---|---|---|---|---|---|
| – | unguided | **0.0953** | 0.267 | 0.344 | 0.101 | **15.98** | 0.697 |
| diffuse | rebake | 0.1099 | 0.270 | 0.345 | 0.124 | 13.80 | 0.678 |
| diffuse | + SuperMat roughness | 0.1108 | **0.136** | **0.340** | 0.104 | 15.12 | 0.728 |
| specular | rebake | 0.1004 | 0.269 | 0.345 | 0.115 | 14.82 | 0.672 |
| specular | + SuperMat roughness | 0.1000 | 0.137 | 0.341 | 0.100 | 15.90 | 0.714 |
| | input as albedo | 0.1072 | 0.147 | 0.566 | **0.081** | 15.94 | **0.731** |

- The specular term removes about two-thirds of the base-color damage (0.1099 → 0.1004) and gains +1.0 dB relit PSNR
  alone, +0.8 dB with the roughness prior. On the metal parts base color now beats unguided (0.059 vs. 0.062).
- Roughness and metallic are not moved by the rebake term itself (0.269 / 0.345, as unguided); only the roughness prior
  improves roughness.
- Order-8 SH gives the same results as order 4 (base color 0.0999, roughness 0.139, metallic 0.340).

## 17. Ground-truth model check of the specular term

`unbake/evaluation/model_check.py`: the rebake loss (×1e-4) with the ground-truth materials against perturbations. The
term can only guide roughness / metallic correctly if the ground truth has the lowest loss.

| setting (specular cauldron, view-blended) | GT | metal parts → non-metal | non-metal → metal | r −0.2 | r +0.2 |
|---|---|---|---|---|---|
| order-4 SH, fitted lighting | 12.21 | **11.16** | 14.61 | 12.24 | 12.84 |
| order-8 SH, fitted lighting | 12.04 | **10.97** | 14.61 | 12.07 | 12.66 |
| 64 spherical Gaussians, fitted lighting | 12.75 | **11.92** | 14.70 | **12.61** | 13.51 |
| order-8 SH + energy compensation | 11.83 | **11.03** | 14.75 | 12.06 | **11.67** |
| order-4 SH, fallback texels excluded | 13.68 | **12.47** | 16.40 | 13.72 | 14.37 |
| order-4 SH, true lighting (env map, fitted scale) | 13.05 | **12.78** | 14.33 | 13.24 | 13.61 |
| order-4 SH, true lighting, emissive voxels excluded | 7.39 | **6.71** | 7.29 | 7.25 | 8.08 |

- No setting passes: the metal parts always score better as non-metal.
- **More detailed lighting** (higher SH order) fits slightly better but does not change the ranking. **Spherical
  Gaussians** fit worse and also rank roughness wrong; **energy compensation** (multiple-scattering GGX) pushes
  roughness too high. Both were removed from the code.
- **Fallback texels:** the bake's camera rig was replicated (`unbake/bake_views.py`): 14.9% unseen texels vs. the
  reported 15.4%. Excluding them does not remove the bias.
- **True lighting:** the environment map projected into SH with the bake's rotation fits the matching diffuse bake as
  well as fitted lighting (19.6 vs. 19.4, with a constant per-channel scale of 0.5); on the specular bake the metal bias
  shrinks from ~9% to ~2%, so fitted lighting exaggerates it.
- **Emission is the main cause:** `cauldron_pbr.glb` has an emissive texture on 9.4% of the surface (fire, liquid).
  The path-traced bake contains their glow and the warm light they cast on the base and the pot. A distant-lighting
  model cannot explain either; excluding the emissive voxels halves the unexplained loss (13.1 → 7.4), but their light
  on the neighbours still breaks the ranking. Since real scans contain emission too, the model has to handle it
  (emitter detection and local lights; not implemented).

## 18. Robust lighting fit (cauldron, 1024³)

- **Setup:** lighting fitted by iteratively reweighted least squares (Geman–McClure, 4 iterations, scale from the
  median absolute deviation), optionally also weighting the rebake loss (`--robust fit|fit+loss`).
- **Files:** `output/guided/robust_*`.

| base color MAE | no robustness | robust fit | robust fit + loss |
|---|---|---|---|
| diffuse cauldron, current method | **0.0890** | 0.0927 | 0.0988 |
| specular cauldron, specular model + roughness prior | **0.0998** | 0.1018 | 0.1039 |

The robust fit makes base color worse in every case (roughness and metallic unchanged) and does not fix the metallic
ranking of experiment 17 either. Down-weighting the worst-explained voxels also removes the ones that constrain the
lighting most (e.g. shadowed regions). It stays off.

## 19. Detail recovery (cauldron, 1024³)

- **Method** (`unbake/detail.py`, `docs/Method.md` Sec. 8.1): per texel of the baked texture, order-2 SH transfer
  with visibility, robust lighting fit against the prediction's base color, shading-normalized albedo D = o / shading;
  only D's high frequencies are transferred: c = c_pred · clamp(D / blur_σ(D), 0.5, 2), blur restricted to the UV
  islands. About 5 s per asset.
- **Input:** the current method's prediction on the diffuse cauldron (experiment 12, smooth normals).
- **Files:** `output/guided/detail_cauldron/` (before / after figures in `before_after/`).

| | base color MAE | relit LPIPS ↓ | relit PSNR ↑ | relit SSIM ↑ | relit MAE ↓ |
|---|---|---|---|---|---|
| input as albedo | 0.1088 | **0.085** | 14.34 | 0.727 | 0.151 |
| guided, before | 0.0890 | 0.101 | **15.19** | 0.718 | 0.124 |
| + detail, σ = 2 | 0.0891 | 0.087 | 15.15 | **0.748** | **0.123** |
| + detail, σ = 4 | 0.0898 | 0.089 | 15.07 | 0.739 | 0.123 |
| + detail, σ = 8 | 0.0918 | 0.093 | 14.90 | 0.731 | 0.125 |
| + detail, σ = 4, factor clamped to 1.5 | **0.0885** | 0.089 | 15.13 | 0.740 | 0.123 |
| full division (D, no high-pass) | 0.1577 | 0.138 | 12.50 | 0.713 | 0.165 |

- σ = 2 is best: relit SSIM above input as albedo for the first time (0.748 vs. 0.727), LPIPS almost level (0.087 vs.
  0.085), base color MAE and relit PSNR unchanged. The relief, brick edges and wood grain are back in the renders.
- The full division is much worse: D's low frequencies still contain the shading the model does not explain
  (emitter light, interreflections), so only the high-frequency transfer works.

## 20. Dataset: detail recovery, smooth-normal relighting (1024³)

- **Setup:** detail recovery (σ = 2) applied to the 80 predictions of experiment 13 and, for a fair comparison, to the
  hybrid of experiment 3 (its base color is unguided TRELLIS.2, equally blurred). Relighting renders now shade with the
  GLBs' smooth vertex normals instead of the renderer's flat face normals (`render_mesh(..., smooth_normals=True)`).
- **Files:** predictions in `output/predictions/{guided_supermat_sv_b1_1024_vis_ss4_detail, hybrid_detail}/`,
  evaluation in `output/guided_eval/eval_smooth/`, table in `output/guided_eval/compare_smooth.md` and `tables_smooth.tex`.

All 80 inputs:

| method | base color MAE | base color PSNR | roughness MAE | metallic MAE | relit LPIPS ↓ | relit PSNR ↑ | relit SSIM ↑ | relit MAE ↓ |
|---|---|---|---|---|---|---|---|---|
| input as albedo | 0.151 | 13.30 | **0.204** | **0.238** | 0.070 | 14.44 | **0.749** | 0.164 |
| TRELLIS.2 unguided 1024 | 0.115 | 15.87 | 0.418 | 0.435 | 0.085 | 14.08 | 0.585 | 0.172 |
| guided, experiment 10 (no visibility) | 0.117 | 15.42 | 0.302 | 0.316 | 0.077 | 14.96 | 0.680 | 0.150 |
| guided, current method (experiment 13) | 0.114 | 15.99 | 0.279 | 0.321 | 0.071 | 15.53 | 0.700 | 0.141 |
| **guided + detail recovery** | **0.108** | **16.24** | 0.279 | 0.321 | 0.070 | 15.56 | 0.716 | 0.141 |
| hybrid | 0.115 | 15.87 | 0.219 | 0.268 | 0.067 | 16.42 | 0.702 | 0.123 |
| **hybrid + detail recovery** | 0.109 | 16.18 | 0.219 | 0.268 | **0.066** | **16.51** | 0.730 | **0.121** |

- **Detail recovery improves base color, not only relighting:** guided 0.114 → 0.108 (+0.25 dB), hybrid 0.115 → 0.109.
  Guided + detail has the best base color of all methods: 6% better than unguided TRELLIS.2, 28% better than input as
  albedo. Relit SSIM rises by 0.016 (guided) and 0.028 (hybrid); LPIPS improves slightly; PSNR is unchanged.
- **Relighting is still led by the hybrid** (with detail: LPIPS 0.066, PSNR 16.51 dB, SSIM 0.730). Its advantage is now
  entirely roughness and metallic (0.219 / 0.268 vs. 0.279 / 0.321). Input as albedo keeps the best SSIM (0.749).
- **Smooth normals** change the dataset numbers by at most ±0.02 dB for every method; earlier conclusions stand.

## 21. SAM parts: part-level roughness / metallic (cauldron, then dataset)

- **Method** (`unbake/evaluation/part_metrics.py`, dataset driver `part_metrics_dataset.py`): SAM ViT-H automatic
  masks on 16 unlit Fibonacci renders (1024²) of the baked input; each pixel takes the smallest covering mask. Texels
  of every UV atlas are projected into the views (depth test, cos > 0.1) and pick up the mask id. Neighbouring texels
  are joined when the views seeing both agree more often than not; components are merged across seams / materials by
  the same vote on their majority mask; small and unseen pieces go to the nearest part. Metrics: surface-area-weighted
  MAE per texel, and after replacing every texel by its part's weighted median ("part-constant"). Oracle: the ground
  truth itself made part-constant. About 2 min per input (SAM ≈ 7 s per view).
- **Also scored:** "SuperMat 16 views, fused", the guidance target itself (the saved per-view SuperMat single-view
  predictions of experiment 10, fused with the prior's cos² weighting directly into texels instead of voxels).
  SuperMat's own baseline folder is a broken symlink, so "SuperMat / hybrid" is read from the hybrid predictions,
  whose roughness / metallic are SuperMat's maps unchanged.
- **Files:** `output/parts/cauldron/`, `output/parts/dataset/` (`part_metrics.md`, per input `report.json`,
  `sam_views.png`, `parts_sam.png`).

**Cauldron** (`examples/`, 25 parts, 14 cover 90% of the area; oracle roughness 0.015, metallic 0.026), roughness MAE:

| method | texel | SAM part median |
|---|---|---|
| constant 0.5 | 0.147 | 0.147 |
| unguided | 0.335 | 0.336 |
| rebake | 0.333 | 0.334 |
| guided (rebake + SuperMat) | 0.156 | **0.112** |
| SuperMat 16 views, fused | 0.166 | 0.148 |

**Dataset, first lighting of each of the 10 assets** (10 inputs; 21.7 parts on average, 11.6 cover 90% of the area;
84% of texels seen by a view):

| method | roughness texel | roughness SAM parts | metallic texel | metallic SAM parts |
|---|---|---|---|---|
| constant (r 0.5, m 0) | 0.204 | 0.204 | 0.238 | 0.238 |
| TRELLIS.2 unguided 1024 | 0.416 | 0.419 | 0.448 | 0.452 |
| SuperMat / hybrid | 0.255 | 0.246 | 0.252 | 0.247 |
| guided (experiment 13) | 0.286 | 0.290 | 0.351 | 0.364 |
| **SuperMat 16 views, fused** | 0.190 | **0.177** | **0.217** | **0.217** |
| *oracle (ground truth part-constant)* | | *0.069* | | *0.043* |

- **The cauldron gain does not generalize.** Part medians help the guided result only on the cauldron (0.156 → 0.113)
  and 69669054 (0.320 → 0.288); on average it gets slightly worse (0.286 → 0.290). On the cauldron the ground truth is
  almost exactly constant per SAM part (oracle 0.015); on four assets the oracle is 0.10–0.14, i.e. SAM's parts mix
  roughness values.
- **The guidance target beats the constants dataset-wide**, the first prediction to do so: fused 16-view SuperMat
  roughness 0.190 (0.177 with parts) vs. 0.204, metallic 0.217 vs. 0.238. The guided method, steered towards exactly
  this target, ends at 0.286 / 0.351: guiding the latent loses most of the prior's information. It is also clearly
  better than SuperMat's own multi-view baseline (0.255 / 0.252).
- **"Ground-truth regions"** (connected regions of quantized ground truth) fragments into ~86k regions on the textured
  ground truths, so it is effectively per texel and not a useful reference; ignore that column in `part_metrics.md`.
- **Next:** guided base color (with detail) + fused 16-view SuperMat roughness / metallic written directly into the UV
  layout, with part medians for roughness as an option. Confirm on all 80 inputs.

## 22. Hybrid with the fused 2D prior: guided base color + fused SuperMat 16-view roughness / metallic (dataset)

- **Method** (`unbake/hybrid_prior.py`): base color from the guided predictions of experiment 13 (with and without
  detail recovery, experiment 20); roughness / metallic from the saved per-view SuperMat single-view predictions
  (16 Fibonacci views, experiment 10) fused per texel of every material with the prior's cos² weighting and depth test,
  unseen texels from the nearest seen texel. No SAM, no part medians. About 4 s per input.
- **Files:** predictions in `output/predictions/{guided+sm16_rm, guided_detail+sm16_rm}/`, evaluation in
  `output/hybrid_prior_eval/eval/` (smooth-normal relighting as in experiment 20; input as albedo reproduces
  experiment 20 exactly).

All 80 inputs (rows other than the two new ones from experiment 20):

| method | base color MAE | roughness MAE | metallic MAE | relit LPIPS ↓ | relit PSNR ↑ | relit SSIM ↑ | relit MAE ↓ |
|---|---|---|---|---|---|---|---|
| input as albedo | 0.151 | 0.204 | 0.238 | 0.070 | 14.44 | 0.749 | 0.164 |
| guided + detail (experiment 20) | 0.108 | 0.279 | 0.321 | 0.070 | 15.56 | 0.716 | 0.141 |
| hybrid + detail (experiment 20) | 0.109 | 0.219 | 0.268 | 0.066 | 16.51 | 0.730 | 0.121 |
| guided base color + fused r/m | 0.114 | **0.186** | **0.234** | 0.064 | 16.75 | 0.750 | 0.125 |
| **guided base color + detail + fused r/m** | **0.108** | **0.186** | **0.234** | **0.062** | **16.81** | **0.767** | 0.124 |

- **First method that is best or tied on every UV metric and on relit LPIPS, PSNR and SSIM**, including SSIM above
  input as albedo (0.767 vs. 0.749). Relit MAE is about level with the hybrid (0.124 vs. 0.121).
- **Roughness** beats the constant 0.5 dataset-wide (0.186 vs. 0.204; better on 50 of 80 inputs) and the hybrid's
  SuperMat maps (0.219; 49 of 80).
- **Metallic** only ties the constant 0 (0.234 vs. 0.238, better on 39 of 80). It helps on the metal-heavy assets
  (e.g. a7da 0.684 → 0.391, cauldron 0.566 → 0.443) and hurts on the metal-free ones, where 0 is nearly perfect
  (e.g. 6966 0.038 → 0.221, afd3 0.030 → 0.176): grey in-between values instead of a metal / non-metal decision
  (experiment 14).
- **Against hybrid + detail, per input:** relit PSNR +0.30 dB on average but better on only 42 of 80 inputs; roughness
  better on 49, metallic on 52.
- Guidance is thus useful for base color, while roughness / metallic are best taken from the fused 2D prior directly
  (experiment 21).

---

## Summary and open items

- **Base color:** guidance with visibility, plus detail recovery, gives the best base color of all methods on the
  dataset (0.108 vs. 0.115 for unguided TRELLIS.2 and 0.151 for input as albedo). Detail recovery also raises relit
  SSIM for every method it is applied to. Smooth vs. face normals makes no measurable difference, in the rebake loss
  or in the relighting evaluation.
- **Roughness:** the 2D prior helps a lot (dataset: 0.418 → 0.279), but guiding the latent towards SuperMat gets only
  part of the way: SuperMat's maps used directly are better (0.219), and the constant 0.5 is better still (0.204).
- **Metallic:** unsolved. Guidance improves it over TRELLIS.2 (0.435 → 0.321), but nothing beats the constant 0
  (0.238). It is a metal / non-metal decision (experiment 14).
- **Relighting:** led by the hybrid with detail recovery (16.51 dB); the guided method with detail reaches 15.56 dB,
  the gap being roughness and metallic. Obvious next step: guided base color with detail + SuperMat's roughness /
  metallic maps.
- **Specular bakes:** the specular rebake term fixes most of the base-color damage the diffuse model causes, but does
  not yet recover metallic: emission (glow and local light from emitters) cannot be explained by distant lighting, and
  real scans contain it too. Model-side fix proposed (emitter detection by an albedo ≤ 1 bound, emitters as local lights
  fitted jointly with the environment); not implemented. A robust lighting fit does not help (experiment 18).
- **Diffuse bake pipeline, suggested changes:** more samples or denoising (Monte Carlo noise is baked in and detail
  recovery transfers it), consistent use of the normal map in bake and evaluation, exposure headroom and HDR storage,
  emission in the bake for realism.
- **Missing files:** `output/predictions/{supermat, trellis_unguided_512, material_anything, rgbx}` and `output/rgbx/`
  were removed during the session, by something other than these experiments. They are needed to re-score those
  baselines; the hybrid predictions in `output/hybrid_test/` still exist.
- **Removed to free disk space** (all regenerable): two partial dataset runs without predictions, cached prior `.pt`
  files in `output/guided/*` and of experiment 10.
- **Disk quota:** the home directory is at its limit (about 72 GB, mostly `~/public`). Runs must not write large
  intermediates.
- **Weights in `/tmp`** (RGB↔X, SuperMat checkpoints, Hugging Face cache) are wiped by container restarts and must be
  re-downloaded.
