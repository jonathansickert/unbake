# UnBake: Experiments (session of 2026-11-14)

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

---

## Summary and open items

- **Base color:** guidance only helps with visibility in the rebake loss (cauldron: 0.0967 → 0.0882). This
  still has to be confirmed on the dataset. `unbake/guided_dataset.py` does not yet pass visibility and the 4×
  sampling through, and the 80 priors need re-fusing for the denser observations.
- **Roughness:** the 2D prior helps a lot, if enough voxels are observed (4× sampling) and the gradients are
  normalized separately. On the cauldron it reaches about the constant baseline.
- **Metallic:** inconclusive. It helped at 512³ on the cauldron and SuperMat's metallic is good dataset-wide, but
  it got worse at 1024³ on the cauldron. To be decided on the dataset, with λ_metallic = 0 as the fallback.
- **The baseline to beat** is the hybrid (relit PSNR 16.42 dB on the dataset); the guided method without
  visibility reached 14.97 dB.
- **Missing files:** `output/predictions/{supermat, trellis_unguided_512, material_anything, rgbx}` and `output/rgbx/`
  were removed during the session, by something other than these experiments. They are needed to re-score those
  baselines and to rebuild the hybrid (`output/hybrid_test/predictions/supermat` is now a broken link).
- **Weights in `/tmp`** (RGB↔X, SuperMat checkpoints, Hugging Face cache) were wiped by the container restart and
  must be re-downloaded for new RGB↔X / SuperMat runs.
