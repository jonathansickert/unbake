# UnBake: Method

UnBake recovers PBR materials (base color $c$, roughness $r$, metallic $m$) for a textured mesh whose texture
has lighting baked in. It is training-free: a frozen 3D generative material prior (the TRELLIS.2 material stage)
is guided at test time by

1. a **rebake loss**: the predicted base color, lit by a lighting fitted in closed form and shadowed by the
   mesh's own geometry, must reproduce the baked texture, and
2. a **2D material prior**: roughness and metallic estimates from a 2D model (SuperMat), predicted on renders
   of the baked asset and fused per voxel with confidence weights.

The output keeps the input mesh's geometry and UV layout. This document describes the method as implemented in
`unbake/guided_trellis2.py` (single asset), `unbake/prior2d.py` (2D prior) and `unbake/guided_dataset.py`
(whole dataset), and lists where it differs from the midterm report.

---

## 1. Problem and image formation

**Input.** A triangle mesh with UVs and a baked texture, exported unlit. The dataset bakes it with Blender Cycles as

$$o(x) = c(x) \odot E(x),$$

the base color times the diffuse irradiance $E$ at surface point $x$, which includes cast shadows, ambient
occlusion and interreflections. The texture is sRGB-encoded and exposure-normalized (99.5th percentile → 1).

**Output.** Base color, roughness and metallic textures in the input's UV layout, at the baked texture's
resolution.

**What the input constrains.** Diffuse baking does not depend on roughness or metallic. So the input constrains
the base color, up to a global scale and color tint per channel that trade off against the lighting, but
carries no direct information on $r$ and $m$. The base color is therefore constrained by the rebake loss, and
$r$ and $m$ by the generative prior and the 2D prior only.

---

## 2. Preprocessing

**Frame.** All geometry is mapped into TRELLIS.2 space (`TrellisFrame`): centered on its bounding box, scaled so
the longest side is $0.99999$, so it lies in $[-0.5, 0.5]^3$, and rotated from glTF Y-up to Z-up,
$(x, y, z) \mapsto (x, -z, y)$.

**Shape latent.** The merged mesh is converted to TRELLIS.2's O-Voxel grid at resolution $R$ ($R = 1024$ by
default; $512$ also supported) and encoded by the frozen shape encoder into the shape latent $z_\text{shape}$.

**Conditioning image.** The baked asset is rendered unlit, i.e. its baked texture without shading, from a
three-quarter view (yaw $45°$, pitch $25°$, 1024 px, 2× supersampling, transparent background). The image is
encoded by DINOv3 into the cross-attention conditioning of the material flow model. This render is the only image
a user with just the baked mesh has.

---

## 3. Observations per voxel

The rebake loss compares decoded voxels with the baked texture, so every voxel of the $R^3$ grid that the
surface passes through gets an observed color and normal.

1. **Surface samples.** For each material, its triangles are rasterized in UV space at $s$ times the texture
   resolution ($s = 4$ by default; *supersampling*). Each sample gets its 3D surface point (interpolated), the
   face normal of its triangle, and the baked color, sampled bilinearly at its UV with glTF repeat wrapping.
   At $s = 1$ the samples lie exactly on texel centers. At $R = 1024$ a 1024² texture does not have enough
   texels for every surface voxel. With $s = 1, 2, 4$ the share of decoded voxels with an observation is about
   13%, 39% and 74% on the cauldron example.
2. **Linear color.** Colors are decoded from sRGB with $\gamma = 2.2$: $o \leftarrow o^{2.2}$.
3. **Per-voxel averaging.** Samples are binned into voxels $\lfloor (p + 0.5) R \rfloor$. Each voxel $i$ gets the
   mean color $o_i$ and the mean normal. Voxels whose normals cancel (mean-normal length below half the sample
   count, i.e. thin parts with opposing surfaces) are dropped. What remains are the observed voxels
   $\mathcal{O}$, with center $x_i$, unit normal $n_i$ and color $o_i$.

---

## 4. Lighting model with visibility

**Shading model.** Lambertian shading under distant lighting expressed in real spherical harmonics up to order 2
(9 coefficients per color channel, $L \in \mathbb{R}^{9 \times 3}$):

$$\hat o_i = c_i \odot \left( t_i^\top L \right), \qquad
  t_i = \frac{4\pi}{K} \sum_{k=1}^{K} V_i(d_k)\, \max(0,\, n_i \cdot d_k)\, Y(d_k) \in \mathbb{R}^9.$$

$t_i$ is the SH projection of the cosine-weighted visibility of voxel $i$, and $Y$ is the real SH basis up to
order 2. Since $\hat o_i$ is linear in $L$, the lighting is fitted in closed form (Sec. 5).

**Visibility $V_i(d)$.** $V_i(d) = 1$ if voxel $i$ sees direction $d$ without the mesh in the way. It is computed
once per asset with orthographic shadow maps:

- $K = 256$ directions $d_k$ on a Fibonacci sphere.
- Per direction, the whole mesh is rasterized orthographically along $d_k$ (half extent $0.9$, which covers the
  object's bounding sphere of radius $0.87$; 2048² depth map, nvdiffrast).
- Voxel centers are offset along the normal by $1.5$ voxels against self-shadowing. A voxel counts as visible if
  its depth along $d_k$ is within 2 shadow-map texels of the depth map.

This lets the fitted lighting explain baked shadows and occlusion, instead of the base color absorbing them.

**Without visibility** ($V \equiv 1$), the transfer reduces to the analytic clamped-cosine convolution
$t_i = A_l\, Y_{lm}(n_i)$ with $A_0 = \pi$, $A_1 = 2\pi/3$, $A_2 = \pi/4$. On unoccluded voxels the Monte Carlo
transfer matches it to 0.2%.

On the cauldron, on average 44–51% of each voxel's hemisphere is unoccluded, and visibility correlates with
baked luminance ($r = 0.38$).

---

## 5. Losses

Let $\mathcal{M}$ be the decoded voxels that have an observation (matched by grid index). Decoded base colors are
linearized like the observations: $c_i \leftarrow \text{clamp}(c_i, 10^{-4}, 1)^{2.2}$.

**Lighting fit.** At every evaluation, per color channel $\kappa$, with $c$ detached:

$$L_{\cdot\kappa} = \arg\min_{\ell} \sum_{i \in \mathcal{M}} \left( c_{i\kappa}\, t_i^\top \ell - o_{i\kappa} \right)^2
  + \lambda_\text{ridge} |\mathcal{M}|\, \|\ell\|^2, \qquad \lambda_\text{ridge} = 10^{-4},$$

solved via the $9 \times 9$ normal equations. The shading $t_i^\top L$ is clamped at $0$.

**Rebake loss.**

$$\mathcal{L}_\text{rebake} = \frac{1}{3|\mathcal{M}|} \sum_{i \in \mathcal{M}} \sum_\kappa
  \text{Huber}_\delta\!\left( c_{i\kappa}\, t_i^\top L_{\cdot\kappa} - o_{i\kappa} \right), \qquad \delta = 0.05.$$

Because $L$ is refit, the loss is invariant to a global scale and a tint per channel of the base color. Those
are left to the generative prior.

**2D prior loss** (Sec. 6), over roughness and metallic only:

$$\mathcal{L}_\text{2D} = \sum_{k \in \{r, m\}} \lambda_k\,
  \frac{\sum_{i \in \mathcal{M}} w^k_i \left| p^k_i - \tilde p^k_i \right|}{\sum_{i \in \mathcal{M}} w^k_i},
  \qquad \lambda_r = \lambda_m = 1,$$

with decoded values $p^k_i$, fused estimates $\tilde p^k_i$ and confidence weights $w^k_i$. The base color is
deliberately not in $\mathcal{L}_\text{2D}$: the 2D models' albedo is too bright and would fight the rebake loss
(see the experiments with RGB↔X).

---

## 6. 2D roughness / metallic prior

**Views.** The baked asset is rendered unlit from $N = 16$ orthographic views, with camera positions on a
Fibonacci sphere (up vector $+Z$, or $+Y$ near the poles). The object is scaled to $0.9$ of the
$[-0.5, 0.5]$ frame, as in SuperMat's setup. Renders are 512², with a depth map per view for visibility.

**Per-view prediction.** SuperMat's single-view network (`supermat.pth`, one diffusion step; RGBA composited on
gray $0.5$) predicts roughness and metallic for every view independently. SuperMat's multi-view network is built
for exactly its own 6 axis-aligned, camera-embedded views, so it cannot use the 16 views. RGB↔X
(`--prior rgbx`) and SuperMat multi-view on 6 views (`--prior supermat`) are available as alternatives.

**Fusion.** Each observed voxel center is projected into every view. A view counts if the voxel passes the depth
test (tolerance $0.01$) and faces the camera with $\cos\theta = n_i \cdot \hat v > 0.1$. Predictions are sampled
bilinearly and averaged with weights $\cos^2\theta$:

$$\tilde p^k_i = \frac{\sum_v \omega_{iv}\, p^k_{iv}}{\sum_v \omega_{iv}}, \qquad
  \sigma^k_i = \sqrt{\frac{\sum_v \omega_{iv} (p^k_{iv} - \tilde p^k_i)^2}{\sum_v \omega_{iv}}}, \qquad
  \omega_{iv} = \cos^2\theta_{iv}\, [\text{visible}].$$

**Confidence.**

$$w^k_i = \frac{1}{1 + (\sigma^k_i / \tau^k)^2}, \qquad \tau^k = \operatorname{median}_i \sigma^k_i,$$

and $w^k_i = 0$ for voxels seen in fewer than 2 views. Views that disagree lower the weight. Note that an error
that is the same in every view, such as a consistently wrong brightness, is not detected.

The fused prior is computed once per asset and voxel set. Per-view predictions are saved, so the prior can be
re-fused for another observation set without running SuperMat again.

---

## 7. Guided sampling

TRELLIS.2 generates the material latent $x$ (normalized, on the shape latent's sparse voxels) with a rectified
flow sampler. In TRELLIS.2's convention

$$x(t) = (1 - t)\, x_0 + (\sigma_\text{min} + (1 - \sigma_\text{min})\, t)\, \epsilon,$$

and the transformer $v_\theta(x(t), t)$ is integrated from $t = 1$ to $t = 0$ with Euler steps. The pipeline
settings are: 12 steps, $t$ rescaled with factor 3, $\sigma_\text{min} = 10^{-5}$, classifier-free guidance
strength 1 (no CFG), and the same seed for guided and unguided samples.

At every step whose $t \in [0, 0.8]$ (steps 6–11, $t \approx 0.75, 0.68, 0.60, 0.50, 0.38, 0.21$):

1. **Clean estimate and noise.** From the model's prediction,
   $\hat x_0 = (1 - \sigma_\text{min})\, x(t) - (\sigma_\text{min} + (1 - \sigma_\text{min}) t)\, v_\theta$ and
   $\hat\epsilon = (1 - t)\, v_\theta + x(t)$.
2. **Correction**, $J = 3$ iterations on $\hat x_0$:
   - Decode $\text{pbr} = D(\hat x_0 \cdot s + \mu) \cdot 0.5 + 0.5$ with the frozen material decoder $D$
     ($\mu, s$: latent normalization). This gives per-voxel base color, metallic, roughness and alpha.
   - Evaluate both losses. For each term, take the gradient with respect to $\hat x_0$ through $D$ only.
     The losses are scaled by $10^4$ before backpropagation, since the decoder runs in fp16 and small gradients
     underflow.
   - **Separately normalized update:**
     $$g = \frac{\nabla \mathcal{L}_\text{rebake}}{\text{rms}(\nabla \mathcal{L}_\text{rebake})}
         + \beta\, \frac{\nabla \mathcal{L}_\text{2D}}{\text{rms}(\nabla \mathcal{L}_\text{2D})}, \qquad
       \hat x_0 \leftarrow \hat x_0 - \eta\, \frac{g}{\text{rms}(g)}, \qquad \eta = 0.05, \ \beta = 1.$$
     The raw gradients differ by about 100× (RMS $\sim 10^{-4}$ vs. $\sim 10^{-2}$ at 1024³). A single combined
     gradient would let $\mathcal{L}_\text{2D}$ dominate and undo the rebake correction. With separate
     normalization, $\beta$ is the 2D prior's share of each step, independent of the losses' magnitudes, and the
     step size does not depend on $\beta$. $\beta = 0$ gives rebake-only guidance.
3. **Continue sampling** from the corrected estimate with the predicted noise, directly at the next timestep:
   $$x(t_\text{prev}) = (1 - t_\text{prev})\, \hat x_0 + (\sigma_\text{min} + (1 - \sigma_\text{min})\, t_\text{prev})\, \hat\epsilon.$$
   This equals an Euler step with the velocity implied by the corrected $\hat x_0$.

At unguided steps, $x(t_\text{prev}) = x(t) - (t - t_\text{prev})\, v_\theta$. All network weights stay frozen. The
decoder weights keep `requires_grad=True` only because flex_gemm's sparse-convolution backward crashes otherwise;
only the gradient with respect to the latent is taken. At $R = 1024$ the decoder uses gradient checkpointing,
since backpropagation does not fit in the 24 GB GPU otherwise.

---

## 8. Output: transfer to the input's UV layout

The final latent is decoded once more. For every texel of each material's texture (at the baked texture's
resolution), the 3D surface point is computed by UV rasterization. Base color, roughness and metallic are
sampled there by trilinear interpolation of the active voxels, renormalized by the interpolated occupancy so
empty neighbors do not darken the result. UV islands are padded by 8 px. The maps replace the baked texture in
the input glTF as a metallic-roughness material (`KHR_materials_unlit` removed), so geometry and UV layout are
unchanged.

---

## 9. Default settings

| | value |
|---|---|
| voxel resolution $R$ | 1024 |
| observation supersampling $s$ | 4 |
| visibility | 256 directions, 2048² shadow maps, 1.5-voxel offset, 2-texel tolerance |
| lighting | order-2 SH, $9 \times 3$, ridge $10^{-4}$ |
| rebake loss | Huber, $\delta = 0.05$, linear color ($\gamma = 2.2$) |
| 2D prior | SuperMat single-view, 16 Fibonacci views, 512², roughness + metallic, $\lambda_r = \lambda_m = 1$ |
| fusion | $\cos^2$ weights, $\cos > 0.1$, depth tolerance $0.01$, ≥ 2 views, $\tau$ = median $\sigma$ |
| guidance | steps with $t \le 0.8$ (6 of 12), $J = 3$, $\eta = 0.05$, $\beta = 1$ |
| sampler | TRELLIS.2 defaults: 12 steps, rescale 3, no CFG, seed 42 |

CLI: `python -m unbake.guided_trellis2 --input <baked.glb> [--gt <pbr.glb>] --out <dir> --resolution 1024
--prior supermat_sv --sm-dir <SuperMat clone> --betas 1 --visibility 1 --supersample 4`.
The script's own defaults are older (512, RGB↔X, no visibility, $s = 1$, $\beta \in \{0.01, 0.1\}$) and are
kept for reproducing earlier runs.

**Implementation status.** `unbake/guided_dataset.py` still calls `voxel_observations` with its defaults (no
visibility, $s = 1$). The 1024³ dataset run in `output/predictions/guided_supermat_sv_b1_1024/` therefore uses
the rebake loss without visibility. Running the dataset with this method needs the two options passed through
and the priors re-fused for the denser observations.

**Run time** (1024³, 24 GB MIG slice of an RTX PRO 6000): about 2.5 min per input for guided sampling, 1–2 s for
visibility, about 20 s for the 16 SuperMat predictions (once the model is loaded), and about 1 min to load the
models.

---

## 10. Changes since the midterm report

| | midterm report | now |
|---|---|---|
| visibility in $t_i$ | described | implemented as above (shadow maps, 256 directions) |
| observations | input texture sampled per voxel | UV rasterization at 4× texture size, bilinear, per-voxel mean |
| 2D prior model | RGB↔X, 16–32 views | SuperMat single-view, 16 views |
| 2D prior channels | $c, r, m$ | $r, m$ only |
| view averaging | "robust average" | $\cos^2$-weighted mean, grazing views dropped |
| update | $\hat x_0 \leftarrow \hat x_0 - \eta \nabla(\mathcal{L}_\text{rebake} + \beta \mathcal{L}_\text{2D})$ | separately normalized gradients, $J = 3$ iterations, fixed step size $\eta$ |
| meaning of $\beta$ | loss weight | share of the 2D prior in each normalized step |
| schedule | "selected steps" | $t \le 0.8$ |
| re-interpolation | $x(t) \leftarrow (1 - t)\hat x_0 + t\hat\epsilon$ | same with $\sigma_\text{min}$, directly at $t_\text{prev}$ |

Unchanged: Lambertian SH model with closed-form lighting fit, Huber rebake loss, confidence weights
$1/(1 + (\sigma/\tau)^2)$ with zero weight below two views, gradients through the frozen decoder only, output in the
input's UV layout.

---

## 11. Known limitations

- **Metallic.** The 2D prior made metallic worse on the cauldron at 1024³ (0.302 → 0.386 MAE), although it helped
  at 512³ and SuperMat's metallic is good dataset-wide. To be decided on the dataset; $\lambda_m = 0$ is the
  fallback.
- **Roughness and metallic are unconstrained by the input** (diffuse baking). They rely on the priors, and the
  constants $r = 0.5$, $m = 0$ remain hard to beat on mostly non-metal assets.
- **Lighting model.** Order-2 SH cannot represent sharp lighting, and interreflections are not modeled; both
  remain partly in the base color.
- **Scale ambiguity.** The rebake loss does not fix the base color's overall brightness or tint per channel; the
  generative prior does.
- **Resolution.** Details smaller than a voxel are lost in the 1024³ grid. Dividing the input texture by the
  fitted shading at texel resolution would recover them, but is not implemented.
- **Unobserved voxels.** Decoded voxels without an observation (26% at $s = 4$ on the cauldron) get no guidance
  and are only indirectly affected through the shared latent.
