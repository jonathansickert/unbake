# Rebake guidance

| Module | Purpose |
|---|---|
| `rebake.py` | Rebake model: texel sampling, visibility transfer to SH, least-squares lighting fit, rebake error, PBR maps of a .glb. |
| `common.py` | Pipeline, conditioning and shape latent as in the unguided baseline, decoder block checkpointing (backprop at 1024 fits in 24 GB), voxel lookup, prediction writer. |
| `latent_opt.py` | Test-time optimization of the TRELLIS.2 texture latent with the rebake loss, latent regularization, albedo anchor and optional metallic prior. |
| `guided_sampling.py` | Rebake guidance inside the rectified-flow sampler, optionally starting from the encoded input. |
| `ensemble.py` | Mean of several predictions per input (e.g. seeds), optionally with constant roughness / metallic. |
| `delight.py` | De-lighting: base color = baked color / shading, with the lighting fitted on a prior prediction, low frequencies from the prior and a fallback to it in shadow. |
| `analyze.py` | Per-input comparison of methods with the scale-invariant base color error. |
| `results/` | Metrics of the first runs (old protocol, see below). |

All scripts run from the repository root as modules and write the prediction layout of
`unbake/predictions.py`. With zero guidance (`latent_opt --steps 0`, `guided_sampling --guided-steps 0`)
they reproduce `baselines.unguided_trellis2` with the same seed.

Best variant so far (base color of three optimized seeds, constant roughness 0.5 / metallic 0, de-lighting):

```sh
for s in 42 1 2; do python -m unbake.method.latent_opt --seed $s; done
python -m unbake.method.ensemble --const-rm --output output/predictions/rebake_opt_ensemble \
    --members output/predictions/rebake_opt output/predictions/rebake_opt_s1 output/predictions/rebake_opt_s2
python -m unbake.method.delight --prior output/predictions/rebake_opt_ensemble --rays 1024 --split 4
python -m unbake.evaluation.evaluate_uv && python -m unbake.evaluation.evaluate_relit
```

## `results/`

Computed with the first version of the evaluation (like `baselines/results/`), and TRELLIS.2 was
conditioned on the dataset renderings, which show the ground-truth PBR asset. Only the relative
comparisons between variants are meaningful; rerun the methods for current numbers.
