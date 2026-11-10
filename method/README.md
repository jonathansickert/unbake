# Rebake guidance

| File | Purpose |
|---|---|
| `rebake.py` | Rebake model: texel sampling, visibility transfer to second-order SH, least-squares lighting fit, rebake error. |
| `common.py` | Decoder block checkpointing (backprop at 1024 fits in 24 GB), voxel lookup, output in the baseline format. |
| `latent_opt.py` | Test-time optimization of the TRELLIS.2 texture latent with the rebake loss, latent regularization, albedo anchor and optional metallic prior. |
| `guided_sampling.py` | Rebake guidance inside the rectified-flow sampler, optionally starting from the encoded input. |
| `ensemble.py` | Mean of several results per input (e.g. seeds), optionally with constant roughness / metallic. |
| `analyze.py` | Per-input comparison with the scale-invariant albedo error. |
| `merge_relit.py` | Merges relighting metrics of repeated evaluations. |
| `results/` | Metrics of all variants on subsets 1 and 2, summary and per input. |

Best variant so far (albedo of three optimized seeds, constant roughness 0.5 / metallic 0):

```sh
for s in 0 1 2; do
  python method/latent_opt.py --data data/unbake_sub10 --out results/opt_seed$s --lambdas 0.1 --anchor 1 --seed $s
done
python method/ensemble.py --const-rm --out results/opt_ensemble --dirs results/opt_seed{0,1,2}/lambda0.1
python baselines/evaluate_uv.py --pred results/opt_ensemble
python baselines/evaluate_relit.py --pred results/opt_ensemble
```
