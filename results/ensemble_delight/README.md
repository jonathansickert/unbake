Results of the branch on all 80 inputs (seed 42 unless an ensemble).

- `main/`: TRELLIS.2, guided sampling, seed ensembles and de-lighting; UV and relighting metrics (`compare.md`).
- `delight_variants/`: de-lighting with sRGB transfer, blur widths, per-region anchor and the ground truth as prior.
- `calibration/`: brightness calibration fitted on the other objects.
- `delit_conditioning/`: TRELLIS.2 conditioned on a render of the de-lit prediction, with and without the baked render as negative condition.
- `step_schedule/`: guided sampling with a step size that shrinks with the posterior variance.

`diagnostics.txt`: scale-invariant base color MAE and per-object differences (bootstrap interval, Wilcoxon p).
Produced by `unbake.evaluation.evaluate_uv`, `evaluate_relit` and `diagnostics`.
