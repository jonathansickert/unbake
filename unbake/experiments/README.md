# Feasibility experiments

| Module | Question |
|---|---|
| `latent_ceiling` | How well can the TRELLIS.2 texture latent represent the ground-truth materials? (writes predictions) |
| `lighting_model` | How much of the bake does Lambertian shading under SH lighting with visibility explain? |
| `rebake_discrimination` | Does the rebake error rank candidate materials by their base color error? |
| `decoder_backprop` | Memory and time of backpropagation through the texture decoder. |
| `confidence` | Does the spread of per-view 2D predictions predict their error? |
| `specular_signal` | Do the bakes contain information about roughness and metallic? |
| `results/` | Results of the first runs on subset 1 (specular signal: subsets 1 and 2), old protocol. |

Run from the repository root, e.g. `python -m unbake.experiments.latent_ceiling`.
