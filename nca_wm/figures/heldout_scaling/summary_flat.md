# Held-out overlay scaling table (reduce=flat)

All runs evaluated on the same 21-game held-out set.

| run | n_games | mask_hidden | n_updates | nca_steps | median 1-NN | mean 1-NN | min | max |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| `multi_scaling_14_v3recipe` | 14 | False | 150000 | 8 | 0.144 | 0.165 | 0.003 | 0.434 |
| `multi_scaling_14_mask_v1` | 14 | True | 80000 | 8 | 0.077 | 0.093 | 0.022 | 0.250 |
| `multi_scaling_14_mask_v2_perstep` | 14 | True | 80000 | 8 | 0.133 | 0.162 | 0.008 | 0.433 |
| `multi_scaling_gallery_v2_nca8` | 59 | False | 80000 | 8 | 0.108 | 0.120 | 0.026 | 0.263 |
| `multi_scaling_gallery_v3_combined` | 59 | False | 150000 | 8 | 0.123 | 0.123 | 0.009 | 0.306 |
