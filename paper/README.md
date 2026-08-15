# Setup

If you want to run commands which reference `oldenburg-2024-logic` (e.g. for reproducing Figure 2D/E), you need to run these two additional commands first in top-level directory of this repository (one-time only):
```
git submodule init
git submodule update
```

# Commands for generating paper figures

| Figure | Filename | Command (run in this directory) |
| ------ | -------- | ------------------------------- |
| 1A inset | `1a.pdf` | <pre>`python plot_ricciardi.py -o figures`</pre> |
| 1B | `plot_rnn_theory/log_normal_vary_n_perturbed/rel_err_n_unperturbed.pdf` | <pre>`python plot_rnn_theory.py resp --dh 20 --n-perturbed 1 10 --log-normal --use-t --t-max 100 --n-trials 300000 -o figures/plot_rnn_theory/log_normal_vary_n_perturbed`</pre> |
| 1C, S1A, S1B | `plot_rnn_theory/log_normal_vary_dh_n_perturbed/*.pdf` | <pre>`python plot_rnn_theory.py resp --dh 10 15 20 25 30 --n-perturbed 1 5 10 20 50 -n 3000 --log-normal --use-t --t-max 100 --n-trials 300000 -o figures/plot_rnn_theory/log_normal_vary_dh_n_perturbed`</pre> |
| 2B | `2b.pdf` | <pre>`python plot_distance.py -s 100 --density 8000 --min-dist 10 --scale 3 -o figures`</pre> |
| 2D, 2E, S2C | `2d.pdf`, `2e.pdf`, `S2c.pdf` | <pre>`python plot_oldenburg_data.py oldenburg-2024-logic --2d -o figures`</pre> |
| 2F | `2f.pdf`, `2f_inset.pdf`, `2f_ori_cmap.pdf`, `2f_ori_cmap.pdf` | <pre>`python plot_model.py -o figures`</pre> |
| 2G, 2H | | See `model_fits/linear_ensemble_space_v_ori_osi_2d` |
| 2I, 2J, 3A, 3B, 3C, 3F, S2H, S2I, S3 | | See `model_fits/nonlinear_ensemble_single_cell_space_v_ori_osi_2d_EI` |
| 3D, 3F inset | `3d.pdf`, `3f_inset.pdf` | <pre>`python plot_mechanism.py -o figures`</pre> |
| 4, 5, S4, S5 | | See `model_fits/nonlinear_ensemble_single_cell_space_v_ori_osi_2d` |
| S1C, S1D | `plot_rnn_theory/log_normal_vary_dh/rel_err_grid_*.pdf` | <pre>`python plot_rnn_theory.py resp --dh 10 15 20 25 30 --log-normal --use-t --t-max 100 --n-trials 300000 -o figures/plot_rnn_theory/log_normal_vary_dh`</pre> |
| S2A, S2B | `S2a.pdf`, `S2b.pdf` | <pre>`python plot_oldenburg_cells.py oldenburg-2024-logic -o figures`</pre> |
| S2D | `S2d.pdf` | <pre>`python plot_oldenburg_data.py oldenburg-2024-logic --2d --rel-ori -o figures`</pre> |
| S2E, S2F | `S2e.pdf`, `S2f.pdf` | <pre>`python plot_oldenburg_ensembles.py oldenburg-2024-logic -o figures`</pre> |
| S2G (left) | `S2g_untuned.pdf` | <pre>`python plot_model_ensembles.py ori -a 0.95 -b 1.22 -k -3.7 10.0 -9.8 -o figures/S2g_untuned.pdf`</pre> |
| S2G (middle) | `S2g_untuned_corrected.pdf` | <pre>`python plot_model_ensembles.py ori -a 0.95 -b 1.22 -k -3.7 10.0 -9.8 --max-ens-osi 0.07 -o figures/S2g_untuned_corrected.pdf`</pre> |
| S2G (right) | `S2g_cotuned.pdf` | <pre>`python plot_model_ensembles.py ori -a 1.32 -b 0.66 -k -7.3 19.0 -12.0 -o figures/S2g_cotuned.pdf`</pre> |

# Commands for generating model fitting data

| Data | Command (run in this directory) |
| ---- | ------------------------------- |
| `ensemble_space_ori_xy_data.pkl` | <pre>`python plot_oldenburg_data.py oldenburg-2024-logic --2d --out-data-dir [path]`</pre> |
| `ensemble_space_v_ori_osi_xy_data.pkl` | <pre>`python plot_oldenburg_data.py oldenburg-2024-logic --rel-ori --2d --out-data-dir [path]`</pre> |
| `zeros_rel_ori_325_PV_SST_VIP.pkl` (regularization) | <pre>`python generate_zeros_data.py 325 0.01 --rel-ori -c PV SST VIP -o [path]`</pre> |
| `zeros_rel_ori_250_PYR.pkl` (regularization) | <pre>`python generate_zeros_data.py 250 0.01 --rel-ori -c PYR -o [path]`</pre> |
