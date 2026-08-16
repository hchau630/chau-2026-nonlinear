# Table of figures

| Figure(s) | Filename(s) (in directory `figures`) |
| --------- | ----------- |
| 4B | `resp_paper_0-9/numerical/dr_10_space-cell_type=PYR.pdf` |
| 4C | `resp_paper_0-9/numerical/dr_10_mean_nearby-cell_type=PYR.pdf` |
| 4D | `cotuned_supp_0-9/numerical/numerical.pdf`, `cotuned_supp2_0-9/numerical/numerical-density=compact.pdf`, `cotuned_supp3_0-9/numerical/numerical-density=compact.pdf` |
| 4E-G (left) | `resp_paper_indiv_0/matrix_quasi_linear_approx/dv_10_space_ori-density=compact.pdf` |
| 4E-G (center left) | `resp_dist_0/matrix_quasi_linear_approx/dv_nearby_all-loss=0.50543832779.pdf` |
| 4E-G (center right) | `resp_dist_0/matrix_quasi_linear_approx/Hdv2_nearby_all-loss=0.50543832779.pdf` |
| 4E-G (right) | `resp_compare_matrix_second_order_decompose_2-2_0/resp_mean_nearby2-loss=5.0543832779e-01-cell_type=PYR.pdf` |
| S4A-E | `dist_0-9/*.pdf` |
| 5A (left) | Same as Figure 4F (center right) |
| 5B | `Lij_0-9/all.pdf` |
| 5C (left) | `resp_paper_0-9/numerical/dr_10_space-cell_type=PV.pdf` |
| 5D (left) | `resp_paper_0-9/numerical/dr_10_space-cell_type=SST.pdf` |
| 5C (right) | `resp_paper2_0-9/numerical/dr_10_mean_nearby-cell_type=PV.pdf` |
| 5D (right) | `resp_paper2_0-9/numerical/dr_10_mean_nearby-cell_type=SST.pdf` |
| 5E, S5 (top) | `mean_variance_corr_0-9/numerical/var_SST-mean_PYR.pdf` |
| 5E, S5 (center) | `mean_variance_corr_0-9/numerical/var_SST-mean_PV.pdf` |
| 5E, S5 (bottom) | `mean_variance_corr_0-9/numerical/var_SST-mean_SST.pdf` |

# Commands for reproducing figures in directory `figures`

These commands were run on a SLURM cluster. If you're not on a SLURM cluster, remove `sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out` from the commands.

| Figures | Command (run in this directory) |
| ------- | ------------------------------- |
| 4E-G (left, center left, center right) | <pre>`for INDICES in 0; do for MODE in matrix_quasi_linear_approx; do for FILENAME in plot/resp_paper_indiv.toml plot/resp_dist.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`</pre> |
| 4E-G (right) | <pre>`for INDICES in 0; do for FILENAME in plot/resp_compare.toml; do MODES='["matrix_second_order_approx_2-2_reclinear_except_PV", "matrix_second_order_approx_2-2_reclinear_except_SST", "matrix_second_order_approx_2-2_reclinear_except_VIP"]' INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_matrix_second_order_decompose_2-2_$INDICES --linfo; done; done`</pre> |
| 4B-D, 5C-F, S5 | <pre>`for INDICES in 0-9; do for MODE in numerical; do for FILENAME in plot/resp_paper.toml plot/resp_paper2.toml plot/cotuned_supp1.toml plot/cotuned_supp2.toml plot/cotuned_supp3.toml plot/mean_variance_corr.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`</pre> |
| S4A-E | <pre>`for INDICES in 0-9; do for FILENAME in plot/params_pre/dist.toml; do INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES --linfo; done; done`</pre> |
| 5B | <pre>`for INDICES in 0-9; do for FILENAME in plot/Lij.toml; do INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES --linfo; done; done`</pre> |

# Commands for computing model perturbation responses in directory `runs`

These commands were run on a SLURM cluster with A40 GPUs. If you're on a SLURM cluster, modify the `--gres` option if necessary. Otherwise, remove `mkdir -p slurm/run/**;` and `sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 --gres=gpu:a40:1 --output slurm/run/**/%A_%a.out` and add the for-loop `for i in {0..9}; do SLURM_ARRAY_TASK_ID=$i [inner-most command]; done`. GPUs with < 48 GB memory will likely not be able to run the commands.

| Figures | Command (run in this directory) |
| ------- | ------------------------------- |
| 4B-C, 4E-G (left, center left, center right), 5C-F | <pre>`for MODE in numerical matrix_quasi_linear_approx; do mkdir -p slurm/run/$MODE; MODE=$MODE PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 --gres=gpu:a40:1 --output slurm/run/$MODE/%A_%a.out niarb run run.toml --linfo --ldebug niarb.integrate; done`</pre> |
| 4D | <pre>`for MODE in numerical; do for CELLTYPE in PV SST VIP; do mkdir -p slurm/run/${MODE}_reclinear_${CELLTYPE}; MODE=$MODE PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 --gres=gpu:a40:1 --output slurm/run/${MODE}_reclinear_${CELLTYPE}/%A_%a.out niarb run run_reclinear_${CELLTYPE}.toml --linfo; done; done`</pre> |
| 4E-G (right) | <pre>`for MODE in matrix_second_order_approx_2-2; do for CELLTYPE in PV SST VIP; do mkdir -p slurm/run/${MODE}_reclinear_except_${CELLTYPE}; MODE=$MODE PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 --gres=gpu:a40:1 --output slurm/run/${MODE}_reclinear_except_${CELLTYPE}/%A_%a.out niarb run run_reclinear_except_${CELLTYPE}.toml --linfo; done; done`</pre> |
| 5B | <pre>`for MODE in matrix_linear_approx; do mkdir -p slurm/run/${MODE}_uniform; MODE=$MODE PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 --gres=gpu:a40:1 --output slurm/run/${MODE}_uniform/%A_%a.out niarb run run_uniform.toml --linfo; done`</pre> |

# Command for producing model fits in directory `fits`

This command was run on a SLURM cluster with A40 GPUs. If you're on a SLURM cluster, modify the `--gres` option if necessary. Otherwise, remove `sbatch --array 0-999 -c 8 --mem-per-cpu=4gb --time 10:00:00 --gres=gpu:a40:1 --output slurm/fit/%A_%a.out` and replace `-N 5` with `-N 5000`. GPUs with < 48 GB memory will likely not be able to run the command. Run the command in this directory.

```
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-999 -c 8 --mem-per-cpu=4gb --time 10:00:00 --gres=gpu:a40:1 --output slurm/fit/%A_%a.out niarb fit fit.toml -N 5 -o fits --linfo --ignore-errors
```

