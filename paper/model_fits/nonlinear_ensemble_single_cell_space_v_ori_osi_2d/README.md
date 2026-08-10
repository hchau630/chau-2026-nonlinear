| Figure | Filename |
|--------|----------|
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
| 5E (left) | `resp_paper_0-9/numerical/dr_10_space-cell_type=VIP.pdf` |
| 5C (right) | `resp_paper2_0-9/numerical/dr_10_mean_nearby-cell_type=PV.pdf` |
| 5D (right) | `resp_paper2_0-9/numerical/dr_10_mean_nearby-cell_type=SST.pdf` |
| 5E (right) | `resp_paper2_0-9/numerical/dr_10_mean_nearby-cell_type=VIP.pdf` |
| 5F, S5 (top) | `mean_variance_corr_0-9/numerical/var_SST-mean_PYR.pdf` |
| 5F, S5 (center top) | `mean_variance_corr_0-9/numerical/var_SST-mean_PV.pdf` |
| 5F, S5 (center bottom) | `mean_variance_corr_0-9/numerical/var_SST-mean_SST.pdf` |
| 5F, S5 (bottom) | `mean_variance_corr_0-9/numerical/var_SST-mean_VIP.pdf` |

All run commands:
| Figures | Command |
|---------|---------|
| 4B-C, 4E-G (left, center left, center right), 5C-F | `for MODE in numerical matrix_quasi_linear_approx; do mkdir slurm/run/$MODE; MODE=$MODE PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/$MODE/%A_%a.out niarb run run.toml --linfo --ldebug niarb.integrate; done` |
| 4D | `for MODE in numerical; do for CELLTYPE in PV SST VIP; do mkdir -p slurm/run/${MODE}_reclinear_${CELLTYPE}; MODE=$MODE PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/${MODE}_reclinear_${CELLTYPE}/%A_%a.out niarb run run_reclinear_${CELLTYPE}.toml --linfo; done; done` |
| 4E-G (right) | `for MODE in matrix_second_order_approx_2-2; do for CELLTYPE in PV SST VIP; do mkdir -p slurm/run/${MODE}_reclinear_except_${CELLTYPE}; MODE=$MODE PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/${MODE}_reclinear_except_${CELLTYPE}/%A_%a.out niarb run run_reclinear_except_${CELLTYPE}.toml --linfo; done; done` |
| 5B | `for MODE in matrix_linear_approx; do mkdir slurm/run/${MODE}_uniform; MODE=$MODE PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/${MODE}_uniform/%A_%a.out niarb run run_uniform.toml --linfo; done` |

Optional run commands:
`for MODE in quasi_linear_approx; do mkdir slurm/run/$MODE; MODE=$MODE PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/$MODE/%A_%a.out niarb run run.toml --linfo; done`
`for MODE in second_order_approx_2-2; do for CELLTYPE in PV SST VIP; do mkdir -p slurm/run/${MODE}_reclinear_except_${CELLTYPE}; MODE=$MODE PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/${MODE}_reclinear_except_${CELLTYPE}/%A_%a.out niarb run run_reclinear_except_${CELLTYPE}.toml --linfo; done; done`
`for MODE in linear_approx; do mkdir slurm/run/${MODE}_uniform; MODE=$MODE PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/${MODE}_uniform/%A_%a.out niarb run run_uniform.toml --linfo; done`
`for MODE in matrix analytical matrix_second_order_approx second_order_approx; do mkdir slurm/run/$MODE; MODE=$MODE PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/$MODE/%A_%a.out niarb run run.toml --linfo; done`
`for MODE in matrix_quasi_linear_approx quasi_linear_approx; do mkdir slurm/run/${MODE}_more; MODE=$MODE PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/${MODE}_more/%A_%a.out niarb run run_more.toml --linfo; done`
`for RFCV in 0.2; do for MODE in numerical; do mkdir slurm/run/${MODE}_disordered_vf-${RFCV}; MODE=$MODE RFCV=$RFCV PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/${MODE}_disordered_vf-${RFCV}/%A_%a.out niarb run run_disordered_vf.toml --linfo --ldebug niarb.integrate; done; done`
`python modify_state_dict.py 0-9`
`for MODE in numerical matrix_quasi_linear_approx; do mkdir slurm/run/${MODE}_SST; MODE=$MODE PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-9 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/${MODE}_SST/%A_%a.out niarb run run_SST.toml --linfo --ldebug niarb.integrate; done`

All plot commands:
| Figures | Command |
|---------|---------|
| 4E-G (left, center left, center right) | `for INDICES in 0; do for MODE in matrix_quasi_linear_approx; do for FILENAME in plot/resp_paper_indiv.toml plot/resp_dist.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done` |
| 4E-G (right) | `for INDICES in 0; do for FILENAME in plot/resp_compare.toml; do MODES='["matrix_second_order_approx_2-2_reclinear_except_PV", "matrix_second_order_approx_2-2_reclinear_except_SST", "matrix_second_order_approx_2-2_reclinear_except_VIP"]' INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_matrix_second_order_decompose_2-2_$INDICES --linfo; done; done` |
| 4B-D, 5C-F, S5 | `for INDICES in 0-9; do for MODE in numerical; do for FILENAME in plot/resp_paper.toml plot/resp_paper2.toml plot/cotuned_supp1.toml plot/cotuned_supp2.toml plot/cotuned_supp3.toml plot/mean_variance_corr.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done` |
| S4A-E | `for INDICES in 0-9; do for FILENAME in plot/params_pre/dist.toml; do INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES --linfo; done; done` |
| 5B | `for INDICES in 0-9; do for FILENAME in plot/Lij.toml; do INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES --linfo; done; done` |

Optional plot commands:
`for INDICES in 0-4 5-9; do for FILENAME in plot/params_pre/param.toml; do INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES --linfo; done; done`
`for INDICES in 0-4 5-9; do for MODE in numerical matrix matrix_quasi_linear_approx matrix_second_order_approx; do for FILENAME in plot/resp_indiv.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`
`for INDICES in 0-4 5-9; do for MODE in numerical analytical quasi_linear_approx second_order_approx; do for FILENAME in plot/resp_indiv.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`
`for INDICES in 0-4 5-9; do for MODE in matrix_quasi_linear_approx; do for FILENAME in plot/more_indiv.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`
`for INDICES in 0-4 5-9; do for MODE in quasi_linear_approx; do for FILENAME in plot/more_indiv.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`
`for INDICES in 0-4 5-9; do for MODE in numerical; do for CELLTYPE in PV SST VIP; do for FILENAME in plot/resp_indiv.toml; do MODE2=${MODE}_reclinear_${CELLTYPE}; INDICES=$INDICES MODE=$MODE2 sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE2 --linfo; done; done; done; done`
`for INDICES in 0; do for MODE in quasi_linear_approx; do for FILENAME in plot/resp_paper_indiv.toml plot/resp_dist.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`
`for INDICES in 0; do for MODE in second_order_approx_2-2_reclinear_except_PV second_order_approx_2-2_reclinear_except_SST second_order_approx_2-2_reclinear_except_VIP; do for FILENAME in plot/resp_paper_indiv.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`
`for INDICES in 0-9; do for MODE in numerical matrix_quasi_linear_approx; do for FILENAME in plot/resp_perturbed.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`
`for INDICES in 0-9; do for MODE in numerical_disordered_vf-0.2; do for FILENAME in plot/resp_paper.toml plot/resp_paper2.toml plot/mean_variance_corr.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`
`for INDICES in 0; do for MODE in numerical_disordered_vf-0.2; do for FILENAME in plot/resp_paper_indiv.toml plot/resp_dist.toml plot/baseline_dist.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`
`for INDICES in 0-9; do for MODE in numerical_SST matrix_quasi_linear_approx_SST; do for FILENAME in plot/resp_paper_no_data.toml plot/resp_perturbed.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`
`for INDICES in 0-9; do for FILENAME in plot/resp_compare_all.toml; do MODES='["matrix_second_order_approx", "numerical"]' INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_matrix_second_order-numerical_$INDICES --linfo; done; done`
`for INDICES in 0-4 5-9; do for FILENAME in plot/resp_compare_indiv.toml; do MODES='["matrix_second_order_approx", "numerical"]' INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_matrix_second_order-numerical_$INDICES --linfo; done; done`
`for INDICES in 0-9; do for MODE in numerical matrix_second_order_approx numerical_reclinear_VIP matrix_second_order_approx_reclinear_VIP; do for FILENAME in plot/resp_indiv_instances.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`

