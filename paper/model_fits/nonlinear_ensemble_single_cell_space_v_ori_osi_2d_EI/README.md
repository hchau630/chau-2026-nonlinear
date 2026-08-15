| Figure | Filename |
|--------|----------|
| 2I | `resp_paper_indiv_0/numerical/dr_10_space-loss=0.50309455395-cell_type=PYR.pdf` |
| 2J | `resp_paper_indiv_0/numerical/dr_10_mean_nearby-loss=0.50309455395-cell_type=PYR.pdf` |
| 3A | `resp_compare_matrix_second_order-numerical_0/resp_space-loss=5.0309455395e-01-cell_type=PYR.pdf` |
| 3B | `resp_compare_matrix_second_order_decompose_0/resp_mean_nearby-loss=5.0309455395e-01-cell_type=PYR.pdf` |
| 3C | `resp_compare_matrix_second_order_decompose_2-2_0/resp_mean_nearby-loss=5.0309455395e-01-cell_type=PYR.pdf` |
| 3F (left) | `resp_paper_indiv_0/matrix_quasi_linear_approx/dv_10_space_ori-loss=0.50309455395-cell_type=PV.pdf` |
| 3F (center left) | `resp_dist/matrix_quasi_linear_approx/dv_nearby_all-loss=0.50309455395.pdf` |
| 3F (center right) | `resp_dist/matrix_quasi_linear_approx/Hdv2_nearby_all-loss=0.50309455395.pdf` |
| 3F (right) | From 3C |
| S2H | `resp_paper_indiv2_0/numerical/dr_10_space-loss=0.50309455395-cell_type=PV.pdf` |
| S2I | `resp_paper_indiv2_0/numerical/dr_10_mean_nearby-loss=0.50309455395-cell_type=PV.pdf` |
| S3A (left) | `resp_compare_matrix_second_order-numerical_0/resp_space-loss=5.0309455395e-01-cell_type=PYR.pdf` |
| S3A (right) | `resp_paper_indiv_0/matrix_second_order_approx/dr_10_mean_nearby-loss=5.0309455395e-01-cell_type=PYR.pdf` |
| S3B (left) | `resp_compare_matrix_quasi_linear-numerical_0/resp_space-loss=5.0309455395e-01-cell_type=PYR.pdf` |
| S3B (right) | `resp_paper_indiv2_0/matrix_quasi_linear_approx/dr_10_mean_nearby-loss=5.0309455395e-01-cell_type=PYR.pdf` |
| S3C (left) | `resp_paper_indiv_0/matrix_quasi_linear_approx/dv_10_space_ori-loss=0.50309455395-cell_type=PYR` |
| S3C (right) | `resp_paper_indiv_0/matrix_quasi_linear_approx/dv_10_space_ori-loss=0.50309455395-cell_type=PV` |

All run commands:
| Figures | Command(s) |
|---------|---------|
| 2I-J, 3A | <pre>`for MODE in numerical; do mkdir slurm/run/$MODE; MODE=$MODE PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-4 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/$MODE/%A_%a.out niarb run run.toml --linfo --ldebug niarb.integrate; done`</pre> |
| 3A-B, 3F, S3 | <pre>`for MODE in matrix_quasi_linear_approx; do mkdir slurm/run/$MODE; MODE=$MODE PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-4 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/$MODE/%A_%a.out niarb run run.toml --linfo; done`<br><br>`for MODE in matrix_second_order_approx matrix_second_order_approx_2-1 matrix_second_order_approx_2-2; do mkdir slurm/run/$MODE; MODE=$MODE PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-4 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/$MODE/%A_%a.out niarb run run.toml --linfo; done`</pre> |
| 3C | <pre>python modify_state_dict.py 0-4<br><br>for MODE in matrix_second_order_approx_2-2; do for FILENAME in linear_PYR linear_PV; do mkdir slurm/run/${MODE}_${FILENAME}; MODE=$MODE PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-4 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/${MODE}_${FILENAME}/%A_%a.out niarb run run_${FILENAME}.toml --linfo; done; done</pre> |

All plot commands:
| Figures | Command |
|---------|---------|
| 2I-J, S2G-H | <pre>for INDICES in 0; do for MODE in numerical; do for FILENAME in plot/resp_paper_indiv.toml plot/resp_paper_indiv2.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done</pre> |
| 3A, S3A (left) | <pre>for INDICES in 0; do for FILENAME in plot/resp_compare.toml; do MODES='["matrix_second_order_approx", "numerical"]' INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_matrix_second_order-numerical_$INDICES --linfo; done; done</pre> |
| S3B (left) | <pre>for INDICES in 0; do for FILENAME in plot/resp_compare.toml; do MODES='["matrix_quasi_linear_approx", "numerical"]' INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_matrix_quasi_linear-numerical_$INDICES --linfo; done; done</pre> |
| S3A (right), S3B (right), S3C, 3F (left) | <pre>for INDICES in 0; do for MODE in matrix_quasi_linear_approx matrix_second_order_approx; do for FILENAME in plot/resp_paper_indiv.toml plot/resp_paper_indiv2.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done</pre> |
| 3B | <pre>for INDICES in 0; do for FILENAME in plot/resp_compare.toml; do MODES='["matrix_second_order_approx", "matrix_quasi_linear_approx", "matrix_second_order_approx_2-1", "matrix_second_order_approx_2-2"]' INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_matrix_second_order_decompose_$INDICES --linfo; done; done</pre> |
| 3C | <pre>for INDICES in 0; do for FILENAME in plot/resp_compare.toml; do MODES='["matrix_second_order_approx_2-2_linear_PV", "matrix_second_order_approx_2-2_linear_PYR"]' INDICES=$INDICES sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_matrix_second_order_decompose_2-2_$INDICES --linfo; done; done</pre> |
| 3F | <pre>for INDICES in 0; do for MODE in matrix_quasi_linear_approx; do for FILENAME in plot/resp_dist.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done</pre> |
