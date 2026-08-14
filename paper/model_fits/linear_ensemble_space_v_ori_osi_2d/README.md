| Figure | Filename |
|--------|----------|
| 2G | `resp_paper_indiv_0/numerical/dr_10_space-loss=0.53821074963-cell_type=PYR.pdf` |
| 2H | `resp_paper_indiv_0/numerical/dr_10_mean_nearby-loss=0.53821074963-cell_type=PYR.pdf` |

All run commands:
| Figures | Command |
|---------|---------|
| 2G-H | `for MODE in numerical; do mkdir slurm/run/$MODE; MODE=$MODE PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-4 -c 8 --mem-per-cpu=4gb --time 20:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/$MODE/%A_%a.out niarb run run.toml --linfo --ldebug niarb.integrate; done` |

All plot commands:
| Figures | Command |
|---------|---------|
| 2G-H | `for INDICES in 0; do for MODE in numerical; do for FILENAME in plot/resp_paper_indiv.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:10:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done` |
