# Setup

To reproduce figures, you need to first download `linear_ensemble_space_v_ori_osi_2d.tar.gz` from https://doi.org/10.5281/zenodo.21987736 and extract its content into this directory. You should get a `runs` directory.

# Table of figures

| Figure | Filename (in directory `figures`) |
| ------ | --------------------------------- |
| 2G | `resp_paper_indiv_0/numerical/dr_10_space-loss=0.53821074963-cell_type=PYR.pdf` |
| 2H | `resp_paper_indiv_0/numerical/dr_10_mean_nearby-loss=0.53821074963-cell_type=PYR.pdf` |

# Commands for reproducing figures in directory `figures`

These commands were run on a SLURM cluster. If you're not on a SLURM cluster, remove `sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out` from the commands.

| Figures | Command (run in this directory) |
| ------- | ------------------------------- |
| 2G-H | <pre>`for INDICES in 0; do for MODE in numerical; do for FILENAME in plot/resp_paper_indiv.toml; do INDICES=$INDICES MODE=$MODE sbatch -c 8 --mem-per-cpu=8gb --time 00:30:00 --output slurm/plot/%A.out niarb plot $FILENAME -o figures/$(basename $FILENAME .toml)_$INDICES/$MODE --linfo; done; done; done`</pre> |

# Commands for computing model perturbation responses in directory `runs`

These commands were run on a SLURM cluster with A40 GPUs. If you're on a SLURM cluster, modify the `--gres` option if necessary. Otherwise, remove `mkdir -p slurm/run/$MODE;` and replace `sbatch --array 0 -c 8 --mem-per-cpu=4gb --time 20:00:00 --gres=gpu:a40:1 --output slurm/run/$MODE/%A_%a.out` with `SLURM_ARRAY_TASK_ID=0`. GPUs with < 48 GB memory will likely not be able to run the commands.

| Figures | Command (run in this directory) |
| ------- | ------------------------------- |
| 2G-H | <pre>`for MODE in numerical; do mkdir -p slurm/run/$MODE; MODE=$MODE PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0 -c 8 --mem-per-cpu=4gb --time 20:00:00 --gres=gpu:a40:1 --output slurm/run/$MODE/%A_%a.out niarb run run.toml --linfo --ldebug niarb.integrate; done`</pre> |

# Command for producing model fits in directory `fits`

This command was run on a SLURM cluster with A40 GPUs. If you're on a SLURM cluster, modify the `--gres` option if necessary. Otherwise, remove `sbatch --array 0-99 -c 8 --mem-per-cpu=4gb --time 2:00:00 --gres=gpu:a40:1 --output slurm/fit/%A_%a.out` and replace `-N 5` with `-N 500`. GPUs with < 48 GB memory will likely not be able to run the command. Run the command in this directory.

```
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-99 -c 8 --mem-per-cpu=4gb --time 2:00:00 --gres=gpu:a40:1 --output slurm/fit/%A_%a.out niarb fit fit.toml -N 5 -o fits --linfo --ignore-errors
```

