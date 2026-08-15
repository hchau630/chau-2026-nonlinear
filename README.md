# About
Code for "A nonlinear inhibition pathway underlying cortical responses to tuned holographic optogenetic perturbations"

> [!NOTE]
> Windows is currently not supported. Please open an issue if you would like to run this code on Windows.

# To reproduce figures...
1. Clone or fork this repository.
2. [Install](https://docs.astral.sh/uv/getting-started/installation/) [uv](https://docs.astral.sh/uv/) if it isn't already installed.
3. Create a virtual environment in `.venv` with the packages specified in `uv.lock` installed by running the command
   ```
   uv sync --locked --extra {CUDA}
   ```
   in this directory, where `{CUDA}` should be replaced by either `cpu`, `cu124`, or `cu128`. Choose `cpu` if you do not have a GPU or you are on macOS. Choose `cu124` if you have a GPU with [compute capability](https://developer.nvidia.com/cuda/gpus) (CC) <= 9.0. Choose `cu128` otherwise (this installs a version of PyTorch that is newer than the one used by me, so reproducibility may not be fully guaranteed).
4. Activate the virtual environment by running
   ```
   source .venv/bin/activate
   ```
   in this directory.
5. Navigate to the directory `paper` and follow the instructions in `paper/README.md`.
