![tests workflow status](https://github.com/hchau630/niarb/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/hchau630/niarb/graph/badge.svg?token=PIC5I838RL)](https://codecov.io/gh/hchau630/niarb)

# About
Code for fitting firing rate models of mouse V1 used in my projects.

# Development
There are two ways to set things up. The preferred method is to use [uv](https://docs.astral.sh/uv/):
1. Clone or fork this repository.
2. [Install](https://docs.astral.sh/uv/getting-started/installation/) uv if it isn't already installed.
3. Create a virtual environment in `.venv` with the packages specified in `uv.lock` installed by running the command
   ```
   uv sync --locked --extra pt{TORCH}{CUDA}
   ```
   in the directory containing this file. `{TORCH}` should be replaced by either `26`, `27`, `28`, `29`, `210`, `211`, or `212`, and `{CUDA}` should be replaced by either `cpu`, `cu124`, `cu126`, `cu128`, or `cu130`. The valid combinations of `{TORCH}` and `{CUDA}` are listed under `[project.optional-dependencies]` in `pyproject.toml`. For example, `--extra pt26cu124` installs `torch==2.6.0+cu124` along with a build of `torch-bessel` compiled for this PyTorch version. 
4. Activate the virtual environment (e.g. `source .venv/bin/activate`) or prepend `uv run --no-sync` to every subsequent script command (e.g. `uv run --no-sync pytest`).

Alternatively, if you don't want to use uv, you can clone or fork this repository, create and activate an empty virtual environment, then install packages with
```
pip install torch=={TORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/{CUDA} -f https://torch-bessel.s3.us-east-2.amazonaws.com/whl/torch-{TORCH_VERSION}%2B{CUDA}.html -e .
```
where `{TORCH_VERSION}` should be replaced by either `2.6.0`, `2.7.1`, `2.8.0`, `2.9.1`, `2.10.0`, `2.11.0`, or `2.12.1`, and `{CUDA}` is the same as before. The downside is that the packages installed will be different from those specified in `uv.lock`, so your development environment will be different from mine.

# Installation
If you just want to install the package instead of developing it (which I don't really recommend, since you'll likely find the package inadequate for your needs in its current form), you can do (command untested)
```
pip install torch=={TORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/{CUDA} -f https://torch-bessel.s3.us-east-2.amazonaws.com/whl/torch-{TORCH_VERSION}%2B{CUDA}.html git+https://github.com/hchau630/niarb
```

# Usage
This package comes with a command line interface (CLI) for fitting models and plotting them with only a configuration file. To fit models, simply write a configuration file, e.g. `fit.toml` (the configuration can be written in JSON or TOML), then do `niarb fit fit.toml -o fits` to fit models according to the specificaation of your configuration file and outputs the results to a directory called `fits`. Similarly, you can use the command `niarb plot {YOUR_CONFIG}` to create various plots, such as distribution of fitted model parameters and the perturbation response of fitted models. For more details on how to write the configuration files, please refer to the examples located in the directory `examples/`.

While there are tons of configuration options available, its primary aim is to provide a quick and simple way to jumpstart your own project rather than cover all possible use cases. For more advanced use cases, it is recommended to use the command line interface code under `src/niarb/cli` as a starting template for your own code.

# Testing
To run tests, simply run the command `pytest` in the project root directory.

# Benchmark
This package is benchmarked with the `asv` package, with the benchmarks located at `benchmarks/`. Benchmark results can be viewed at https://hchau630.github.io/niarb-old.
