# Examples
To run the following examples, first ensure that you have properly installed the `niarb` package following the instructions [here](https://github.com/hchau630/niarb#usage), and then change the working directory to `niarb/examples/ricciardi_ring`.
| Filename    | Figure (`figures/`) | Description                                                                              |
|-------------|---------------------|------------------------------------------------------------------------------------------|
| `fit.py`    | `fit.pdf`           | Uses the package's high-level functions to fit an E-I ring model to pseudo data. Note that you might need to rerun it a couple of times to get the nice figure shown. |
| `fit.toml`  | N/A                 | Fit 10 E-I ring models with command `niarb fit fit.toml`                                 |
| `dist.toml` | `*_dist.pdf`        | Plot distribution of parameters of models in `fits/` with command `niarb plot dist.toml --show` |
| `resp.toml` | `mean_response.pdf`, `responses.pdf` | Plot response of top-5 best-fit models in `fits/` with command `niarb plot resp.toml --show`    |