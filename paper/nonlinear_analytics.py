import argparse
from pathlib import Path
import logging
import pprint
from itertools import product
import math
from typing import Iterator, Iterable, Sequence

import torch
from torch import Tensor
import pandas as pd
import xarray as xr
from scipy import special, integrate
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from niarb import nn, io
from niarb.nn.modules.v1 import UV_decomposition
from niarb.nn.modules.pipeline import Scaler
from niarb.special.resolvent import laplace_r
from niarb.cell_type import CellType

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("conf", type=Path)
    parser.add_argument("--perturbed", action="store_true")
    parser.add_argument("--EI", action="store_true")
    parser.add_argument("-r", type=float, nargs="+", default=[22.5, 400, 5.0])
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--log-level", "--ll", dest="log_level", default="INFO")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logging.getLogger("matplotlib").setLevel("INFO")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    conf = io.load_config(args.conf)
    pipeline = conf["pipeline"]
    state_dict = conf["state_dict"]
    space_extent = conf["space_extent"]
    window = conf["window"]
    perturbations_extent = conf["perturbations_extent"]
    kappas = conf["kappas"]
    K = conf["K"]
    N = conf["N"]

    N = N * math.prod(window) / math.prod(space_extent)
    space_extent = window

    d = len(space_extent)

    # initialize pipeline
    if isinstance(pipeline, dict):
        pipeline = nn.Pipeline(**pipeline)
    logger.debug(f"pipeline:\n{pipeline}")

    # load state dict
    if isinstance(state_dict, (str, Path)):
        state_dict = torch.load(state_dict, map_location="cpu")
    logger.debug(f"state dict:\n{pprint.pformat(state_dict)}")

    nn.load_state_dict(pipeline, state_dict)

    V = math.prod(space_extent)
    Lmean, tLmean, Lcoef, lamb, prob, dV, dh, g, dg = compute_vars(pipeline, V, N)
    logger.debug(f"Lmean:\n{Lmean}")
    # logger.debug(f"Lcoef:\n{Lcoef}")
    logger.debug(f"lamb:\n{lamb}")

    r = torch.arange(*args.r)
    θ = torch.tensor([0, torch.pi / 2, torch.pi])
    rr, θθ = r[:, None], θ[None, :]
    Lspace, Lspace_conv = compute_spatial_kernel(d, Lcoef, lamb, rr)  # (2, n, n, *)
    L = compute_linear_response(Lspace, θθ)  # (n, n, *)

    Lspace = dh * dV[None, None, :, None, None] / (2 * torch.pi) * Lspace
    Lspace_conv = dh * dV[None, None, :, None, None] / (2 * torch.pi) * Lspace_conv
    L = dh * dV[None, :, None, None] * L

    # dist_space = pdf_2d(*perturbations_extent, rr)  # (*)
    # dist_ori = pdf_vonmises(perturbations_kappa, θθ)  # (*)
    # dist = dist_space * dist_ori  # (*)

    # _r = torch.arange(0, *args.r[1:])
    # plt.plot(_r, pdf_2d(*perturbations_extent, _r))
    # plt.xlabel("Distance (μm)")
    # plt.ylabel("Probability density")
    # plt.show()

    gshape = L.shape[:2]
    fig, axes = plt.subplots(*gshape, figsize=(2.5 * gshape[1], 2 * gshape[0]))
    cell_types = [ct.name for ct in pipeline.model.cell_types]
    for a, b in ndindex(axes.shape):
        ax = axes[a, b]
        for i in range(len(θ)):
            #     ax.plot(
            #         r, (L * dist)[a, b, :, i], label=f"{θ[i].item() * 90 / torch.pi:.0f}°"
            #     )
            ax.plot(r, L[a, b, :, i], label=f"{θ[i].item() * 90 / torch.pi:.0f}°")
        # ax.plot(
        #     r, (Lspace * dist_space)[0, a, b], label=r"$\tilde{L}_{0 \alpha \beta}(r)$"
        # )
        ax.plot(r, Lspace[0, a, b], label=r"$\tilde{L}_{0 \alpha \beta}(r)$")
        # ax.plot(
        #     r, (Lspace * dist_space)[1, a, b], label=r"$\tilde{L}_{1 \alpha \beta}(r)$"
        # )
        ax.plot(r, Lspace[1, a, b], label=r"$\tilde{L}_{1 \alpha \beta}(r)$")
        ax.axhline(
            0, color=rcParams["grid.color"], linewidth=rcParams["grid.linewidth"]
        )
        ax.set_title(f"{cell_types[b]} → {cell_types[a]}")
        # ax.set_xlabel("Distance (μm)")
        # ax.set_ylabel("Linear response")
    fig.supxlabel("Distance (μm)")
    fig.supylabel("Linear response")
    plt.legend(title="Δθ")
    fig.tight_layout()
    if args.out:
        plt.savefig(args.out / "response_space.pdf", bbox_inches="tight")
    plt.show() if args.show else plt.clf()

    # fig, axes = plt.subplots(*L.shape[:2], figsize=(5, 4))
    # cell_types = ["E", "I"]
    # for a, b in ndindex(axes.shape):
    #     ax = axes[a, b]
    #     ax.plot(r, Lspace_conv[1, a, b])
    #     ax.axhline(
    #         0, color=rcParams["grid.color"], linewidth=rcParams["grid.linewidth"]
    #     )
    #     ax.set_title(f"{cell_types[b]} → {cell_types[a]}")
    #     ax.set_xlabel("Distance (μm)")
    #     ax.set_ylabel(r"$\tilde{L}^{(2)}_{1 \alpha \beta}(r)$")
    # plt.tight_layout()
    # plt.show()

    for a, b in ndindex(L.shape[:2]):
        plt.plot(
            r,
            # (Lspace_conv * dist_space)[1, a, b],
            Lspace_conv[1, a, b],
            label=f"{cell_types[b]} → {cell_types[a]}",
        )
    plt.axhline(0, color=rcParams["grid.color"], linewidth=rcParams["grid.linewidth"])
    plt.xlabel("Distance (μm)")
    plt.ylabel(r"$\tilde{L}^{(2)}_{1 \alpha \beta}(r)$")
    plt.legend()
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out / "conv_response_space.pdf", bbox_inches="tight")
    plt.show() if args.show else plt.clf()

    prefactor0 = K / (prob * N)  # (n,)
    prefactor1 = dh * prefactor0  # (n,)
    prefactor21 = dh * dg / g * V * prefactor0**2  # (n,)
    prefactor22 = dh**2 / g**2 * V * prefactor0**2  # (n,)
    print(f"{dh=}, {g=}, {dg=}")

    mean_Lspace, mean_Lspace_conv = compute_expected_spatial_kernel(
        d, *perturbations_extent, Lcoef, lamb
    )  # (2, n, n)

    df = {}
    for kappa_cat, kappa in kappas.items():
        mean_Lori = (special.i1(kappa) / special.i0(kappa)) ** 2
        mean1 = prefactor1 * (Lmean if args.perturbed else tLmean)  # (n, n)
        mean21 = (
            prefactor21
            * (Lmean if args.perturbed else tLmean)
            * (mean_Lspace[0].diag() + 2 * mean_Lspace[1].diag() * mean_Lori)
        )  # (n, n)
        mean22 = prefactor22 * (
            Lmean @ (mean_Lspace_conv[0] + 2 * mean_Lspace_conv[1] * mean_Lori)
        )  # (n, n)
        if args.EI:
            mean22E = (
                prefactor22
                * Lmean[:, :1]
                * (mean_Lspace_conv[0] + 2 * mean_Lspace_conv[1] * mean_Lori)[:1, :]
            )  # (n, n)
            mean22I = (
                prefactor22
                * Lmean[:, 1:]
                * (mean_Lspace_conv[0] + 2 * mean_Lspace_conv[1] * mean_Lori)[1:, :]
            )  # (n, n)
            torch.testing.assert_close(mean22, mean22E + mean22I)
        print(kappa_cat)
        print("2,1")
        print(mean_Lspace[0].diag())
        print(mean_Lspace[1].diag() * mean_Lori)
        print(mean_Lspace[0].diag() + 2 * mean_Lspace[1].diag() * mean_Lori)
        print(Lmean if args.perturbed else tLmean)
        print(
            (Lmean if args.perturbed else tLmean)[0, 0]
            * (mean_Lspace[0].diag() + 2 * mean_Lspace[1].diag() * mean_Lori)[0]
        )
        print(mean21)
        # print("2,2")
        # print(mean_Lspace_conv[0])
        # print(mean_Lspace_conv[1] * mean_Lori)
        # print(mean_Lspace_conv[0] + 2 * mean_Lspace_conv[1] * mean_Lori)
        # print(Lmean)
        # print(
        #     Lmean[0, 0]
        #     * (mean_Lspace_conv[0] + 2 * mean_Lspace_conv[1] * mean_Lori)[0, 0]
        # )
        # print(mean22E)
        # print(mean22I)
        # print(mean22)
        # print(prefactor21, prefactor22)

        if args.EI:
            mean = torch.stack([mean1, mean21, mean22E, mean22I], dim=0)  # (4, n, n)
            order = ["1", "2,1", "2,2,E", "2,2,I", "Total"]
        else:
            mean = torch.stack([mean1, mean21, mean22], dim=0)  # (3, n, n)
            order = ["1", "2,1", "2,2", "Total"]
        mean = torch.cat([mean, mean.sum(dim=0, keepdim=True)], dim=0)  # (4, n, n)
        mean = xr.DataArray(
            mean.numpy(),
            coords={
                "Order": order,
                "Measured": cell_types,
                "Perturbed": cell_types,
            },
        )
        mean = mean.to_dataframe(name="Mean response").reset_index()
        df[kappa_cat] = mean
    df = (
        pd.concat(df)
        .reset_index(level=0, names="Ensemble tuning")
        .reset_index(drop=True)
    )

    g = sns.catplot(
        df,
        kind="bar",
        x="Order",
        y="Mean response",
        hue="Ensemble tuning",
        col="Perturbed",
        row="Measured",
        height=2,
        aspect=1.25,
        sharey=False,
    )
    g.tight_layout()
    if args.out:
        g.savefig(args.out / "mean_response.pdf")
    plt.show() if args.show else plt.clf()


def ndindex(shape: Iterable[int]) -> Iterator[Sequence[int]]:
    yield from product(*map(range, shape))


def laplace_r_conv(d, l0, l1, r):
    l0, l1 = torch.as_tensor(l0), torch.as_tensor(l1)
    l0, l1, r = torch.broadcast_tensors(l0, l1, r)
    return torch.where(
        l0 == l1,
        laplace_r(d - 2, l0, r) / (4 * torch.pi),
        (laplace_r(d, l0, r) - laplace_r(d, l1, r)) / (l1 - l0),
    )


def compute_vars(
    pipeline: nn.Pipeline, vol: float, N: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    model = pipeline.model
    scaler = pipeline.scaler
    assert isinstance(model, nn.V1)
    assert isinstance(scaler, Scaler)
    assert isinstance(model.sign, Tensor)
    # assert set(model.variables) == {"cell_type", "space", "ori"}
    # assert set(model.cell_types) == {CellType.PYR, CellType.PV}
    n = model.n

    W = model.gW * model.sign[..., None, :]
    W = torch.stack([W, W * model.kappa])  # (2, n, n)
    Lmean = torch.linalg.inv(torch.eye(n) - W[0])  # (n, n)
    tLmean = Lmean - torch.eye(n)

    U, V, S = UV_decomposition(W, model.sigma)  # (2, n, m), (2, m, n), (m, m)
    m = U.shape[-1]
    lamb, P = torch.linalg.eig(torch.linalg.inv(S) - V @ U)  # (2, m), (2, m, m)
    UP = U.to(P.dtype) @ P  # (2, n, m)
    PinvV = torch.linalg.inv(P) @ V.to(P.dtype)  # (2, m, n)
    Lcoef = torch.empty((2, n, n, m), dtype=P.dtype)
    for i, a, b, p in ndindex(Lcoef.shape):
        Lcoef[i, a, b, p] = UP[i, a, p] * PinvV[i, p, b]

    # prob = torch.tensor([CellType.PYR.prob, 1 - CellType.PYR.prob])
    total_E_prob = sum(ct.prob for ct in CellType if ct.sign == 1)
    total_I_prob = sum(ct.prob for ct in CellType if ct.sign == -1)
    subset_E_prob = sum(ct.prob for ct in model.cell_types if ct.sign == 1)
    subset_I_prob = sum(ct.prob for ct in model.cell_types if ct.sign == -1)
    E_ratio = total_E_prob / subset_E_prob if subset_E_prob > 0 else float("nan")
    I_ratio = total_I_prob / subset_I_prob if subset_I_prob > 0 else float("nan")
    prob = [
        ct.prob * E_ratio if ct.sign == 1 else ct.prob * I_ratio
        for ct in model.cell_types
    ]
    prob = torch.tensor(prob)
    prob = prob / prob.sum()
    dV = vol * 2 * torch.pi / (prob * N)

    dh = scaler.scale
    g = model.gain()
    dg = model.gain(dh) - g
    # dh = model.f(model.vf + dh) - model.f(model.vf)
    Lmean, tLmean, Lcoef, lamb = (
        Lmean.detach(),
        tLmean.detach(),
        Lcoef.detach(),
        lamb.detach(),
    )
    dh, g, dg = dh.detach(), g.detach(), dg.detach()
    return Lmean, tLmean, Lcoef, lamb, prob, dV, dh, g, dg


@torch.inference_mode()
def compute_spatial_kernel(
    d: int, Lcoef: Tensor, lamb: Tensor, r: Tensor
) -> tuple[Tensor, Tensor]:
    assert Lcoef.ndim == 4 and lamb.ndim == 2
    assert Lcoef.shape[0] == lamb.shape[0] == 2
    assert Lcoef.shape[1] == Lcoef.shape[2]
    assert Lcoef.shape[-1] == lamb.shape[-1]
    nones = (None,) * r.ndim

    Lcoef = Lcoef[(...,) + nones]  # (2, n, n, m, *)
    lamb = lamb[(slice(None), None, None, slice(None)) + nones]  # (2, 1, 1, m, *)

    Lspace = (Lcoef * laplace_r(d, lamb, r)).sum(dim=3).real  # (2, n, n, *)

    # (2, n, n, m, m, *)
    Lcoef_prod = Lcoef[:, :, :, :, None, ...] * Lcoef[:, :, :, None, :, ...]
    lamb0, lamb1 = lamb[:, :, :, :, None, ...], lamb[:, :, :, None, :, ...]
    # (2, n, n, *)
    Lspace_conv = (Lcoef_prod * laplace_r_conv(d, lamb0, lamb1, r)).sum(dim=(3, 4)).real

    return Lspace, Lspace_conv


def compute_spatial_kernel_scalar(
    d: int, Lcoef: Tensor, lamb: Tensor, r: float
) -> float:
    assert Lcoef.ndim == 1 and lamb.ndim == 1 and Lcoef.shape == lamb.shape
    r = torch.tensor(r)

    return (Lcoef * laplace_r(d, lamb, r)).sum().real.item()


def compute_spatial_kernel_conv_scalar(
    d: int, Lcoef: Tensor, lamb: Tensor, r: float
) -> float:
    assert Lcoef.ndim == 1 and lamb.ndim == 1 and Lcoef.shape == lamb.shape
    r = torch.tensor(r)
    Lcoef_prod = Lcoef[:, None] * Lcoef[None, :]
    lamb0, lamb1 = lamb[:, None], lamb[None, :]

    return (Lcoef_prod * laplace_r_conv(d, lamb0, lamb1, r)).sum().real.item()


def compute_expected_spatial_kernel_scalar(
    d: int, a: float, b: float, Lcoef: Tensor, lamb: Tensor
) -> tuple[float, float]:
    def integrand1(r):
        return compute_spatial_kernel_scalar(d, Lcoef, lamb, r) * pdf_2d(
            a, b, torch.tensor(r)
        )

    out1 = integrate.quad(integrand1, 0, math.sqrt(a**2 + b**2))[0]

    def integrand2(r):
        return compute_spatial_kernel_conv_scalar(d, Lcoef, lamb, r) * pdf_2d(
            a, b, torch.tensor(r)
        )

    out2 = integrate.quad(integrand2, 0, math.sqrt(a**2 + b**2))[0]
    return out1, out2


@torch.inference_mode()
def compute_expected_spatial_kernel(
    d: int, a: float, b: float, Lcoef: Tensor, lamb: Tensor
) -> tuple[Tensor, Tensor]:
    assert Lcoef.ndim == 4 and lamb.ndim == 2
    assert Lcoef.shape[0] == lamb.shape[0] == 2
    assert Lcoef.shape[1] == Lcoef.shape[2]
    assert Lcoef.shape[-1] == lamb.shape[-1]
    lamb = lamb[:, None, None, :].broadcast_to(Lcoef.shape)
    shape = Lcoef.shape[:-1]
    out1, out2 = torch.empty(shape), torch.empty(shape)

    for idx in ndindex(shape):
        out1[idx], out2[idx] = compute_expected_spatial_kernel_scalar(
            d, a, b, Lcoef[idx], lamb[idx]
        )
    return out1, out2


@torch.inference_mode()
def compute_linear_response(Lspace: Tensor, dtheta: Tensor) -> Tensor:
    assert Lspace.ndim >= 3
    assert Lspace.shape[0] == 2
    assert Lspace.shape[1] == Lspace.shape[2]
    torch.broadcast_shapes(Lspace.shape[3:], dtheta.shape)  # ensure broadcastable

    return (Lspace[0] + 2 * Lspace[1] * torch.cos(dtheta)) / (2 * torch.pi)


def pdf_2d(a: float, b: float, r: Tensor) -> Tensor:
    """Probability density function of the distance between two points in a rectangle.

    For the derivation of this formula, see my answer on Math Stack Exchange:
    https://math.stackexchange.com/questions/798655/square-line-picking/5037020#5037020

    Args:
        a: Length of the rectangle.
        b: Width of the rectangle.
        r: Distance between two points.

    Returns:
        Probability density function of the distance.

    """
    if a > b:
        a, b = b, a

    prefactor = 4 * r / (a**2 * b**2)
    case0 = torch.pi * a * b / 2 - (a + b) * r + r**2 / 2
    case1 = (
        a * b * (torch.pi / 2 - torch.arccos(a / r))
        - b * (r - torch.sqrt(r**2 - a**2))
        - a**2 / 2
    )
    case2 = (
        a * b * (torch.arcsin(b / r) - torch.arccos(a / r))
        - b * (b / 2 - torch.sqrt(r**2 - a**2))
        - a * (a / 2 - torch.sqrt(r**2 - b**2))
        - r**2 / 2
    )

    mask0 = (r >= 0) & (r < a)
    mask1 = (r >= a) & (r < b)
    mask2 = (r >= b) & (r < math.sqrt(a**2 + b**2))
    out = torch.zeros_like(r)
    out[mask0] = case0[mask0]
    out[mask1] = case1[mask1]
    out[mask2] = case2[mask2]
    return prefactor * out


def pdf_vonmises(κ: float, θ: Tensor) -> Tensor:
    """Probability density function of sum of two i.i.d. von Mises random variables.

    Args:
        κ: Concentration parameter.
        θ: Angle.

    Returns:
        Probability density function of the angle.

    """
    return special.i0(2 * κ * torch.cos(θ / 2)) / (2 * torch.pi * special.i0(κ) ** 2)


if __name__ == "__main__":
    main()
