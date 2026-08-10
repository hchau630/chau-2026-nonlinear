import argparse
import sys
import math
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from skimage import measure
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from niarb import nn, random
from niarb.cell_type import CellType
from niarb.nn.modules.frame import ParameterFrame
from niarb.zero_crossing import find_n_crossings
from mpl_config import GRID_COLOR, GRID_WIDTH, get_sizes, get_cbar_configs, set_rcParams


def sample_W(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float | None = None,
    scale: float = 5.0,
) -> np.ndarray:
    """Sample random 3x3 W matrices satisfying x, y, z conditions.

    The W matrix satisfies the sign constraints that the first column is non-negative,
    the remaining two columns are non-positive, and W_{00} > 1. x = Tr(W), z = det(W),
    and y is the sum of all three principal minors of W of order 1.

    Args:
        x, y, z: Broadcastable ndarrays with broadcasted shape (*).
        a (optional): If not None, fix 1 - W_{11} + W_{10} * W_{21} / W_{20} to a.
        scale (optional): Scale factor for the random weights.

    Returns:
        ndarray with shape (*, 3, 3) of sampled W matrices.

    """
    x, y, z = np.broadcast_arrays(x, y, z)
    shape = x.shape

    if np.prod(shape) == 0:
        return np.empty((*shape, 3, 3))

    W = scale * np.abs(np.random.randn(3, 3, *shape))
    W[:, 1:] = -W[:, 1:]  # non-positive second and third columns
    W[0, 0] += 1.0  # W_{00} > 1

    # Solve for W_{20}, W_{21}, W_{22} such that x, y, z conditions are satisfied.
    # The equations are obtained with Mathematica.
    if a is None:
        W[2, 0] = -(
            (
                -(W[0, 0] ** 2 * W[0, 2] * W[1, 0])
                - W[0, 1] * W[0, 2] * W[1, 0] ** 2
                - W[0, 0] * W[0, 2] * W[1, 0] * W[1, 1]
                - W[0, 2] * W[1, 0] * W[1, 1] ** 2
                + W[0, 0] ** 3 * W[1, 2]
                + 2 * W[0, 0] * W[0, 1] * W[1, 0] * W[1, 2]
                + W[0, 1] * W[1, 0] * W[1, 1] * W[1, 2]
                + W[0, 0] * W[0, 2] * W[1, 0] * x
                + W[0, 2] * W[1, 0] * W[1, 1] * x
                - W[0, 0] ** 2 * W[1, 2] * x
                - W[0, 1] * W[1, 0] * W[1, 2] * x
                - W[0, 2] * W[1, 0] * y
                + W[0, 0] * W[1, 2] * y
                - W[1, 2] * z
            )
            / (
                -(W[0, 2] ** 2 * W[1, 0])
                + W[0, 0] * W[0, 2] * W[1, 2]
                - W[0, 2] * W[1, 1] * W[1, 2]
                + W[0, 1] * W[1, 2] ** 2
            )
        )
        W[2, 1] = -(
            (
                -(W[0, 0] * W[0, 1] * W[0, 2] * W[1, 0])
                - 2 * W[0, 1] * W[0, 2] * W[1, 0] * W[1, 1]
                - W[0, 2] * W[1, 1] ** 3
                + W[0, 0] ** 2 * W[0, 1] * W[1, 2]
                + W[0, 1] ** 2 * W[1, 0] * W[1, 2]
                + W[0, 0] * W[0, 1] * W[1, 1] * W[1, 2]
                + W[0, 1] * W[1, 1] ** 2 * W[1, 2]
                + W[0, 1] * W[0, 2] * W[1, 0] * x
                + W[0, 2] * W[1, 1] ** 2 * x
                - W[0, 0] * W[0, 1] * W[1, 2] * x
                - W[0, 1] * W[1, 1] * W[1, 2] * x
                - W[0, 2] * W[1, 1] * y
                + W[0, 1] * W[1, 2] * y
                + W[0, 2] * z
            )
            / (
                -(W[0, 2] ** 2 * W[1, 0])
                + W[0, 0] * W[0, 2] * W[1, 2]
                - W[0, 2] * W[1, 1] * W[1, 2]
                + W[0, 1] * W[1, 2] ** 2
            )
        )
        W[2, 2] = -W[0, 0] - W[1, 1] + x
    else:
        W[2, 0] = (
            W[1, 0]
            * (
                W[0, 0] * W[0, 1] * W[1, 0]
                - W[0, 0] ** 2 * (-1 + a + W[1, 1])
                - W[0, 0] * (-1 + a + W[1, 1]) * (W[1, 1] - x)
                - W[0, 1] * W[1, 0] * (-1 + a - W[1, 1] + x)
                - (-1 + a) * (W[1, 1] ** 2 - W[1, 1] * x + y)
                - z
            )
        ) / (
            (
                1
                + a**2
                - W[0, 1] * W[1, 0]
                + W[0, 0] * (-1 + W[1, 1])
                - W[1, 1]
                + a * (-2 + W[0, 0] + W[1, 1])
            )
            * W[1, 2]
        )
        W[2, 1] = -(
            (
                (-1 + a + W[1, 1])
                * (
                    -(W[0, 0] * W[0, 1] * W[1, 0])
                    + W[0, 0] ** 2 * (-1 + a + W[1, 1])
                    + W[0, 0] * (-1 + a + W[1, 1]) * (W[1, 1] - x)
                    + W[0, 1] * W[1, 0] * (-1 + a - W[1, 1] + x)
                    + (-1 + a) * (W[1, 1] ** 2 - W[1, 1] * x + y)
                    + z
                )
            )
            / (
                (
                    1
                    + a**2
                    - W[0, 1] * W[1, 0]
                    + W[0, 0] * (-1 + W[1, 1])
                    - W[1, 1]
                    + a * (-2 + W[0, 0] + W[1, 1])
                )
                * W[1, 2]
            )
        )
        W[2, 2] = -W[0, 0] - W[1, 1] + x
        W[0, 2] = -(
            (
                W[1, 2]
                * (
                    -(W[0, 0] ** 3 * (-1 + a + W[1, 1]))
                    - W[0, 0] * W[0, 1] * W[1, 0] * (-2 + 2 * a + W[1, 1] + x)
                    + W[0, 0] ** 2 * (W[0, 1] * W[1, 0] + (-1 + a + W[1, 1]) * x)
                    - W[0, 0] * (-1 + a + W[1, 1]) * y
                    + W[0, 1]
                    * W[1, 0]
                    * (W[0, 1] * W[1, 0] + W[1, 1] - a * W[1, 1] + (-1 + a) * x + y)
                    + (-1 + a + W[1, 1]) * z
                )
            )
            / (
                W[1, 0]
                * (
                    -(W[0, 0] * W[0, 1] * W[1, 0])
                    + W[0, 0] ** 2 * (-1 + a + W[1, 1])
                    + W[0, 0] * (-1 + a + W[1, 1]) * (W[1, 1] - x)
                    + W[0, 1] * W[1, 0] * (-1 + a - W[1, 1] + x)
                    + (-1 + a) * (W[1, 1] ** 2 - W[1, 1] * x + y)
                    + z
                )
            )
        )

    # move axis to get correct output shape
    W = np.moveaxis(W, (0, 1), (-2, -1))  # (*shape, 3, 3)

    # resample W for instances where the sign constraints are not satisfied
    mask = (W[..., 2, 0] < 0) | (W[..., 2, 1] > 0) | (W[..., 2, 2] > 0)
    if a is not None:
        mask |= W[..., 0, 2] > 0

    W[mask] = sample_W(x[mask], y[mask], z[mask], a=a, scale=scale)

    # check that the x, y, z conditions are satisfied
    kwargs = {"rtol": 1e-4, "atol": 1e-6}
    np.testing.assert_allclose(np.linalg.trace(W), x, **kwargs)
    np.testing.assert_allclose(
        np.linalg.det(W[..., :2, :2])
        + np.linalg.det(W[..., 1:, 1:])
        + np.linalg.det(W[..., ::2, ::2]),
        y,
        **kwargs,
    )
    np.testing.assert_allclose(np.linalg.det(W), z, **kwargs)
    if a is not None:
        np.testing.assert_allclose(
            1 - W[..., 1, 1] + W[..., 1, 0] * W[..., 2, 1] / W[..., 2, 0], a, **kwargs
        )

    return W


# def eigvals2x2(W, rho, as_complex=False):
#     S = np.diag([1 / rho, rho])
#     M = (np.eye(2) - W) @ np.linalg.inv(S)
#     tr, det = np.trace(M, axis1=-2, axis2=-1), np.linalg.det(M)
#     D = tr**2 - 4 * det
#     if (D < 0).any() or as_complex:
#         D = D.astype(complex)
#     l0 = 0.5 * (tr - np.sqrt(D))
#     l1 = 0.5 * (tr + np.sqrt(D))
#     return l0, l1


def response(
    d: int,
    W: ArrayLike,
    sigma: ArrayLike,
    r: ArrayLike,
    cell_types: Sequence[CellType | str],
    tau: Sequence[float],
    stability_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-cell excitatory neuron perturbation response.

    Args:
        d: Number of spatial dimensions.
        W: Connectivity matrices with with shape (*, n, n).
        sigma: Connection width matrices with shape (*, 1 | n, 1 | n).
        r: Distances with shape (N,) at which responses are calculated.
        tau: Time constants for all the cell types.

    Returns:
        Tuple of two ndarrays (is_stable, dr) where is_stable has shape (*) and dr has
        shape (*, n, N).

    """
    n = len(cell_types)
    W = torch.tensor(W, dtype=torch.float)
    sigma = torch.tensor(sigma, dtype=torch.float)
    r = torch.tensor(r, dtype=torch.float)

    if W.ndim < 2 or W.shape[-2:] != (n, n):
        raise ValueError(f"W must have shape (*, n, n),  but {W.shape=}.")

    if sigma.ndim < 2:
        raise ValueError("sigma must have at least 2 dimensions")

    if sigma.shape[-2:] == (n, n):
        symmetry = None
    elif sigma.shape[-2:] == (1, n):
        symmetry = "pre"
    elif sigma.shape[-2:] == (n, 1):
        symmetry = "post"
    elif sigma.shape[-2:] == (1, 1):
        symmetry = "full"
    else:
        raise ValueError(f"Invalid shape for sigma: {sigma.shape}")

    shape = torch.broadcast_shapes(W.shape[:-2], sigma.shape[:-2])
    W = W.broadcast_to(shape + W.shape[-2:])
    sigma = sigma.broadcast_to(shape + sigma.shape[-2:])

    if (W[..., :, 0] < 0).any():
        raise ValueError("Excitatory connections must be non-negative.")

    if (W[..., :, 1:] > 0).any():
        raise ValueError("Inhibitory connections must be non-positive.")

    W[..., :, 1:] = -W[..., :, 1:]  # could also just take abs()

    model = nn.V1(
        ["cell_type", "space"],
        cell_types=cell_types,
        init_stable=False,
        sigma_symmetry=symmetry,
        batch_shape=shape,
        tau=tau,
    )
    model.load_state_dict({"gW": W, "sigma": sigma}, strict=False)

    is_stable = model.spectral_summary(kind="J").abscissa < 0
    if stability_only:
        return is_stable.numpy(), None

    space = torch.tensor(np.r_[0, r], dtype=torch.float)  # (N + 1,)
    space = torch.stack(
        [space, *([torch.zeros_like(space)] * (d - 1))], dim=-1
    )  # (N + 1, d)
    dh = torch.zeros((n, space.shape[0]))  # (n, N + 1,)
    dh[0, 0] = 1.0  # Single-cell excitatory neuron perturbation

    # Note that dV does not matter since we are only interested in the response shape
    x = ParameterFrame(
        {
            "cell_type": torch.tensor([[0], [1], [2]]),  # (n, 1)
            "space": space[None, ...],  # (1, N + 1, d)
            "dV": torch.tensor([[1.0]]),  # (1, 1)
            "dh": dh,  # (n, N + 1)
        },
        ndim=2,
    )  # (n, N + 1)

    with torch.inference_mode():
        out = model(
            x, ndim=x.ndim, check_circulant=False, to_dataframe=False
        )  # (*, n, N + 1)
    dr = out["dr"]  # (*, n, N + 1)

    return is_stable.numpy(), dr[..., 1:].numpy()  # (*,), (*, n, N)


def plot_implicit_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    volume: np.ndarray,
    level: float,
    ax: Axes3D,
    **kwargs,
) -> Poly3DCollection:
    if x.ndim != 3 or y.ndim != 3 or z.ndim != 3 or volume.ndim != 3:
        raise ValueError("x, y, z, and volume must be 3D arrays.")

    if not (x.shape == y.shape == z.shape == volume.shape):
        raise ValueError("x, y, z, and volume must have the same shape.")

    # if (
    #     np.diff(x, n=2, axis=0).any()
    #     or np.diff(y, n=2, axis=1).any()
    #     or np.diff(z, n=2, axis=2).any()
    # ):
    #     raise ValueError("x, y, z must be uniformly spaced.")

    if x.shape[0] < 2 or y.shape[1] < 2 or z.shape[2] < 2:
        raise ValueError("x, y, z must have at least 2 points in each dimension.")

    spacing = [
        x[1, 0, 0] - x[0, 0, 0],
        y[0, 1, 0] - y[0, 0, 0],
        z[0, 0, 1] - z[0, 0, 0],
    ]

    verts, faces, _, _ = measure.marching_cubes(volume, level, spacing=spacing)
    verts -= np.array([x[0, 0, 0], y[0, 0, 0], z[0, 0, 0]])  # subtract origin

    return ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], **kwargs)


def plot_phase_diagram_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xr1: np.ndarray,
    xr2: np.ndarray,
    xr3: np.ndarray,
    is_stable: np.ndarray,
    discriminant: np.ndarray,
    is_pos: np.ndarray | None,
    col_wrap: int = 5,
):
    c = np.zeros_like(xr1, dtype=np.long)  # no crossings
    c[~np.isnan(xr1)] = 1  # has at least 1 crossing
    c[~np.isnan(xr2)] = 2  # has at least 2 crossings
    c[~np.isnan(xr3)] = 3  # has at least 3 crossings

    fig, axes = plt.subplots(int(math.ceil(y.shape[1] / col_wrap)), col_wrap)
    for i, ax in enumerate(axes.flat):
        assert isinstance(ax, Axes)
        ax.contourf(
            x[:, i, :],
            z[:, i, :],
            c[:, i, :],
            levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            colors=["C2", "C1", "C0", "C3"],
        )
        ax.contourf(
            x[:, i, :],
            z[:, i, :],
            is_stable[:, i, :],
            levels=[-0.5, 0.5],
            colors=["black"],
        )
        ax.contour(
            x[:, i, :],
            z[:, i, :],
            discriminant[:, i, :],
            levels=[0],
            colors=GRID_COLOR,
            linewidths=GRID_WIDTH,
        )
        if is_pos is not None:
            ax.contour(
                x[:, i, :],
                z[:, i, :],
                is_pos[:, i, :],
                levels=[0.5],
                colors="purple",
                linewidths=GRID_WIDTH,
            )
        ax.set_title(r"$\sum_i M_{ii} = %.3g$" % y[0, i, 0])

    return fig, axes


def plot_phase_diagram_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xr1: np.ndarray,
    xr2: np.ndarray,
    ax: Axes3D,
) -> Sequence[Poly3DCollection]:
    c = np.zeros_like(xr1, dtype=np.long)  # no crossings
    c[~np.isnan(xr1)] = 1  # has at least 1 crossing
    c[~np.isnan(xr2)] = 2  # has at least 2 crossings

    out = []
    for ci, color in zip([0.5, 1.5], ["C1", "C0"]):
        out.append(plot_implicit_surface(x, y, z, c, ci, ax, color=color))

    return out


def pos_condition(l: np.ndarray, a: float, s: float) -> np.ndarray:
    l = np.moveaxis(l, -1, 0)  # (3, *shape)
    # ensure l[2].imag >= l[1].imag
    l[1:, np.imag(l[1]) > np.imag(l[2])] = l[1:, np.imag(l[1]) > np.imag(l[2])].conj()
    s = 1 / s**2

    c0 = (l[0] - a) * (l[0] - s) / ((l[1] - l[0]) * (l[2] - l[0]))
    c2 = (l[2] - a) * (l[2] - s) / ((l[2] - l[1]) * (l[2] - l[0]))
    c02 = c0 / c2
    c12 = -((l[1] - a) * (l[1] - s) * (l[2] - l[0])) / (
        (l[2] - a) * (l[2] - s) * (l[1] - l[0])
    )
    cmax = np.maximum(1 + c12, 0)
    cmin = np.zeros_like(cmax)
    lt = -(np.sqrt(l[2]) - np.sqrt(l[0])) / (np.sqrt(l[1]) - np.sqrt(l[0]))
    smin = np.log(lt / c12) / (np.sqrt(l[2]) - np.sqrt(l[1]))
    cmin = np.where(c12 <= lt, 1 + c12, cmin)
    cmin = np.where(
        (lt < c12) & (c12 < 0),
        np.exp(-(np.sqrt(l[2]) - np.sqrt(l[0])) * smin)
        + c12 * np.exp(-(np.sqrt(l[1]) - np.sqrt(l[0])) * smin),
        cmin,
    )
    out_real = (c0 > 0) & ((-c02 < cmin) | (-c02 > cmax))

    c02 = np.abs(c02)
    argc2 = np.angle(c2)
    imsqrtl2 = np.imag(np.sqrt(l[2]))
    k = np.real(np.sqrt(l[2])) - np.sqrt(l[0])
    smin = (argc2 - np.atan(k / imsqrtl2)) / imsqrtl2
    smin = np.where(smin <= 0, smin + np.pi / imsqrtl2, smin)
    smin2 = smin + np.pi / imsqrtl2
    cmin = np.minimum(
        np.cos(argc2),
        np.minimum(
            np.cos(argc2 - imsqrtl2 * smin) * np.exp(-k * smin),
            np.cos(argc2 - imsqrtl2 * smin2) * np.exp(-k * smin2),
        ),
    )
    out_imag = (np.abs(c0.imag) < 1e-10) & (c0.real > 0) & (-c02 / 2 < cmin)

    is_real = np.all(l.imag == 0, axis=0)
    out = np.where(is_real, out_real, out_imag)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numerical", "--num", dest="numerical", action="store_true")
    parser.add_argument("--pos", type=str, choices=["necessary", "sufficient", "iff"])
    parser.add_argument("-x", type=float, nargs=2, default=(-5, 5))
    parser.add_argument("-y", type=float, nargs=2, default=(-5, 5))
    parser.add_argument("-z", type=float, nargs=2, default=(-5, 5))
    parser.add_argument("-a", type=float)
    parser.add_argument("--cell-types", type=str, nargs=3, default=["PYR", "PV", "SST"])
    parser.add_argument("--cell-type", "-c", type=str, default="PYR")
    parser.add_argument("-d", type=int, default=2)
    parser.add_argument("-N", type=int, default=100)
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--rmax", type=float, default=3000.0)
    parser.add_argument("--rN", type=int, default=3000)
    parser.add_argument("--tau", type=float, nargs=3, default=(1.0, 0.5, 1.0))
    parser.add_argument(
        "--spacing",
        "-s",
        choices=["log", "halflog", "neghalflog", "symlog", "linear", "halflinear"],
        default="log",
    )
    parser.add_argument("--levels", "-l", type=float, nargs=2, default=[-1.0, 1.0])
    parser.add_argument("--N-levels", "-n", type=int, default=9)
    parser.add_argument("--linthresh", "-t", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--out", "-o", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    x = np.linspace(*args.x, args.N)
    y = np.linspace(*args.y, args.N)
    z = np.linspace(*args.z, args.N)
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    # characteristic polynomial is t^3 - x * t^2 + y * t - z
    discriminant = x**2 * y**2 - 4 * y**3 - 4 * x**3 * z - 27 * z**2 + 18 * x * y * z

    # get the 3x3 W matrix
    if args.depth:
        sys.setrecursionlimit(args.depth)
    with random.set_seed(args.seed):
        W = sample_W(x, y, z, args.a)

    # necessary condition for stability
    # is_stable = 1 - x + y - z > 0
    eigvals = np.linalg.eigvals(np.eye(3) - W)  # (N, N, N, 3)
    is_stable_necessary = ((eigvals.imag != 0) | (eigvals.real > 0)).all(axis=-1)
    if args.a is not None and args.pos:
        eigvals = np.take_along_axis(
            eigvals, np.sqrt(eigvals).real.argsort(axis=-1), axis=-1
        )
        s0, s1 = 1.0, args.a
        if s1 < s0:
            s0, s1 = s1, s0

        if args.pos == "necessary":
            is_pos = (eigvals[..., 0].real < min(s0, s1)) & (eigvals[..., 0].imag == 0)
        elif args.pos == "sufficient":
            is_pos = (
                (eigvals[..., 0].real < s0)
                & (s0 < eigvals[..., 1].real)
                & (eigvals[..., 1].real < s1)
                & (s1 < eigvals[..., 2].real)
            )
        else:
            if args.cell_type != "SST":
                raise ValueError("pos='iff' is only implemented for cell_type=SST")
            is_pos = pos_condition(eigvals, args.a, 1.0)
    else:
        is_pos = None

    # compute various zero crossings
    if args.numerical:
        # first compute perturbation response
        sigma = np.array([[args.s0]])
        r = np.linspace(0, args.rmax, args.rN + 1)[1:]
        is_stable, _dr = response(args.d, W, sigma, r, args.cell_types, args.tau)
        if args.d > 1:
            # dr must be +inf at r = 0 theoretically
            r = np.insert(r, 0, 0.0)
            _dr = np.insert(_dr, 0, np.inf, axis=-1)
        dr = {ct: _dr[..., i, :] for i, ct in enumerate(args.cell_types)}

        # print(eigvals)
        # for ct in args.cell_types:
        #     plt.plot(r[200:500], dr[ct][0, 0, 0, 200:500], label=ct)
        #     # plt.plot(r[1:500], dr[ct][0, -1, -2, 1:500], label=ct)
        # plt.legend()
        # plt.gca().axhline(0, color=GRID_COLOR, linewidth=GRID_WIDTH)
        # plt.show()

        # mask = (
        #     (find_n_crossings(r, dr["PYR"]) < 50)[0]
        #     & (find_n_crossings(r, dr["SST"]) > 150)[0]
        #     & is_stable
        # )
        # print(np.stack([x[mask], y[mask], z[mask]]).T)
        # print(dr[args.cell_type][0, -1, -1])
        # print(dr[args.cell_type][0, -1, -2])
        # print(dr[args.cell_type][1, -1, -1])
        # print(dr[args.cell_type][1, -1, -2])
        xr1, xr2, xr3 = find_n_crossings(r, dr[args.cell_type], n=3)
        s = np.prod(sigma) ** (1 / sigma.size)
        xr1, xr2, xr3 = xr1 / s, xr2 / s, xr3 / s
        # print(xr1[0, -1, -1], xr2[0, -1, -1], xr3[0, -1, -1])
        # print(xr1[0, -1, -2], xr2[0, -1, -2], xr3[0, -1, -2])
        # print(xr1[1, -1, -1], xr2[1, -1, -1], xr3[1, -1, -1])
        # print(xr1[1, -1, -2], xr2[1, -1, -2], xr3[1, -1, -2])
        # print(xr1, xr2, xr3)
        # print(is_pos)
    else:
        raise NotImplementedError()

    # return

    # set some plotting defaults
    set_rcParams()
    # figsize, rect = get_sizes(1, 1, 1, 1)
    # fig = plt.figure(figsize=figsize)
    # ax = fig.add_axes(rect, projection="3d")
    # assert isinstance(ax, Axes3D)

    # plot the phase diagram
    # plot_phase_diagram_3d(x, y, z, xr1, xr2, ax)
    fig, axes = plot_phase_diagram_2d(
        x, y, z, xr1, xr2, xr3, is_stable_necessary, discriminant, is_pos
    )

    # plot instability region
    # plot_implicit_surface(x, y, z, is_stable, 0.5, ax, color="black")

    # plot analytic phase boundaries
    # ylim = ax.get_ylim()
    # _x = np.linspace(*ax.get_xlim())
    # ax.plot(_x, _x**2 / 4, color=GREY, linewidth=GRID_WIDTH)
    # _x = np.linspace(ax.get_xlim()[0], 0)
    # ax.plot(_x, np.zeros_like(_x), color=GREY, linewidth=GRID_WIDTH)
    # ax.set_ylim(*ylim)

    # plot region where wII > 0 or wEI > 0
    # make zorder a big number so that it is plotted on top of everything
    # ax.contourf(x, y, W[..., 0, 1], levels=[0, 1e8], colors=["purple"], zorder=100)
    # ax.contourf(x, y, W[..., 1, 1], levels=[0, 1e8], colors=["grey"], zorder=101)

    # nicer looking y-axis
    # ax.set_yticks([ylim[0], sum(ylim) / 2, ylim[1]])
    # ax.yaxis.set_major_formatter("{x:g}")

    # add labels
    # ax.set_xlabel(r"$\mathrm{tr}(\mathbf{W})$")
    # ax.set_ylabel(r"$\sum_i M_{ii}$")
    # ax.set_zlabel(r"$\mathrm{det}(\mathbf{W})$")
    fig.supxlabel(r"$\mathrm{tr}(\mathbf{W})$")
    fig.supylabel(r"$\mathrm{det}(\mathbf{W})$")

    # save figure
    if args.out:
        fig.savefig(args.out, metadata={"Subject": " ".join(["python"] + sys.argv)})

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
