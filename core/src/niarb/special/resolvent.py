import math
from collections.abc import Callable
from functools import partial
from numbers import Number
from typing import Concatenate

import torch
import torch.utils.checkpoint
from torch import Tensor
from torch.autograd.function import FunctionCtx

from niarb.linalg import is_diagonal
from niarb.utils import cast, take_along_dims

from .core import irkd, k0, kd, scaled_kd

__all__ = ["laplace", "laplace_beltrami", "laplace_r", "mixture"]


twopi = 2 * torch.pi
sqrthalfpi = math.sqrt(torch.pi / 2)


def _laplace_r_forward(d: int, s: Tensor, r: Tensor) -> Tensor:
    r"""Forward computation of (2\pi)^(d/2) * laplace_r without sigularity handling."""
    if d == 0:
        return r / s * kd(0, s * r)
    if d == 1:
        return sqrthalfpi * torch.exp(-s * r) / s
    if d == 2:
        return k0(s * r)
    if d == 3:
        return sqrthalfpi * torch.exp(-s * r) / r
    if d % 2 == 0:
        return (s / r) ** int(d / 2 - 1) * kd(d, s * r)
    return s ** int((d - 3) / 2) * r ** int((1 - d) / 2) * scaled_kd(d, s * r)


class LaplaceRNeg(torch.autograd.Function):
    r"""Autograd function for (2\pi)^(d/2) * laplace_r for d <= 0."""

    generate_vmap_rule = True

    @staticmethod
    def forward(d: int, s: Tensor, r: Tensor) -> Tensor:
        out = _laplace_r_forward(d, s, r)

        # (2\pi)^(d/2) lim_{r -> 0^+} G_d(r; l) for d <= 1. Calculated by Mathematica.
        limit = math.gamma(1 - d / 2) * s ** (d - 2) / 2 ** (d / 2) if d <= 1 else 0
        return torch.where(r != 0, out, limit)

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: tuple[int, Tensor, Tensor], _: Tensor):
        d, s, r = inputs
        ctx.d = d
        ctx.dtype_s = s.dtype
        ctx.dtype_r = r.dtype

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(s, r)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor | None) -> tuple[None, Tensor, Tensor]:
        out = [None, None, None]
        if grad is None:
            return tuple(out)

        d = ctx.d
        s, r = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            out[1] = (-s * LaplaceRNeg.apply(d - 2, s, r)).conj() * grad
            out[1] = out[1].to(ctx.dtype_s)
        if ctx.needs_input_grad[2]:
            out[2] = (-r * LaplaceRNeg.apply(d + 2, s, r)).conj() * grad
            out[2] = out[2].to(ctx.dtype_r)
        return tuple(out)


class LaplaceRPos(torch.autograd.Function):
    r"""Autograd function for (2\pi)^(d/2) * laplace_r for d >= 2."""

    generate_vmap_rule = True

    @staticmethod
    def forward(d: int, s: Tensor, r: Tensor, singularity: Tensor | Number) -> Tensor:
        out = _laplace_r_forward(d, s, r)
        return torch.where(r != 0, out, singularity)

    @staticmethod
    def setup_context(
        ctx: FunctionCtx, inputs: tuple[int, Tensor, Tensor, Tensor | Number], _: Tensor
    ):
        d, s, r, singularity = inputs
        ctx.d = d
        ctx.dtype_s = s.dtype
        ctx.dtype_r = r.dtype
        if isinstance(singularity, Tensor):
            ctx.dtype_sing = singularity.dtype

        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            ctx.save_for_backward(s, r)
        elif ctx.needs_input_grad[3]:
            ctx.save_for_backward(r)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad: Tensor | None
    ) -> tuple[None, Tensor | None, Tensor | None, Tensor | None]:
        out = [None, None, None, None]
        if grad is None:
            return tuple(out)

        d = ctx.d
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            s, r = ctx.saved_tensors
        else:
            r = ctx.saved_tensors[0]

        if ctx.needs_input_grad[1]:
            out[1] = (-s * LaplaceRPos.apply(d - 2, s, r, 0)).conj() * grad
            out[1] = out[1].to(ctx.dtype_s)
        if ctx.needs_input_grad[2]:
            out[2] = (-r * LaplaceRPos.apply(d + 2, s, r, 0)).conj() * grad
            out[2] = out[2].to(ctx.dtype_r)
        if ctx.needs_input_grad[3]:
            out[3] = torch.where(r == 0, grad, torch.zeros_like(r))
            out[3] = out[3].to(ctx.dtype_sing)
        return tuple(out)


# @profile
def laplace_r(
    d: int,
    l: Number | Tensor,
    r: Tensor,
    dr: float | Tensor = 0.0,
    is_sqrt: bool = False,
    validate_args: bool = True,
) -> Tensor:
    r"""Radial component of the kernel of the resolvent of the Laplacian.

    Radial component of the kernel of the resolvent of the
    laplace operator in d dimensions, i.e.
    $\bra{x}(l - \nabla^2)^{-1}\ket{y} = laplace(d, l, ||x - y||)$
    but with the modification that the output is finite when d > 1 and r == 0,
    due to the fact that the resolvent diverges when d > 1 and r == 0.
    Explicitly, the equation is given by
    $1 / (2\pi)^{d/2} (\sqrt{l} / r)^{d/2 - 1} K_{d/2 - 1}(\sqrt{l}r)$
    The resolvent of the Laplacian is only well-defined for
    $\lambda \in \mathbb{C} \ (-\infty, 0]$, but here we will just plug in negative
    $\lambda$ into the above expression directly.

    Args:
        d: Dimension of the space.
        l: Parameter $\lambda$.
        r: Tensor of distances, must be non-negative and real.
        dr (optional): Small non-negative number to avoid singularity at r = 0. Must be
            0.0 if d <= 1, as there is no singularity in these cases.
        is_sqrt (optional): Whether l is already the square root of the parameter.
            Useful when using laplace_r inside vmap, as it avoids a .item() call
            when l is a Tensor.
        validate_args (optional): Whether to validate the arguments.

    Returns:
        Tensor with shape broadcast(*, **)

    """
    if validate_args:
        if not isinstance(d, int):
            raise ValueError(f"d must be an integer, but {d=}.")

        if r.is_complex():
            raise TypeError(f"r must be real, but {r.dtype=}.")

        if r.requires_grad and (r == 0).any():
            raise NotImplementedError(
                "Backpropagtion with respect to r when r contains 0 is currently"
                " untested."
            )

        if not (r >= 0).all():
            raise ValueError(f"r must be non-negative, but {r.min()=}.")

        if d <= 1 and not (isinstance(dr, float) and dr == 0.0):
            raise ValueError(f"dr must be 0.0 when d <= 1, but {dr=}.")

    if is_sqrt:
        s = l
    elif isinstance(l, torch.Tensor) and not l.is_complex() and (l < 0).any():
        # cast real tensors with negative elements to corresponding complex dtype
        # so that the square root operation does not result in NaNs.
        s = l.to(l.dtype.to_complex()) ** 0.5
    else:
        # s = torch.sqrt(l.to(torch.cdouble)).to(torch.cfloat) if l.is_complex() else l**0.5
        s = l**0.5

    if d == 1:
        # 1D case needs to be handled separately
        return 1 / (2 * s) * torch.exp(-s * r)

    s = torch.as_tensor(s).to(r.device)
    prefactor = twopi ** (-d / 2)

    if d <= 0:
        return prefactor * LaplaceRNeg.apply(d, s, r)

    singularity = 0.0
    if not isinstance(dr, float) or dr != 0.0:
        dr = torch.as_tensor(dr)
        singularity = d * irkd(d, s * dr) / (dr**d * s**2)

    return prefactor * LaplaceRPos.apply(d, s, r, singularity)


def laplace(l: Number | Tensor, r: Tensor, **kwargs) -> Tensor:
    r"""Resolvent of the Laplacian, $R(\lambda; \nabla^2)$.

    Args:
        l: Parameter $\lambda$.
        r: Tensor with shape (*, d). Must be non-negative and real.

    Returns:
        Tensor with shape broadcast(l.shape, r.shape

    """
    return laplace_r(r.shape[-1], l, r.norm(dim=-1), **kwargs)


def laplace_beltrami(g, l, r, **kwargs):
    L, V = torch.linalg.eig(g)
    sqrt_g = V @ (L**0.5) @ torch.linalg.inv(V)
    r = sqrt_g @ r
    return torch.linalg.det(g) ** 0.5 * laplace_r(
        r.shape[-1], l, r.norm(dim=-1), **kwargs
    )


# @profile
def mixture(
    S: Tensor,
    U: Tensor,
    V: Tensor,
    R: Callable[Concatenate[Tensor, ...], Tensor],
    l: Number | Tensor,
    i: Tensor,
    j: Tensor,
    *args,
    checkpoint: bool = False,
    real: bool = False,
) -> Tensor:
    r"""Compute a mixture of resolvents.

    Computes $UP(l)R(L(l); D)P(l)^{-1}V$ where $P(l)L(l)P(l)^{-1}$ is the
    eigendecomposition of $S^{-1} + lVU$.

    Args:
        S: Tensor with shape (*S, m, m)
        U: Tensor with shape (*U, n, m)
        V: Tensor with shape (*V, m, n)
        R: Resolvent of D, which takes l as its first argument.
        l: Number or tensor with shape (*l, 1 | n, 1 | m)
        i: Tensor with dtype torch.long with shape i
        j: Tensor with dtype torch.long with shape j
        args: Remaining arguments to R, with shapes (*a, ?)
        checkpoint (optional): If True, uses activation checkpointing to save a lot of
          GPU memory if gradient is required.
        real (optional): If True, returns the real part of output. This is more memory-
          efficient than calling output.real.

    Returns:
        Tensor with shape SUVlija

    """
    m = S.shape[-1]

    Sinv = cast(torch.linalg.inv, S)  # (*S, m, m)
    if (
        (isinstance(l, Number) and l == 0) or (isinstance(l, Tensor) and (l == 0).all())
    ) and is_diagonal(Sinv):
        # sometimes linalg.eig cause backprop issues, so avoid it if possible
        L = Sinv.diagonal(dim1=-2, dim2=-1)  # (*S, m)
        P = torch.eye(m, dtype=U.dtype, device=U.device)  # (m, m)
    else:
        # cast to double for eigenvalue computation since near-degenerate eigenvalues
        # can cause linalg_eig_backward error if evaluated with single precision
        M = Sinv + V @ (l * U)
        L, P = cast(torch.linalg.eig, M.double())  # (*SUVl, m), (*SUVl, m, m)
        L = L.to(torch.cfloat) if L.is_complex() else L.float()
        P = P.to(torch.cfloat) if P.is_complex() else P.float()

    PinvV = cast(
        torch.matmul, cast(torch.linalg.inv, P), V.to(P.dtype)
    )  # (*SUVl, m, n)
    UP = cast(torch.matmul, U.to(P.dtype), P)  # (*SUVl, n, m)

    # reduce memory usage by half if real
    # TODO: Also reduce memory usage by half if complex by taking advantage of
    # conjugate symmetry in diagonalization of real matrices
    if L.isreal().all():
        L, PinvV, UP = L.real, PinvV.real, UP.real

    if (flag := is_diagonal(UP)) or is_diagonal(PinvV):
        # faster path for special case where either UP or PinvV is diagonal
        # (occurs when computing the weight matrix)
        A = PinvV if flag else UP  # (*SUVl, n, n)
        A = take_along_dims(A, i[..., None, None], j[..., None, None])  # SUVlij

        B, i = (UP, i) if flag else (PinvV, j)  # (*SUVl, n, n), i or j
        B = B.diagonal(dim1=-2, dim2=-1)  # (*SUVl, n)
        B = take_along_dims(B, i[..., None])  # SUVli or SUVlj
        L = take_along_dims(L, i[..., None])  # SUVli or SUVlj

        out = R(L, *args)  # SUVlia or SUVlja
        out = A * B * out  # SUVlija

    else:
        UP = take_along_dims(UP, i[..., None, None], dims=(-2,))  # (*SUVli, m)
        PinvV = take_along_dims(PinvV, j[..., None, None], dims=(-1,))  # (*SUVlj, m)
        func = partial(_func, real=real)
        if checkpoint:
            # Saves a ton of backprop GPU memory especially if m is large.
            # The naive implementation will store the output R(l[..., i], *args) in GPU
            # memory for every i to compute gradient, thus consuming a lot of memory.
            out = sum(
                torch.utils.checkpoint.checkpoint(
                    func,
                    UP[..., i],
                    PinvV[..., i],
                    L[..., i],
                    R,
                    *args,
                    use_reentrant=False,
                )
                for i in range(m)
            )  # SUVlija
        else:
            out = sum(
                func(UP[..., i], PinvV[..., i], L[..., i], R, *args) for i in range(m)
            )  # SUVlija
    return out


def _func(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    f: Callable[Concatenate[Tensor, ...], Tensor],
    *args,
    real: bool = False,
) -> Tensor:
    if not real:
        return f(c, *args) * a * b
    out = f(c, *args) * a
    if not (out.is_complex() and b.is_complex()):
        return out.real * b.real
    return (out.real * b.real) - (out.imag * b.imag)
