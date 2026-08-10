import math
from collections.abc import Callable, Hashable
from numbers import Integral, Number

import torch
from torch import Tensor

from niarb.tensors import categorical

__all__ = [
    "Add",
    "Compose",
    "Identity",
    "Match",
    "Mul",
    "Pow",
    "Prod",
    "Sub",
    "Sum",
    "TrueDiv",
]


class FunctionMixin:
    """
    A Mixin class that adds algebraic operations to torch.nn.Module
    """

    def __add__(self, g):
        if g == 0:
            return self
        return Add(self, g)

    def __radd__(self, g):
        if g == 0:
            return self
        return Add(self, g)

    def __sub__(self, g):
        if g == 0:
            return self
        return Sub(self, g)

    def __rsub__(self, g):
        if g == 0:
            return self
        return Sub(self, g)

    def __mul__(self, g):
        if g == 1:
            return self
        return Mul(self, g)

    def __rmul__(self, g):
        if g == 1:
            return self
        return Mul(self, g)

    def __truediv__(self, g):
        if g == 1:
            return self
        return TrueDiv(self, g)

    def __rtruediv__(self, g):
        if g == 1:
            return self
        return TrueDiv(self, g)

    def __pow__(self, p):
        if p == 1:
            return self
        return Compose(Pow(p), self)


class Function(FunctionMixin, torch.nn.Module):
    pass


def _validate_moment_orders(n: list[int]) -> None:
    if any(not isinstance(k, Integral) or k < 0 for k in n):
        raise ValueError(f"Moment orders must be non-negative integers, but got {n}.")


def _nan_gaussian_moments(
    n: list[int],
    mu: Tensor,
    sigma: Tensor,
    jacobian: bool,
    *args: Tensor,
) -> list[Tensor] | tuple[list[Tensor], list[Tensor], list[Tensor]]:
    _validate_moment_orders(n)
    mu = torch.broadcast_tensors(mu, sigma, *args)[0]
    moments = [torch.full_like(mu, torch.nan) for _ in n]
    if not jacobian:
        return moments
    return moments, [v.clone() for v in moments], [v.clone() for v in moments]


def _raw_gaussian_moments(
    n: list[int], mu: Tensor, sigma: Tensor, jacobian: bool
) -> list[Tensor] | tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Raw moments of N(mu, sigma**2), retaining only requested dependencies."""
    _validate_moment_orders(n)
    mu, sigma = torch.broadcast_tensors(mu, sigma)
    if not n:
        return ([], [], []) if jacobian else []

    needed = set(n)
    if jacobian:
        needed.update(k - 1 for k in n if k >= 1)
        needed.update(k - 2 for k in n if k >= 2)

    saved: dict[int, Tensor] = {}
    m_nm2 = torch.ones_like(mu)
    if 0 in needed:
        saved[0] = m_nm2

    max_needed = max(needed)
    if max_needed >= 1:
        m_nm1 = mu
        if 1 in needed:
            saved[1] = m_nm1
        for k in range(2, max_needed + 1):
            m_n = mu * m_nm1 + (k - 1) * sigma.square() * m_nm2
            if k in needed:
                saved[k] = m_n
            m_nm2, m_nm1 = m_nm1, m_n

    moments = [saved[int(k)] for k in n]
    if not jacobian:
        return moments

    zeros = torch.zeros_like(mu)
    moments_mu = [zeros if k == 0 else int(k) * saved[int(k) - 1] for k in n]
    moments_sigma = [
        zeros if k < 2 else int(k) * (int(k) - 1) * sigma * saved[int(k) - 2] for k in n
    ]
    return moments, moments_mu, moments_sigma


class Identity(FunctionMixin, torch.nn.Identity):
    def inv(self):
        return Identity()

    def nth_deriv(self, n, x):
        if n > 1:
            return torch.zeros_like(x)
        return torch.ones_like(x)

    def gaussian_moments(
        self, n: list[int], mu: Tensor, sigma: Tensor, jacobian: bool = False
    ):
        return _raw_gaussian_moments(n, mu, sigma, jacobian)


class Pow(Function):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return x**self.p

    def inv(self):
        return Pow(1 / self.p)

    def nth_deriv(self, n, x):
        if isinstance(n, int) and isinstance(self.p, int) and n > self.p:
            return torch.zeros_like(x)

        # this fails on float self.p
        # c = math.prod(range(self.p, self.p - n, -1))
        c = math.prod(self.p - k for k in range(n))
        return c * x ** (self.p - n)

    def gaussian_moments(
        self, n: list[int], mu: Tensor, sigma: Tensor, jacobian: bool = False
    ):
        _validate_moment_orders(n)
        if not isinstance(self.p, Integral) or self.p < 0:
            raise NotImplementedError(
                "Gaussian moments are only implemented for non-negative integer "
                f"powers, but got p={self.p}."
            )
        return _raw_gaussian_moments(
            [int(self.p) * int(k) for k in n], mu, sigma, jacobian
        )


class BinOp(Function):
    def __init__(self, f, g):
        if not isinstance(f, torch.nn.Module):
            raise TypeError(
                f"f must be an instance of torch.nn.Module, but {type(f)=}."
            )

        super().__init__()
        self.f = f
        self.g = g


class Add(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) + self.g(*args, **kwargs)


class Sub(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) - self.g(*args, **kwargs)


class Mul(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) * self.g(*args, **kwargs)


class TrueDiv(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) / self.g(*args, **kwargs)


class Sum(Function):
    def __init__(self, funcs):
        super().__init__()
        self.funcs = torch.nn.ModuleDict(funcs)

    def forward(self, *args, **kwargs):
        return sum(func(*args, **kwargs) for func in self.funcs.values())


class Prod(Function):
    def __init__(self, funcs):
        super().__init__()
        self.funcs = torch.nn.ModuleDict(funcs)

    def forward(self, *args, **kwargs):
        return math.prod(func(*args, **kwargs) for func in self.funcs.values())


class Compose(Function):
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        self.f = f
        self.args = torch.nn.ModuleList(args)
        self.kwargs = torch.nn.ModuleDict(kwargs)

    def forward(self, *args, **kwargs):
        return self.f(
            *[v(*args, **kwargs) for v in self.args],
            **{k: v(*args, **kwargs) for k, v in self.kwargs.items()},
        )

    def inv(self):
        if len(self.args) != 1 and len(self.kwargs) != 0:
            raise NotImplementedError()

        return Compose(self.args[0].inv(), self.f.inv())

    def nth_deriv(self, n, *args, **kwargs):
        # Compute the n-th derivative assuming all functions are scalar-valued and
        # only the first function has non-zero derivative.
        if len(self.args) < 1:
            raise ValueError("At least one function is required for the derivative.")

        if not (hasattr(self.f, "nth_deriv") and hasattr(self.args[0], "nth_deriv")):
            raise ValueError("Not all functions have a derivative method.")

        g = self.args[0]
        args_ = [v(*args, **kwargs) for v in self.args]
        kwargs_ = {k: v(*args, **kwargs) for k, v in self.kwargs.items()}

        if n == 1:
            return self.f.nth_deriv(1, *args_, **kwargs_) * g.nth_deriv(
                1, *args, **kwargs
            )

        if n == 2:
            return self.f.nth_deriv(2, *args_, **kwargs_) * g.nth_deriv(
                1, *args, **kwargs
            ) ** 2 + self.f.nth_deriv(1, *args_, **kwargs_) * g.nth_deriv(
                2, *args, **kwargs
            )

        raise NotImplementedError("only n=1 and n=2 are implemented for nth_deriv.")

    def gaussian_moments(
        self,
        n: list[int],
        mu: Tensor,
        sigma: Tensor,
        jacobian: bool = False,
        *args: Tensor,
        **kwargs,
    ):
        _validate_moment_orders(n)
        supported = (
            isinstance(self.f, Pow)
            and isinstance(self.f.p, Integral)
            and self.f.p >= 0
            and len(self.args) == 1
            and len(self.kwargs) == 0
            and hasattr(self.args[0], "gaussian_moments")
        )
        if not supported:
            return _nan_gaussian_moments(n, mu, sigma, jacobian, *args)

        return self.args[0].gaussian_moments(
            [int(self.f.p) * int(k) for k in n],
            mu,
            sigma,
            jacobian,
            *args,
            **kwargs,
        )


class Match(Function):
    def __init__(
        self,
        cases: dict[Hashable, Callable[[Tensor], Tensor]],
        default: Callable[[Tensor], Tensor],
    ):
        super().__init__()
        self.cases = cases
        self.default = default

    def forward(self, x: Tensor, key: Tensor) -> Tensor:
        key, x = torch.broadcast_tensors(key, x)
        out = self.default(x)
        for k, v in self.cases.items():
            if not isinstance(k, Number):
                # this is not normally needed, but inside of torch.func.grad or
                # torch.func.vmap, the dispatch does not work properly, see
                # https://github.com/pytorch/pytorch/issues/149788
                k = categorical.tensor(0, categories=(k,), device=x.device)
            mask = key == k
            out = torch.where(mask, v(x), out)
        return out

    def gaussian_moments(
        self,
        n: list[int],
        mu: Tensor,
        sigma: Tensor,
        jacobian: bool = False,
        key: Tensor | None = None,
    ):
        _validate_moment_orders(n)
        if key is None:
            raise TypeError("Match.gaussian_moments requires the dispatch key.")

        key, mu, sigma = torch.broadcast_tensors(key, mu, sigma)

        def branch_moments(f):
            if not hasattr(f, "gaussian_moments"):
                return _nan_gaussian_moments(n, mu, sigma, jacobian)
            return f.gaussian_moments(n, mu, sigma, jacobian)

        out = branch_moments(self.default)
        if jacobian:
            moments, moments_mu, moments_sigma = out
        else:
            moments = out

        for k, v in self.cases.items():
            if not isinstance(k, Number):
                k = categorical.tensor(0, categories=(k,), device=mu.device)
            mask = key == k
            branch = branch_moments(v)
            if jacobian:
                branch_values, branch_mu, branch_sigma = branch
                moments = [
                    torch.where(mask, new, old)
                    for old, new in zip(moments, branch_values)
                ]
                moments_mu = [
                    torch.where(mask, new, old)
                    for old, new in zip(moments_mu, branch_mu)
                ]
                moments_sigma = [
                    torch.where(mask, new, old)
                    for old, new in zip(moments_sigma, branch_sigma)
                ]
            else:
                moments = [
                    torch.where(mask, new, old) for old, new in zip(moments, branch)
                ]

        if jacobian:
            return moments, moments_mu, moments_sigma
        return moments
