import math

import torch
from scipy import special
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch.types import _size


def beta_cdf(self, value: Tensor) -> Tensor:
    if (
        value.requires_grad
        or self.concentration1.requires_grad
        or self.concentration0.requires_grad
    ):
        raise NotImplementedError(
            "The beta CDF is neither differentiable with respect to its inputs nor "
            "parameters."
        )
    return (
        special.betainc(
            self.concentration1.cpu(), self.concentration0.cpu(), value.cpu()
        )
        .to(value.device)
        .to(value.dtype)
    )


class RectLinePicking(Distribution):
    """
    Distribution of distances between two uniformly sampled points in a rectangle.

    Args:
        a: Length of one of the rectangle's sides.
        b: Length of the other side.
        validate_args (optional): If True, validate arguments at runtime.

    """

    arg_constraints = {"a": constraints.positive, "b": constraints.positive}  # noqa: RUF012
    has_rsample = True
    a: Tensor
    b: Tensor

    def __init__(
        self, a: float | Tensor, b: float | Tensor, validate_args: bool | None = None
    ):
        self.a, self.b = broadcast_all(a, b)
        batch_shape = self.a.shape
        super().__init__(batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(0.0, (self.a**2 + self.b**2).sqrt())

    def rsample(self, sample_shape: _size = ()) -> Tensor:
        shape = (2, *self._extended_shape(sample_shape))
        x1, y1 = self.a * torch.rand(shape, device=self.a.device, dtype=self.a.dtype)
        x2, y2 = self.b * torch.rand(shape, device=self.b.device, dtype=self.b.dtype)
        return ((x1 - y1) ** 2 + (x2 - y2) ** 2).sqrt()

    def log_prob(self, r: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(r)

        a, b = torch.minimum(self.a, self.b), torch.maximum(self.a, self.b)

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

        mask0 = (r >= self.support.lower_bound) & (r < a)
        mask1 = (r >= a) & (r < b)
        mask2 = (r >= b) & (r <= self.support.upper_bound)
        out = torch.zeros_like(r)
        out[mask0] = case0[mask0]
        out[mask1] = case1[mask1]
        out[mask2] = case2[mask2]
        return (prefactor * out.clip(min=0.0)).log()

    def cdf(self, r: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(r)

        a, b = torch.minimum(self.a, self.b), torch.maximum(self.a, self.b)

        prefactor = 4 / (a**2 * b**2)
        sqrtba = torch.sqrt(b**2 - a**2)
        sqrtra = torch.sqrt(r**2 - a**2)
        sqrtrb = torch.sqrt(r**2 - b**2)
        asinab = torch.asin(a / b)
        acosab = torch.acos(a / b)
        asinbr = torch.asin(b / r)
        acosar = torch.acos(a / r)

        case0 = r**2 / 24 * (6 * a * b * torch.pi - 8 * (a + b) * r + 3 * r**2)
        case1 = (
            3 * a**4
            + a**3 * b * (4 - 3 * torch.pi)
            + 4 * b * r**2 * (sqrtra - r)
            + a**2 * (2 * b * sqrtra - 3 * r**2)
            + 6 * a * b * r**2 * torch.arcsin(a / r)
        ) / 12 + ((6 * torch.pi - 8) * b - 5 * a) * a**3 / 24
        case2 = (
            4 * a**2 * b * (sqrtra - sqrtba)
            + 8 * b * (r**2 * sqrtra - b**2 * sqrtba)
            + 4 * a * sqrtrb * (b**2 + 2 * r**2)
            - 3 * (r**2 - b**2) * (2 * a**2 + 3 * b**2 + r**2)
            - 6 * a * b**3 * torch.pi
            + 12 * a * b * (b**2 * acosab + r**2 * (asinbr - acosar))
        ) / 24 + (
            a**4
            + 8 * b**3 * (sqrtba - b)
            + 2 * a**2 * b * (2 * sqrtba - 3 * b)
            + 12 * a * b**3 * asinab
        ) / 24

        mask0 = (r >= self.support.lower_bound) & (r < a)
        mask1 = (r >= a) & (r < b)
        mask2 = (r >= b) & (r <= self.support.upper_bound)
        out = torch.zeros_like(r)
        out[mask0] = case0[mask0]
        out[mask1] = case1[mask1]
        out[mask2] = case2[mask2]
        return prefactor * out.clip(min=0.0)


class EllipsoidConstraint(constraints.Constraint):
    event_dim = 1

    def __init__(self, r):
        self.r = r
        super().__init__()

    def check(self, value: Tensor) -> Tensor:
        return torch.linalg.vector_norm(value / self.r, dim=-1) < 1.0


class UniformEllipsoid(Distribution):
    """
    Sample points uniformly within an n-ellipsoid.

    Args:
        r: Half lengths of each axis.
        validate_args (optional): If True, validate arguments at runtime.

    """

    arg_constraints = {"r": constraints.independent(constraints.positive, 1)}  # noqa: RUF012
    has_rsample = True
    r: Tensor
    d: int

    def __init__(self, r: Tensor, validate_args: bool | None = None):
        if r.ndim < 1:
            raise ValueError("r must be at least one-dimensional.")
        batch_shape, event_shape = r.shape[:-1], r.shape[-1:]
        self.r = r
        self.d = event_shape[0]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        return EllipsoidConstraint(self.r)

    def rsample(self, sample_shape: _size = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        kwargs = {"device": self.r.device, "dtype": self.r.dtype}
        out = torch.randn(shape, **kwargs)
        out.div_(torch.linalg.vector_norm(out, dim=-1, keepdim=True))
        out.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)  # handle edge case zero norm
        out.mul_(torch.rand(shape[:-1] + (1,), **kwargs).pow_(1 / shape[-1]))
        out.mul_(self.r)
        return out

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)

        # volume = (pi^{d/2} / Gamma(d/2 + 1)) * r.prod(dim=-1)
        # log_prob = log(1 / volume) = -log(volume)
        # = log(Gamma(d/2 + 1)) - d/2 * log(pi) - r.log().sum(dim=-1)
        out = math.lgamma(self.d / 2 + 1) - math.log(math.pi) * self.d / 2
        return self.r.log().sum(dim=-1).neg_().add_(out).broadcast_to(value.shape[:-1])

    def volume(self) -> Tensor:
        out = self.r.prod(dim=-1)
        out.mul_(math.pi ** (self.d / 2) / math.gamma(self.d / 2 + 1))
        return out
