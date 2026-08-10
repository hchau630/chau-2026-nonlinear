import math
from numbers import Number

import torch
from ricciardi import ricciardi

from ..parameter import Parameter
from .functions import Function, FunctionMixin, _validate_moment_orders

__all__ = ["SSN", "Rectified", "Ricciardi", "Threshold"]


class Threshold(FunctionMixin, torch.nn.Threshold):
    """
    f(x) = x if x > threshold else value
    """

    def inv(self):
        if self.inplace:
            raise NotImplementedError()

        return InvThreshold(self.threshold, self.value)

    def nth_deriv(self, n, x):
        if self.inplace:
            raise NotImplementedError()

        if n == 1:
            return (x > self.threshold).to(x.dtype)
        return torch.zeros_like(x)


class InvThreshold(Function):
    def __init__(self, threshold, value):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x):
        if (x <= self.threshold).any() or (x == self.value).any():
            raise ValueError(
                f"""Function is not invertible for inputs less than or
                equal to {self.threshold}, or inputs equal to {self.value},
                but got {x.min()=} and {x[x == self.value].numel()=}."""
            )
        return x


class Rectified(Threshold):
    """
    f(x) = x if x > threshold else threshold
    """

    def __init__(self, threshold=0.0, inplace=False):
        super().__init__(threshold, threshold, inplace=inplace)

    def gaussian_moments(
        self,
        n: list[int],
        mu: torch.Tensor,
        sigma: torch.Tensor,
        jacobian: bool = False,
    ):
        if self.threshold != 0.0:
            raise NotImplementedError(
                "Gaussian moments are only implemented for threshold=0.0, but "
                f"got threshold={self.threshold}."
            )

        _validate_moment_orders(n)
        mu, sigma = torch.broadcast_tensors(mu, sigma)
        if not n:
            return ([], [], []) if jacobian else []

        positive_orders = [int(k) for k in n if k > 0]
        if not positive_orders:
            moments = [torch.ones_like(mu) for _ in n]
            if not jacobian:
                return moments
            zeros = [torch.zeros_like(mu) for _ in n]
            return moments, zeros, [v.clone() for v in zeros]

        nonzero_sigma = sigma != 0
        safe_sigma = torch.where(nonzero_sigma, sigma, torch.ones_like(sigma))
        standardized = mu / safe_sigma
        cdf = torch.special.ndtr(standardized)
        pdf = torch.exp(-standardized.square() / 2) / math.sqrt(2 * math.pi)

        # Use the continuous sigma -> 0 limits, including the symmetric value at
        # the nondifferentiable point mu=sigma=0.
        zero_scale_cdf = torch.where(
            mu > 0,
            torch.ones_like(mu),
            torch.where(mu < 0, torch.zeros_like(mu), torch.full_like(mu, 0.5)),
        )
        zero_scale_pdf = torch.where(
            mu == 0,
            torch.full_like(mu, 1 / math.sqrt(2 * math.pi)),
            torch.zeros_like(mu),
        )
        cdf = torch.where(nonzero_sigma, cdf, zero_scale_cdf)
        pdf = torch.where(nonzero_sigma, pdf, zero_scale_pdf)

        needed = set(positive_orders)
        if jacobian:
            needed.update(k - 1 for k in positive_orders if k >= 2)
            needed.update(k - 2 for k in positive_orders if k >= 2)

        # Here t_k = E[X**k 1{X > 0}].  The requested zeroth ReLU moment is
        # instead E[relu(X)**0] = 1, so t_0 is only an internal recurrence term.
        saved: dict[int, torch.Tensor] = {0: cdf}
        t_nm2 = cdf
        t_nm1 = mu * cdf + sigma * pdf
        if 1 in needed:
            saved[1] = t_nm1
        for k in range(2, max(needed) + 1):
            t_n = mu * t_nm1 + (k - 1) * sigma.square() * t_nm2
            if k in needed:
                saved[k] = t_n
            t_nm2, t_nm1 = t_nm1, t_n

        moments = [torch.ones_like(mu) if k == 0 else saved[int(k)] for k in n]
        if not jacobian:
            return moments

        zeros = torch.zeros_like(mu)
        moments_mu = [
            zeros if k == 0 else cdf if k == 1 else int(k) * saved[int(k) - 1]
            for k in n
        ]
        moments_sigma = [
            zeros
            if k == 0
            else pdf
            if k == 1
            else int(k) * (int(k) - 1) * sigma * saved[int(k) - 2]
            for k in n
        ]
        return moments, moments_mu, moments_sigma


class Ricciardi(Function):
    """Ricciardi nonlinearity with an optimizable scale parameter.

    Args:
        scale (optional): Initial value of the scale parameter.
        requires_optim (optional): Whether the scale parameter requires optimization.
        bounds (optional): Lower and upper bounds of the scale parameter.
        tag (optional): Tag of the scale parameter.
        unitless (optional): If True, then the function input is expected to be a unitless
          quantity where a value of 1.0 corresponds to an input with the same magnitude as
          the standard deviation of the input noise, sigma. This is good for simulation
          since we generally want to keep quantities around 1 to avoid numerical issues.
        sigma (optional): Standard deviation of the input noise.
        V_r (optional): Resting potential.
        theta (optional): Threshold potential.
        **kwargs: Additional arguments to pass to ricciardi.

    """

    def __init__(
        self,
        scale=0.025,
        requires_optim=False,
        bounds=(0.0, torch.inf),
        tag="alpha",
        unitless=True,
        sigma=0.01,  # in volts
        V_r=0.01,  # in volts
        theta=0.02,  # in volts
        **kwargs,
    ):
        super().__init__()
        self.init_scale = scale
        self.scale = Parameter(
            torch.empty(()), requires_optim=requires_optim, bounds=bounds, tag=tag
        )
        if unitless:
            V_r = V_r / sigma
            theta = theta / sigma
            sigma = 1.0
        self.kwargs = kwargs | {"sigma": sigma, "V_r": V_r, "theta": theta}
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.scale, self.init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * ricciardi(x, **self.kwargs)


def SSN(p: Number = 2, **kwargs) -> torch.nn.Module:
    return Rectified(**kwargs) ** p
