import copy
from collections.abc import Sequence

import torch

__all__ = ["Parameter"]


class Parameter(torch.nn.Parameter):
    """
    A subclass of torch.nn.Parameter that has an additional attribute
    requires_optim, which is a torch.BoolTensor that specifies
    which elements of the data should be optimized.
    If bounds is not None, bounds should be either be a tuple or
    a torch.Tensor with size (*data.shape, 2), which specifies the
    lower and upper bound of either all elements or each individual
    element respectively.
    """

    def __new__(cls, data, requires_optim=True, **kwargs):
        requires_grad = torch.as_tensor(requires_optim).any().item()
        return super().__new__(cls, data, requires_grad)

    def __init__(
        self,
        data: torch.Tensor,
        requires_optim: bool | Sequence[bool | Sequence] | torch.Tensor = True,
        bounds: Sequence[float | Sequence] | torch.Tensor = (-torch.inf, torch.inf),
        tag: str | None = None,
    ):
        requires_optim = torch.as_tensor(requires_optim, dtype=torch.bool)
        self._requires_optim = requires_optim.broadcast_to(data.shape).clone()

        bounds = torch.as_tensor(bounds, dtype=data.dtype)
        self.bounds = bounds.broadcast_to((*data.shape, 2)).clone()
        self.tag = tag

    def __repr__(self):
        return (
            f"{super().__repr__()}"
            f"\nrequires_optim=\n{self.requires_optim}"
            f"\nbounds=\n{self.bounds}"
            f"\ntag={self.tag}"
        )

    def __deepcopy__(self, memo):
        """
        Implementation adapted from PyTorch's implementation of torch.nn.Parameter.__deepcopy__
        This is needed to ensure correctness when copying models with Parameters.
        """
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format),
                requires_optim=copy.deepcopy(self._requires_optim),
                bounds=copy.deepcopy(self.bounds),
                tag=copy.deepcopy(self.tag),
            )
            memo[id(self)] = result
            return result

    @property
    def requires_optim(self):
        return self._requires_optim

    @requires_optim.setter
    def requires_optim(self, value):
        self.requires_grad = torch.any(value).item()
        self._requires_optim = value
