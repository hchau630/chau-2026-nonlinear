import functools
import itertools
import math

import torch

from .base import _rebuild, tensor_class_factory
from .base_circulant import BaseCirculantTensor


def tensor(
    data, dtype=None, device=None, requires_grad=False, pin_memory=False, **kwargs
):
    data = torch.tensor(data, dtype=dtype, device=device, pin_memory=pin_memory)
    return _instantiate(data, requires_grad=requires_grad, **kwargs)


def as_tensor(data, dtype=None, device=None, **kwargs):
    data = torch.as_tensor(data, dtype=dtype, device=device)
    return _instantiate(data, **kwargs)


def _instantiate(
    data, requires_grad=False, cdim=(-1,), bdim1=None, bdim2=None, ndim=None
):
    from niarb.utils import normalize_dim  # avoid circular import

    cdim = normalize_dim(cdim, data.ndim, neg=True)

    if bdim1 is None and bdim2 is None and ndim is None:
        raise ValueError("Must specify at least one of bdim1, bdim2, or ndim.")

    if ndim is None:
        ndim = len(cdim) + (len(bdim1) if bdim2 is None else len(bdim2))

    if bdim1 is not None:
        bdim1 = normalize_dim(bdim1, data.ndim, neg=True)

    if bdim2 is None:
        if len(cdim) == ndim:
            bdim2 = ()
        else:
            dim = cdim + (bdim1 or ())
            bdim2 = tuple(d for d in range(-data.ndim, 0) if d not in dim)
            bdim2 = bdim2[-ndim + len(cdim) :]
    else:
        bdim2 = normalize_dim(bdim2, data.ndim, neg=True)

    if bdim1 is None:
        if len(cdim) == ndim:
            bdim1 = ()
        else:
            bdim1 = tuple(d for d in range(-data.ndim, 0) if d not in cdim + bdim2)
            bdim1 = bdim1[-ndim + len(cdim) :]

    if tuple(sorted(cdim + bdim1 + bdim2)) != tuple(
        range(-len(cdim) - len(bdim1) - len(bdim2), 0)
    ):
        raise ValueError(
            "Circulant and block dimensions must be unique, contiguous,"
            f" and start from the last dimension, but {cdim=}, {bdim1=}, {bdim2=}."
        )

    if len(bdim1) != len(bdim2):
        raise ValueError(
            f"Block dimensions must have equal length, but {bdim1=}, {bdim2=}."
        )

    ndim = len(cdim) + len(bdim1) + len(bdim2)
    shift = itertools.accumulate(d in bdim1 for d in range(-1, -ndim - 1, -1))
    shift = tuple(shift)[::-1]

    vec_cdim = tuple(d + shift[d] for d in cdim)
    vec_bdim = tuple(d + shift[d] for d in bdim2)
    vec_bshape = tuple(data.shape[d] for d in bdim2)

    data = data.movedim(cdim + bdim2, vec_cdim + vec_bdim)

    supercls = tensor_class_factory(
        BaseCirculantTensor, nc=len(cdim), block=len(bdim1) > 0, orig_dtype=data.dtype
    )
    cls = tensor_class_factory(
        CirculantTensor, supercls, vec_bdim=vec_bdim, vec_bshape=vec_bshape
    )

    return cls(data, requires_grad=requires_grad)


def _vector_from_shape(data, bdim, offset=0):
    bdim = tuple(d - offset for d in bdim)

    data = data.movedim(bdim, tuple(range(-len(bdim) - offset, -offset)))
    if len(bdim) > 0:
        offset_shape = data.shape[-offset:] if offset > 0 else ()
        data = data.reshape((*data.shape[: -len(bdim) - offset], -1, *offset_shape))
    return data


def _vector_to_shape(data, bdim, bshape, offset=0):
    bdim = tuple(d - offset for d in bdim)

    if len(bdim) > 0:
        offset_shape = data.shape[-offset:] if offset > 0 else ()
        data = data.reshape((*data.shape[: -1 - offset], *bshape, *offset_shape))
    data = data.movedim(tuple(range(-len(bdim) - offset, -offset)), bdim)
    return data


class CirculantTensor(BaseCirculantTensor):
    def __new__(cls, data, **kwargs):
        ndim = cls.nc + len(cls.vec_bdim)
        cdim = tuple(d for d in range(-ndim, 0) if d not in cls.vec_bdim)
        bdim1 = tuple(range(-ndim - len(cls.vec_bdim), -ndim))
        bdim2 = cls.vec_bdim
        bshape = cls.vec_bshape

        if not all(
            data.size(d1) == data.size(d2) == s
            for d1, d2, s in zip(bdim1, bdim2, bshape)
        ):
            raise ValueError(
                f"Shape of block dimensions {bdim1}, {bdim2} must equal {bshape}, but"
                f" {data.shape=}."
            )

        dim = cdim + bdim1 + bdim2

        data = data.movedim(dim, tuple(range(-len(dim), 0)))
        if len(bdim1 + bdim2) > 0:
            data = data.reshape(
                (*data.shape[: -len(bdim1 + bdim2)], math.prod(bshape), -1)
            )

        return super().__new__(cls, data, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in HANDLED_FUNCTIONS:
            result = HANDLED_FUNCTIONS[func](cls, types, args=args, kwargs=kwargs)

        elif func.__name__ in ALLOWED_FUNCTIONS:
            result = super().__torch_function__(func, types, args, kwargs)

        else:
            raise NotImplementedError()
        return result

    def __reduce_ex__(self, proto):
        # override __reduce_ex__ to support pickling
        func, args = super().__reduce_ex__(proto)
        args = list(args)
        args[1] = {"vec_bdim": self.vec_bdim, "vec_bshape": self.vec_bshape}
        return _rebuild, (func, tuple(args), CirculantTensor)

    @property
    def vec_ndim(self):
        return self.nc + len(self.vec_bdim)

    @property
    def vec_cdim(self):
        return tuple(d for d in range(-self.vec_ndim, 0) if d not in self.vec_bdim)

    @property
    def vec_shape(self):
        shape = self.cshape + self.vec_bshape
        vec_shape = [None] * len(shape)
        for i, d in enumerate(self.vec_cdim + self.vec_bdim):
            vec_shape[d] = shape[i]
        return torch.Size(vec_shape)

    def dense(self, keep_shape=True):
        out = super().dense(keep_shape=True)

        out = out.reshape(
            *self.batchshape,
            *self.cshape,
            *self.vec_bshape,
            *self.cshape,
            *self.vec_bshape,
        )
        source_dim = tuple(range(self.nbatch, out.ndim))
        target_dim = self.vec_cdim + self.vec_bdim
        target_dim = tuple(d - self.vec_ndim for d in target_dim) + target_dim
        out = out.movedim(source_dim, target_dim)

        if keep_shape:
            return out

        return out.reshape(*self.batchshape, math.prod(self.cshape + self.bshape), -1)


HANDLED_FUNCTIONS = {}
ALLOWED_FUNCTIONS = {"__get__"}


def implements(torch_functions):
    """Register a list of torch functions to override"""

    def decorator(func):
        for torch_function in torch_functions:
            HANDLED_FUNCTIONS[torch_function] = functools.partial(func, torch_function)

    return decorator


@implements(
    [
        torch.t,
        torch.Tensor.t,
        torch.adjoint,
        torch.Tensor.adjoint,
        torch.stack,
        torch.linalg.eigvals,
        torch.linalg.eigvalsh,
        torch.linalg.inv,
        torch.Tensor.add,
        torch.add,
        torch.Tensor.sub,
        torch.sub,
        torch.Tensor.mul,
        torch.mul,
        torch.Tensor.div,
        torch.div,
    ]
)
def inherited_func(func, cls, types, args=(), kwargs=None):
    return super(CirculantTensor, cls).__torch_function__(
        func, types, args=args, kwargs=kwargs
    )


@implements(
    [
        torch.matmul,
        torch.linalg.matmul,
        torch.Tensor.matmul,
        torch.linalg.solve,
        torch.Tensor.solve,
    ]
)
def batched_matrix_op(func, cls, types, args=(), kwargs=None):
    """Batched matrix operation between a circulant tensor A and a tensor B.

    Args:
        A (CirculantTensor): Circulant tensor.
        B (Tensor): Circulant tensor or tensor with shape (*, *shape) or (*, *shape, k).

    Returns:
        Circulant tensor or tensor with shape (*, *shape) or (*, *shape, k).

    """
    A, B = args

    if isinstance(B, BaseCirculantTensor):
        return super(CirculantTensor, cls).__torch_function__(
            func, types, args=args, kwargs=kwargs
        )

    if B.ndim < A.vec_ndim:
        raise ValueError(
            f"Expected at least {A.vec_ndim=} dimensions for B, but {B.ndim=}."
        )
    offset = int(A.vec_shape != B.shape[-A.vec_ndim :])

    B = _vector_from_shape(B, cls.vec_bdim, offset=offset)
    out = super(CirculantTensor, cls).__torch_function__(
        func, types, args=(A, B), kwargs=kwargs
    )
    out = _vector_to_shape(out, cls.vec_bdim, cls.vec_bshape, offset=offset)

    return out


def eye_like(data: CirculantTensor, requires_grad: bool = False) -> CirculantTensor:
    kwargs = {
        "dtype": data.dtype,
        "device": data.device,
        "requires_grad": requires_grad,
    }

    if data.block:
        I = torch.eye(data.shape[-1], **kwargs)
    else:
        I = torch.ones((), **kwargs)
    I = I.broadcast_to(data.shape[-data.nc - 2 * data.nb :])

    return I.as_subclass(type(data))


def diag_like(data: CirculantTensor, diag: torch.Tensor) -> CirculantTensor:
    if diag.ndim < len(data.vec_bshape):
        raise ValueError(
            f"Expected at least {len(data.vec_bshape)=} dimensions for diag, "
            f"but {diag.ndim=}."
        )

    bshape = diag.shape[: -len(data.vec_bshape)] if data.block else diag.shape
    out = diag.broadcast_to((*bshape, *data.vec_bshape))
    out = out.reshape((*bshape, *((1,) * data.nc), -1)).diag_embed()
    if not data.block:
        out = out.squeeze((-2, -1))
    out = out.broadcast_to((*bshape, *data.shape[-data.nc - 2 * data.nb :]))
    out = out.to(data.dtype)

    return out.as_subclass(type(data))
