from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike


def _sign_change(x: ArrayLike) -> np.ndarray:
    """Return a mask of sign changes in x along the last dimension.

    This is a surprisingly annoying problem. Solutions found using stackexchange
    all fail in different edge cases. For example, solutions like
    `np.diff(np.signbit(x), axis=-1) != 0`
    fail on inputs like [-1, 0, -1]. The key idea of this implementation is to
    modify np.sign(x) by replacing values of 0 with the sign of the last nonzero value.

    Args:
        x: ndarray with shape (*, N)

    Returns:
        ndarray with shape (*, N - 1) of sign changes in x.

    """
    x = np.sign(np.asarray(x))

    idx = np.broadcast_to(np.arange(x.shape[-1]), x.shape)
    idx = np.where(x == 0, -1, idx)
    idx = np.maximum(np.maximum.accumulate(idx, axis=-1), 0)

    x = np.take_along_axis(x, idx, axis=-1)
    return (x[..., :-1] != x[..., 1:]) & (x[..., :-1] != 0)


@overload
def find_n_crossings(
    x: ArrayLike,
    y: ArrayLike,
    n: int = ...,
    return_indices: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def find_n_crossings(
    x: ArrayLike,
    y: ArrayLike,
    n: int = ...,
    return_indices: Literal[True] = ...,
) -> tuple[np.ndarray, np.ndarray]: ...


def find_n_crossings(
    x: ArrayLike, y: ArrayLike, n: int = 1, return_indices: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Args:
        x: array-like with shape (N)
        y: array-like with shape (*, N)
        n (optional): Find the first n zero crossings. If n == -1, find all zero
          crossings.
        return_indices (optional): If True, return the indices of the zero crossings.

    Returns:
        ndarray with shape (n, *) of the first n zero crossings of the function.
        If return_indices is True, also return a ndarray with shape (n, *) of the
        indices of the zero crossings.

    """
    # if n < 1 and n != -1:
    #     raise ValueError("n must be -1 or greater than or equal to 1")

    if n < 1:
        raise ValueError("n must greater than or equal to 1")

    x, y = np.asarray(x), np.asarray(y)

    # get midpoints
    mid = (x[1:] + x[:-1]) / 2  # (N - 1,)

    idx, indices, crossings = np.empty(y.shape[:-1], dtype=np.long), [], []
    while len(crossings) < n:
        if len(crossings) == 0:
            # get mask of where y changes sign
            mask = _sign_change(y).astype(np.long)  # (*, N - 1)

        else:
            # set the mask to zero at the index of nonzero element found
            np.put_along_axis(mask, idx[..., None], 0, axis=-1)  # (*, N - 1)

        # find the next index of nonzero element by using the fact that argmax returns
        # the first occurrence of the maximum value. If there are no nonzero elements,
        # then idx is 0.
        idx = mask.argmax(axis=-1)  # (*,)

        # mask of whether the nth crossing exists
        has_crossing = mask.any(axis=-1)  # (*,)

        # get nth crossing if it exists, else NaN
        crossing = np.where(has_crossing, mid[idx], np.nan)  # (*,)

        indices.append(idx)
        crossings.append(crossing)

        # if len(crossings) == n or (n == -1 and not has_crossing.any()):
        #     break

    if return_indices:
        return np.stack(crossings), np.stack(indices)
    return np.stack(crossings)
