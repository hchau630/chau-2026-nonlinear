import numpy as np


def bisect(func, a, b, fa=None, fb=None, args=(), tol=1e-8):
    """
    Find a root of a function using the bisection method.

    Args:
        func: callable
        a: np.ndarray
        b: np.ndarray
        fa: np.ndarray
        fb: np.ndarray
        tol: float

    Returns:
        np.ndarray

    """
    if (b <= a).any():
        raise ValueError("b must be greater than a")

    a, b, *args = np.broadcast_arrays(a, b, *args)
    out = np.full_like(a, np.nan)
    if fa is None:
        fa = func(a, *args)
    if fb is None:
        fb = func(b, *args)
    valid = fa * fb < 0
    a, b, fa, fb = a[valid], b[valid], fa[valid], fb[valid]
    args = [arg[valid] for arg in args]
    while (b - a > tol).any():
        c = (a + b) / 2
        fc = func(c, *args)
        left = fa * fc < 0
        right = ~left
        b[left], fb[left] = c[left], fc[left]
        a[right], fa[right] = c[right], fc[right]
    out[valid] = (a + b) / 2
    return out


def func(r, a, θ):
    return a * np.sin(r) / np.cos(r - θ) + r


def case_one(a, θ, **kwargs):
    crit1 = θ + np.arccos(-np.sqrt(-a * np.cos(θ)))
    crit2 = θ - np.arccos(-np.sqrt(-a * np.cos(θ))) + 2 * np.pi
    crit3 = θ - np.arccos(np.sqrt(-a * np.cos(θ))) + 2 * np.pi
    pole1 = θ + 3 * np.pi / 2
    pole2 = pole1 + np.pi
    infs = np.full_like(a, np.inf)
    r1 = bisect(func, crit1, crit2, args=(a, θ), **kwargs)
    r2 = bisect(func, pole1, crit3, fa=infs, args=(a, θ), **kwargs)
    r3 = bisect(func, pole1, pole2, fa=infs, fb=-infs, args=(a, θ), **kwargs)
    r = np.stack([r1, r2, r3], axis=0)
    out = np.nanmin(r, axis=0)
    return out


def case_two(a, θ, **kwargs):
    mask = θ > np.pi / 2
    rmin = θ + np.pi / 2
    rmin[mask] -= np.pi
    rmax = rmin + np.pi
    infs = np.full_like(a, np.inf)
    fa, fb = -infs, infs
    fa[mask], fb[mask] = -fa[mask], -fb[mask]
    return bisect(func, rmin, rmax, fa=fa, fb=fb, args=(a, θ), **kwargs)


def first_zero(a, θ, **kwargs):
    """Find the first positive zero of f(r) = a*sin(r)/cos(r-θ) + r, a > 1.

    Args:
        a: np.ndarray
        θ: np.ndarray

    Returns:
        np.ndarray

    """
    if (a <= 1).any():
        raise ValueError("a must be greater than 1")

    if (θ < -np.pi).any() or (θ > np.pi).any():
        raise ValueError("θ must be in (-π, π)")

    a, θ = np.broadcast_arrays(a, θ)

    out = np.full_like(a, np.nan)
    mask = θ < -np.pi / 2
    out[mask] = case_one(a[mask], θ[mask], **kwargs)
    out[~mask] = case_two(a[~mask], θ[~mask], **kwargs)
    return out
