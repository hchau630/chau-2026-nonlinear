from typing import Sequence
import argparse

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

rng = np.random.default_rng()


def random_points(lengths: Sequence[float], size: Sequence[int] = ()) -> np.ndarray:
    """Generate random points in a box.

    Args:
        lengths: Lengths of the box.

    Returns:
        Random points in the box.

    """
    return rng.random(size + (len(lengths),)) * np.array(lengths)


def pdf_2d(a: float, b: float, r: np.ndarray) -> np.ndarray:
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
    case0 = np.pi * a * b / 2 - (a + b) * r + r**2 / 2
    case1 = (
        a * b * (np.pi / 2 - np.arccos(a / r))
        - b * (r - np.sqrt(r**2 - a**2))
        - a**2 / 2
    )
    case2 = (
        a * b * (np.arcsin(b / r) - np.arccos(a / r))
        - b * (b / 2 - np.sqrt(r**2 - a**2))
        - a * (a / 2 - np.sqrt(r**2 - b**2))
        - r**2 / 2
    )

    mask0 = (r >= 0) & (r < a)
    mask1 = (r >= a) & (r < b)
    mask2 = (r >= b) & (r < np.sqrt(a**2 + b**2))
    out = np.zeros_like(r)
    out[mask0] = case0[mask0]
    out[mask1] = case1[mask1]
    out[mask2] = case2[mask2]
    return prefactor * out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lengths", type=float, nargs="+")
    parser.add_argument("-N", type=int, default=1000)
    parser.add_argument("-b", "--bins", type=int, default=100)
    args = parser.parse_args()

    points = random_points(args.lengths, size=(args.N,))
    distances = spatial.distance.cdist(points, points)

    plt.hist(distances.flatten(), bins=args.bins, density=True)
    r = np.linspace(0, np.sqrt(sum(l**2 for l in args.lengths)), 1000)
    plt.plot(r, pdf_2d(*args.lengths, r))
    plt.show()


if __name__ == "__main__":
    main()
