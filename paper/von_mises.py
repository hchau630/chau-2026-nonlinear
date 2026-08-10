from typing import Sequence
import argparse

import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt

rng = np.random.default_rng()


def pdf(κ: float, θ: np.ndarray) -> np.ndarray:
    """Probability density function of sum of two i.i.d. von Mises random variables.

    Args:
        κ: Concentration parameter.
        θ: Angle.

    Returns:
        Probability density function of the angle.

    """
    return special.i0(2 * κ * np.cos(θ / 2)) / (2 * np.pi * special.i0(κ) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kappa", type=float)
    parser.add_argument("-N", type=int, default=5000)
    parser.add_argument("-b", "--bins", type=int, default=100)
    args = parser.parse_args()

    samples = stats.vonmises(loc=0, kappa=args.kappa).rvs(args.N)
    distances = (samples[:, None] - samples[None, :]) % (2 * np.pi)

    plt.hist(distances.flatten(), bins=args.bins, density=True)
    θ = np.linspace(0, 2 * np.pi, 1000)
    plt.plot(θ, pdf(args.kappa, θ))
    plt.show()


if __name__ == "__main__":
    main()
