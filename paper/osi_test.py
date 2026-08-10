import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from niarb import special


def main():
    a0, b0 = 0.88, 1.07
    a1, b1 = 1.2, 0.85
    N = 10000
    osi_dist = torch.distributions.Beta(a0, b0)
    new_osi_dist = torch.distributions.Beta(a1, b1)
    osi = osi_dist.sample((N,)).double()
    prob = special.ubeta(a1 - a0 + 1, b1 - b0 + 1, osi)
    prob = prob / prob.sum()
    print(osi)
    print(prob)
    plt.scatter(osi, prob)
    plt.show()
    new_osi = np.random.choice(osi, p=prob, size=N)
    print(new_osi)
    res = stats.kstest(new_osi, "beta", args=(a1, b1))
    new_osi_expected = new_osi_dist.sample((N,))
    plt.hist(new_osi, label="OSI", alpha=0.5)
    plt.hist(new_osi_expected, label="expected OSI", alpha=0.5)
    plt.legend()
    plt.show()
    print(res.pvalue)


if __name__ == "__main__":
    main()
