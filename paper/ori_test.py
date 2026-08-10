import torch
import matplotlib.pyplot as plt

from niarb.tensors import periodic
from niarb.nn import functional


def main():
    N = 1000
    M = 4
    dist = torch.distributions.Uniform(-torch.pi, torch.pi)
    # ori = dist.sample(M, N))
    # ori_mean = ori.mean(dim=0)
    # rel_ori = (ori - ori_mean).abs()
    ori = periodic.as_tensor(dist.sample((M, N, 1)), extents=[(-torch.pi, torch.pi)])
    ori_mean = ori.cmean(dim=0)
    rel_ori = functional.diff(ori, ori_mean).norm(dim=-1)  # (M, N)
    # print(ori[:, :2, 0].tensor)
    # print(ori_mean[:2, 0].tensor)
    plt.hist(ori.flatten(), alpha=0.5, label="ori")
    plt.hist(rel_ori.flatten(), alpha=0.5, label="rel_ori")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
