import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from niarb import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", "-o", type=Path)
    args = parser.parse_args()
    x = torch.linspace(-1.0, 3.0, 100)
    f = nn.Ricciardi(scale=1.0)
    plt.figure(figsize=(1, 0.75))
    plt.plot(x, f(x))
    plt.xticks([])
    plt.yticks([])
    if args.show:
        plt.show()
    if args.out:
        plt.savefig(args.out / "1a.pdf")


if __name__ == "__main__":
    main()
