import logging

import torch

from niarb import integrate, nn, optimize


class SSN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        init_W = torch.tensor(
            [
                [0.0, -0.1],
                [0.1, 0.0],
            ]
        )
        bounds_W = torch.tensor(
            [
                [[0, torch.inf], [-torch.inf, 0]],
                [[0, torch.inf], [-torch.inf, 0]],
            ]
        )
        self.W = nn.Parameter(init_W, bounds=bounds_W)

    def f(self, x):
        return x.clip(min=0) ** 2

    def forward(self, h, **kwargs):
        x0 = torch.zeros((2,))

        def func(_, x, W=self.W, h=h):
            return self.f(W @ x + h) - x

        r = integrate.odeint_ss(func, x0, **kwargs).x

        return r


class MSELoss(torch.nn.Module):
    def forward(self, x, y):
        return (x - y).norm()


def main():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel("INFO")

    model = SSN()
    criterion = MSELoss()

    h = torch.tensor([1.0, 0.0])
    target = torch.tensor([0.5, 0.5])

    optimizer = optimize.Optimizer(model, criterion)

    success, loss = optimizer(h, target)

    return loss, model.state_dict()


if __name__ == "__main__":
    main()
