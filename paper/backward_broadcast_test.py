import torch

from niarb.special.resolvent import laplace_r


class Add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y

    @staticmethod
    def backward(ctx, grad):
        return grad, grad


def main():
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor([1.0, 2.0], requires_grad=True)
    z = torch.tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    out = Add.apply(Add.apply(x, y), z)
    out.sum().backward()
    print(x.grad, y.grad, z.grad)
    # d = 2
    # r = torch.linspace(0, 1, 100)
    # l = torch.tensor(1.0, requires_grad=True)
    # out = laplace_r(d, l, r)
    # out.sum().backward()
    # print(l.grad)


if __name__ == "__main__":
    main()
