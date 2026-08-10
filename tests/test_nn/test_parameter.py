import copy

import pytest
import torch

from niarb import nn


class TestParameter:
    @pytest.fixture
    def parameter(self):
        return nn.Parameter(
            torch.tensor([1.0, 2.0]),
            requires_optim=torch.Tensor([True, False]),
            bounds=[(0.0, torch.inf), (-torch.inf, 0.0)],
            tag="hi",
        )

    def test_init(self):
        parameter = nn.Parameter(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            requires_optim=[False, True],
            bounds=[(0.0, torch.inf), (-torch.inf, 0.0)],
            tag="hi",
        )
        expected_requires_optim = torch.tensor([[False, True], [False, True]])
        assert (parameter.requires_optim == expected_requires_optim).all()
        expected_bounds = torch.tensor(
            [
                [[0.0, torch.inf], [-torch.inf, 0.0]],
                [[0.0, torch.inf], [-torch.inf, 0.0]],
            ]
        )
        assert (parameter.bounds == expected_bounds).all()

    def test_deepcopy(self, parameter):
        y = parameter.mean()
        y.backward()
        parameter_copy = copy.deepcopy(parameter)
        assert (parameter_copy == parameter).all()
        assert parameter_copy.tag == "hi"
        assert (parameter_copy.requires_optim == torch.tensor([True, False])).all()
        assert (
            parameter_copy.bounds == torch.Tensor([[0.0, torch.inf], [-torch.inf, 0.0]])
        ).all()
        # deepcopy should not preserve gradient
        assert parameter.grad is not None and parameter_copy.grad is None
