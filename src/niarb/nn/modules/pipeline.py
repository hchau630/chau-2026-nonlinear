from collections.abc import Iterable, Sequence

import pandas as pd
import tdfl
import torch

from ..parameter import Parameter
from .activations import Ricciardi
from .analysis import TensorDataFrameAnalysis
from .containers import FanOut, NamedSequential
from .frame import ParameterFrame
from .functions import Identity, Match


class Scaler(torch.nn.Module):
    def __init__(
        self,
        init_scale: float = 1.0,
        var: str = "dh",
        requires_optim: bool | Sequence[bool] | torch.Tensor = False,
        bounds: Sequence[float | Sequence] | torch.Tensor = (0.0, torch.inf),
        keep_scale: bool = False,
        tag: str = "dh",
    ):
        super().__init__()
        self.scale = Parameter(
            torch.empty(()), requires_optim=requires_optim, bounds=bounds, tag=tag
        )
        self.var = var
        self.init_scale = init_scale
        self.keep_scale = keep_scale
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.scale, self.init_scale)

    def forward(
        self, x: ParameterFrame | tdfl.DataFrame
    ) -> ParameterFrame | tdfl.DataFrame:
        if isinstance(x, ParameterFrame):
            x = x.copy()
        elif isinstance(x, tdfl.DataFrame):
            x = tdfl.DataFrame(x, copy=False)  # shallow copy
        else:
            raise TypeError()

        x[self.var] = self.scale * x[self.var]
        if self.keep_scale:
            if self.var == self.scale.tag:
                raise ValueError(
                    "keep_scale cannot be True when self.var == self.scale.tag"
                )
            if isinstance(x, ParameterFrame):
                x[self.scale.tag] = self.scale[(None,) * x.ndim]
            else:
                x[self.scale.tag] = self.scale

        return x


class ToTensor(torch.nn.Module):
    def __init__(self, var: str = "dr"):
        super().__init__()
        self.var = var

    def forward(self, x: Iterable[tdfl.DataFrame]) -> torch.Tensor:
        return torch.cat([xi[self.var] for xi in x])


class Pipeline(NamedSequential):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        data: Iterable[pd.DataFrame] | None = None,
        scaler: torch.nn.Module | dict | None = None,
        pre_analysis: torch.nn.Module | None = None,
        analysis: torch.nn.Module | None = None,
        y: str = "dr",
        yerr: str = "dr_se",
        estimator: str = "mean",
    ):
        if data is not None and analysis is not None:
            raise ValueError("`data` and `analysis` cannot both be provided.")

        if scaler is None:
            scaler = {}

        if isinstance(scaler, dict):
            scaler = Scaler(**scaler)

        if data:
            data = list(data).copy()
            for i, df in enumerate(data):
                if y in df.columns:
                    df = df.drop(columns=y)
                if yerr in df.columns:
                    df = df.drop(columns=yerr)
                data[i] = df

        modules = {}
        modules["scaler"] = scaler
        modules["model"] = model
        if pre_analysis:
            modules["pre_analysis"] = pre_analysis
        if data:
            modules["analysis"] = FanOut(
                [TensorDataFrameAnalysis(x=df, y=y, estimator=estimator) for df in data]
            )
            modules["to_tensor"] = ToTensor(var=y)
        elif analysis:
            modules["analysis"] = analysis

        super().__init__(modules)

    def scale_parameters(self, scale: float):
        with torch.no_grad():
            if isinstance(self.model.f, Identity):
                self.scaler.scale *= scale
            elif hasattr(self.model.f, "inv"):
                k = self.model.f.inv()(torch.tensor(scale))
                self.scaler.scale *= k
                self.model.vf *= k
            elif isinstance(self.model.f, Ricciardi):
                self.model.f.scale *= scale
            elif isinstance(self.model.f, Match):
                if not isinstance(self.model.f.default, Ricciardi) or not all(
                    isinstance(v, Ricciardi) for v in self.model.f.cases.values()
                ):
                    raise NotImplementedError()
                self.model.f.default.scale *= scale
                for v in self.model.f.cases.values():
                    v.scale *= scale
            else:
                raise RuntimeError()
