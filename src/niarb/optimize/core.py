import functools
import logging
import time
from collections.abc import Sequence
from typing import Any

import hyclib as lib
import numpy as np
import scipy.optimize as sp_opt
import torch

from niarb import exceptions
from niarb.nn.parameter import Parameter

from .constraint import Constraint

logger = logging.getLogger(__name__)


def _ndarray_to_tuple(f):
    @functools.wraps(f)
    def wrapped_f(x):
        return f(tuple(x.tolist()))

    return wrapped_f


class Optimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        regularizer: torch.nn.Module | None = None,
        method: str | None = None,
        constraints: Sequence[Constraint] = (),
        tol: float | None = None,
        options: dict | None = None,
        use_autograd: bool = True,
        timeout: float = 1000.0,
        cache_size: int = 128,
    ):
        """Initialize optimizer.

        Args:
            model: Model to optimize.
            criterion: Loss function.
            regularizer (optional): Regularization term.
            method (optional): 'method' argument of scipy.optimize.minimize.
            constraints (optional): Constraints.
            tol (optional): 'tol' argument of scipy.optimize.minimize.
            options (optional): 'options' argument of scipy.optimize.minimize.
            use_autograd (optional): Whether to use autograd.
            timeout (optional): Time (in seconds) before optimization times out.
            cache_size (optional): Cache size (in number of calls).

        """
        constraint_names = [repr(c) for c in constraints]
        if len(constraint_names) != len(set(constraint_names)):
            raise ValueError("Constraint names must be unique.")

        self.model = model
        self.criterion = criterion
        self.regularizer = regularizer
        self.method = method
        self.tol = tol
        self.options = options
        self.use_autograd = use_autograd
        self.timeout = timeout
        self.cache_size = cache_size
        c0, c1 = [], []
        for c in constraints:
            if c.requires_graph:
                c0.append(c)
            else:
                c1.append(c)
        self.constraints = c0 + c1
        self.num_requires_graph = len(c0)

    @property
    def params(self):
        """
        Returns a 1D-tensor of optimizable parameters
        """
        params = []
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter) and torch.any(param.requires_optim):
                params.append(param[param.requires_optim])
        return torch.cat(params)

    @property
    def params_grad(self):
        """
        Returns a 1D-tensor of optimizable parameters gradients
        """
        params_grad = []
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter) and torch.any(param.requires_optim):
                if (
                    param.grad is None
                ):  # parameter has no gradient, probably because the objective function is independent of the parameter
                    params_grad.append(
                        torch.zeros(
                            len(param.requires_optim.nonzero()), device=param.device
                        )
                    )
                else:
                    params_grad.append(param.grad[param.requires_optim])
        return torch.cat(params_grad)

    @params.setter
    def params(self, params):
        """
        Updates the optimizable parameters with params, which is a 1D-tensor.
        """
        assert len(params) == len(self.params)

        i = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if isinstance(param, Parameter) and torch.any(param.requires_optim):
                    N = param[param.requires_optim].numel()
                    param[param.requires_optim] = torch.tensor(
                        params[i : i + N], dtype=param.dtype, device=param.device
                    )
                    i += N

    @property
    def bounds(self):
        bounds = []
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter):
                bounds += param.bounds[param.requires_optim].tolist()
        return bounds

    def state_dict(self):
        state_dict = {}
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter):
                state_dict[name] = param.data
        return state_dict

    def make_constraints(self, compute_quantities) -> dict[str, dict[str, Any]]:
        constraints = {}
        for i, c in enumerate(self.constraints):

            def constraint_func(params, i=i):
                return compute_quantities(params)["constraint_vals"][i]

            constraint_type = "eq" if c.is_equality else "ineq"
            constraint = {"type": constraint_type, "fun": constraint_func}

            if self.use_autograd:

                def constraint_jac(params, i=i):
                    return compute_quantities(params)["constraint_grads"][i]

                constraint["jac"] = constraint_jac
            constraints[repr(c)] = constraint

        return constraints

    def make_compute_quantities(self, x, y, **kwargs):
        def compute_quantities(params, x, y, **kwargs):
            with torch.set_grad_enabled(self.use_autograd):
                logger.debug(f"Computing quantities with {params=}...")
                self.params = params

                constraint_vals = [None] * len(self.constraints)
                handles: list[torch.utils.hooks.RemovableHandle] = []
                for i, constraint in enumerate(self.constraints):

                    def hook(*args, func=constraint, i=i):
                        constraint_vals[i] = func(*args)

                    # always_call=True ensures constraint evaluation occurs even when
                    # the model's forward pass raises an error
                    handles.append(
                        self.model.get_submodule(
                            constraint.module_name
                        ).register_forward_hook(hook, always_call=True)
                    )

                try:
                    y_pred = self.model(x, **kwargs)

                except exceptions.SimulationError as err:
                    logger.debug(
                        "Exception encountered when running model with params"
                        f" {self.params}: {err}"
                    )
                    loss = torch.tensor(np.inf)
                else:
                    loss = self.criterion(y_pred, y)
                finally:
                    for handle in handles:
                        handle.remove()

                pure_loss_item = loss.item()

                if self.regularizer is not None:
                    loss += self.regularizer(self.model)

                loss_item = loss.item()
                assert all(cv is not None for cv in constraint_vals)
                constraint_val_items = [cv.item() for cv in constraint_vals]

                if self.use_autograd:
                    grads = []
                    for i, tensor in enumerate([loss] + constraint_vals):
                        tname = "loss" if i == 0 else repr(self.constraints[i - 1])
                        retain_graph = i < self.num_requires_graph
                        if tensor.grad_fn is None:  # non-differentiable
                            grad = [np.nan for _ in range(len(params))]

                        else:
                            self.model.zero_grad()
                            logger.debug(
                                f"Beginning backward of {tname} with {retain_graph=}..."
                            )
                            tensor.backward(retain_graph=retain_graph)
                            logger.debug("Backward complete")

                            grad = self.params_grad
                            if (~grad.isfinite()).any():
                                raise exceptions.OptimizationError(
                                    "Gradient contains non-finite values:"
                                    f" {grad.tolist()}"
                                )
                            grad = grad.tolist()
                        grads.append(grad)

                    return {
                        "pure_loss": pure_loss_item,
                        "loss": loss_item,
                        "constraint_vals": constraint_val_items,
                        "grad": grads[0],
                        "constraint_grads": grads[1:],
                    }

                return {
                    "pure_loss": pure_loss_item,
                    "loss": loss_item,
                    "constraint_vals": constraint_val_items,
                }

        func = functools.partial(compute_quantities, x=x, y=y, **kwargs)
        func = functools.lru_cache(maxsize=self.cache_size)(func)
        return _ndarray_to_tuple(func)

    def make_minimizer_func(self, compute_quantities):
        def minimizer_func(params):
            quantities = compute_quantities(params)
            if self.use_autograd:
                return quantities["loss"], quantities["grad"]
            return quantities["loss"]

        return minimizer_func

    def make_callback(self, compute_quantities, constraints, hist, start_time):
        def callback(params):
            if (time_taken := time.time() - start_time) > self.timeout:
                raise exceptions.OptimizationError(
                    f"self.optimize has been running for {time_taken} seconds and timed"
                    " out."
                )

            # Compute loss, hopefully result is in cache
            loss_item = compute_quantities(params)["pure_loss"]

            # Compute constraints, hopefully results are in cache
            all_satisfied = True
            constraint_values_dict = {}

            for cname, constraint in constraints.items():
                val = constraint["fun"](params)
                if constraint["type"] == "eq":
                    satisfied = val == 0
                elif constraint["type"] == "ineq":
                    satisfied = val >= 0
                else:
                    raise RuntimeError()

                constraint_values_dict[cname] = val
                all_satisfied = all_satisfied and satisfied

            hist["loss"].append(loss_item)
            hist["satisfied"].append(all_satisfied)
            hist["params"].append(params)

            # logger.info(f"Loss: {loss_item}")
            logger.info(
                f"Loss: {loss_item}. Constraints:"
                f" {list(constraint_values_dict.values())}."
            )
            logger.debug(f"Constraints:\n{lib.pprint.pformat(constraint_values_dict)}")
            logger.debug(f"Cache info: {compute_quantities.__wrapped__.cache_info()}")
            logger.debug(f"self.params={self.params.detach().cpu()}")

        return callback

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> tuple[bool, float]:
        logger.info("Started optimizing...")
        logger.debug(lib.pprint.pformat(self.state_dict()))

        hist = {"loss": [], "satisfied": [], "params": []}
        start_time = time.time()

        compute_quantities = self.make_compute_quantities(x, y, **kwargs)
        minimizer_func = self.make_minimizer_func(compute_quantities)
        constraints = self.make_constraints(compute_quantities)
        callback = self.make_callback(compute_quantities, constraints, hist, start_time)

        try:
            result = sp_opt.minimize(
                minimizer_func,
                self.params.tolist(),
                method=self.method,
                jac=self.use_autograd,
                bounds=self.bounds,
                constraints=list(constraints.values()),
                tol=self.tol,
                callback=callback,
                options=self.options,
            )
            if not result.success:
                raise exceptions.OptimizationError(result.message)

        except exceptions.OptimizationError as err:
            logger.info(f"Optimization failed due to exception: {err}")

            logger.debug(f"Loss hist: {hist['loss']}")
            logger.debug(f"Satisfied hist: {hist['satisfied']}")

            if len(hist["loss"]) == 0:
                logger.info("No result returned since len(hist['loss']) = 0.")
                return False, np.inf

            loss_hist = np.array(hist["loss"])
            satisfied_hist = np.array(hist["satisfied"])
            params_hist = np.array(hist["params"])

            satisfied_loss_hist = loss_hist[
                satisfied_hist
            ]  # get only the losses where constraint is satisfied
            satisfied_params_hist = params_hist[satisfied_hist]

            logger.debug(f"Satisfied loss hist: {satisfied_loss_hist}")

            if len(satisfied_loss_hist) == 0:
                logger.info("No result returned since len(satisfied_loss_hist) = 0.")
                return False, np.inf

            idx = np.argmin(satisfied_loss_hist)

            self.params = satisfied_params_hist[idx]
            loss = satisfied_loss_hist[idx]

            logger.info(f"Returning result during optimization. Loss: {loss}.")
            logger.debug(self.params.detach().cpu())

            return True, loss

        self.params = result.x
        loss = compute_quantities(result.x)["pure_loss"]

        logger.info(f"Finished optimization successfully. Loss: {loss}")
        logger.debug(self.params.detach().cpu())

        return True, loss
