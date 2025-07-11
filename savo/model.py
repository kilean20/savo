from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Tuple, Dict
from functools import partial
import io
import contextlib
import warnings
import re
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.optim import Adam
from torch.distributions import Normal

import gpytorch
from gpytorch.lazy import LazyTensor
from gpytorch.mlls import ExactMarginalLogLikelihood

import botorch
from botorch.models import SingleTaskGP, HigherOrderGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.higher_order_gp import FlattenedStandardize
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim.core import OptimizationResult 

from linear_operator.settings import _fast_solves


escaped_message = re.escape("added jitter of")
warnings.filterwarnings("ignore", message=f".*{escaped_message}.*")


def remove_near_duplicates(train_x, train_y, train_yvar=None, tol=1e-9):
    # Round the data to specified tolerance
    rounded = torch.round(train_x / tol) * tol

    # Use unique with return_inverse to simulate np.unique(..., return_index=True)
    # We'll keep the first occurrence of each unique row
    unique_rows, inverse_indices = torch.unique(rounded, dim=0, return_inverse=True)
    first_indices = []
    seen = set()

    for i, idx in enumerate(inverse_indices.tolist()):
        if idx not in seen:
            first_indices.append(i)
            seen.add(idx)

    unique_indices = torch.tensor(first_indices, dtype=torch.long, device=train_x.device)

    # Filter out the duplicates
    train_x = train_x[unique_indices]
    train_y = train_y[unique_indices]
    if train_yvar is not None:
        train_yvar = train_yvar[unique_indices]

    return train_x, train_y, train_yvar

class LossCollectorCallback:
    """
    A simple callback class to collect and store the loss at each
    optimization step.
    """
    def __init__(self):
        self.losses = []
        self.steps = []

    def __call__(self, parameters: Dict[str, torch.Tensor], result: OptimizationResult) -> None:
        self.steps.append(result.step)
        self.losses.append(result.fval)
        self.runtime = result.runtime
        self.message = result.message

    def get_and_clear(self):
        """Returns the collected data and clears the internal lists."""
        data = (self.steps, self.losses)
        self.steps = []
        self.losses = []
        return data, self.runtime, self.message


class Model(ABC):
    def __init__(self,
                x: torch.Tensor,
                y: torch.Tensor,
                ):
        """
        Abstract base class for models.
        """
        assert x.dim() == y.dim() == 2
        self.n_input_dim = x.shape[1]
        self.n_output_dim = y.shape[1]

    @abstractmethod
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """
        Fit the model to training data.
        """
        pass

    @abstractmethod
    def __call__(self,
                x: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions using the model.
        """
        pass


class GaussianProcess(Model):
    def __init__(self,
                x: torch.Tensor,
                y: torch.Tensor,
                yvar: Optional[torch.Tensor] = None,
                objective: Optional[Callable] = None,
                obj_func_noise: Optional[float] = None,
                train_epochs: Optional[int] = 200,
                ):
        """
        Gaussian Process regression model.
        """
        super().__init__(x,y)
        self.obj_func_noise = obj_func_noise
        self.train_epochs = train_epochs
        self.fit(x,y,yvar,train_epochs=train_epochs)
        if self.n_output_dim > 1:
            assert objective is not None, "Objective function must be provided for multi-output Gaussian Process."
        self.objective = objective

    def fit(self,
            x: torch.Tensor,
            y: torch.Tensor,
            yvar: Optional[torch.Tensor] = None,
            train_epochs: Optional[int] = None,
            ) -> None:
        """
        Fit a Gaussian Process regressor to the training data.
        """
        x = x.view(-1, self.n_input_dim)
        y = y.view(-1, self.n_output_dim)
        if yvar is not None:
            yvar = yvar.view(-1, self.n_output_dim)

        x, y, yvar = remove_near_duplicates(x, y, yvar)
        self.train_x = x
        self.train_y = y
        self.train_yvar = yvar

        if self.obj_func_noise is not None and yvar is None:
            yvar = torch.full_like(y, self.obj_func_noise)

        if train_epochs is None:
            train_epochs = self.train_epochs

        loss_collector = LossCollectorCallback()
        if self.n_output_dim == 1:
            covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.n_input_dim)
            )
            self.model = SingleTaskGP(
                    train_X=x,
                    train_Y=y,
                    train_Yvar=yvar,
                    covar_module=covar_module,
                    input_transform=Normalize(d=x.shape[-1]),
                    outcome_transform=Standardize(m=y.shape[-1])
                    )

            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            # fit_gpytorch_mll(mll)
            
            with _fast_solves(True):
                fit_gpytorch_mll_torch(
                    mll, step_limit=train_epochs, optimizer=partial(Adam, lr=0.2), callback=loss_collector,
                )

        else:
            self.model = HigherOrderGP(
                train_X=x,
                train_Y=y,
                latent_init='gp',
                input_transform=Normalize(d=x.shape[-1]),
                outcome_transform=FlattenedStandardize(y.shape[1:]),
            )
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            with _fast_solves(True):
                fit_gpytorch_mll_torch(
                    mll, step_limit=train_epochs, optimizer=partial(Adam, lr=0.2), callback=loss_collector,
                )
        self.loss_history = loss_collector
        # print(f"Model training completed in {self.loss_history.runtime:.2f} seconds. Message: {self.loss_history.message}")
        # fig, ax = plt.subplots(figsize=(4, 2))
        # ax.plot(self.loss_history.losses)
        # ax.set_title("Loss Curve During Model Fitting")
        # ax.set_xlabel("Iteration")
        # ax.set_ylabel("Loss (Negative Marginal Log Likelihood)")
        # ax.grid(True)
        # plt.show()
        # plt.close(fig)


    def __call__(self,
                x: torch.Tensor,
                n_sample = 16,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions using the model.
        """
        posterior = self.model.posterior(x)
        return posterior.mean, posterior.variance


    def get_grad(self,
                x: torch.Tensor,
                n_samples: int = 1,
                ) -> torch.Tensor:
        """
        Calculate gradients of the model predictions with respect to input data.
        """
        x_ = x.clone().requires_grad_(True)
        if n_samples == 1:
            y, _ = self(x_)
            if self.objective is not None:
                y = self.objective(y)
            y = y.sum()
            y.backward()
            dydx = x_.grad
        else:
            posterior = self.model.posterior(x_)
            samples = posterior.rsample(torch.Size([n_samples]))
            if self.objective is not None:
                samples = self.objective(samples).view(n_samples, -1, 1)

            dydx = torch.autograd.grad(
                outputs=samples,
                inputs=x_,
                grad_outputs=torch.ones_like(samples),
                retain_graph=False,
                create_graph=False,
            )[0]/n_samples
        return dydx

    def _get_KXX_inv(self):
        """
        Get the inverse of the noisy kernel matrix, (K_XX + sigma^2*I)^-1.
        """
        L_inv_upper = self.model.prediction_strategy.covar_cache.detach()
        return L_inv_upper @ L_inv_upper.transpose(-1, -2)

    def _get_KxX_dx(self, xn: torch.Tensor) -> torch.Tensor:
        """
        Computes the derivative of the kernel K(x,X) w.r.t. x.
        Uses an efficient analytic formula for RBF kernels, and falls back
        to autograd for other kernel types. Operates on normalized data.
        """
        covar_module = self.model.covar_module
        is_rbf = (
            hasattr(covar_module, "base_kernel") and
            isinstance(covar_module.base_kernel, gpytorch.kernels.RBFKernel)
        )

        if is_rbf:
            if xn.dim() == 2:
                xn = xn.unsqueeze(0)
            Xn = self.model.train_inputs[0]
            lengthscale = covar_module.base_kernel.lengthscale.detach()
            K_xX = covar_module(xn, Xn).to_dense()
            diff = xn.unsqueeze(-2) - Xn.unsqueeze(-3)
            derivative_term = diff * K_xX.unsqueeze(-1)
            K_xX_dx = (-1 / lengthscale ** 2) * derivative_term
            return K_xX_dx.squeeze(0).permute(0, 2, 1)
        else:
            # ROBUST FALLBACK: Abandon vmap and loop over the batch dimension.
            from torch.func import jacrev
            if xn.dim() != 2:
                raise ValueError(f"Expected xn to be a 2D tensor, but got shape {xn.shape}")
            
            Xn = self.model.train_inputs[0]
            
            def single_x_kernel_fn(x_i):
                return covar_module(x_i.unsqueeze(0), Xn).to_dense().squeeze(0)

            jacobians = [jacrev(single_x_kernel_fn)(x_i) for x_i in xn]
            
            k_xX_dx_jac = torch.stack(jacobians, dim=0) 

            return k_xX_dx_jac.permute(0, 2, 1)


    def _get_Kxx_dx2(self) -> torch.Tensor:
        """
        Computes the second derivative (Hessian) of the kernel K(x,x')
        w.r.t. x and x', evaluated at x=x'.
        Uses an efficient analytic formula for RBF kernels, and falls back
        to autograd for other stationary kernel types.
        """
        covar_module = self.model.covar_module
        is_rbf = (
            hasattr(covar_module, "base_kernel") and
            isinstance(covar_module.base_kernel, gpytorch.kernels.RBFKernel)
        )

        if is_rbf:
            lengthscale = covar_module.base_kernel.lengthscale.detach()
            outputscale = covar_module.outputscale.detach()
            hessian = (torch.eye(self.n_input_dim, device=lengthscale.device) / lengthscale ** 2) * outputscale
            return hessian
        else:
            from torch.func import jacrev
            
            n_dims = self.n_input_dim
            device = self.train_x.device
            x0 = torch.zeros(1, n_dims, device=device)
            
            # FINAL FIX: Explicitly convert to a dense tensor before squeezing.
            grad_k_wrt_xp_fn = jacrev(
                lambda x, xp: covar_module(x, xp).to_dense().squeeze(),
                argnums=1
            )
            hessian_fn = jacrev(grad_k_wrt_xp_fn, argnums=0)
            hessian = hessian_fn(x0, x0)
            return hessian.squeeze()

    def get_gradient_posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the posterior mean and covariance of the GP's gradient.
        """
        if self.n_output_dim > 1:
            raise NotImplementedError("get_gradient_posterior is only implemented for single-output GPs.")

        self.model.eval()
        input_transform = self.model.input_transform
        x_scale = input_transform.ranges
        outcome_transform = self.model.outcome_transform
        y_std = outcome_transform.stdvs

        xn = self.model.input_transform(x)
        with torch.no_grad():
            self.model.posterior(self.model.train_inputs[0])

        K_inv = self._get_KXX_inv()
        K_xX_dx = self._get_KxX_dx(xn)
        Kxx_dx2 = self._get_Kxx_dx2()
        train_yn_col = self.model.train_targets.view(-1, 1)

        alpha = K_inv @ train_yn_col
        mu_grad_normalized = (K_xX_dx @ alpha).squeeze(-1)
        cov_term = torch.bmm(K_xX_dx @ K_inv, K_xX_dx.transpose(-1, -2))
        Sigma_grad_normalized = Kxx_dx2 - cov_term
        Sigma_grad_normalized = Sigma_grad_normalized.clamp_min(1e-9)

        scaling_factor = y_std / x_scale
        mu_grad = mu_grad_normalized * scaling_factor
        scaling_outer = scaling_factor.t() @ scaling_factor
        Sigma_grad = Sigma_grad_normalized * scaling_outer

        return mu_grad.detach(), Sigma_grad.detach()


    def get_maximum_probable_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the Most Probable Ascent direction vector.
        """
        mu_grad, Sigma_grad = self.get_gradient_posterior(x)

        assert Sigma_grad.dim() == 3, "Sigma_grad must be a 3D tensor (n, d, d)"
        assert mu_grad.dim() == 2, "mu_grad must be a 2D tensor (n, d)"
        assert Sigma_grad.shape[0] == mu_grad.shape[0], "Batch dimensions of Sigma_grad and mu_grad must match"
        assert Sigma_grad.shape[1] == Sigma_grad.shape[2], "Sigma_grad matrices must be square"
        assert Sigma_grad.shape[1] == mu_grad.shape[1], "Dimensions of Sigma_grad and mu_grad must match"

        n, d = mu_grad.shape

        with torch.no_grad():
            try:
                jitter = 1e-9 * torch.eye(d, device=Sigma_grad.device, dtype=Sigma_grad.dtype)
                L = torch.linalg.cholesky(Sigma_grad + jitter.unsqueeze(0))
                mu_grad_reshaped = mu_grad.unsqueeze(-1)
                v_raw = torch.cholesky_solve(mu_grad_reshaped, L).squeeze(-1)
            except torch.linalg.LinAlgError:
                warnings.warn("Cholesky decomposition failed. Falling back to linalg.solve().")
                jitter = 1e-6 * torch.eye(d, device=Sigma_grad.device, dtype=Sigma_grad.dtype)
                v_raw = torch.linalg.solve(Sigma_grad + jitter.unsqueeze(0), mu_grad)

            v = torch.nn.functional.normalize(v_raw, p=2, dim=-1)
            dot_product = torch.sum(mu_grad * v, dim=-1, keepdim=True)
            mu_grad_projected_on_v = dot_product * v

            # Calculate the argument for the CDF: sqrt(μ^T Σ⁻¹ μ) = sqrt(μ^T v_raw)
            mu_dot_v_raw = torch.sum(mu_grad * v_raw, dim=-1)
            cdf_arg = torch.sqrt(torch.clamp_min(mu_dot_v_raw, 0.0))
            # Compute the probability using the standard normal CDF, Φ(cdf_arg)
            probability = Normal(0, 1).cdf(cdf_arg)

        return mu_grad_projected_on_v, probability