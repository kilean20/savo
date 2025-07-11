from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Tuple, Dict
from collections import OrderedDict
from functools import partial
import io
import contextlib
import warnings
import re

import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

import gpytorch
from gpytorch.lazy import LazyTensor
from gpytorch.mlls import ExactMarginalLogLikelihood
# from gpytorch.likelihoods import GaussianLikelihood
# from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

from botorch.models import SingleTaskGP, HigherOrderGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch import fit_fully_bayesian_model_nuts
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.acquisition.objective import GenericMCObjective

from linear_operator.settings import _fast_solves

escaped_message = re.escape("added jitter of")
warnings.filterwarnings("ignore", message=f".*{escaped_message}.*")



# TODO: train multiple models of different kernels (including RBFWithLinearTransformationKernel) in parallel and get best model in terms of data likelihood



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


class Model(ABC):
    def __init__(self,
                x: torch.Tensor,
                y: torch.Tensor,
                ):
        """
        Abstract base class for models.

        Parameters
        ----------
        x : torch.Tensor
            Training input data.
        y : torch.Tensor
            Corresponding target values.
        """
        assert x.dim() == y.dim() == 2
        self.n_input_dim = x.shape[1]#if x.ndim > 1 else 1
        self.n_output_dim = y.shape[1]# if y.ndim > 1 else 1
        # print("y.shape",y.shape)
        # self.x = x.view(-1, self.n_input_dim)
        # self.y = y.view(-1, self.n_output_dim)
        self._set_standardization_params(x, y)


    def _set_standardization_params(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Set standardization parameters for input and output data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        y : torch.Tensor
            Output data.
        """
        self.x_min = x.min(dim=0).values.reshape(1, -1)
        self.x_diff = x.max(dim=0).values.reshape(1, -1) - self.x_min   + 1e-6

        self.y_mean = y.mean(dim=0).reshape(1, -1)
        self.y_std = y.std(dim=0).reshape(1, -1) + 1e-6

        if torch.any(self.x_diff == 0):
            raise ValueError("Input data has zero range in one or more dimensions.")
        if torch.any(self.y_std == 0):
            raise ValueError("Output data has zero standard deviation in one or more dimensions.")
        

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Normalized input data.
        """
        return (x - self.x_min) / self.x_diff
    
    def unnormalize_x(self, xn: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize input data.

        Parameters
        ----------
        xn : torch.Tensor
            Normalized input data.

        Returns
        -------
        torch.Tensor
            Unnormalized input data.
        """
        return xn * self.x_diff + self.x_min

    def standardize_y(self, y: torch.Tensor) -> torch.Tensor:
        """
        Standarize output data.

        Parameters
        ----------
        y : torch.Tensor
            Output data.

        Returns
        -------
        torch.Tensor
            Standarized output data.
        """
        return (y - self.y_mean) / self.y_std

    def unstandardize_y(self, yn: torch.Tensor) -> torch.Tensor:
        """
        UnStandarize output data.

        Parameters
        ----------
        yn : torch.Tensor
            Standarized output data.

        Returns
        -------
        torch.Tensor
            UnStandarized output data.
        """
        return yn * self.y_std + self.y_mean
    
    def standardize_yvar(self, yvar: torch.Tensor) -> torch.Tensor:
        """
        Standardize output variance.

        Parameters
        ----------
        yvar : torch.Tensor
            Output variance.

        Returns
        -------
        torch.Tensor
            Standardized output variance.
        """
        return yvar / (self.y_std ** 2)
    
    def unstandardize_yvar(self, yvar: torch.Tensor) -> torch.Tensor:
        """
        Unstandardize output variance.

        Parameters
        ----------
        yvar : torch.Tensor
            Standardized output variance.

        Returns
        -------
        torch.Tensor
            Unstandardized output variance.
        """
        return yvar * (self.y_std ** 2) 


    @abstractmethod
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """
        Fit the model to training data.

        Parameters
        ----------
        x : torch.Tensor
            Training input data.
        y : torch.Tensor
            Corresponding target values.
        """
        pass

    @abstractmethod
    def __call__(self, 
                x: torch.Tensor,
                ) -> Dict[ str, torch.Tensor]:
        """
        Make predictions using the model.

        Parameters
        ----------
        x : torch.Tensor
            Input data for which predictions are made.
        return_var : bool, optional
            Whether to return prediction variances, by default True.
        return_grad : bool, optional
            Whether to return gradients of predictions, by default False.
        return_grad_var : bool, optional
            Whether to return variances of gradients, by default False.

        Returns
        -------
        Dict[ str, torch.Tensor]
        """
        pass


class GaussianProcess(Model):
    def __init__(self, 
                x: torch.Tensor,
                y: torch.Tensor,
                yvar: Optional[torch.Tensor] = None,
                objective: Optional[Callable] = None,
                obj_func_noise: Optional[float] = None,
                ):
        """
        Gaussian Process regression model.

        Parameters
        ----------
        x : torch.Tensor
            Training input data.
        y : torch.Tensor
            Corresponding target values.
        """
        super().__init__(x,y)
        self.obj_func_noise = obj_func_noise
        self.fit(x,y,yvar)
        if self.n_output_dim > 1:
            assert objective is not None, "Objective function must be provided for multi-output Gaussian Process."
            # def objective_wrapper(x):
                
            #     orig_shape = x.shape
            #     if x.dim() == 2:
            #         batch_size, n_output_dim =  x.shape
            #         n_sample = 1
            #         q = 1
            #     elif x.dim() == 3:
            #         batch_size, q, n_output_dim = x.shape
            #         n_sample = 1
            #     elif x.dim() == 4:
            #         n_sample, batch_size, q, n_output_dim = x.shape
            #     else:
            #         raise ValueError(f"Unsupported input shape {x.shape} for objective function.")
            #     x = x.view(-1, n_output_dim)
            #     composite_obj = objective(x)  # objective should take input of shape (batch_size, n_output_dim) or (n_sample, batch_size, n_output_dim)
            #     return composite_obj.view(n_sample,batch_size,q)
                    
            # self.MCObjective = GenericMCObjective(objective_wrapper) # TODO: test GenericMCObjective 

            self.objective = objective
        else:
            # self.MCObjective = None
            self.objective = None

    def fit(self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            yvar: Optional[torch.Tensor] = None,
            ) -> None:
        """
        Fit a Gaussian Process regressor to the training data.

        Parameters
        ----------
        x : torch.Tensor
            Training input data.
        y : torch.Tensor
            Corresponding target values.
        """
        x = x.view(-1, self.n_input_dim)
        y = y.view(-1, self.n_output_dim)
        self._set_standardization_params(x, y)

        xn = self.normalize_x(x)
        yn = self.standardize_y(y)
        if yvar is not None:
            yvar = yvar.view(-1, self.n_output_dim)
            yvarn = self.standardize_yvar(yvar) 
        else:
            yvarn = None

        xn,yn,yvarn = remove_near_duplicates(xn,yn,yvarn)
        self.train_xn = xn
        self.train_yn = yn
        self.train_yvarn = yvarn

        if self.obj_func_noise is not None:
            yvar = torch.full_like(yn, self.obj_func_noise)
            yvarn = self.standardize_yvar(yvar) 

        if self.n_output_dim == 1:
            # covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.n_input_dim))


            covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.n_input_dim)
            )

            self.model = SingleTaskGP(
                    train_X=xn,
                    train_Y=yn,
                    train_Yvar=yvarn,
                    # likelihood=GaussianLikelihood(),
                    covar_module=covar_module,
                    )

            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)

            # TODO: add option for SaasFullyBayesianSingleTaskGP when input dimension is too high. 
            # SaasFullyBayesianSingleTaskGP training is too slow... simplify training using gradient decsent instead of NUTS
            # self.model = SaasFullyBayesianSingleTaskGP(
            #         train_X=xn,
            #         train_Y=yn,
            #         train_Yvar=yvarn,
            #         # likelihood=GaussianLikelihood(),
            #         # covar_module=covar_module,
            #         )
            # fit_fully_bayesian_model_nuts(
            #                             self.model,
            #                             warmup_steps=32,
            #                             num_samples=16,
            #                             thinning=16,
            #                             disable_progbar=True)

            
        else:
            # Higher-order GP for multi-output
            self.model = HigherOrderGP(
                train_X=xn,
                train_Y=yn,
                # train_Yvar=yvarn,
                latent_init='gp',
            )

            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            # TODO: improve HOGP training speed: limit epochs, adjust lr, etc
            with _fast_solves(True):
                fit_gpytorch_mll_torch(
                    mll, step_limit=1000, optimizer=partial(Adam, lr=0.01)
                )
            # TODO: add option for SaasFullyBayesianMultiTaskGP

    def __call__(self, 
                x: torch.Tensor,
                n_sample = 16,
                ) -> Dict[ str, torch.Tensor]:
        """
        Make predictions using the model.

        Parameters
        ----------
        x : torch.Tensor
            Input data for which predictions are made.
        Returns
        -------
        Dict[ str, torch.Tensor]
        """
        xn = self.normalize_x(x)   
        pred = self.model(xn)
        y = self.unstandardize_y(pred.mean)
        yvar = self.unstandardize_yvar(pred.variance)
        return y, yvar
    

    def get_grad(self, 
                x: torch.Tensor, 
                #  prior_mean_model_grad: Optional[Callable] = None
                n_samples: int = 1,
                ) -> torch.Tensor:
        """
        Calculate gradients of the model predictions with respect to input data.
        """
        batch_size = x.shape[0]
        x_ = x.clone().requires_grad_(True)  # x is shape of (batch_size, self.n_input_dim)
        if n_samples == 1:
            y, yvar = self(x_)
            if self.objective is not None:
                y = self.objective(y)  # TODO verify shape of output from GenericMCObjective
            y = y.sum()
            y.backward()
            dydx = x_.grad

        else:
            xn = self.normalize_x(x_)
            posterior = self.model.posterior(xn)
            samples = posterior.rsample(torch.Size([n_samples]))  #(n_samples, batch_size, self.n_output_dim) 
            # print(f"Shape of samples: {samples.shape}")
            samples = self.unstandardize_y(samples) #(n_samples*batch_size, self.n_output_dim)
            # print(f"Shape of samples after unstandardize: {samples.shape}")
            if self.objective is not None:
                samples = self.objective(samples).view(n_samples, -1, 1) #(n_samples, batch_size, 1) 
                # print(f"Shape of samples after objective: {samples.shape}")

            dydx = torch.autograd.grad(
                outputs=samples,
                inputs=x_,
                grad_outputs=torch.ones_like(samples),
                retain_graph=False,
                create_graph=False,
            )[0]/n_samples
        
        # print("dydx.shape",dydx.shape)
        # print(f"Shape of grads: {dydx.shape}") # (batch_size, self.n_input_dim) 
            # torch.autograd.grad returns a tuple of gradients.  
            # Each element corresponds to one tensor in the inputs argument. 
            # For a single input tensor, use grads[0] to get the gradient.
        return dydx # (batch_size, self.n_input_dim)  # averging over samples effectively reduce magnitude of the gradient from variation of gradients
    

    # def get_gradient_posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     self.model.eval()
    #     n_pts, n_dim = x.shape
    #     xn = self.normalize_x(x).clone().requires_grad_(True)
    #     mean_module, covar_module = self.model.mean_module, self.model.covar_module

    #     posterior_at_train = self.model.posterior(self.train_xn)
    #     K_XX_noisy = posterior_at_train.covariance_matrix
    #     y_minus_mu = self.train_yn - self.model.mean_module(self.train_xn)
        
    #     if isinstance(K_XX_noisy, LazyTensor):
    #         alpha = K_XX_noisy.inv_matmul(y_minus_mu)
    #     else:
    #         alpha = torch.linalg.solve(K_XX_noisy, y_minus_mu)

    #     mean_x = mean_module(xn)
    #     # --- FIX: Add allow_unused=True ---
    #     mean_grad = torch.autograd.grad(mean_x.sum(), xn, create_graph=True, allow_unused=True)[0]
    #     if mean_grad is None:
    #         mean_grad = torch.zeros_like(xn)

    #     K_xX_grad = torch.zeros(n_pts, n_dim, self.train_xn.shape[0], device=xn.device, dtype=xn.dtype)
    #     for i in range(n_pts):
    #         x_i = xn[i:i+1].detach().requires_grad_(True)
    #         k_iX = covar_module(x_i, self.train_xn)
    #         for j in range(self.train_xn.shape[0]):
    #             grad = torch.autograd.grad(k_iX[0, j], x_i, retain_graph=True, allow_unused=True)[0]
    #             if grad is None:
    #                 grad = torch.zeros_like(x_i)
    #             K_xX_grad[i, :, j] = grad.squeeze(0)

    #     mu_grad_normalized = mean_grad + torch.einsum('idm,mo->id', K_xX_grad, alpha)

    #     K_xx_grad_grad = torch.zeros(n_pts, n_dim, n_dim, device=xn.device, dtype=xn.dtype)
    #     for i in range(n_pts):
    #         def k_func(x_flat):
    #             return covar_module(x_flat.view(1, -1), x_flat.view(1, -1)).evaluate().squeeze()
    #         K_xx_grad_grad[i] = torch.autograd.functional.hessian(k_func, xn[i])

    #     K_xX_grad_T = K_xX_grad.permute(0, 2, 1)

        
    #     if isinstance(K_XX_noisy, LazyTensor):
    #         A_T = K_XX_noisy.inv_matmul(K_xX_grad_T)
    #     else:
    #         A_T = torch.linalg.solve(K_XX_noisy, K_xX_grad_T)
            
  
    #     cov_term = torch.bmm(A_T.permute(0, 2, 1), K_xX_grad_T)
    #     Sigma_grad_normalized = K_xx_grad_grad - cov_term

    #     scaling_factor = self.y_std / self.x_diff
    #     mu_grad = mu_grad_normalized * scaling_factor
    #     scaling_outer = scaling_factor.t() @ scaling_factor
    #     Sigma_grad = Sigma_grad_normalized * scaling_outer.unsqueeze(0)

    #     return mu_grad.detach(), Sigma_grad.detach()

# Add these methods inside your GaussianProcess class

    def _get_KXX_inv(self):
        """
        Get the inverse of the noisy kernel matrix, (K_XX + sigma^2*I)^-1,
        using the cached Cholesky decomposition from the prediction strategy.
        This is efficient and numerically stable.

        Returns:
            The inverse of the noisy kernel matrix.
        """
        # The prediction_strategy is populated when model.posterior() is called.
        # It caches the Cholesky decomposition of the noisy kernel matrix.
        # covar_cache is L_inv_upper, where L is the lower Cholesky factor.
        L_inv_upper = self.model.prediction_strategy.covar_cache.detach()
        # K_inv = L_inv_upper @ L_inv_upper.T
        return L_inv_upper @ L_inv_upper.transpose(-1, -2)

# In your GaussianProcess class, replace the old version of this function.

    def _get_KxX_dx_analytical(self, xn: torch.Tensor) -> torch.Tensor:
        """
        Computes the analytic derivative of the RBF kernel K(x,X) w.r.t. x.
        Assumes a gpytorch.kernels.ScaleKernel wrapping a gpytorch.kernels.RBFKernel.

        Args:
            xn: (n x D) Normalized test points.

        Returns:
            (n x D x N) The derivative of K(x,X) w.r.t. x.
        """
        # Ensure we have an RBF kernel
        if not isinstance(self.model.covar_module.base_kernel, gpytorch.kernels.RBFKernel):
            raise NotImplementedError("_get_KxX_dx_analytical is only implemented for RBF kernels.")

        Xn = self.train_xn
        n, D = xn.shape
        N = Xn.shape[0]  # N is the number of training points

        # Get kernel parameters from the trained model
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()

        # K(x, X)
        K_xX = self.model.covar_module(xn, Xn).evaluate()

        # Efficiently compute the derivative using broadcasting
        # The shapes must be (n, 1, D) and (1, N, D) to broadcast to (n, N, D)
        # The original code incorrectly used 'n' in the view for 'Xn'.
        diff = xn.view(n, 1, D) - Xn.view(1, N, D)
        
        # The einsum performs a batched scaling of each difference vector
        # by the corresponding kernel value k(x_i, X_j).
        # The result has shape (n, N, D).
        derivative_term = torch.einsum("nmd,nm->nmd", diff, K_xX)

        # Final scaling by -1/l^2
        K_xX_dx = (-1 / lengthscale ** 2) * derivative_term
        
        # We need the result in shape (n, D, N) for subsequent matrix multiplication
        return K_xX_dx.permute(0, 2, 1)

    def _get_Kxx_dx2_analytical(self) -> torch.Tensor:
        """
        Computes the analytic second derivative (Hessian) of the RBF kernel
        K(x,x') w.r.t. x and x', evaluated at x=x'.

        Returns:
            (D x D) The second derivative matrix.
        """
        if not isinstance(self.model.covar_module.base_kernel, gpytorch.kernels.RBFKernel):
            raise NotImplementedError("_get_Kxx_dx2_analytical is only implemented for RBF kernels.")
            
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
        outputscale = self.model.covar_module.outputscale.detach()
        
        # For RBF kernel, d^2k(x,x')/dxdx'|_{x=x'} = (sigma_f^2 / l^2) * I
        # where sigma_f is the outputscale.
        hessian = (torch.eye(self.n_input_dim, device=lengthscale.device) / lengthscale ** 2) * outputscale
        return hessian
    
# Add this main method to your GaussianProcess class.
# You can rename it to get_gradient_posterior to replace your old one.

    def _get_KXX_inv(self):
        """
        Get the inverse of the noisy kernel matrix, (K_XX + sigma^2*I)^-1,
        using the cached Cholesky decomposition from the prediction strategy.
        This is efficient and numerically stable.

        Returns:
            The inverse of the noisy kernel matrix.
        """
        # The prediction_strategy is populated when model.posterior() is called.
        # It caches the Cholesky decomposition of the noisy kernel matrix.
        # covar_cache is L_inv_upper, where L is the lower Cholesky factor.
        L_inv_upper = self.model.prediction_strategy.covar_cache.detach()
        # K_inv = L_inv_upper @ L_inv_upper.T
        return L_inv_upper @ L_inv_upper.transpose(-1, -2)

    def _get_KxX_dx_analytical(self, xn: torch.Tensor) -> torch.Tensor:
        """
        Computes the analytic derivative of the RBF kernel K(x,X) w.r.t. x.
        Assumes a gpytorch.kernels.ScaleKernel wrapping a gpytorch.kernels.RBFKernel.

        Args:
            xn: (n x D) Normalized test points.

        Returns:
            (n x D x N) The derivative of K(x,X) w.r.t. x.
        """
        # Ensure we have an RBF kernel
        if not isinstance(self.model.covar_module.base_kernel, gpytorch.kernels.RBFKernel):
            raise NotImplementedError("_get_KxX_dx_analytical is only implemented for RBF kernels.")

        Xn = self.train_xn
        n, D = xn.shape
        N = Xn.shape[0]

        # Get kernel parameters from the trained model
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()

        # K(x, X)
        K_xX = self.model.covar_module(xn, Xn).evaluate()

        # Efficiently compute the derivative using broadcasting
        diff = xn.view(n, 1, D) - Xn.view(1, N, D)
        
        # The einsum is correct but can be written more explicitly with broadcasting
        # derivative_term = torch.einsum("nmd,nm->nmd", diff, K_xX)
        derivative_term = diff * K_xX.unsqueeze(-1) # (n, N, D) * (n, N, 1) -> (n, N, D)

        # Final scaling by -1/l^2. Broadcasting handles scalar or ARD lengthscales.
        K_xX_dx = (-1 / lengthscale ** 2) * derivative_term
        
        # We need the result in shape (n, D, N) for subsequent matrix multiplication
        return K_xX_dx.permute(0, 2, 1)

    def _get_Kxx_dx2_analytical(self) -> torch.Tensor:
        """
        Computes the analytic second derivative (Hessian) of the RBF kernel
        K(x,x') w.r.t. x and x', evaluated at x=x'.

        Returns:
            (D x D) The second derivative matrix.
        """
        if not isinstance(self.model.covar_module.base_kernel, gpytorch.kernels.RBFKernel):
            raise NotImplementedError("_get_Kxx_dx2_analytical is only implemented for RBF kernels.")
            
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
        outputscale = self.model.covar_module.outputscale.detach()
        
        # For RBF kernel, d^2k(x,x')/dxdx'|_{x=x'} = (sigma_f^2 / l^2) * I
        # where sigma_f is the outputscale.
        hessian = (torch.eye(self.n_input_dim, device=lengthscale.device) / lengthscale ** 2) * outputscale
        return hessian
    
    def get_gradient_posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the posterior mean and covariance of the GP's gradient
        using direct analytical formulas for the RBF kernel derivatives. This
        implementation is adapted from the local-bo-mpd repository.

        Args:
            x: (n x D) Test points in the original, un-normalized space.

        Returns:
            A tuple containing:
            - mu_grad (torch.Tensor): The posterior mean of the gradient. Shape (n, D).
            - Sigma_grad (torch.Tensor): The posterior covariance of the gradient. Shape (n, D, D).
        """
        if self.n_output_dim > 1:
            raise NotImplementedError("get_gradient_posterior_analytical is only implemented for single-output GPs.")

        self.model.eval()
        
        # 1. Normalize inputs
        xn = self.normalize_x(x)

        # 2. CRUCIAL STEP: Call posterior to populate the prediction strategy cache.
        with torch.no_grad():
            self.model.posterior(self.train_xn)

        # 3. Get necessary components from the model in the NORMALIZED space
        K_inv = self._get_KXX_inv()
        K_xX_dx = self._get_KxX_dx_analytical(xn)
        Kxx_dx2 = self._get_Kxx_dx2_analytical()

        # 4. Compute posterior mean and covariance in the NORMALIZED space
        # Ensure train_yn is a column vector for matmul
        train_yn_col = self.train_yn.view(-1, 1)
        
        # alpha = (K_XX + sigma^2*I)^-1 * y_n
        alpha = K_inv @ train_yn_col
        
        # mu_grad_n = dK/dx_n @ alpha
        mu_grad_normalized = (K_xX_dx @ alpha).squeeze(-1)

        ## FIX: The matrix multiplication for the covariance term was likely the source of error.
        ## It should be a batched matrix multiplication (BMM).
        ## K_xX_dx is (n, D, N), K_inv is (N, N), K_xX_dx.transpose is (n, N, D)
        ## The result of `(K_xX_dx @ K_inv)` is (n, D, N).
        ## We then perform a BMM: `(n, D, N) @ (n, N, D)` -> `(n, D, D)`.
        cov_term = torch.bmm(K_xX_dx @ K_inv, K_xX_dx.transpose(-1, -2))
        
        # Sigma_grad_n = d^2k/dx_n^2 - (dK/dx_n @ K_inv @ (dK/dx_n)^T)
        # Kxx_dx2 is (D,D), cov_term is (n,D,D). Broadcasting handles the subtraction.
        Sigma_grad_normalized = Kxx_dx2 - cov_term
        Sigma_grad_normalized = Sigma_grad_normalized.clamp_min(1e-9)

        # 5. Un-scale the results to the original data space using the chain rule
        # For mu_grad: dy/dx = (dy/dy_n) * (dy_n/dx_n) * (dx_n/dx)
        # dy/dy_n = y_std, dx_n/dx = 1/x_diff
        scaling_factor = self.y_std / self.x_diff
        mu_grad = mu_grad_normalized * scaling_factor

        # For Sigma_grad: Cov(dy/dx) = J * Cov(dy_n/dx_n) * J^T
        # where J is a diagonal matrix with scaling_factor on the diagonal.
        # This results in element-wise multiplication: Sigma_ij * s_i * s_j
        scaling_outer = scaling_factor.t() @ scaling_factor
        Sigma_grad = Sigma_grad_normalized * scaling_outer

        return mu_grad.detach(), Sigma_grad.detach()
    
    
    def get_maximum_probable_gradient(self, x: torch.Tensor, minimize: bool = True) -> torch.Tensor:
        """
        ## FIX: This method has been completely replaced to match the reference.
        
        Get the Most Probable Descent (or Ascent) direction vector.

        This computes the MPD direction v = -Σ⁻¹μ / ||Σ⁻¹μ||, which provides an
        uncertainty-aware unit vector for the direction of movement.

        Args:
            x: Test points, shape (n, d) where n is number of points, d is input dimension.
            minimize: If True, return descent direction. If False, return ascent direction.

        Returns:
            v: The normalized MPD unit vector directions, shape (n, d).
        """
        mu_grad, Sigma_grad = self.get_gradient_posterior(x)
        
        assert Sigma_grad.dim() == 3, "Sigma_grad must be a 3D tensor (n, d, d)"
        assert mu_grad.dim() == 2, "mu_grad must be a 2D tensor (n, d)"
        assert Sigma_grad.shape[0] == mu_grad.shape[0], "Batch dimensions of Sigma_grad and mu_grad must match"
        assert Sigma_grad.shape[1] == Sigma_grad.shape[2], "Sigma_grad matrices must be square"
        assert Sigma_grad.shape[1] == mu_grad.shape[1], "Dimensions of Sigma_grad and mu_grad must match"
        
        n, d = mu_grad.shape
        
        # 1. Solve for the unnormalized direction v_raw = Sigma_inv * mu
        # Using cholesky_solve is often more stable than linalg.solve for SPD matrices.
        try:
            # Add jitter for numerical stability before Cholesky decomposition
            jitter = 1e-9 * torch.eye(d, device=Sigma_grad.device, dtype=Sigma_grad.dtype)
            L = torch.linalg.cholesky(Sigma_grad + jitter.unsqueeze(0))
            
            # Reshape mu_grad for batched solve
            mu_grad_reshaped = mu_grad.unsqueeze(-1) # Shape (n, d, 1)
            
            # v_raw = Sigma_inv * mu_grad
            v_raw = torch.cholesky_solve(mu_grad_reshaped, L).squeeze(-1) # Shape (n, d)
        except torch.linalg.LinAlgError:
            # Fallback to linalg.solve if Cholesky fails for any reason
            warnings.warn("Cholesky decomposition failed. Falling back to linalg.solve().")
            jitter = 1e-6 * torch.eye(d, device=Sigma_grad.device, dtype=Sigma_grad.dtype)
            v_raw = torch.linalg.solve(Sigma_grad + jitter.unsqueeze(0), mu_grad)

        # 2. Normalize the direction to get a unit vector. This is the MPD.
        v = torch.nn.functional.normalize(v_raw, p=2, dim=-1)

        return v