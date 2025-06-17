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
        self.x = x.view(-1, self.n_input_dim)
        self.y = y.view(-1, self.n_output_dim)
        self._set_standardization_params(x, y)

        xn = self.normalize_x(self.x)
        yn = self.standardize_y(self.y)
        if yvar is not None:
            if yvar.shape != self.y.shape:
                raise ValueError("yvar must have the same shape as y")
            yvar = yvar.view(-1, self.n_output_dim)
            yvarn = self.standardize_yvar(yvar) 
        else:
            yvarn = None

        xn,yn,yvarn = remove_near_duplicates(xn,yn,yvarn)

        if self.obj_func_noise is not None:
            yvar = torch.full_like(yn, self.obj_func_noise)
            yvarn = self.standardize_yvar(yvar) 

        if self.n_output_dim == 1:
            # covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.n_input_dim))
            self.model = SingleTaskGP(
                    train_X=xn,
                    train_Y=yn,
                    train_Yvar=yvarn,
                    # likelihood=GaussianLikelihood(),
                    # covar_module=covar_module,
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
        yvar = self.unstandardize_yvar(pred.var)
        return y, yvar
    

    def get_grad(self, 
                x: torch.Tensor, 
                #  prior_mean_model_grad: Optional[Callable] = None
                n_samples: int = 32,
                ) -> torch.Tensor:
        """
        Calculate gradients of the model predictions with respect to input data.
        """
        batch_size = x.shape[0]
        x_ = x.clone().requires_grad_(True)  # x is shape of (batch_size, self.n_input_dim)
        # y, yvar = self(x_)
        # if self.objective is not None:
        #     y = self.objective(y)  # TODO verify shape of output from GenericMCObjective
        # y = y.sum()
        # y.backward()
        # dydx = x_.grad

        # if not return_grad_var:
        #     return dydx
        # else:
            # x_ = x.clone().requires_grad_(True)
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
        )[0]
        # print("dydx.shape",dydx.shape)
        # print(f"Shape of grads: {dydx.shape}") # (batch_size, self.n_input_dim) 
            # torch.autograd.grad returns a tuple of gradients.  
            # Each element corresponds to one tensor in the inputs argument. 
            # For a single input tensor, use grads[0] to get the gradient.
        return dydx # (batch_size, self.n_input_dim)  # averging over samples effectively reduce magnitude of the gradient from variation of gradients