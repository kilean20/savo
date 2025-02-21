from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Tuple
import numpy as np
from collections import OrderedDict

import botorch
import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_mll

import io
import contextlib
import warnings
import re

# @contextlib.contextmanager    
# def suppress_outputs():
#     with contextlib.redirect_stdout(io.StringIO()):
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             yield


escaped_message = re.escape(" scaling the input ")   # suppress botorch warning regarding input scaling
warnings.filterwarnings("ignore", message=f".*{escaped_message}.*")

#from .kernel import RBFWithLinearTransformationKernel

# TODO: train multiple models of different kernels (including RBFWithLinearTransformationKernel) in parallel and get best model in terms of data likelihood


class Model(ABC):
    def __init__(self, ndim: int):
        """
        Abstract base class for models.

        Parameters
        ----------
        ndim : int
            Number of dimensions for the model.
        """
        assert type(ndim) == int
        self.ndim = ndim
        self.x_mean = np.zeros((1, ndim))
        self.x_std = np.ones((1, ndim))
        self.y_mean = np.zeros((1, ndim))
        self.y_std = np.ones((1, ndim))

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize input data.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Normalized input data.
        """
        return (x - self.x_mean) / self.x_std

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        """
        Normalize output data.

        Parameters
        ----------
        y : np.ndarray
            Output data.

        Returns
        -------
        np.ndarray
            Normalized output data.
        """
        return (y - self.y_mean) / self.y_std

    def unnormalize_x(self, xn: np.ndarray) -> np.ndarray:
        """
        Unnormalize input data.

        Parameters
        ----------
        xn : np.ndarray
            Normalized input data.

        Returns
        -------
        np.ndarray
            Unnormalized input data.
        """
        return xn * self.x_std + self.x_mean

    def unnormalize_y(self, yn: np.ndarray) -> np.ndarray:
        """
        Unnormalize output data.

        Parameters
        ----------
        yn : np.ndarray
            Normalized output data.

        Returns
        -------
        np.ndarray
            Unnormalized output data.
        """
        return yn * self.y_std + self.y_mean

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Fit the model to training data.

        Parameters
        ----------
        x : np.ndarray
            Training input data.
        y : np.ndarray
            Corresponding target values.
        """
        pass

    @abstractmethod
    def __call__(self, 
                 x: np.ndarray,
                 return_var: bool = True,
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the model.

        Parameters
        ----------
        x : np.ndarray
            Input data for which predictions are made.
        return_var : bool, optional
            Whether to return prediction variances, by default True.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Mean predictions and (optionally) variances.
        """
        pass

class GaussianProcess(Model):
    def __init__(self, 
                 ndim: int, 
                 prior_mean_model: Optional[Callable] = None,
                 prior_mean_model_grad: Optional[Callable] = None
                ):
        """
        Gaussian Process regression model.

        Parameters
        ----------
        ndim : int
            Number of dimensions.
        prior_mean_model : Optional[Callable], optional
            Prior mean function for the GP, by default None.
        prior_mean_model_grad : Optional[Callable], optional
            Gradient of the prior mean function, by default None.
        """
        super().__init__(ndim=ndim)
        self.prior_mean_model = prior_mean_model
        self.prior_mean_model_grad = prior_mean_model_grad

    def fit(self, 
            x: np.ndarray, 
            y: np.ndarray, 
            covar_module: Optional[gpytorch.kernels.Kernel] = None, 
            verbose: bool = False,
            debug = False,
            ) -> None:
        """
        Fit a Gaussian Process regressor to the training data.

        Parameters
        ----------
        x : np.ndarray
            Training input data.
        y : np.ndarray
            Corresponding target values.
        covar_module : Optional[gpytorch.kernels.Kernel], optional
            Covariance module for the GP, by default None.
        verbose : bool, optional
            Whether to print verbose information, by default False.
        """
        x = np.array(x).reshape(-1, self.ndim)
        y = np.array(y).reshape(-1, 1)
        if self.prior_mean_model is not None:
            y = y - self.prior_mean_model(x).reshape(-1, 1)
        self.x_mean = np.mean(x, axis=0).reshape(1, -1)
        self.x_std = np.std(x, axis=0).reshape(1, -1)
        self.y_mean = np.mean(y, axis=0).reshape(1, -1)
        self.y_std = np.std(y, axis=0).reshape(1, -1)

        xn = self.normalize_x(x)
        yn = self.normalize_y(y)
        
        if debug:
            print('[debug][savo][model]fit...')
            print('  xn.shape,yn.shape',xn.shape,yn.shape)

#         with suppress_outputs():
        self.model = SingleTaskGP(
                  train_X=torch.tensor(xn),
                  train_Y=torch.tensor(yn),
                  likelihood=GaussianLikelihood(),
                  covar_module=covar_module,
                  )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the Gaussian Process model.

        Parameters
        ----------
        x : np.ndarray
            Input data for which predictions are made.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Mean predictions and variances.
        """
        x = np.array(x).reshape(-1, self.ndim)
        if self.prior_mean_model is not None:
            y_prior = self.prior_mean_model(x).flatten()
        xn = self.normalize_x(x)
        with torch.no_grad():
            pred = self.model(torch.tensor(xn))

        y = pred.mean.numpy() * self.y_std + self.y_mean
        if self.prior_mean_model is not None:
            y += y_prior

        yvar = pred.variance.numpy() * self.y_std**2

        return y, yvar

    def get_grad(self, 
                 x: np.ndarray, 
                 prior_mean_model_grad: Optional[Callable] = None
                ) -> np.ndarray:
        """
        Calculate gradients of the model predictions with respect to input data.

        Parameters
        ----------
        x : np.ndarray
            Input data for which gradients are computed.
        prior_mean_model_grad : Optional[Callable], optional
            Gradient of the prior mean
        """
        prior_mean_model_grad = prior_mean_model_grad or self.prior_mean_model_grad
        x = np.array(x).reshape(-1, self.ndim)
        if self.prior_mean_model is not None:
            if prior_mean_model_grad is None:
                raise ValueError("prior_mean_model_grad function needs to be provided")
            else:
                grad_prior = prior_mean_model_grad(x)
        else:
            grad_prior = 0

        xn = self.normalize_x(x)
        x_ = torch.tensor(xn)
        x_.requires_grad = True
        y_ = self.model(x_).mean.sum()
        y_.backward()
        return x_.grad.detach().numpy()*self.y_std/self.x_std + grad_prior