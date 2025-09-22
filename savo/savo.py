'''
(S)urrogate Gradient
(A)ssisted
(V)ery high dimension
(O)ptimization
'''
# import matplotlib.pyplot as plt
from typing import List, Union, Optional, Callable, Tuple, Dict
from copy import deepcopy as copy
import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
import sys




import os
from datetime import datetime
# script_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(script_dir,'../../', 'machineIO'))
# from machineIO.construct_machineIO import Evaluator
# from machineIO.utils import df_mean_var, df_mean

from .model import GaussianProcess
from .util import df_mean_var, get_primes


class savo:
    def __init__(self,
                 control_CSETs: List[str],
                 control_RDs: List[str],
                 control_min: List[float],
                 control_max: List[float],
                 control_maxstep: List[float],

                 objective_names: List[str],
                 composite_objective_name: str,
                 
                #  evaluator: Evaluator,
                 evaluator: Callable,
                 obj_func: Optional[Callable] = None,
                 obj_func_grad: Optional[Callable] = None,
                 obj_func_noise: Optional[float] = None,

                 n_train_data: Optional[int] = None,
                 model_train_budget: Optional[int] = 200,
                 n_grad_ave: Optional[int] = None,
                 lr: Optional[float] = None,
                 lrES: Optional[float] = None,
                 gain_scale_ES: Optional[float] = 1.0,
                 optimizer: Optional[Callable] = None,
                 prev_history: Optional[Dict] = None,
                 
                 clip_gradient: Optional[bool] = False,
                 use_ctrRD_to_train: bool = True,
                 minimize: Optional[bool] = False,
                 adapt_ES_gain: Optional[bool] = False,
                 asynchronous: Optional[bool] = False,

                 **kwargs,
                 ):  
        self.now = datetime.now()
        self.control_CSETs = control_CSETs
        self.control_RDs = control_RDs
        self.control_maxstep = np.array(control_maxstep).reshape(-1)
        self.control_min = np.array(control_min).reshape(-1)
        self.control_max = np.array(control_max).reshape(-1)
        self.ndim = len(self.control_CSETs)
        assert len(self.control_RDs) == self.ndim, "control_RDs must have same length as control_CSETs"
        assert len(self.control_maxstep) == self.ndim, "control_maxstep must have same length as control_CSETs"
        assert len(self.control_min) == self.ndim, "control_min must have same length as control_CSETs"
        assert len(self.control_max) == self.ndim, "control_max must have same length as control_CSETs"
        assert evaluator.control_CSETs == control_CSETs, "evaluator.control_CSETs must match control_CSETs"
        assert evaluator.control_RDs == control_RDs, "evaluator.control_RDs must match control_RDs"
        self.objective_names = objective_names
        assert isinstance(composite_objective_name, str), "composite_objective_name must be a string"
        self.composite_objective_name = composite_objective_name
        self.evaluator = evaluator
        df = evaluator.read()  # read from machine and process the data
        assert set(control_CSETs).issubset(df.columns), "control_CSETs must be a subset of evaluators return df columns"
        assert set(control_RDs).issubset(df.columns), "control_RDs must be a subset of evaluators return df columns"    
        assert set(objective_names).issubset(df.columns), "objective_names must be a subset of evaluators return df columns"
        assert composite_objective_name in df.columns, "composite_objective_name must be in evaluators return df columns"
        mean, var = df_mean_var(df)
        self.x = mean[self.control_CSETs].values
        self.xrd = mean[self.control_RDs].values  # readback of set values
        self.y = mean[self.composite_objective_name]
        self.yvar = var[self.composite_objective_name]

        self.obj_func = obj_func
        self.obj_func_grad = obj_func_grad
        self.obj_func_noise = obj_func_noise
        if self.obj_func is not None:
            self.multi_y = mean[self.objective_names]
            self.multi_yvar = var[self.objective_names]
        
        self.n_train_data = n_train_data or min(10 * self.ndim, 1024) 
        self.model_train_budget = model_train_budget or 200
        self.n_grad_ave = n_grad_ave or 4
        self.lr = lr or 1e-2
        self.lrES = lrES or 1e-2
        self.optimizer = optimizer

        self.minimize = minimize
        self.clip_gradient = clip_gradient
        self.use_ctrRD_to_train = use_ctrRD_to_train
        self.adapt_ES_gain = adapt_ES_gain
        self.asynchronous = asynchronous
    
        if prev_history is not None:
            self.history = copy(prev_history)
            assert 'x' in self.history and 'xrd' in self.history and 'y' in self.history and 'yvar' in self.history and 'multi_y' in self.history and 'multi_yvar' in self.history# and 'model' in self.history
            assert len(self.history['x']) == len(self.history['xrd']) == len(self.history['y'])
            if not 'ES' in self.history:
                self.history['ES'] = { 't'  :[],
                                       'kES':[],
                                       'aES':[],}
                self.t = 0
            else:
                self.t = self.history['ES']['t'][-1]
                self.kES = self.history['ES']['kES'][-1]
                self.aES = self.history['ES']['aES'][-1]
        else:
            # ES dithering freq = 2pi*nu/10,  nu in [0.5,1], factor 10 for at least 10 iteration for smoothing (with dt=1).
            # self.wES = 2*np.pi * (0.5*(np.arange(self.ndim) +0.5) / self.ndim + 0.5)/ 10
            sqrt_primes = np.sqrt(get_primes(self.ndim))#[::-1]
            np.random.shuffle(sqrt_primes)
            max_freq = np.max(sqrt_primes)
            self.wES = 2 * np.pi * sqrt_primes / (10*max_freq)
            
            # self.aES = self.control_maxstep  # for dx = aES*Cos(...)
            # for dx = (aES*wES)**0.5*Cos(...)  -> stepsize = (aES*wES)**0.5*cos(...) -> stepsize**2 = (aES*wES)*(1/2)
            self.aES = (2*self.control_maxstep)**2/np.mean(self.wES)

            # for dx = aES*Cos(...)
            # ES gain, assuming change of objective in each iter is Delta(obj) ~ O(sum((alpha_i*w_i)**2))
            # and kES*Delta(obj) ~ wES*(dt=1), so that phase change from gain is about same order
            # of dithering phase change.
            # self.kES = gain_scale_ES*3*self.wES/(np.sum((self.aES*self.wES)**2))

            # for dx = (aES*wES)**0.5*Cos(...)
            # we need, kES << 1/(aES*wES)
            # self.kES = gain_scale_ES*self.wES/np.sqrt(np.sum((self.aES*self.wES)**2))
            self.kES = 2*self.lrES/self.aES


            self.history = {
                'cpu_time':[],
                'x': [],
                'xrd': [],
                'y': [],
                'yvar': [],
                'multi_y':[],
                'multi_yvar':[],
                'dydx':[],
                'probability':[],
                'cov_trace':[],
                'model_fitloss_histroy':[],
				'model_fit_time':[],
                'ES': {'t':[0],
                       'kES':[self.kES],
                       'aES':[self.aES],}
            }
            self.t = 0
        self.norm_aES = np.linalg.norm(self.aES)
        

        #set current ctr set and do not wait to read
        self.future = self.evaluator.submit(self.x)
        self.process_evaluator_future()
        self.TurBO_failure_counter = 0
        self.TurBO_success_counter = 0  
        self.TurBO_failure_tolerance = 3
        self.TurBO_success_tolerance = 3
        self.TurBO_length = self.control_maxstep.copy()
        

    def process_evaluator_future(self):
        now = datetime.now()
        df, ramping_df = self.evaluator.get_result(self.future)
        self.future = None
        mean, var = df_mean_var(df)
        self.x = mean[self.control_CSETs].values
        self.xrd = mean[self.control_RDs].values  # readback of set values
        self.y = mean[self.composite_objective_name]
        self.yvar = var[self.composite_objective_name]
        self.history['x'].append(copy(self.x))
        self.history['xrd'].append(copy(self.xrd))
        self.history['y'].append(copy(self.y))
        self.history['yvar'].append(copy(self.yvar))
        self.history['cpu_time'].append((now - self.now).total_seconds())
        if self.obj_func is not None:
            self.multi_y = mean[self.objective_names]
            self.multi_yvar = var[self.objective_names]
            self.history['multi_y'].append(copy(self.multi_y))
            self.history['multi_yvar'].append(copy(self.multi_yvar))

        self.now = now

    def train_model(self,
                train_x: Optional[torch.Tensor] = None,
                train_y: Optional[torch.Tensor] = None,
                train_yvar: Optional[torch.Tensor] = None,
                n_train_data: Optional[int] = None,
                model_train_budget: Optional[int] = None,
        ):
        n_train_data = n_train_data or self.n_train_data
        n_train_data = min(n_train_data, 400)
        model_train_budget = model_train_budget or self.model_train_budget
        if train_x is None:
            if self.use_ctrRD_to_train:
                train_x    = torch.tensor(np.array(self.history['xrd' ][-n_train_data:]),dtype=torch.float64)
            else:
                train_x    = torch.tensor(np.array(self.history['x'   ][-n_train_data:]),dtype=torch.float64)
            
            if self.obj_func is None:
                train_y    = torch.tensor(np.array(self.history['y'   ][-n_train_data:]),dtype=torch.float64).view(-1,1)
                train_yvar = torch.tensor(np.array(self.history['yvar'][-n_train_data:]),dtype=torch.float64).view(-1,1)
            else:
                train_y    = torch.tensor(np.array(self.history['multi_y'   ][-n_train_data:]),dtype=torch.float64)
                train_yvar = torch.tensor(np.array(self.history['multi_yvar'][-n_train_data:]),dtype=torch.float64)
        else:
            assert train_y is not None, "train_y must be provided if train_x is provided"
            assert train_x.shape[0] == train_y.shape[0], "train_x and train_y must have same number of rows"
            assert train_x.shape[1] == self.ndim, "train_x must have same number of columns as control_CSETs"
            train_x = torch.as_tensor(train_x,dtype=torch.float64)
            train_y = torch.as_tensor(train_y,dtype=torch.float64)
            if train_yvar is not None:
                train_yvar = torch.as_tensor(train_yvar,dtype=torch.float64)
        if train_yvar is not None:
            if train_yvar.abs().max() < 1e-9:
                train_yvar = None

        self.model = GaussianProcess(train_x,train_y,train_yvar,self.obj_func,obj_func_noise=self.obj_func_noise, train_epochs=model_train_budget)
        self.history['model_fitloss_histroy'].append(self.model.loss_history.losses)
        self.history['model_fit_time'].append(self.model.loss_history.runtime)


    # def _adaptive_kES(self, update_rate=0.01, grad=None):
    #     if len(self.history['y']) < 16:
    #         self.history['ES']['kES'].append(None)
    #         return
    #     grad = grad or self.obj_func_grad

    #     if grad:
    #         kES_step = 1.41421 / np.abs(self.aES * grad(self.x))  - self.kES
    #     else:
    #         y = self.history['y'][-16:]
    #         std = np.std(y)
    #         variation = std
    #         kES_step = 0.1 * self.wES/(variation + 1e-15)  - self.kES
    #     # slow update on ES gain
    #     self.kES += np.clip(kES_step,
    #                         a_min=-update_rate * self.kES,
    #                         a_max= update_rate * self.kES)
        
    #     self.history['ES']['t'].append(copy(self.t))
    #     self.history['ES']['aES'].append(copy(self.aES))
    #     self.history['ES']['kES'].append(copy(self.kES))


    def _get_ES_setp(self):
        self.t += 1
        if self.minimize:
            # return self.aES * np.sin(self.t * self.wES + self.kES * self.y)
            return (self.aES*self.wES)**0.5 * np.sin(self.t * self.wES + self.kES * self.y)
            # return self.dtES*(self.aES*self.wES)**0.5*np.cos(self.dtES*self.t*self.wES + self.kES*self.y)
        else:
            return (self.aES*self.wES)**0.5 * np.sin(self.t * self.wES - self.kES * self.y)
            # return self.dtES*(self.aES*self.wES)**0.5*np.cos(self.dtES*self.t*self.wES - self.kES*self.y)
        



    def _get_dydx(self, n_grad_ave, penalize_uncertain_gradient):
        # neighboring points (staggered points toward historical trajectory) for averaged gradient.
        x_ = np.zeros((n_grad_ave,len(self.x)))
        x_[0 ,:] = self.x
        x_[1:,:] = 0.5*(self.x.reshape(1,-1) + np.array(self.history['x'][-n_grad_ave+1:]))
        if penalize_uncertain_gradient:
            dydx, probability = self.model.get_maximum_probable_gradient(torch.from_numpy(x_))
            mask = probability < 0.65
            self.history['probability'].append(probability.mean().item()    )
            probability[mask] = 0.0
            dydx = dydx * probability.view(-1,1)
        else:
            dydx = self.model.get_grad(torch.from_numpy(x_))
        cov_trace = self.model.get_gradient_covariance_trace(torch.from_numpy(x_))
        self.history['cov_trace'].append(cov_trace.mean().item())
        self.history['dydx'].append(dydx.mean(axis=0))
        return dydx


    def _get_SG_step(self,
                 n_grad_ave = None,
                 optimizer = None,
                 penalize_uncertain_gradient = False,
                 ):
        n_grad_ave = n_grad_ave or self.n_grad_ave
        optimizer = optimizer or self.optimizer
        dydx = self._get_dydx(n_grad_ave, penalize_uncertain_gradient=penalize_uncertain_gradient).detach().numpy()
        
        if optimizer is not None:
            SG_step = optimizer(dydx.mean(axis=0))
        else:
            SG_step = dydx.mean(axis=0)
        
        if self.minimize:
            return -SG_step
        else:
            return  SG_step
        

    def TurBO_step(self):
        """
        Trust Region Bayesian Optimization (TurBO) step.
        Uses a single trust region box around the current best point.
        """
        # Train the model first
        self.train_model()
        
        # Get current best point and value
        if len(self.history['y']) == 0:
            current_best_x = self.x.copy()
            current_best_y = self.y
        else:
            if self.minimize:
                best_idx = np.argmin(self.history['y'])
                current_best_y = self.history['y'][best_idx]
            else:
                best_idx = np.argmax(self.history['y'])
                current_best_y = self.history['y'][best_idx]
            
            if self.use_ctrRD_to_train:
                current_best_x = np.array(self.history['xrd'][best_idx])
            else:
                current_best_x = np.array(self.history['x'][best_idx])
        
        # Define trust region bounds around current best point
        tr_lb = np.maximum(current_best_x - self.TurBO_length, self.control_min)
        tr_ub = np.minimum(current_best_x + self.TurBO_length, self.control_max)
        
        # Convert to torch tensors
        tr_bounds = torch.tensor(np.column_stack([tr_lb, tr_ub]).T, dtype=torch.float64)
        
        # Set up acquisition function (Upper Confidence Bound)
        # beta = 2.0  # Exploration parameter
        best_f = torch.tensor(current_best_y, dtype=torch.float64)
        if self.minimize:
            # For minimization, we want to maximize the negative of the function
            acq_func = ExpectedImprovement(self.model.model, best_f=best_f, maximize=False)
            # acq_func = ExpectedImprovement(self.model.model, beta=beta, maximize=False)
        else:
            acq_func = ExpectedImprovement(self.model.model, best_f=best_f, maximize=True)
            # acq_func = ExpectedImprovement(self.model.model, beta=beta, maximize=True)
        
        # Optimize acquisition function within trust region
        try:
            candidate, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=tr_bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
            )
            
            # Convert back to numpy and update position
            next_x = candidate.detach().numpy().flatten()
            
            # Evaluate the candidate point
            self.x = next_x
            self.future = self.evaluator.submit(self.x)
            self.process_evaluator_future()
            
            # Update trust region based on success/failure
            improvement_threshold = 1e-3 * np.abs(current_best_y) if current_best_y != 0 else 1e-3
            
            if self.minimize:
                improved = (self.y < current_best_y - improvement_threshold)
            else:
                improved = (self.y > current_best_y + improvement_threshold)
            
            if improved:
                # Success: expand trust region
                self.TurBO_success_counter += 1
                self.TurBO_failure_counter = 0
                
                if self.TurBO_success_counter >= self.TurBO_success_tolerance:
                    self.TurBO_length = 2.0 * self.TurBO_length
                    self.TurBO_success_counter = 0
            else:
                # Failure: shrink trust region
                self.TurBO_failure_counter += 1
                self.TurBO_success_counter = 0
                
                if self.TurBO_failure_counter >= self.TurBO_failure_tolerance:
                    self.TurBO_length = 0.5 * self.TurBO_length
                    self.TurBO_failure_counter = 0
                    
        except Exception as e:
            print(f"TurBO optimization failed: {e}")
            # Fallback: random point within trust region
            self.x = tr_lb + np.random.rand(len(tr_lb)) * (tr_ub - tr_lb)
            self.future = self.evaluator.submit(self.x)
            self.process_evaluator_future()
        

    def step(self,
             lr: Optional[float] = None,
             lrES_scaler: Optional[float] = None,
             optimizer: Optional[Callable] = None,
             penalize_uncertain_gradient: bool = False,
             normalize_gradient_step: bool = False,
             lr_adapt2obj_params: Optional[List[float]] = [10,0.8],
             asynchronous: Optional[bool] = None,
             TurBO: Optional[bool] = False,
            ):
        if TurBO:
            self.TurBO_step()
            return
        
        if lr is None:
            lr = self.lr
        if lrES_scaler is None:
            lrES_scaler = 1.0
            
        if lr_adapt2obj_params is not None:

            reduction = 1/(1+np.exp(lr_adapt2obj_params[0]*(self.y- lr_adapt2obj_params[1])))

            lrES_scaler = lrES_scaler*reduction**2
            lr = lr*reduction

        if asynchronous is None:
            asynchronous = self.asynchronous
                
        if self.future is None:
            dxES = self._get_ES_setp()
            self.x += dxES*lrES_scaler
            self.future = self.evaluator.submit(self.x)

        if not asynchronous:
            self.process_evaluator_future()

        dxES = self._get_ES_setp()
        if lr > 0:
            self.train_model()
            dxSG = self._get_SG_step(optimizer=optimizer,penalize_uncertain_gradient=penalize_uncertain_gradient)
            if normalize_gradient_step:
                norm_dxSG = np.linalg.norm(dxSG)
                if norm_dxSG < 1e-15:
                    dxSG = 0
                else:
                    dxSG = dxSG / (norm_dxSG + 1e-15) * self.norm_aES
            if self.clip_gradient:
                dxSG = np.clip(dxSG,a_min = -self.control_maxstep,a_max= self.control_maxstep)
        else:
            dxSG = 0

        if asynchronous:
            self.process_evaluator_future()

        self.x += dxSG*lr + dxES*lrES_scaler
        self.x = np.clip(self.x, a_min=self.control_min, a_max=self.control_max)
        self.future = self.evaluator.submit(self.x)

    def evaluate(self,x):
        self.x = x
        self.future = self.evaluator.submit(self.x)
        self.process_evaluator_future()
        return -self.y