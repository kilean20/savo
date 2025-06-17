'''
(S)urrogate Gradient
(A)ssisted
(V)ery high dimension
(O)ptimization
'''
from typing import List, Union, Optional, Callable, Tuple, Dict
from copy import deepcopy as copy
import numpy as np
import torch
import sys


import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir,'../../', 'machineIO'))
from machineIO.construct_machineIO import Evaluator
from machineIO.utils import df_mean_var, df_mean

from .model import GaussianProcess
from .util import obj_func_wrapper
from .optim import Adam


class savo:
    def __init__(self,
                 control_CSETs: List[str],
                 control_RDs: List[str],
                 control_min: List[float],
                 control_max: List[float],
                 control_maxstep: List[float],

                 objective_PVs: List[str],
                 composite_objective_name: str,
                 
                 evaluator: Evaluator,
                 obj_func: Optional[Callable] = None,
                 obj_func_grad: Optional[Callable] = None,

                 n_grad_data: Optional[int] = None,
                 n_grad_ave: Optional[int] = None,
                 lr: Optional[float] = None,
                 optimizer: Optional[Callable] = None,
                 prev_history: Optional[Dict] = None,

                 clip_gradient: Optional[bool] = True,
                 use_ctrRD_to_train: bool = True,
                 minimize: Optional[bool] = False,
                 
                 ):
        '''
        '''        
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
        self.objective_PVs = objective_PVs
        assert isinstance(composite_objective_name, str), "composite_objective_name must be a string"
        self.composite_objective_name = composite_objective_name
        self.evaluator = evaluator
        df = evaluator.read()  # read from machine and process the data
        assert set(control_CSETs).issubset(df.columns), "control_CSETs must be a subset of evaluators return df columns"
        assert set(control_RDs).issubset(df.columns), "control_RDs must be a subset of evaluators return df columns"    
        assert set(objective_PVs).issubset(df.columns), "objective_PVs must be a subset of evaluators return df columns"
        assert composite_objective_name in df.columns, "composite_objective_name must be in evaluators return df columns"
        mean, var = df_mean_var(df)
        self.x = mean[self.control_CSETs].values
        self.xrd = mean[self.control_RDs].values  # readback of set values
        self.y = mean[self.composite_objective_name]
        self.yvar = var[self.composite_objective_name]
        
        self.n_grad_data = n_grad_data or min(10 * self.ndim, 1024) 
        self.n_grad_ave = n_grad_ave or 16
        self.lr = lr or 1e-2
        self.optimizer = optimizer or Adam()

        self.minimize = minimize
        self.clip_gradient = clip_gradient
        self.use_ctrRD_to_train = use_ctrRD_to_train

        self.adam_m = 0
        self.adam_v = 0
        self.adam_t = 0
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
    
        if prev_history is not None:
            self.history = copy(prev_history)
            assert 'x' in self.history and 'xrd' in self.history and 'y' in self.history and 'yvar' in self.history# and 'model' in self.history
            assert len(self.history['x']) == len(self.history['xrd']) == len(self.history['y']) == len(self.history['yvar'])
            if not 'ES' in self.history:
                self.history['ES'] = { 't'  :[],
                                       'kES':[],
                                       'aES':[],}
                self.t = 0
            else:
                self.t = self.history['ES']['t'][-1]
        else:
            self.history = {
                'x': [],
                'xrd': [],
                'y': [],
                'yvar': [],
                # 'model': {
                #     'constraints': {},
                #     'objective': {}
                # },
                'ES': {'t':[],
                       'kES':[],
                       'aES':[],}
            }
            self.t = 0
        
        #set current ctr set and do not wait to read
        self.future = self.evaluator.submit(self.x)
        self.process_evaluator_future()

    def process_evaluator_future(self):
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

            
    def runES(self,
              n_loop: int,
              aES: Optional[float] = None,
              kES: Optional[float] = None,
              adaptive_ES: Optional[bool] = False,
              ):
        '''
        Run (syncrhonous)  Extreemum Seeking (ES) optimization loop.
        
        Args:
            n_loop: int, number of iterations to run.
            aES: float or list of floats, amplitude of dithering. If None, use default value.
            kES: float or list of floats, gain of dithering. If None, use default value.
            adaptive_ES: bool, whether to adaptively adjust the gain of dithering based on the objective function value.
        '''

        if not hasattr(self, "wES"):
            # ES dithering freq = 2pi*nu/10,  nu in [0.5,1], factor 10 for at least 10 iteration for smoothing (with dt=1).
            self.wES = 2*np.pi * (0.5*(np.arange(self.ndim) +0.5) / self.ndim + 0.5)/ 10
                
        if aES is None:
            if not hasattr(self, "aES"):
                # dithering amp. This ensures maximum dx in each iter is less than control_maxstep.
                self.aES = np.asarray(self.control_maxstep)
        else:
            self.aES = np.array(aES).reshape(-1)

        if kES is None:
            if not hasattr(self, "kES"):
                # ES gain, assuming change of objective in each iter is Delta(obj) ~ O(0.02) and kES*Delta(obj) ~ 0.1*wES*dt, so that phase change from gain is about 10% of dithering phase change.
                self.kES = 0.1*self.wES/0.02  
        elif isinstance(kES, (float, int)):
            self.kES = kES * np.ones(self.ndim)
        else:
            self.kES = np.array(kES).reshape(-1)
            
        if not hasattr(self, 't'):
            self.t = 0
            
        if adaptive_ES:
            self.history['ES']['t'].append(copy(self.t))
            self._adaptive_kES()

        for n in range(n_loop):
            # ES step
            self.x += self._get_ES_setp()
            self.x = np.clip(self.x, a_min=self.control_min, a_max=self.control_max)
            self.process_evaluator_future()
            self.future = self.evaluator.submit(self.x)
            self.process_evaluator_future()

    def run_savo(self,n_loop:int,
                 lambdaES: Optional[float] = 0,
                 n_grad_data= None,
                 n_grad_ave = None,
                 lr = None,
                 optimizer = None,
                 acquisition = None,
                 debug = False,
                 adam = False,
                 reset_adam_hyper = False,
                 ): 

        if adam and reset_adam_hyper:
            self.adam_m = 0
            self.adam_v = 0
            self.adam_t = 0
            self.adam_beta1 = 0.9
            self.adam_beta2 = 0.999
        
        for i in range(n_loop):
            dxES = lambdaES*self._get_ES_setp()#adaptive_ES=adaptive_ES)
            dxSG = self._get_SG_step(n_grad_data= n_grad_data,
                                     n_grad_ave = n_grad_ave,
                                     lr = lr,
                                     optimizer = optimizer,
                                     acquisition = acquisition,
                                     debug = debug)
            if adam:
                self.adam_t += 1
                self.adam_m = self.adam_beta1*self.adam_m + (1.-self.adam_beta1)*dxSG
                self.adam_v = self.adam_beta2*self.adam_v + (1.-self.adam_beta2)*dxSG*dxSG
                dxSG = self.adam_m/((1.-self.adam_beta1**self.adam_t)*
                                     (np.sqrt(self.adam_v/(1.-self.adam_beta2**self.adam_t))+1e-6))
                
            
            if self.clip_gradient:
                dxSG = np.clip(dxSG,a_min = -self.control_maxstep,a_max= self.control_maxstep)
            self.x += dxSG + dxES
            if self.x_bounds is not None:
                self.x = np.clip(self.x, a_min=self.x_bounds[:,0], a_max=self.x_bounds[:,1])
            self.y, self.y_err, self.x_RD = self.obj_func(self.x)
            self.history['x'].append(self.x_RD)
            self.history['y'].append(self.y)
            # TODO: implement surrogate model error using y_err.

   
    def _adaptive_kES(self, update_rate=0.01, grad=None):
        if len(self.history['y']) < 16:
            self.history['ES']['kES'].append(None)
            return
        grad = grad or self.obj_func_grad
        if grad:
            kES_step = 1.41421 / np.abs(self.aES * grad(self.x))  - self.kES
        else:
            y = self.history['y'][-16:]
            if self.apply_bilog:
                y = np.sign(y)*np.log(1+np.abs(y))
            std = np.std(y)
            variation = std
            kES_step = 0.1 * self.wES/(variation + 1e-15)  - self.kES
        # slow update on ES gain
        self.kES += np.clip(kES_step,
                            a_min=-update_rate * self.kES,
                            a_max= update_rate * self.kES)
        self.history['ES']['kES'].append(copy(self.kES))
        
        
    def _get_ES_setp(self):
        self.t += 1
        if self.minimize:
            return self.aES * np.cos(self.t * self.wES + self.kES * self.y)
        else:
            return self.aES * np.cos(self.t * self.wES - self.kES * self.y)
            
            
    def _get_SG_step(self,
                 n_grad_data= None,
                 n_grad_ave = None,
                 lr = None,
                 optimizer = None,
                 acquisition = None,
                 debug = False,
                 ):        
        n_grad_data = n_grad_data or self.n_grad_data
        n_grad_ave = n_grad_ave or self.n_grad_ave
        lr = lr or self.lr
        optimizer = optimizer or self.optimizer
         
        train_x = np.array(self.history['x'][-n_grad_data:])
        train_y = np.array(self.history['y'][-n_grad_data:])[:,None]
        if self.apply_bilog:
            train_y = np.sign(train_y)*np.log(1+np.abs(train_y))
        #if max_y_err is None or max_y_err==0:
        #    max_y_err = np.std(train_y[-16:])
            
        self.model = GaussianProcess(self.ndim)
        
        if debug:
            print('[debug][savo]_get_SG_step...')
            print('  train_x.shape,train_y.shape',train_x.shape,train_y.shape)
            
        # train surrogate model.     
        self.model.fit(train_x,train_y,debug=debug)
        
        # neighboring points for averaged gradient.  
        x_ = np.zeros((n_grad_ave,len(self.x)))
        x_[0 ,:] = self.x
        x_[1:,:] = 0.5*(self.x.reshape(1,-1) + np.array(self.history['x'][-n_grad_ave+1:]))
        x_grad = self.model.get_grad(x_).mean(axis=0)
        
#         #  gradient normalization for robust learining rate regularity independent of obj_func of choice.
#         if hasattr(self,'grad_norm'):
#             # slow update on grad_norm
#             self.grad_norm = 0.99*self.grad_norm + 0.01*np.mean(np.abs(x_grad)/self.control_maxstep)
#         else:
#             self.grad_norm = np.mean(np.abs(x_grad)/self.control_maxstep) 
#         x_grad /= (self.grad_norm + 1e-15)
        if optimizer is not None:
            x_grad = optimizer(x_grad)
        
        if acquisition :
            dx = acquisition(model=self.model,
                             x=self.x,
                             control_maxstep=self.control_maxstep,
                             x_grad=x_grad,
                             lr = lr)
        else:
            dx = lr*x_grad
            
        if self.minimize:
            return -dx
        else:
            return  dx