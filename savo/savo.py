'''
(S)urrogate Gradient
(A)ssisted
(V)ery high dimension
(O)ptimization
'''
from typing import List, Union, Optional, Callable, Tuple, Dict
from copy import deepcopy as copy
import numpy as np
import time
import concurrent

from .model import GaussianProcess
from .util import obj_func_wrapper
from .optim import Adam


class savo:
    def __init__(self,
                 obj_func: Callable,
                 x0: List[float],
                 max_dx: List[float],
                 x_bounds: Optional[List[Tuple[float]]] = None,
                 
                 # n_init: Optional[float] = None,
                 # x_train: Optional[List[List[float]]] = None,
                 # y_train: Optional[List[List[float]]] = None,
                 
                 n_grad_data: Optional[int] = None,
                 n_grad_ave: Optional[int] = None,
                 lr: Optional[float] = None,
                 optimizer: Optional[Callable] = None,
                 clip_gradient_step: Optional[bool] = True,
                 
                 prev_history: Optional[Dict] = None,
                 obj_func_grad = None,
                 apply_bilog = False,
                 minimize: Optional[bool] = False,
                 
                 ):
        '''
        obj_func: objective function evaluator
                      input of shape x0,
                      output of float:objective value  
                             or list[float:objective value, list[float]:decision readback]
                             or list[float:objective value, float:objective rms noise, list[float]:decision readback]
                             or [dict] of keys: objective value, objective rms noise, decision readback
        x0 [array of shape (dim,)]: initial decision parameters.
        max_dx [array of shape (dim,)]: approximate maximum step size for decision parameters.
        prev_history [dict]: previous history data of savo.
       
        n_grad_data [int]: number of most recent data points to train local surrogate model
        n_grad_ave  [int]: number of points near current set to calculate averaged gradient
        lr [float]: nominal learning rate for gradient decent. 
        optimizer [object]: learning rate controller.
        obj_func_grad [Callable]: Exact gradient of obj_func, if available. Benchmark purpose. 
        apply_bilog [bool]: regularize obj_funcs
        '''
        self.x = np.array(x0).reshape(-1)
        self.max_dx = np.array(max_dx).reshape(-1)
        self.obj_func, self.y, self.y_err, self.x_RD = obj_func_wrapper(obj_func, self.x)
        self.ndim = len(self.x)

        self.x_bounds = x_bounds
        if self.x_bounds is not None:
            self.x_bounds = np.array(self.x_bounds)
            if np.any(self.x_bounds[:,0] > self.x) or np.any(self.x > self.x_bounds[:,1]):
                raise ValueError("the initial decision x is out of the bounds")

        self.n_grad_data = n_grad_data or min(10*self.ndim, 1024) 
        self.n_grad_ave = n_grad_ave or 16
        self.lr = lr or 1e-2
        self.optimizer = optimizer or Adam()
      
        self.minimize = minimize
        self.apply_bilog = apply_bilog
        self.clip_gradient_step = clip_gradient_step

        self.adam_m = 0
        self.adam_v = 0
        self.adam_t = 0
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999

        if prev_history is not None:
            self.history = copy(prev_history)
            assert 'x' in self.history and 'y' in self.history and 'model' in self.history
            assert len(self.history['x']) == len(self.history['y'])
            if not 'ES' in self.history:
                self.history['ES'] = { 't'  :[],
                                       'kES':[],
                                       'aES':[],}
                self.t = 0
            else:
                self.t = self.history['ES']['t'][-1]
        else:
            self.history = {
                'x': [copy(self.x)],
                'y': [copy(self.y)],
                'model': {
                    'constraints': {},
                    'objective': {}
                },
                'ES': {'t':[],
                       'kES':[],
                       'aES':[],}
            }
            self.t = 0
            
    def runES(self,
              n_loop: int,
              aES: Optional[float] = None,
              kES: Optional[float] = None,
              adaptive_ES: Optional[bool] = False,
              ):


        if not hasattr(self, "wES"):
            # ES dithering freq = 2pi*nu/10,  nu in [0.5,1], factor 10 for at least 10 iteration for smoothing (with dt=1).
            self.wES = 2*np.pi * (0.5*(np.arange(self.ndim) +0.5) / self.ndim + 0.5)/ 10
        
        
        if aES is None:
            if not hasattr(self, "aES"):
                # dithering amp. This ensures maximum dx in each iter is less than max_dx.
                self.aES = self.max_dx  
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
            if self.x_bounds is not None:
                self.x = np.clip(self.x, a_min=self.x_bounds[:,0], a_max=self.x_bounds[:,1])
           
            self.y, self.y_err, self.x_RD = self.obj_func(self.x)
            self.history['x'].append(copy(self.x_RD))
            self.history['y'].append(copy(self.y))

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
                
            
            if self.clip_gradient_step:
                dxSG = np.clip(dxSG,a_min = -self.max_dx,a_max= self.max_dx)
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
        y = self.y
        if self.apply_bilog:
            y = np.sign(y)*np.log(1+np.abs(y))
        if self.minimize:
            return self.aES * np.cos(self.t * self.wES + self.kES * y)
        else:
            return self.aES * np.cos(self.t * self.wES - self.kES * y)
            
            
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
#             self.grad_norm = 0.99*self.grad_norm + 0.01*np.mean(np.abs(x_grad)/self.max_dx)
#         else:
#             self.grad_norm = np.mean(np.abs(x_grad)/self.max_dx) 
#         x_grad /= (self.grad_norm + 1e-15)
        if optimizer is not None:
            x_grad = optimizer(x_grad)
        
        if acquisition :
            dx = acquisition(model=self.model,
                             x=self.x,
                             max_dx=self.max_dx,
                             x_grad=x_grad,
                             lr = lr)
        else:
            dx = lr*x_grad
            
        if self.minimize:
            return -dx
        else:
            return  dx