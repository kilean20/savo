from typing import List, Union, Optional, Callable, Tuple, Dict
from copy import deepcopy as copy
import numpy as np
import time


class ES:
    def __init__(self,
                 obj_func: Callable,
                 x0: List[float],
                 max_dx: List[float],
                 x_bounds: Optional[List[Tuple[float]]] = None,
                 apply_bilog = False,
                 minimize: Optional[bool] = False,
                 prev_history: Optional = None,
                 ):
        self.x = np.array(x0).reshape(-1)
        self.max_dx = np.array(max_dx).reshape(-1)
        self.obj_func = obj_func        
        self.ndim = len(self.x)

        self.x_bounds = x_bounds
        if self.x_bounds is not None:
            self.x_bounds = np.array(self.x_bounds)
            if np.any(self.x_bounds[:,0] > self.x) or np.any(self.x > self.x_bounds[:,1]):
                raise ValueError("the initial decision x is out of the bounds")

        self.minimize = minimize
        self.apply_bilog = apply_bilog
        
        self.y = self.obj_func(self.x)
        self.y0 = self.y
        assert type(self.y) in [float, np.float32, np.float64]

        if prev_history is not None:
            self.history = copy(prev_history)
            assert 'x' in self.history and 'y' in self.history 
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
                #ensures maximum dx in each iter is less than max_dx.
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
            self.y = self.obj_func(self.x)
            self.history['x'].append(copy(self.x))
            self.history['y'].append(copy(self.y))

   
    def _adaptive_kES(self, update_rate=0.01, grad=None):
        if len(self.history['y']) < 16:
            self.history['ES']['kES'].append(None)
            return
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
        y = self.y - self.y0
        if self.apply_bilog:
            y = np.sign(y)*np.log(1+np.abs(y))
            
#         print("self.aES,self.wES,self.kES,self.t,self.y",self.aES,self.wES,self.kES,self.t,self.y)
        if self.minimize:
            return self.aES * np.sin(self.t * self.wES + self.kES * y)
        else:
            return self.aES * np.sin(self.t * self.wES - self.kES * y)