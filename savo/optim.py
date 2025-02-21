import numpy as np

class Adam:
    def __init__(self,
#         betas=(0.9, 0.999), 
        betas=(0.8, 0.999), 
        eps=1e-08, 
        ):    
        self.m = 0
        self.v = 0
        self.t = 0
        self.betas = betas
        self.eps = eps
    
    def __call__(self,grad):
        self.t += 1
        self.m = self.betas[0]*self.m + (1.-self.betas[0])*grad
        self.v = self.betas[1]*self.v + (1.-self.betas[1])*grad*grad
        return self.m/((1.-self.betas[0]**self.t)*
               (np.sqrt(self.v/(1.-self.betas[1]**self.t))+self.eps))