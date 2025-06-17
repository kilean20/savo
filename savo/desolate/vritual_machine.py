import numpy as np
from typing import List, Union, Optional, Callable
import pandas as pd
from IPython.display import display
import time


class RandomNeuralNetwork:
    def __init__(self, n_input: int, n_output: int, hidden_layers: Optional[List[int]] = None, 
                 activation_functions: Optional[List[str]] = None):
        self.n_input = n_input
        self.n_output = n_output
        
        # Determine the minimum and maximum number of nodes in hidden layers based on input and output dimensions
        min_node = int(np.clip(np.log2((n_input+n_output)**0.2), a_min=4, a_max=11))
        max_node = int(np.clip(np.log2((n_input+n_output)), a_min=min(min_node+3,10), a_max=12))
        
        # Initialize hidden_layers if not provided with random values within a specified range
        if hidden_layers is None:
            hidden_layers = [2**np.random.randint(min_node, max_node) for _ in range(np.random.randint(4, 7))]  # 4, 5, or 6 layers
        
        # Calculate the total number of layers and layer dimensions
        self.n_layers: int = len(hidden_layers) + 2
        self.layer_dims: List[int] = [n_input] + hidden_layers + [n_output]
        
        # Initialize activation functions if not provided with random choices for hidden layers and None for the output layer
        if activation_functions is None:
            activation_functions = [np.random.choice(['elu', 'sin', 'cos', 'tanh', 'sinc']) for i in range(self.n_layers-1)]
            activation_functions.append(None)  # no activation on the last layer
        
        # Store layer activation functions and initialize network parameters
        self.activation_functions: List[Union[str, None]] = activation_functions
        self.parameters: dict = self.initialize_parameters()
        
        # Generate random inputs for normalization calculation
        self.mean_output: float = 0
        self.std_output: float = 1
        random_inputs = np.random.randn(1024, n_input)
        random_outputs = self(random_inputs)
        self.mean_output = np.mean(random_outputs, axis=0)
        self.std_output = np.std(random_outputs, axis=0)
        

    def initialize_parameters(self) -> dict:
        # Initialize weights and biases for each layer using He initialization
        parameters = {}
        for l in range(1, self.n_layers):
            scale_weights = np.sqrt(2.0 / self.layer_dims[l - 1])  # He initialization for weights
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * scale_weights

            # Initialize biases with small random values
            scale_biases = np.sqrt(0.5 / self.layer_dims[l])
            parameters[f'b{l}'] = np.random.randn(self.layer_dims[l], 1) * scale_biases

        return parameters
    
    def normalize_output(self, output: np.ndarray) -> np.ndarray:
        # Normalize output based on mean and standard deviation
        return (output - self.mean_output) / self.std_output

    def activate(self, Z: np.ndarray, activation_function: Union[str, None]) -> np.ndarray:
        # Apply activation functions based on the specified function
        if activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation_function == 'tanh':
            return np.tanh(Z)
        elif activation_function == 'relu':
            return np.maximum(0, Z)
        elif activation_function == 'elu':
            return np.where(Z > 0, Z, np.exp(Z) - 1)
        elif activation_function == 'sin':
            return np.sin(Z)
        elif activation_function == 'cos':
            return np.cos(Z)
        elif activation_function == 'sinc':
            return np.sinc(Z)
        else:
            return Z

    def __call__(self, X: np.ndarray) -> np.ndarray:
        
        X = np.array(X)
        assert X.shape[1] == self.n_input
        # Perform forward propagation through the neural network
        A = X.T  # Transpose the input to make it compatible with matrix multiplication
        for l in range(1, self.n_layers):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b
            A = self.activate(Z, self.activation_functions[l])
            
        # Transpose the result back to (n_batch, n_output) and then normalize
        return self.normalize_output(A.T)
    
    
    
    
    
def zscore_mean(x: np.ndarray, abs_z: Optional[float] = None) -> float:
    """
    Calculate the mean of the input array 'x' within a specified z-score range.

    Parameters:
    - x (numpy.ndarray): Input array.
    - abs_z (float, optional): Absolute value of the z-score threshold. If None, return the mean of 'x' without filtering.

    Returns:
    - float: Mean of the input array 'x' within the specified z-score range, or the overall mean if 'abs_z' is None.
    """

    x = np.array(x)
    mean = np.mean(x)

    if abs_z is None:
        return mean

    std = np.std(x)

    if np.any(std == 0.):
        return mean

    zscore_filter = np.logical_and(mean - abs_z * std < x, x < mean + abs_z * std)
    filtered_mean = np.mean(x[zscore_filter])

    return filtered_mean


class VM:
    """
    VM (Virtual Machine) class for simulating a virtual environment with decision variables and objectives.

    Parameters:
    - x0 (List[float]): Initial values of decision variables.
    - decision_CSETs (Optional[List[str]]): List of decision variable names.
    - objective_RDs (Optional[List[str]]): List of objective names.
    - objective_RDs_mean (Optional[List[float]]): Mean values for normalization of objective results.
    - objective_RDs_std (Optional[List[float]]): Standard deviation values for normalization of objective results.
    - decision_min (Optional[List[float]]): Minimum values for decision variables.
    - decision_max (Optional[List[float]]): Maximum values for decision variables.
    - fun (Optional): Custom function for objective evaluation.
    - dt (Optional[float]): Time step for simulation.

    Attributes:
    - decision_CSET_vals (List[float]): Current values of decision variables.
    - decision_CSETs (List[str]): List of decision variable names.
    - objective_RDs (List[str]): List of objective names.
    - y_mean (np.ndarray): Mean of the simulated objective results.
    - y_std (np.ndarray): Standard deviation of the simulated objective results.
    - fun (function): Normalized objective function.
    - objective_RD_vals (np.ndarray): Current normalized objective values.
    - dt (float): Time step for simulation.
    - t (float): Current simulation time.
    - history (pd.DataFrame): Simulation history as a pandas DataFrame.
    """

    def __init__(self,
                 x0: List[float],
                 decision_CSETs: Optional[List[str]],
                 objective_RDs: Optional[List[str]],
                 objective_RDs_mean: Optional[List[float]] = None,
                 objective_RDs_std: Optional[List[float]] = None,
#                  objective_RDs_noise: Optional[List[float]] = None,
                 decision_min: Optional[List[float]] = None,
                 decision_max: Optional[List[float]] = None,
                 fun: Optional[Callable] = None,
                 dt: Optional[float] = 0.2,
                 fetch_data_time_span: Optional[float] = 0.2,
                 verbose: Optional[bool]  = False,
                 real_time_delay: Optional[float] = 0.01,
                 ):
        # Initialize decision variables and objectives
        self._test = True
        self.decision_CSET_vals = x0
        self.decision_CSETs = decision_CSETs
        self.decision_RDs   = [pv.replace('_CSET','_RD') for pv in decision_CSETs]
        self.objective_RDs = objective_RDs
        self.fetch_data_time_span = fetch_data_time_span
        self._verbose = verbose
        assert len(self.decision_CSET_vals) == len(self.decision_CSETs)
        
        # Initialize objective function if not provided
        if decision_min is not None:
            assert decision_max is not None
            decision_min = np.array(decision_min).reshape(1,-1)
            decision_max = np.array(decision_max).reshape(1,-1)
        else:
            decision_min = -1
            decision_max = 1
            
        if fun is None:
            randNN = RandomNeuralNetwork(len(self.decision_CSETs), len(self.objective_RDs))
            x_test = np.random.randn(1024, len(self.decision_CSETs))
            y_test = randNN(x_test)
            self.y_mean = np.mean(y_test, axis=0)
            self.y_std = np.std(y_test, axis=0)
            if objective_RDs_std is not None:
                self.y_std /= np.array(objective_RDs_std)
            if objective_RDs_mean is not None:
                self.objective_RDs_mean = np.array(objective_RDs_mean)
            else:
                self.objective_RDs_mean = 0

            # Define normalized objective function
            def fun(x):
                assert np.size(x) == len(self.decision_CSETs)
                x = 2*(np.array(x).reshape(1,-1)-decision_min)/(decision_max-decision_min)-1
                return (randNN(x)[0] - self.y_mean)/self.y_std  + self.objective_RDs_mean
        
        self.fun = fun
        
        # Initialize objective values
        self.objective_RD_vals = self.fun(self.decision_CSET_vals)
        assert len(self.objective_RD_vals) == len(self.objective_RDs)
        
        # Initialize simulation parameters
        self.dt = dt or 0.2
        self.t = 0
        self.real_time_delay = real_time_delay
        
        # Initialize simulation history
        self.history = pd.DataFrame(np.hstack((self.decision_CSET_vals,
                                               self.decision_CSET_vals,
                                               self.objective_RD_vals)).reshape(1,-1), 
                                    columns = np.hstack((self.decision_CSETs,self.decision_RDs,self.objective_RDs)))

    def __call__(self):
        """
        Update the simulation state by advancing the time step and calculating new objective values.
        """
        self.t += self.dt
        self.objective_RD_vals = self.fun(self.decision_CSET_vals)
        #self.history = self.history.append({**dict(zip(self.decision_CSETs, self.decision_CSET_vals)),
        #                                    **dict(zip(self.objective_RDs, self.objective_RD_vals))}, ignore_index=True)
        df = pd.DataFrame({**dict(zip(self.decision_CSETs, self.decision_CSET_vals)),
                           **dict(zip(self.decision_RDs  , self.decision_CSET_vals)),
                           **dict(zip(self.objective_RDs , self.objective_RD_vals))},index=[len(self.history)])
        self.history = pd.concat([self.history, df], ignore_index=True)
        
    def caput(self,pvname,value,verbose=None):
        
        verbose = verbose or self._verbose
        if verbose:
            print('ramping...')
            display(pd.DataFrame(np.array(value).reshape(1,-1), columns=pvname))
            
        for i, pv in enumerate(self.decision_CSETs):
            if pv == pvname:
                self.decision_CSET_vals[i] = value
                
        if verbose:
            print('done')
                
    def caget(self,pvname):
        t0 = self.t
        self()
        time.sleep(self.real_time_delay)
        return self.history[pvname].iloc[-1]
                
    def ensure_set(self, 
                   setpoint_pv: Union[str, List[str]], 
                   readback_pv: Union[str, List[str]], 
                   goal: Union[float, List[float]], 
                   tol: Union[float, List[float]] = 0.01, 
                   timeout: float = 10.0, 
                   verbose: bool = None):
        """
        Ensure that setpoint values reach the specified goals for given time steps.

        Parameters:
        - setpoint_pv (Union[str, List[str]]): Setpoint variable name(s).
        - readback_pv (Union[str, List[str]]): Readback variable name(s).
        - goal (Union[float, List[float]]): Target value(s) for setpoint variables.
        - tol (Union[float, List[float]], optional): Tolerance for reaching the target values.
        - timeout (float, optional): Maximum time allowed for reaching the target values.
        - verbose (bool, optional): Print verbose messages.
        """
        verbose = verbose or self._verbose
        if verbose:
            print('ramping...')
            display(pd.DataFrame(np.array(goal).reshape(1,-1), columns=setpoint_pv))
        
        for i, pv in enumerate(self.decision_CSETs):
            for j, pv_sp in enumerate(setpoint_pv):
                if pv == pv_sp:
                    self.decision_CSET_vals[i] = goal[j]
                    
    
    def fetch_data(self, 
                   pvlist: List[str], 
                   time_span: float = None, 
                   abs_z: Optional[float] = None, 
                   with_data=False, 
                   verbose=False):
        """
        Fetch simulated data for specified variable names over a specified time span.

        Parameters:
        - pvlist (List[str]): List of variable names to fetch data for.
        - time_span (float, optional): Time span for data fetching.
        - abs_z (float, optional): Absolute value for data normalization.
        - with_data (bool, optional): Include raw data in the output.
        - verbose (bool, optional): Print verbose messages.

        Returns:
        Tuple[List[float]]: A tuple containing averaged data and raw data (optional).
        """
        verbose = verbose or self._verbose
        if verbose:
            print(f'reading ...')
        
        time_span = time_span or self.fetch_data_time_span
        time.sleep(self.real_time_delay)
        t0 = self.t
        raw_data = {pv: [] for pv in pvlist}
        while self.t - t0 <= time_span:
            self()
            for pv in pvlist:
                if pv in self.history:
                    raw_data[pv].append(self.history[pv].iloc[-1])
                else:
                    raw_data[pv].append(0.5)
                        
        ave_data = np.array([zscore_mean(raw_data[pv], abs_z) for pv in pvlist])
        nmax = 0
        for k in raw_data.keys():
            if len(raw_data[k])>nmax:
                nmax = len(raw_data[k])
            raw_data[k] += [len(raw_data[k]), np.nanmean(raw_data[k]), np.nanstd(raw_data[k])]
            
        raw_data = pd.DataFrame(raw_data, index=np.arange(nmax).tolist() + ['#','mean','std']).T
        
        if verbose:
            display(pd.DataFrame(np.array(ave_data).reshape(1,-1), columns=pvlist))
        
        if with_data:
            return ave_data, raw_data
        else:
            return ave_data, None
        