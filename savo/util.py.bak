import re
import io
import os
import sys
import json
import warnings
import contextlib
from typing import List, Optional, Tuple, Dict, Union, Callable
import numpy as np
import torch
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))
try:
    from IPython.display import display
except ImportError:
    display = print

@contextlib.contextmanager    
def suppress_outputs():
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

_name_conversions =(
    (':PSQ_' , ':Q_'  ),
    (':PSQ_' , ':QV_' ),
    (':PSQ_' , ':QH_' ),
    (':PSC2_', ':DCH_'),
    (':PSC1_', ':DCV_'),
)
SPEED_OF_LIGHT = 299_792_458.0  # m/s
ATOMIC_MASS_UNIT_eV = 931_494_320.0  # eV/c²


class dictClass(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


# Brho calculation with input compatibility
def calculate_Brho(E_MeV_u, mass_number, charge_number) -> torch.Tensor:
    E_MeV_u = torch.as_tensor(E_MeV_u, dtype=torch.float64)
    mass_number = torch.as_tensor(mass_number, dtype=torch.float64)
    charge_number = torch.as_tensor(charge_number, dtype=torch.float64)

    E = E_MeV_u * 1e6 + ATOMIC_MASS_UNIT_eV  # Total energy in eV
    p = mass_number * torch.sqrt(E**2 - ATOMIC_MASS_UNIT_eV**2) / SPEED_OF_LIGHT  # Momentum in eV/c
    return p / charge_number

# Beta-gamma calculation
def calculate_betagamma(E_MeV_u, mass_number) -> torch.Tensor:
    E_MeV_u = torch.as_tensor(E_MeV_u, dtype=torch.float64)
    mass_number = torch.as_tensor(mass_number, dtype=torch.float64)

    E = E_MeV_u * 1e6 + ATOMIC_MASS_UNIT_eV
    return torch.sqrt(E**2 - ATOMIC_MASS_UNIT_eV**2) / ATOMIC_MASS_UNIT_eV

# ELU with gradient support and input safety
def elu(x) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64)
    return torch.where(x > 0, x, torch.exp(x) - 1)

# Cyclic distance with tensor conversion and gradient support
def cyclic_distance(x, y, period, eps=1e-2) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    period = torch.as_tensor(period, dtype=torch.float64)
    delta = (x - y + period / 2) % period - period / 2
    return torch.abs(delta)

def cyclic_mean(x: Union[List[float], np.ndarray, torch.Tensor], Lo: float, Hi: float) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64)
    period = Hi - Lo
    if x.numel() == 0:
        return torch.tensor(float('nan'), dtype=torch.float64)
    elif x.numel() == 1:
        return x.clone()
    angles = 2 * torch.pi * (x - Lo) / period
    mean_angle = torch.angle(torch.mean(torch.exp(1j * angles)))
    mean = (mean_angle % (2 * torch.pi)) / (2 * torch.pi) * period + Lo
    return mean


def cyclic_mean_var(x: Union[List[float], np.ndarray, torch.Tensor], Lo: float, Hi: float) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(x, dtype=torch.float64)
    period = Hi - Lo
    if x.numel() == 0:
        return torch.tensor(float('nan'), dtype=torch.float64), torch.tensor(float('nan'), dtype=torch.float64)
    elif x.numel() == 1:
        return x.clone(), torch.tensor(0.0, dtype=torch.float64)
    mean = cyclic_mean(x, Lo, Hi)
    var = torch.mean(cyclic_distance(x, mean, period) ** 2)
    return mean, var

def cyclic_difference(x, y, Lo: float, Hi: float, smooth_sign=False, k=100.0) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    period = Hi - Lo
    x_ang = 2 * torch.pi * (x - Lo) / period
    y_ang = 2 * torch.pi * (y - Lo) / period
    distance = cyclic_distance(x, y, period)
    if smooth_sign:
        # Use a smooth approximation of sign: tanh(k * x)
        sign_approx = torch.tanh(k * torch.sin(y_ang - x_ang))
        return distance * sign_approx
    else:
        # Original sign function (not differentiable at zero)
        return distance * torch.sign(torch.sin(y_ang - x_ang))
    

def df_mean(df):
    mean = df.mean()
    for col in df.columns:
        if 'PHASE' in col:
            if 'BPM' in col:
                mean[col] = cyclic_mean(df[col], -90, 90)
            else:
                mean[col] = cyclic_mean(df[col], -180, 180)
    return mean

def df_mean_var(df):
    mean = df.mean()
    mean.name = df.index[0]
    var = df.var()
    var.name = df.index[0]
    for col in df.columns:
        if 'obj' not in col:
            if 'PHASE' in col:
                if 'BPM' in col:
                    mean[col], var[col]= cyclic_mean_var(df[col], -90, 90)
                else:
                    mean[col], var[col]= cyclic_mean_var(df[col], -180, 180)
    return mean, var

            
# Nelder-Mead optimization
def NelderMead(
    loss_ftn: Callable,
    x0: np.ndarray,
    simplex_size: float = 0.05,
    bounds: Optional[List[Tuple[float, float]]] = None,
    tol: float = 1e-4
) -> optimize.OptimizeResult:
    n = len(x0)
    initial_simplex = np.vstack([x0] * (n + 1))

    if bounds is None:
        for i in range(n):
            initial_simplex[i + 1, i] += simplex_size
    else:
        bounds = np.array(bounds)
        assert np.all(x0 <= bounds[:, 1]) and np.all(bounds[:, 0] <= x0)
        for i in range(n):
            dx = simplex_size * (bounds[i, 1] - bounds[i, 0])
            initial_simplex[i + 1, i] = np.clip(x0[i] + dx, bounds[i, 0], bounds[i, 1])

    result = optimize.minimize(
        loss_ftn, x0, method='Nelder-Mead', bounds=bounds, tol=tol,
        options={'initial_simplex': initial_simplex}
    )

    return result


# Check if input is a list of lists
def is_list_of_lists(input_list: Union[list, np.ndarray]) -> bool:
    if not isinstance(input_list, list):
        return False
    return all(isinstance(item, list) for item in input_list)


# Convert list of dicts to pandas DataFrame
def from_listdict_to_pd(data: List[Dict]) -> pd.DataFrame:
    all_keys = set().union(*data)
    dict_of_lists = {key: [d.get(key, np.nan) for d in data] for key in all_keys}
    max_length = max(len(lst) for lst in dict_of_lists.values())
    for key in dict_of_lists:
        dict_of_lists[key] += [np.nan] * (max_length - len(dict_of_lists[key]))
    return pd.DataFrame(dict_of_lists)


def get_Dnum_from_pv(pv: str) -> int or None:
    """
    Extracts the D number from a PV string.
    Args:
        pv (str): The PV string.
    Returns:
        int or None: The extracted D number or None if not found.
    """
    try:
        match = re.search(r"_D(\d{4})", pv)
        if match:
            return int(match.group(1))
        else:
            return None
    except AttributeError:
        return None
    

def split_name_field_from_PV(PV: str, 
                           return_device_name: bool =True) -> tuple:
    """
    Splits the PV into name and key components.

    Args:
        PV (str): The PV string.

    Returns:
        tuple: A tuple containing the name and key components.
    """
    # Find the index of the first colon
    first_colon_index = PV.find(':')

    if first_colon_index == -1:
        print(f"Name of PV: {PV} is not found")
        return None, None
    
    if return_device_name:
        for dev_name, phys_name in _name_conversions:
            PV = PV.replace(phys_name,dev_name)

    second_colon_index = PV.find(':', first_colon_index + 1)
    if second_colon_index != -1:
        return PV[:second_colon_index], PV[second_colon_index + 1:]
    else:
        return PV, None
    

D_NUM_PATTERN = re.compile(r'\D(\d{4})$')  # Precompiled
def sort_by_Dnum(strings: list[str]) -> list[str]:
    """Sort BPM names by trailing 4-digit number."""
    return sorted(strings, key=lambda s: int(m.group(1)) if (m := D_NUM_PATTERN.search(s)) else 0)
    
# def sort_by_Dnum(strings):
#     """
#     Sort a list of PVs by dnum.
#     """
#     # Define a regular expression pattern to extract the 4-digit number at the end of each string
#     pattern = re.compile(r'\D(\d{4})$')

#     # Define a custom sorting key function that extracts the 4-digit number using the regex pattern
#     def sorting_key(s):
#         match = pattern.search(s)
#         if match:
#             return int(match.group(1))
#         return 0  # Default value if no match is found

#     # Sort the strings based on the custom sorting key
#     sorted_strings = sorted(strings, key=sorting_key)
#     return sorted_strings
    
    
# Calculate mismatch factor between two Courant-Snyder parameters
def calculate_mismatch_factor(cs_ref: Tuple[float, float, float], cs_test: Tuple[float, float, float]) -> float:
    alpha_ref, beta_ref, nemit_ref = [np.float64(val) for val in cs_ref]
    alpha, beta, nemit = [np.float64(val) for val in cs_test]
    gamma_ref = (1 + alpha_ref**2) / beta_ref
    gamma = (1 + alpha**2) / beta
    R = beta_ref * gamma + beta * gamma_ref - 2 * alpha_ref * alpha
    Mx = max(0.5 * (R + max(R**2 - 4, 0)**0.5), 1)**0.5 - 1
    return max(nemit/nemit_ref,nemit_ref/nemit) * Mx


# def calculate_MMD4D(cs_ref : Tuple[float, float, float, float, float, float], 
#                     cs_test: Tuple[float, float, float, float, float, float]) -> float:
#     alfx1, betx1, emitx1, alfy1, bety1, emity1 = [np.float64(val) for val in cs_ref]
#     alfx2, betx2, emitx2, alfy2, bety2, emity2 = [np.float64(val) for val in cs_test]
#     gamx1 = (1.0+alfx1**2)/betx1
#     gamy1 = (1.0+alfy1**2)/bety1
#     gamx2 = (1.0+alfx2**2)/betx2
#     gamy2 = (1.0+alfy2**2)/bety2
#     Rx = max(0.5*(gamx2*betx1 + gamx1*betx2 - 2*alfx1*alfx2),1.0)
#     Ry = max(0.5*(gamy2*bety1 + gamy1*bety2 - 2*alfy1*alfy2),1.0)
#     denom1 = (emitx1**2 + 4 * Rx * emitx1 * emitx2 + 4 * emitx2**2) * (emity1**2 + 4 * Ry * emity1 * emity2 + 4 * emity2**2)
#     denom2 = (4 * emitx1**2 + 4 * Rx * emitx1 * emitx2 + emitx2**2) * (4 * emity1**2 + 4 * Ry * emity1 * emity2 + emity2**2)
#     MMD4D = ( 1.0/9.0 + emitx1*emity1 *(1.0/denom1**0.5 - 2.0 / denom2**0.5) )**0.5
#     return MMD4D    

def calculate_MMD4D_from_covs(xcov1,ycov1,
                              xcov2,ycov2,
                              xcov_ref=None,ycov_ref=None)-> float:
    
    if xcov_ref is None:
        xcov_ref = xcov1
    if ycov_ref is None:
        ycov_ref = ycov1
        
    I2 = np.eye(2)  # 2x2 identity matrix
    
    # Compute S_x and S_y as inverses of Sigma matrices for index 0
    S_x = np.linalg.inv(xcov_ref)
    S_y = np.linalg.inv(ycov_ref)


    # Compute determinants for x components
    det1_x = np.linalg.det(I2 + 2 * S_x @ xcov1)
    det2_x = np.linalg.det(I2 + 2 * S_x @ xcov2)
    det3_x = np.linalg.det(I2 + S_x @ xcov1 + S_x @ xcov2)
    
    # Compute determinants for y components
    det1_y = np.linalg.det(I2 + 2 * S_y @ ycov1)
    det2_y = np.linalg.det(I2 + 2 * S_y @ ycov2)
    det3_y = np.linalg.det(I2 + S_y @ ycov1 + S_y @ ycov2)
    
    # Compute the three terms of the MMD formula
    term1 = 1 / np.sqrt(det1_x * det1_y)
    term2 = 1 / np.sqrt(det2_x * det2_y)
    term3 = -2 / np.sqrt(det3_x * det3_y)
    
    # Sum the terms to get the final MMD
    return (term1 + term2 + term3)**0.5


def sigma(alpha, beta, epsilon):
    """
    Create a 2x2 Sigma matrix based on parameters alpha, beta, and epsilon.
    
    Args:
        alpha (float): The alpha parameter.
        beta (float): The beta parameter.
        epsilon (float): The epsilon scaling factor.
    
    Returns:
        numpy.ndarray: A 2x2 matrix scaled by epsilon.
    """
    return epsilon * np.array([[beta, -alpha], [-alpha, (1 + alpha**2) / beta]])

def calculate_MMD4D(cs1 : Tuple[float, float, float, float, float, float], 
                    cs2 : Tuple[float, float, float, float, float, float],
                    cs_ref : Optional[Tuple[float, float, float, float, float, float]] = None)-> float:
    
    if cs_ref is None:
        cs_ref = cs1
    αx0, βx0, εx0, αy0, βy0, εy0 = cs_ref
    αx1, βx1, εx1, αy1, βy1, εy1 = cs1
    αx2, βx2, εx2, αy2, βy2, εy2 = cs2
        
    xcov_ref = sigma(αx0, βx0, εx0)
    ycov_ref = sigma(αy0, βy0, εy0)
    xcov1 = sigma(αx1, βx1, εx1)
    ycov1 = sigma(αy1, βy1, εy1)
    xcov2 = sigma(αx2, βx2, εx2)
    ycov2 = sigma(αy2, βy2, εy2)
    
    return calculate_MMD4D_from_covs(xcov1,ycov1,
                                   xcov2,ycov2,
                                   xcov_ref,ycov_ref)


from itertools import combinations


# Backtracking function to check if a clique of size n exists
def has_clique(n, current_set, start_index, D, min_dist):
    """
    Check if there exists a clique of size n where all pairwise distances >= min_dist.
    
    Args:
        n (int): Desired size of the subset.
        current_set (list): Current set of vector indices.
        start_index (int): Starting index to consider next vectors.
        D (np.ndarray): Distance matrix.
        min_dist (float): Minimum distance threshold.
    
    Returns:
        bool: True if a clique of size n exists, False otherwise.
    """
    m = D.shape[0]
    if len(current_set) == n:
        return True
    for i in range(start_index, m):
        # Check if vector i can be added (distance to all in current_set >= min_dist)
        if all(D[i, j] >= min_dist for j in current_set):
            if has_clique(n, current_set + [i], i + 1, D, min_dist):
                return True
    return False

# Backtracking function to find a clique of size n
def find_clique(n, current_set, start_index, D, min_dist):
    """
    Find a subset of size n where all pairwise distances >= min_dist.
    
    Args:
        n (int): Desired size of the subset.
        current_set (list): Current set of vector indices.
        start_index (int): Starting index to consider next vectors.
        D (np.ndarray): Distance matrix.
        min_dist (float): Minimum distance threshold.
    
    Returns:
        list: Subset of n indices forming a clique, or None if none exists.
    """
    m = D.shape[0]
    if len(current_set) == n:
        return current_set
    for i in range(start_index, m):
        if all(D[i, j] >= min_dist for j in current_set):
            result = find_clique(n, current_set + [i], i + 1, D, min_dist)
            if result is not None:
                return result
    return None

def select_n_most_distant_mmd4d_covs(n,xcovs,ycovs,xcov_ref,ycov_ref):
    m = len(xcovs)
    if n > m:
        raise ValueError("n cannot be greater than m")

    # Step 1: Precompute the distance matrix
    D = np.zeros((m, m))
    for i, j in combinations(range(m), 2):
        dist = calculate_MMD4D_from_covs(xcovs[i], ycovs[i], 
                                         xcovs[j], ycovs[j],
                                         xcov_ref, ycov_ref)
        D[i, j] = dist
        D[j, i] = dist  # Symmetric matrix

    # Step 2: Extract unique pairwise distances
    upper_tri = D[np.triu_indices(m, k=1)]  # Upper triangle excluding diagonal
    unique_d = np.sort(np.unique(upper_tri))  # Sorted in ascending order

    # Step 3: Binary search to find the largest feasible minimum distance
    low, high = 0, len(unique_d) - 1
    optimal_dist = unique_d[0]  # Initialize with smallest distance
    while low <= high:
        mid = (low + high + 1) // 2
        candidate_dist = unique_d[mid]
        if has_clique(n, [], 0, D, candidate_dist):
            optimal_dist = candidate_dist
            low = mid + 1  # Search for a larger distance
        else:
            high = mid - 1  # Search for a smaller distance

    # Step 4: Find the subset achieving the optimal distance
    selected_indices = find_clique(n, [], 0, D, optimal_dist)
    if selected_indices is None:
        raise RuntimeError("No subset found, which should not happen with valid input")

    return selected_indices





    
# Generate ellipse points based on Courant-Snyder parameters
def generate_ellipse(alpha: float, beta: float, nemit: float, bg: float) -> np.ndarray:
    gamma = (1 + alpha**2) / beta
    cov_matrix = np.array([
        [nemit * beta, -nemit * alpha * 1e-3],
        [-nemit * alpha * 1e-3, nemit * gamma * 1e-6]
    ]) / bg * 1e6
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(t), np.sin(t)])
    ellipse = np.linalg.cholesky(cov_matrix).dot(circle)
    return ellipse
    

# Plot beam ellipse
def plot_beam_ellipse(alpha: float, beta: float, nemit: float, bg: float, direction: str = 'x',
                      ax: Optional[plt.Axes] = None, fig=None, **kwargs):
    ellipse = generate_ellipse(alpha, beta, nemit, bg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.3))
    ax.plot(ellipse[0, :], ellipse[1, :] * 1e3, **kwargs)
    ax.set_xlabel(f"{direction}  (mm)")
    ax.set_ylabel(f"{direction}' (mrad)")
    ax.grid(True)
    

# Plot beam ellipse from covariance matrix
def plot_beam_ellipse_from_cov(cov: np.ndarray, direction: str = 'x',
                               ax: Optional[plt.Axes] = None, fig=None, **kwargs):
    assert cov.shape == (2, 2), "Covariance matrix must be 2x2"
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(t), np.sin(t)])
    cov = cov.copy()
    fail = True
    while fail:
        try:
            ellipse = np.linalg.cholesky(cov).dot(circle)
            fail = False
        except np.linalg.LinAlgError:
            cov[0, 0] += 1e-6
            cov[1, 1] += 1e-6

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3.3))
    ax.plot(ellipse[0, :] * 1e3, ellipse[1, :] * 1e3, **kwargs)
    ax.set_xlabel(f"{direction}  (mm)")
    ax.set_ylabel(f"{direction}' (mrad)")
    ax.grid(True)
    
    
# def get_ISAAC_preset(keywords: List[str],    
#                      E_MeV_u: float,
#                      mass_number: int,
#                      charge_number: int):
#     with open(os.path.join(script_dir, 'scan_temp_conf.json'), 'r') as f:
#         presets = json.load(f)
#     preset = None
#     for key in presets.keys():
#         is_key = np.all([k in key for k in keywords])
#         if is_key:
#             preset = presets[key]
#             match = re.search(r'\((\d+)[A-Za-z]*(\d+)\+\)', key)
#             if match:
#                 pre_mass_number = int(match.group(1))
#                 pre_charge_number = int(match.group(2))
#             else:
#                 raise ValueError(f"Could not extract mass and charge from key: {key}")
#             break
#     if preset == None:
#         raise ValueError("preset not found")

#     pre_E_MeV_u = preset['E']
#     assert eval(preset['qA_ratio']) == pre_charge_number/pre_mass_number
#     Brho = calculate_Brho(E_MeV_u, mass_number, charge_number)
#     pre_Brho = calculate_Brho(pre_E_MeV_u, pre_mass_number, pre_charge_number)
#     df = pd.DataFrame(np.array(preset['put_pv_valll'])*Brho/pre_Brho, index=preset['put_pv_name']).T
#     if 'BDS' in key:
#         df = df.iloc[[1,2,3,6,7,8]]
#     return df


def get_ISAAC_preset(n_init:int,
                     keywords: List[str],    
                     E_MeV_u: float,
                     mass_number: int,
                     charge_number: int):
    with open(os.path.join(script_dir, 'scan_temp_conf.json'), 'r') as f:
        presets = json.load(f)
    preset = None
    for key in presets.keys():
        is_key = np.all([k in key for k in keywords])
        if is_key:
            preset = presets[key]
            match = re.search(r'\((\d+)[A-Za-z]*(\d+)\+\)', key)
            if match:
                pre_mass_number = int(match.group(1))
                pre_charge_number = int(match.group(2))
            else:
                print(f"Could not extract mass and charge from preset key: {key}")
                return None
            break
    if preset == None:
        print("preset not found")
        return None

    pre_E_MeV_u = preset['E']
    if eval(preset['qA_ratio']) != pre_charge_number/pre_mass_number:
        print("prest inforation of charge to mass ratio is not consistent")
        return None
    Brho = calculate_Brho(E_MeV_u, mass_number, charge_number)
    pre_Brho = calculate_Brho(pre_E_MeV_u, pre_mass_number, pre_charge_number)
    df = pd.DataFrame(np.array(preset['put_pv_valll'])*Brho/pre_Brho, index=preset['put_pv_name']).T
    if 'BDS' in key:
        if n_init <= 2:
            iloc = [2,7]
        elif n_init <= 4:
            iloc = [1,2,7,8]
        else:
            iloc = [1,2,3,6,7,8]
        df = df.iloc[iloc]
    return df.iloc[:min(n_init,len(df))]


def is_valid_vector(x, length=68):
    """Check if x is a NumPy array of the given length with no NaNs."""
    return isinstance(x, np.ndarray) and (len(x) == length) and (not np.isnan(x).any())

def validate_df_rows(df, vector_cols, lengths, scalar_cols=None):
    """
    Filter rows by iterating over each row once.
    vector_cols: list of lists of columns. Each sublist is a group where all columns must be valid vectors.
    lengths: list of expected lengths for each group.
    scalar_cols: list of scalar column names; if None, all columns not in any vector group.
    """
    # Determine scalar_cols only once.
    if scalar_cols is None:
        all_vector_cols = {col for group in vector_cols for col in group}
        scalar_cols = list(set(df.columns) - all_vector_cols)
    
    valid_indices = []
    for idx, row in df.iterrows():
        valid = True
        # Check each vector group.
        for group, l in zip(vector_cols, lengths):
            for col in group:
                x = row[col]
                # Inline the check:
                if not (isinstance(x, np.ndarray) and (x.shape[0] == l) and (not np.isnan(x).any())):
                    valid = False
                    break
            if not valid:
                break
        
        # Check scalar columns.
        if valid and not pd.notna(row[scalar_cols]).all():
            valid = False
        
        if valid:
            valid_indices.append(idx)
    
    return df.loc[valid_indices]
    
def _rightmost_zero(n):
    """Position of the lowest 0-bit in the binary representation of integer `n`."""
    s = np.binary_repr(n)
    i = s[::-1].find("0")
    if i == -1:
        i = len(s)
    return i


def _generator(dimension, skip=0):
    """Generator for the Sobol sequence"""
    DIMS = 23  # maximum number of dimensions
    BITS = 30  # maximum number of bits

    if not (1 <= dimension <= DIMS):
        raise ValueError("Sobol: dimension must be between 1 and %i." % DIMS)

    # initialize direction numbers
    V = np.zeros((DIMS, BITS), dtype=int)
    data = np.array([
                        [7,1,1,0,0,0,0,0,0,0,0,0,0,0],
                        [11,1,3,7,0,0,0,0,0,0,0,0,0,0],
                        [13,1,1,5,0,0,0,0,0,0,0,0,0,0],
                        [19,1,3,1,1,0,0,0,0,0,0,0,0,0],
                        [25,1,1,3,7,0,0,0,0,0,0,0,0,0],
                        [37,1,3,3,9,9,0,0,0,0,0,0,0,0],
                        [59,1,3,7,13,3,0,0,0,0,0,0,0,0],
                        [47,1,1,5,11,27,0,0,0,0,0,0,0,0],
                        [61,1,3,5,1,15,0,0,0,0,0,0,0,0],
                        [55,1,1,7,3,29,0,0,0,0,0,0,0,0],
                        [41,1,3,7,7,21,0,0,0,0,0,0,0,0],
                        [67,1,1,1,9,23,37,0,0,0,0,0,0,0],
                        [97,1,3,3,5,19,33,0,0,0,0,0,0,0],
                        [91,1,1,3,13,11,7,0,0,0,0,0,0,0],
                        [109,1,1,7,13,25,5,0,0,0,0,0,0,0],
                        [103,1,3,5,11,7,11,0,0,0,0,0,0,0],
                        [115,1,1,1,3,13,39,0,0,0,0,0,0,0],
                        [131,1,3,1,15,17,63,13,0,0,0,0,0,0],
                        [193,1,1,5,5,1,59,33,0,0,0,0,0,0],
                        [137,1,3,3,3,25,17,115,0,0,0,0,0,0],
                        [145,1,1,7,15,29,15,41,0,0,0,0,0,0],
                        [143,1,3,1,7,3,23,79,0,0,0,0,0,0],
                        [241,1,3,7,9,31,29,17,0,0,0,0,0,0],
                    ], dtype=int)
    poly = data[:, 0]
    V[:, :13] = data[:, 1:14]
    V[0, :] = 1
    for i in range(1, dimension):
        m = len(np.binary_repr(poly[i])) - 1
        include = np.array([int(b) for b in np.binary_repr(poly[i])[1:]])
        for j in range(m, BITS):
            V[i, j] = V[i, j - m]
            for k in range(m):
                if include[k]:
                    V[i, j] = np.bitwise_xor(V[i, j], 2 ** (k + 1) * V[i, j - k - 1])
    V = V[:dimension] * 2 ** np.arange(BITS)[::-1]

    point = np.zeros(dimension, dtype=int)

    # fast-forward
    for i in range(skip):
        point = np.bitwise_xor(point, V[:, _rightmost_zero(i)])

    # start sampling
    for i in range(skip, 2 ** BITS):
        point = np.bitwise_xor(point, V[:, _rightmost_zero(i)])
        yield point / 2 ** BITS


def _get_sobol_sample(n_points, dimension,skip=0):
    """Generate a Sobol point set.
    Parameters
    ----------
    dimension : int
        Number of dimensions
    n_points : int, optional
        Number of points to sample
    skip : int, optional
        Number of points in the sequence to skip, by default 0
    Returns
    -------
    array, shape=(n_points, dimension)
        Samples from the Sobol sequence.
    """
    sobol = _generator(dimension, skip)
    points = np.empty((n_points, dimension))
    for i in range(n_points):
        points[i] = next(sobol)
    return points


def _init_population_qmc(n, d, qmc_engine='sobol',seed=None):
    """Initializes the population with a QMC method.
    QMC methods ensures that each parameter is uniformly
    sampled over its range.
    Parameters
    ----------
    qmc_engine : str
        The QMC method to use for initialization. Can be one of
        ``latinhypercube``, ``sobol`` or ``halton``.
    """
    try:
        from scipy.stats import qmc
        # Create an array for population of candidate solutions.
        if qmc_engine == 'latinhypercube':
            sampler = qmc.LatinHypercube(d=d, seed=seed)
        elif qmc_engine == 'sobol':        
            sampler = qmc.Sobol(d=d, seed=seed)
        elif qmc_engine == 'halton':
            sampler = qmc.Halton(d=d, seed=seed)
        else:
            raise ValueError("qmc_engine",qmc_engine,"is not recognized.")
        return sampler.random(n=n)
    except:
        print("scipy version mismatch. 'scipy.stat.qmc' is not imported. Using custom halton seqeunce instead")
        return _get_sobol_sample(n,d)


def proximal_ordered_init_sampler(n,
                                  bounds,
                                  x0,
                                  ramping_rate=None,
                                  polarity_change_time=None,
                                  method='sobol',seed=None):
    if n<1:
        raise ValueError('n must be larger than 0') 
    bounds = np.array(bounds,dtype=np.float64)
    d = len(bounds)
    x0 = np.atleast_2d(x0)
    _,xd = x0.shape
    assert _==1
    assert xd==d
    with suppress_outputs():
        samples = list(_init_population_qmc(n, d, method, seed) * 
                       (bounds[:, 1] - bounds[:, 0])[None, :] + bounds[:, 0][None, :])
    return order_samples( samples,
                          x0=x0,
                          ramping_rate=ramping_rate,
                          polarity_change_time=polarity_change_time)

def order_samples(samples,
                  x0,
                  ramping_rate=None,
                  polarity_change_time=None):
    
    ordered_samples = []
    x0 = np.atleast_2d(x0)
    _,d = x0.shape
    assert _ == 1
    samples = np.atleast_2d(samples)
    n = len(samples)
    if n == 1:
        return samples
    samples = samples.tolist()
    if ramping_rate is None:
        ramping_rate = np.ones(d)*5
    polarity_change_time = polarity_change_time or 15
    
    while len(ordered_samples)<n:
        x = np.array(samples)
        sign_x  = np.sign(x)
        sign_x0 = np.sign(x0)
        is_not_zero = np.logical_and(sign_x != 0, sign_x0 != 0)
        is_pol_crossing = (sign_x != sign_x0)
        distance = polarity_change_time * np.all(np.logical_and(is_pol_crossing, is_not_zero), axis=1).astype(np.float64)
        distance += np.max( np.abs(x-x0)/ramping_rate, axis=1)
        ordered_samples.append(samples.pop(np.argmin(distance)))
        x0 = ordered_samples[-1]
        
    return np.array(ordered_samples)



def progressbar(it, prefix="", size=40, out=sys.stdout): # Python3.6+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {int(j/count*100)}%/{100}%", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


def plot_2D_projection(
                        func,
                        bounds,
                        dim_xaxis = 0,
                        dim_yaxis = 1,
                        grid_ponits_each_dim = 25, 
                        project_maximum = True,
                        project_mode = None,
                        fixed_values_for_each_dim = None, 
                        overdrive = False,
                        fig = None,
                        ax = None,
                        colorbar = True,
                        dtype = np.float32 ):
        '''
        bounds : shape (dim,2) array
        fixed_values_for_each_dim: dict of key: dimension, val: value to fix for that dimension
        '''
        
        n_fixed = 0
        if fixed_values_for_each_dim is not None:
            n_fixed = len(fixed_values_for_each_dim.keys())
            assert dim_xaxis not in fixed_values_for_each_dim.keys()
            assert dim_yaxis not in fixed_values_for_each_dim.keys()
        else:
            fixed_values_for_each_dim = {}
            
        dim = len(bounds)  
        
        project_minimum = False
        project_mean = False
        if project_mode is not None:
            if project_mode in ['MAX', 'max', 'Max']:
                project_maximum = True
            elif project_mode in ['MIN', 'min', 'Min']:
                project_minimum = True
                project_maximum = False
            elif project_mode in ['MEAN', 'mean', 'Mean','Average', 'Ave', 'ave']:
                project_mean = True,
                project_maximum = False
        if dim > 2+n_fixed:
            assert project_minimum + project_maximum + project_mean == 1
        
        batch_size = 1
        for n in range(dim-2-n_fixed):
            batch_size*=grid_ponits_each_dim
            if batch_size*dim > 2e4:
                if overdrive or not (project_minimum or project_mean):
                    batch_size = int(batch_size/grid_ponits_each_dim)
                    print("starting projection plot...")
                    break
                else:
                    raise RuntimeError("Aborting: due to high-dimensionality and large number of grid point, minimum or mean projection plot may take long time. Try to reduce 'grid_ponits_each_dim' or turn on 'overdrive' if long time waiting is OK'")
        n_batch = int(grid_ponits_each_dim**(dim-n_fixed-2)/batch_size)
        linegrid = np.linspace(0,1,grid_ponits_each_dim)
        x_grid = np.zeros((grid_ponits_each_dim*grid_ponits_each_dim,dim))
        y_grid = np.zeros((grid_ponits_each_dim*grid_ponits_each_dim))
        
        n = 0
        for i in progressbar(range(grid_ponits_each_dim)):
            bounds_xaxis = bounds[dim_xaxis,:]
            for j in range(grid_ponits_each_dim):
                bounds_yaxis = bounds[dim_yaxis,:]
                x_grid[n,dim_xaxis] = linegrid[i]*(bounds_xaxis[1]-bounds_xaxis[0])+bounds_xaxis[0]
                x_grid[n,dim_yaxis] = linegrid[j]*(bounds_yaxis[1]-bounds_yaxis[0])+bounds_yaxis[0]
                if (project_minimum or project_maximum or project_mean) and dim > 2+n_fixed:
                    inner_grid = []
                    for d in range(dim):
                        if d == dim_xaxis:
                            inner_grid.append([x_grid[n,dim_xaxis]])
                        elif d == dim_yaxis:
                            inner_grid.append([x_grid[n,dim_yaxis]])
                        elif d in fixed_values_for_each_dim.keys():
                            inner_grid.append([fixed_values_for_each_dim[d]])
                        else:
                            inner_grid.append(np.linspace(bounds[d,0],
                                                          bounds[d,1],
                                                          grid_ponits_each_dim))
                    inner_grid = np.meshgrid(*inner_grid)
                    inner_grid = np.array(list(list(x.flat) for x in inner_grid),dtype=dtype).T
                    
                    y = []
                    for b in range(n_batch):
                        i1 = b*batch_size
                        i2 = i1 + batch_size
                        x_batch = inner_grid[i1:i2]
                        y.append(func(x_batch))
                    y = np.concatenate(y, axis=0)

                    if project_minimum or project_maximum:
                        if project_minimum:
                            arg_project = np.nanargmin
                        elif project_maximum:
                            arg_project = np.nanargmax
                        iproj = arg_project(y)
                        y_grid[n] = y[iproj]
                        x_grid[n,:] = inner_grid[iproj]
                    elif project_mean:
                        y_grid[n] = np.nanmean(y_mean)
                n+=1
                
        if dim==2+n_fixed:
            if fixed_values_for_each_dim is not None:
                for dim,val in fixed_values_for_each_dim.items():
                    x_grid[:,dim]  = val   
            y_grid  = func(x_grid)
            
        iNaN = np.any([np.isnan(x_grid).any(axis=1),np.isnan(y_grid)],axis=0)
        x_grid = x_grid[~iNaN,:]
        y_grid = y_grid[~iNaN  ]
                           
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.5,3),dpi=96)
        cs = ax.tricontourf(x_grid[:,dim_xaxis], x_grid[:,dim_yaxis], y_grid, levels=64, cmap="viridis");
        if colorbar:
            fig.colorbar(cs,ax=ax,shrink=0.95)
            
        return cs