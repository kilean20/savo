import numpy as np


def obj_func_wrapper(obj_func, x_CSET):
    """
    Wrapper function for evaluating an objective function with various output formats.

    This function takes an objective function `obj_func` and a test input `x_CSET`. It calls the
    `obj_func` with `x_CSET` and processes the output to ensure consistency in the format of the
    result. The function handles different types of output from `obj_func`, including floats,
    NumPy arrays, lists, and dictionaries, and returns a standardized result that is
        : [float]objective value, [float]objective rms noise, [array]decision readback.

    Parameters:
    - obj_func (callable): A user-defined objective function that takes a NumPy array as input
      and returns a result.
    - x_CSET (array-like): The input to the objective function, typically a NumPy array.

    Returns:
    - obj_func_wrapp (callable): A wrapper function that takes an input array and returns a
      standardized tuple containing the objective value, objective error, and decision readback.
      The format of the output depends on the type of `obj_func`'s output.
    - y (float): The objective value obtained from `obj_func`.
    - y_err (float): The objective error (rms noise) obtained from `obj_func`.
    - x_RD (array-like): The decision readback obtained from `obj_func`.
    """

   
    x_CSET = np.array(x_CSET).reshape(-1)
    assert len(x_CSET) > 1
   
    out = obj_func(x_CSET)
    errmsg = "output of obj_func must be \n" \
            + "\t[float] objective value  \n" \
            + "\t[list[float]] objective value, [list[float]] decision readback \n" \
            + "\t[list[float]] objective value, [float] objective rms noise, [list[float]] decision readback" \
            + "\t[dict] of keys: objective value, objective rms noise, decision readback"

    if isinstance(out, float):
        def obj_func_wrapp(x):
            return obj_func(x), 0, x
        y = out
        y_err = 0
        x_RD = x_CSET  # Corrected variable name
   
    elif isinstance(out, np.ndarray):
        try:
            y = float(out)
            y_err = 0
            x_RD = x_CSET  # Corrected variable name
            def obj_func_wrapp(x):
                return float(obj_func(x)), 0, x
        except:
            raise ValueError(errmsg)
           
    elif isinstance(out, list):
        try:
            y = float(out[0])
        except:
            raise ValueError(errmsg)
        try:
            y_err = float(out[1])
            assert len(np.array(out[2]).reshape(-1)) == len(x_CSET)
            x_RD = np.array(out[2]).reshape(-1)
            def obj_func_wrapp(x):
                out = obj_func(x)
                return float(out[0]), float(out[1]), np.array(out[2]).reshape(-1)
        except:
            y_err = 0
            assert len(np.array(out[1]).reshape(-1)) == len(x_CSET)
            x_RD = np.array(out[1]).reshape(-1)
            def obj_func_wrapp(x):
                out = obj_func(x)
                return float(out[0]), 0, np.array(out[1]).reshape(-1)

    elif isinstance(out, dict):
        try:
            y = float(out['objective value'])
        except:
            raise ValueError(errmsg)
        assert len(np.array(out['decision readback']).reshape(-1)) == len(x_CSET)
        x_RD = np.array(out['decision readback']).reshape(-1)
        try:
            y_err = float(out['objective rms noise'])
            def obj_func_wrapp(x):
                out = obj_func(x)
                return float(out['objective value']), float(out['objective rms noise']), np.array(out['decision readback']).reshape(-1)
        except:
            y_err = 0
            def obj_func_wrapp(x):
                out = obj_func(x)
                return float(out['objective value']), 0, np.array(out['decision readback']).reshape(-1)

    else:
        raise ValueError(errmsg)
   
    return obj_func_wrapp, y, y_err, x_RD