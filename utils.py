import numpy as np
import functools
from types import FunctionType, MethodType

def one_dimensional(array: np.ndarray) -> np.ndarray:
    """
    Check if an array is really one dimensional. This could be various
    shapes, e.g (N, 1, 1), (1, N, 1) etc. If array is not 1d, raise an
    error. Else return the array flattened to (N, )

    Parameters
    ----------
    array               A numpy array that should be 1d

    Returns
    -------
    array_flattened     The same array flattened to shape (N, )

    """

    # nothing to do
    if len(array.shape) == 1:
        return array

    # if two or more of the dimensions are greater than 1
    elif (np.array(array.shape) != 1).sum() > 1:
        raise ValueError('Array must be one-dimensional but it has shape {}'.format(array.shape))

    # else return flattened array
    else:
        return array.reshape(-1)


def handle_floats(func: callable):
    """
    Decorator to for functions designed to input and output numpy arrays only. Adds
    ability to handle floats/ints
    """

    if isinstance(func, MethodType):

        @functools.wraps(func)
        def wrapper(self, array: np.ndarray, *args, **kwargs):

            if isinstance(array, np.ndarray):
                return func(self, array, *args, **kwargs)

            elif isinstance(array, (int, float)):
                return func(self, np.array([array]), *args, **kwargs)[0]

            else:
                raise ValueError('Illegal argument to {}'.format(func))

    elif isinstance(func, FunctionType):

        @functools.wraps(func)
        def wrapper(array: np.ndarray, *args, **kwargs):

            if isinstance(array, np.ndarray):
                return func(array, *args, **kwargs)

            elif isinstance(array, (int, float)):
                return func(np.array([array]), *args, **kwargs)[0]

            else:
                raise ValueError('Illegal argument to {}'.format(func))

    else:
        raise ValueError()

    return wrapper


