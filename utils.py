import numpy as np
import functools
from numbers import Number

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
    Decorator for functions designed to input and output numpy arrays only. Adds
    ability to handle floats/ints. Decorated function must have a numpy array as
    first argument, then unlimited args/kwargs. It must also return an array of
    the same shape.
    """

    # we need to differentiate between methods and functions because the first array argument is significant
    # i.e. sometiems arg[0] is array, sometimes self. Maybe a better way is possible. TODO review

    # is the __qualname__ func or class.func?
    if func.__name__ != func.__qualname__:

        @functools.wraps(func)
        def wrapper(self, array: np.ndarray, *args, **kwargs):

            # no change
            if isinstance(array, np.ndarray):
                return func(self, array, *args, **kwargs)

            # transform to arraqy of shape (1, ) and take zeroth element
            elif isinstance(array, Number):
                return func(self, np.array([array]), *args, **kwargs)[0]

            else:
                raise ValueError('Illegal argument to {}. Expected numpy array or float, but got {}'.format(func, type(array)))

    # as before with small changes
    else:

        @functools.wraps(func)
        def wrapper(array: np.ndarray, *args, **kwargs):

            if isinstance(array, np.ndarray):
                return func(array, *args, **kwargs)

            elif isinstance(array, Number):
                return func(np.array([array]), *args, **kwargs)[0]

            else:
                raise ValueError('Illegal argument to {}. Expected numpy array or float, but got {}'.format(func, type(array)))

    return wrapper


def domain(a: float, b:float, outside_domain: float=np.nan):
    """
    A decorator, with two arguments, for functions which take a numpy array as their
    first argument. The domain is enforced, meaning the corresponding output element
    for any elements not falling within this range get replaced. By default, they
    become nan, but this can be configured via tha outside_domain argument. This
    is appropriate only for functions returning an array of the same shape as the
    input array

    Parameters
    ----------
    a               Lower domain bound
    b               Upper domain bound
    outside_domain

    """

    # return this decorator
    def decorator(func: callable):

        # again, I see no better way to do this that differentiate between methods and functions
        # is the __qualname__ func or class.func?
        if func.__name__ != func.__qualname__:

            @functools.wraps(func)
            def wrapper(self, array: np.ndarray, *args, **kwargs):

                if isinstance(array, np.ndarray):
                    out = np.empty_like(array)
                    illegal = (array < a) | (array > b)
                    out[illegal] = outside_domain
                    out[~illegal] = func(self, array[~illegal], *args, **kwargs)
                    return out

                elif isinstance(array, Number):
                    if array > b or array < a:
                        return outside_domain
                    return func(self, np.array([array]), *args, **kwargs)[0]

                else:
                    raise ValueError('Illegal argument to {}. Expected numpy array or float, but got {}'.format(func, type(array)))

        else:

            @functools.wraps(func)
            def wrapper(array: np.ndarray, *args, **kwargs):

                if isinstance(array, np.ndarray):
                    out = np.empty_like(array)
                    illegal = (array < a) | (array > b)
                    out[illegal] = outside_domain
                    out[~illegal] = func(array[~illegal], *args, **kwargs)
                    return func(array, *args, **kwargs)

                elif isinstance(array, Number):
                    if array > b or array < a:
                        return outside_domain
                    return func(np.array([array]), *args, **kwargs)[0]

                else:
                    raise ValueError('Illegal argument to {}. Expected numpy array or float, but got {}'.format(func, type(array)))

        return wrapper

    return decorator


def support(a: float, b:float):
    """
    Quick way to enforce support on distributinon pdfs by calling
    domain, and setting outside_domain to 0.

    Parameters
    ----------
    a               Lower domain bound
    b               Upper domain bound
    """

    return domain(a, b, 0)

