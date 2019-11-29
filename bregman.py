import numpy as np

bregman_functions = {'Euclidean': {'gamma': lambda x: x ** 2,
                                   'gamma prime': lambda x: 2 * x,
                                   'gamma prime inverse': lambda x: x / 2},

                     'Geometric': {'gamma': lambda x: x * np.log(x) - x + 1,
                                   'gamma prime': lambda x: np.log(x),
                                   'gamma prime inverse': lambda x: np.exp(x)},

                     'Harmonic' : {'gamma': lambda x: -np.log(x) + x - 1,
                                   'gamma prime': lambda x: -1 / x + 1,
                                   'gamma prime inverse': lambda x: 1 / (1 - x)},
                     }

