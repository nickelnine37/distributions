import numpy as np
from multivariate import MultivariateDistribution, MultivariateNormal, MultivariateT
from distributions import Normal, StudentT

class Copula:

    def __init__(self):

        self.d = None
        self.generator = None
        self.marginals = None

    def sample(self, N: int):

        if any([q is None for q in [self.d, self.generator, self.marginals]]):
            raise Exception('Instantiate Copula of specific type')

        samples = self.generator.sample(N)
        return np.array([self.marginals[i].cdf(samples[:, i]) for i in range(self.d)]).T


class GaussianCopula(Copula):

    def __init__(self, Σ: np.ndarray):
        super().__init__()

        self.d = Σ.shape[0]
        self.generator = MultivariateNormal(np.zeros(self.d), Σ)
        self.marginals = [Normal(0, Σ[i, i] ** 0.5) for i in range(self.d)]


class StudentTCopula(Copula):

    def __init__(self, v: float, Σ: np.ndarray):
        super().__init__()

        self.d = Σ.shape[0]
        self.generator = MultivariateT(v, np.zeros(self.d), Σ)
        self.marginals = [StudentT(v, 0, Σ[i, i] ** 0.5) for i in range(self.d)]






if __name__ == '__main__':

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from distributions import Laplace, Gamma, Exponential
    from multivariate import CopulaDistribution
    import time

    # sample = pd.DataFrame(GaussianCopula(np.array([[ 1,  0.9],
    #                                                 [0.9,  1 ]])).sample(10000), columns=['x', 'y'])

    # fig, ax = plt.subplots()
    #
    # plt.scatter(sample[:, 0], sample[:, 1])
    # ax.set_aspect('equal')
    # plt.show()

    # sns.jointplot(x='x', y='y', data=sample, ratio=2, joint_kws={'alpha': 0.3, 's': 10},
    #               marginal_kws={'bins': 50})
    # plt.show()

    copula = StudentTCopula(1.5, np.array([[ 1,  0.2],
                                         [0.2,  1 ]]))

    dist = CopulaDistribution(marginals=[Laplace(0.0008, 0.005), Laplace(0.001, 0.008)], copula=copula)

    N = 10000000
    t0 = time.time()
    sample = dist.sample(N)
    print('time for {} samples: {:.2f}s'.format(N, time.time() - t0))

    vs = np.linspace(dist.marginals[1].quantile(0.01), dist.marginals[1].quantile(0.99), 99)
    es = []
    for v in vs:
        es.append(sample[:, 0][sample[:, 1] < v].mean())

    plt.plot(vs, es)
    plt.show()
    # import yfinance
    #
    # d1 = yfinance.Ticker('BA').history(period='max', interval='1d')['Close'].pct_change()[1:].rename('boeing')
    # d2 = yfinance.Ticker('AIR.PA').history(period='max', interval='1d')['Close'].pct_change()[1:].rename('airbus')
    #
    # d = pd.concat([d1, d2], axis=1, join='inner')
    #
    #
    # sns.jointplot(x='boeing', y='airbus', data=d, ratio=2, joint_kws={'alpha': 0.3, 's': 10},
    #               marginal_kws={'bins': 100})
    # plt.show()

