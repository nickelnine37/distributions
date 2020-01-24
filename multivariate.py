import numpy as np
from numpy import pi, exp
from scipy.special import gamma
from scipy.linalg import cholesky, det, inv
from scipy.stats import multivariate_normal as mvn
from utils import handle_floats, domain, one_dimensional

class MultivariateDistribution:

    def __init__(self):
        self.d = None

    def pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, N):
        raise NotImplementedError


class IndependantMultivariateDistribution(MultivariateDistribution):

    def __init__(self, distributions: list):
        super().__init__()

        self.d = len(distributions)
        self.distributions = distributions

    def pdf(self, x: np.ndarray) -> np.ndarray:

        if x.shape == (self.d, ):
            return np.prod([distribution.pdf(x[i]) for i, distribution in enumerate(self.distributions)])

        elif len(x.shape) == 2 and x.shape[1] == self.d:
            return np.prod([distribution.pdf(x[:, i]) for i, distribution in enumerate(self.distributions)], axis=0)

        else:
            raise ValueError('Input to multivariate pdf must have shape ({0}, ) or (N, {0})'.format(self.d))

    def cdf(self, x: np.ndarray):

        if x.shape == (self.d,):
            return np.prod([distribution.cdf(x[i]) for i, distribution in enumerate(self.distributions)])

        elif len(x.shape) == 2 and x.shape[1] == self.d:
            return np.prod([distribution.cdf(x[:, i]) for i, distribution in enumerate(self.distributions)], axis=0)

        else:
            raise ValueError('Input to multivariate cdf must have shape ({0}, ) or (N, {0})'.format(self.d))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:

        if x.shape == (self.d, ):
            return np.sum([distribution.log_pdf(x[i]) for i, distribution in enumerate(self.distributions)])

        elif len(x.shape) == 2 and x.shape[1] == self.d:
            return np.sum([distribution.log_pdf(x[:, i]) for i, distribution in enumerate(self.distributions)], axis=0)

        else:
            raise ValueError('Input to multivariate log pdf must have shape ({0}, ) or (N, {0})'.format(self.d))

    def sample(self, N):
        return np.stack([distribution.sample(N) for distribution in self.distributions], axis=-1)


class MultivariateNormal(MultivariateDistribution):

    def __init__(self, μ: np.ndarray, Σ: np.ndarray):
        super().__init__()

        self.μ = one_dimensional(μ)
        self.d = len(self.μ)
        assert Σ.shape == (self.d, self.d)
        self.Σ = Σ

        # save these calculations until we need them
        self.Σ_det = None   # Covariance determinant
        self.Σ_inv = None   # Covariance inverse
        self.A = None       # Cholesky decomposition

    def pdf(self, x: np.ndarray):

        try:
            x = one_dimensional(x)
        except ValueError:
            pass

        if self.Σ_det is None:
            self.Σ_det = det(self.Σ)
            self.Σ_inv = inv(self.Σ)

        if x.shape == (self.d, ):
            return (2 * pi) ** (-self.d / 2) * self.Σ_det ** -0.5 * exp(-0.5 * (x - self.μ) @ self.Σ_inv @ (x - self.μ))

        elif len(x.shape) == 2 and x.shape[1] == self.d:
            return (2 * pi) ** (-self.d / 2) * self.Σ_det ** -0.5 * exp(-0.5 * ((x - self.μ) * (self.Σ_inv @ (x - self.μ).T).T).sum(1))

        else:
            raise ValueError('Input to multivariate pdf must have shape ({0}, ) or (N, {0})'.format(self.d))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return mvn(mean=self.μ, cov=self.Σ).cdf(x)

    def sample(self, N: int):

        if self.A is None:
            self.A = cholesky(self.Σ, lower=False)

        return np.random.normal(size=(N, self.d)) @ self.A + self.μ


class MultivariateT(MultivariateDistribution):

    def __init__(self, v: float, μ: np.ndarray, Σ: np.ndarray):
        super().__init__()

        self.v = v
        self.μ = one_dimensional(μ)
        self.d = len(self.μ)

        assert Σ.shape == (self.d, self.d)
        self.Σ = Σ

        # save these calculations until we need them
        self.Σ_det = None   # Covariance determinant
        self.Σ_inv = None   # Covariance inverse
        self.A = None       # Cholesky decomposition

    def pdf(self, x: np.ndarray):

        x = one_dimensional(x)

        # now do the calculations if needs be
        if self.Σ_det is None:
            self.Σ_det = det(self.Σ)
            self.Σ_inv = inv(self.Σ)

        c = gamma((self.d + self.v) / 2) / (gamma(self.v / 2) * ((self.v * pi) ** (self.d / 2)) * self.Σ_det ** 0.5)
        return c * (1 + (1 / self.v) * (x - self.μ) @ self.Σ_inv @ (x - self.μ)) ** (-(self.v + self.d) / 2)

    def sample(self, N: int):

        # https://stackoverflow.com/questions/41957633/sample-from-a-multivariate-t-distribution-python

        if self.A is None:
            self.A = cholesky(self.Σ, lower=True)

        x = np.random.chisquare(self.v, N) / self.v
        return self.μ + (self.A @ np.random.normal(size=(self.d, N))).T / (x ** 0.5)[:, None]


class CopulaDistribution(MultivariateDistribution):

    def __init__(self, marginals: list, copula):
        super().__init__()

        self.copula = copula
        self.marginals = marginals
        self.d = len(marginals)
        assert self.d == copula.d, 'Copula and marginals should have same dimensionality'

    def sample(self, N: int):
        copula_sample = self.copula.sample(N)
        return np.array([self.marginals[i].quantile(copula_sample[:, i]) for i in range(self.d)]).T



if __name__ == '__main__':

    from distributions import Gamma, Normal, InverseGamma, Laplace, Exponential

    dists = [Laplace(1, 1), Normal(0, 5), Exponential(2)]
    d = IndependantMultivariateDistribution(dists)

    x = np.stack([np.linspace(-5, 5, 1000) for _ in range(3)], axis=-1)
    print(d.pdf(np.array([1, 2, 3])))
    # print(d.log_pdf(x))
    print(d.sample(100))