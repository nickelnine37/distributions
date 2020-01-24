import numpy as np
from numpy import pi, exp, log, inf
from typing import Union
from scipy.special import erf, erfinv, gamma, gammainc, beta
from scipy.interpolate import interp1d
from scipy.integrate import quad as integrate
from scipy.stats import t
from utils import handle_floats, domain, support

class UnivariateDistribution:

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs

    def pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @domain(0, 1)
    def quantile(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:
        raise NotImplementedError

    def E(self, f: callable) -> float:
        """
        The expected value of a function of random variable X
        """
        return integrate(lambda x: f(x) * self.pdf(x), a=-np.inf, b=np.inf)[0]

    def E_α(self, f: callable, α: float) -> float:
        """
        The expected value of a function of random variable X, given
        that the random variable is above the quantile q_α
        """
        q_α = self.quantile(α)
        return integrate(lambda x: f(x) * self.pdf(x), a=q_α, b=np.inf)[0] / (1 - q_α)

    def BSQ(self, γ_prime: callable, γ_prime_inv: callable, α: float) -> float:
        return γ_prime_inv(self.E_α(γ_prime, α))

    @classmethod
    def fit(cls, data: np.ndarray, method='MLE'):
        raise NotImplementedError


class Normal(UnivariateDistribution):

    def __init__(self, μ: float, σ: float):
        super().__init__(μ=μ, σ=σ)

        self.μ = μ
        self.σ = σ

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (self.σ * (2 * pi) ** 0.5) ** -1 * exp(- (x - self.μ) ** 2 / (2 * self.σ ** 2))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + erf((x - self.μ) / (self.σ * 2 ** 0.5)))

    @domain(0, 1)
    def quantile(self, u: np.ndarray) -> np.ndarray:
        return self.μ + self.σ * 2 ** 0.5 * erfinv(2 * u - 1)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return - (x - self.μ) ** 2 / (2 * self.σ ** 2) - 0.5 * log(2 * pi * self.σ ** 2)

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:
        return np.random.normal(loc=self.μ, scale=self.σ, size=size)

    @classmethod
    def fit(cls, data: np.ndarray, method='MLE'):
        return cls(data.mean(), data.std())

    def __repr__(self):
        return 'Normal Distribution: μ = {}, σ = {}'.format(self.μ, self.σ)


class Uniform(UnivariateDistribution):

    def __init__(self, a: float, b: float):
        super().__init__()

        assert b > a
        self.a = a
        self.b = b

    def pdf(self, x: np.ndarray) -> np.ndarray:

        out = np.empty_like(x)
        out[x < self.a] = 0
        out[x > self.b] = 0
        out[(x >= self.a) & (x <= self.b)] = 1 / (self.b - self.a)

        return out

    def cdf(self, x: np.ndarray) -> np.ndarray:

        out = np.empty_like(x)
        out[x < self.a] = 0
        out[x > self.b] = 1
        out[(x >= self.a) & (x <= self.b)] = (x - self.a) / (self.b - self.a)

        return out

    @domain(0, 1)
    def quantile(self, u: np.ndarray) -> np.ndarray:
        return (1 - u) * self.a + u * self.b

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:
        return np.random.uniform(self.a, self.b, size=size)


class Laplace(UnivariateDistribution):

    def __init__(self, μ: float, b: float):
        super().__init__(μ=μ, b=b)

        self.μ = μ
        self.b = b

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (2 * self.b) ** -1 * exp(-np.abs(x - self.μ) / self.b)

    @handle_floats
    def cdf(self, x: np.ndarray) -> np.ndarray:

        out = np.empty_like(x)
        out[x < self.μ] = 0.5 * exp((x[x < self.μ] - self.μ) / self.b)
        out[x >= self.μ] = 1 - 0.5 * exp((self.μ - x[x >= self.μ]) / self.b)

        return out

    @domain(0, 1)
    def quantile(self, u: np.ndarray) -> np.ndarray:
        return self.μ - self.b * np.sign(u - 0.5) * np.log(1 - 2 * np.abs(u - 0.5))

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return -np.abs(x - self.μ) / self.b - log(2 * self.b)

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:
        return np.random.laplace(loc=self.μ, scale=self.b, size=size)

    def __repr__(self):
        return 'Laplace Distribution: μ = {}, b = {}'.format(self.μ, self.b)


class Exponential(UnivariateDistribution):

    def __init__(self, λ: float):
        super().__init__(λ=λ)
        self.λ = λ

    @support(0, inf)
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self.λ * exp(- self.λ * x[x >= 0])

    @support(0, inf)
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return 1 - exp(- self.λ * x[x >= 0])

    @domain(0, 1)
    def quantile(self, u: np.ndarray) -> np.ndarray:
        return - np.log(1 - u) / self.λ

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:
        return np.random.exponential(scale=self.λ**-1, size=size)

    def __repr__(self):
        return 'Exponential distribution: λ = {}'.format(self.λ)


class Gamma(UnivariateDistribution):

    def __init__(self, α: float,  β: float):
        super().__init__(α=α, β=β)
        self.α = α
        self.β = β
        self.c = β ** α / gamma(α)

    @support(0, inf)
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return  self.c * x ** (self.α - 1) * exp(-self.β * x)

    @support(0, inf)
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return gammainc(self.α, self.β * x)

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:
        return np.random.gamma(self.α, 1 / self.β, size=size)

    @domain(0, 1)
    def quantile(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError("No quantile for gamma yet :'(")

    def __repr__(self):
        return 'Gamma distribution: α = {}, β= {}'.format(self.α, self.β)


class InverseGamma(UnivariateDistribution):

    def __init__(self, α: float,  β: float):
        super().__init__(α=α, β=β)
        self.α = α
        self.β = β
        self.c = β ** α / gamma(α)

    @support(0, inf)
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return  self.c * x ** (-self.α - 1) * exp(-self.β / x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return gammainc(self.α, self.β / x)

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:
        return 1 / np.random.gamma(self.α, 1 / self.β, size=size)

    @domain(0, 1)
    def quantile(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError("No quantile for inverse gamma yet :'(")

    def __repr__(self):
        return 'Inverse gamma distribution: α = {}, β= {}'.format(self.α, self.β)


class Lognormal(UnivariateDistribution):

    def __init__(self, μ: float, σ: float):
        super().__init__(μ=μ, σ=σ)

        self.μ = μ
        self.σ = σ

    @support(0, inf)
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (x * self.σ * (2 * pi) ** 0.5) * exp(- (log(x) - self.μ) ** 2 / (2 * self.σ ** 2))

    @support(0, inf)
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return 0.5 + 0.5 * erf((log(x) - self.μ) / (2 ** 0.5* self.σ))

    @domain(0, 1)
    def quantile(self, u: np.ndarray) -> np.ndarray:
        return exp(self.μ + 2 ** 0.5 * self.σ * erfinv(2 * u - 1))


class StudentT(UnivariateDistribution):

    def __init__(self, v: float, μ: float=0, σ: float=1):
        super().__init__()

        self.v = v
        self.μ = μ
        self.σ = σ

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (self.v ** 0.5 * beta(0.5, self.v / 2)) ** -1 * (1 + (((x - self.μ) / self.σ) ** 2) / self.v) ** (- 0.5 * (self.v + 1))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return t(df=self.v, loc=self.μ, scale=self.σ).cdf(x)

    def quantile(self, u: np.ndarray) -> np.ndarray:
        return t(df=self.v, loc=self.μ, scale=self.σ).ppf(x)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return t(df=self.v, loc=self.μ, scale=self.σ).logpdf(x)


class WeightedMixture(UnivariateDistribution):

    def __init__(self, distribtutions: list, weights: list):
        super().__init__(distribtutions=distribtutions, weights=weights)

        assert len(distribtutions) == len(weights)
        assert sum(weights) == 1

        self.distributions = distribtutions
        self.weights = weights
        self.n_dists = len(weights)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return sum([weight * distribution.pdf(x) for weight, distribution in zip(self.weights, self.distributions)])

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return sum([weight * distribution.cdf(x) for weight, distribution in zip(self.weights, self.distributions)])

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:

        if isinstance(size, int):
            n_samples = size
        elif isinstance(size, (tuple, list)):
            n_samples = np.prod(size)
        else:
            raise ValueError

        samples = np.concatenate([distribution.sample(n_samples)[:, None] for distribution in self.distributions], axis=-1)
        indicies = np.random.choice(self.n_dists, p=self.weights, size=n_samples)

        return samples[np.arange(n_samples), indicies].reshape(size)

    def quantile(self, u: np.ndarray, tol: float=1e-6, steps: int=10000) -> np.ndarray:

        x0 = min([distribution.quantile(tol) for distribution in self.distributions])
        x1 = max([distribution.quantile(1-tol) for distribution in self.distributions])
        x = np.linspace(x0, x1, steps)

        return interp1d(self.cdf(x), x)(u)

    def __repr__(self):
        return ('Mixed Distribution: ' + ('{}, ' * self.n_dists)[:-2] + '. With weights: ' + ('{}, ' * self.n_dists)[:-2]).format(*[dist.__repr__() for dist in self.distributions], *self.weights)




if __name__ == '__main__':

    import matplotlib.pyplot as plt

    samples = np.random.normal(3.5, 4.2, 1000)

    d = Normal.fit(samples)
    x = np.linspace(d.μ - 4 * d.σ, d.μ + 4 * d.σ, 10001)


    plt.hist(samples, bins=50, density=True)
    plt.plot(x, d.pdf(x))
    plt.show()



