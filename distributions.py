import numpy as np
from scipy.special import erf, erfinv
from typing import Union
from scipy.interpolate import interp1d
from scipy.integrate import quad as integrate
from utils import handle_floats

class UnivariateDistribution:

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs

    def pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inv_cdf(self, u: np.ndarray) -> np.ndarray:
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
        q_α = self.inv_cdf(α)
        return integrate(lambda x: f(x) * self.pdf(x), a=q_α, b=np.inf)[0] / (1 - q_α)

    def BSQ(self, γ_prime: callable, γ_prime_inv: callable, α: float) -> float:
        return γ_prime_inv(self.E_α(γ_prime, α))


class Normal(UnivariateDistribution):

    def __init__(self, μ: float, σ: float):
        super().__init__(μ=μ, σ=σ)

        self.μ = μ
        self.σ = σ

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (self.σ * (2 * np.pi) ** 0.5) ** -1 * np.exp(- (x - self.μ) ** 2 / (2 * self.σ ** 2))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + erf((x - self.μ) / (self.σ * 2 ** 0.5)))

    def inv_cdf(self, u: np.ndarray) -> np.ndarray:
        return self.μ + self.σ * 2 ** 0.5 * erfinv(2 * u - 1)

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:
        return np.random.normal(loc=self.μ, scale=self.σ, size=size)

    def __repr__(self):
        return 'Normal Distribution: μ = {}, σ = {}'.format(self.μ, self.σ)


class Laplace(UnivariateDistribution):

    def __init__(self, μ: float, b: float):
        super().__init__(μ=μ, b=b)

        self.μ = μ
        self.b = b

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (2 * self.b) ** -1 * np.exp(-np.abs(x - self.μ) / self.b)

    @handle_floats
    def cdf(self, x: np.ndarray) -> np.ndarray:

        out = np.empty_like(x)
        out[x < self.μ] = 0.5 * np.exp((x[x < self.μ] - self.μ) / self.b)
        out[x >= self.μ] = 1 - 0.5 * np.exp((self.μ - x[x >= self.μ]) / self.b)

        return out

    def inv_cdf(self, u: np.ndarray) -> np.ndarray:
        return self.μ - self.b * np.sign(u - 0.5) * np.log(1 - 2 * np.abs(u - 0.5))

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:
        return np.random.laplace(loc=self.μ, scale=self.b, size=size)

    def __repr__(self):
        return 'Laplace Distribution: μ = {}, b = {}'.format(self.μ, self.b)


class Exponential(UnivariateDistribution):

    def __init__(self, λ: float):
        super().__init__(λ=λ)
        self.λ = λ

    @handle_floats
    def pdf(self, x: np.ndarray) -> np.ndarray:

        out = np.empty_like(x)
        out[x >= 0] = self.λ * np.exp(- self.λ * x[x >= 0])
        out[x < 0] = 0

        return out

    @handle_floats
    def cdf(self, x: np.ndarray) -> np.ndarray:

        out = np.empty_like(x)
        out[x >= 0] = 1 - np.exp(- self.λ * x[x >= 0])
        out[x < 0] = 0

        return out

    def inv_cdf(self, u: np.ndarray) -> np.ndarray:
        return - np.log(1 - u) / self.λ

    def sample(self, size: Union[tuple, int]=None) -> np.ndarray:
        return np.random.exponential(scale=self.λ**-1, size=size)

    def __repr__(self):
        return 'Exponential distribution: λ = {}'.format(self.λ)


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

    def inv_cdf(self, u: np.ndarray, tol: float=1e-6, steps: int=10000) -> np.ndarray:

        x0 = min([distribution.inv_cdf(tol) for distribution in self.distributions])
        x1 = max([distribution.inv_cdf(1-tol) for distribution in self.distributions])
        x = np.linspace(x0, x1, steps)

        return interp1d(self.cdf(x), x)(u)

    def __repr__(self):
        return ('Mixed Distribution: ' + ('{}, ' * self.n_dists)[:-2] + '. With weights: ' + ('{}, ' * self.n_dists)[:-2]).format(*[dist.__repr__() for dist in self.distributions], *self.weights)


if __name__ == '__main__':

    d1 = Exponential(λ=5)
    bs = np.linspace(1, 3, 101)
    print(d1.pdf(bs))
    # bsq85 = np.array([d1.BSQ(lambda x: x ** b, lambda x: x ** 1 / b, 0.85) for b in bs])


