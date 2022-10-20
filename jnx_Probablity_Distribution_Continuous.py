import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap, jit, grad, random, lax, scipy

from jax.lax import switch


class ContinuousDistributions:
    def __init__(self,
                 mu: jnp.ndarray = None,
                 variance: jnp.ndarray = None,
                 sigma: jnp.ndarray = None,
                 lower: jnp.ndarray = None,
                 upper: jnp.ndarray = None,
                 variant_chains: bool = False,
                 activate_jit: bool = False,
                 nchains: int = 1,
                 rng: int = 1) -> None:

        self.key = random.PRNGKey(rng)

        if isinstance(mu, (jnp.ndarray, float, int)):
            self.mu = mu
        elif mu is None:
            self.mu = None
        else:
            raise Exception('The value of mu is not specified correctly!')

        if isinstance(sigma, (jnp.ndarray, float, int)) and isinstance(variance, (jnp.ndarray, float, int)):
            raise Exception('Please Enter either variance or standard deviation!')

        if isinstance(sigma, (jnp.ndarray, float, int)) and not isinstance(variance, (jnp.ndarray, float, int)):
            if sigma > 0:
                self.sigma = sigma
                self.variance = sigma ** 2
            else:
                raise Exception('The standard deviation should be a positive value!')

        if not isinstance(sigma, (jnp.ndarray, float, int)) and isinstance(variance, (jnp.ndarray, float, int)):
            if variance > 0:
                self.sigma = jnp.sqrt(variance)
                self.variance = variance
            else:
                raise Exception('The standard deviation should be a positive value!')

        if sigma is None and variance is None:
            self.sigma = None
            self.variance = None

        if isinstance(lower, (jnp.ndarray, float, int)):
            self.lower = lower
        elif lower is None:
            self.a = None
        else:
            raise Exception('The value of lower is not specified correctly!')

        if isinstance(upper, (jnp.ndarray, float, int)):
            self.upper = upper
        elif upper is None:
            self.upper = None
        else:
            raise Exception('The value of upper is not specified correctly!')

        if isinstance(variant_chains, bool):
            self.variant_chains = variant_chains
        else:
            raise Exception('Please specify whether the number of chains are fixed or variant during simulation !')

        if isinstance(activate_jit, bool):
            self.activate_jit = activate_jit
        else:
            raise Exception('Please specify the activation of the just-in-time evaluation!')

    def parallelization(self):
        if not self.variant_chains:
            # when the number of parallel evaluation is fixed. Useful for MCMC
            if self.activate_jit:
                self.pdf = jit(vmap(self.pdf_, in_axes=[0], out_axes=0))
                self.diff_pdf = jit(vmap(grad(self.diff_pdf_), in_axes=[0], out_axes=0))
                self.log_pdf = jit(vmap(self.log_pdf_, in_axes=[0], out_axes=0))
                self.diff_log_pdf = jit(vmap(grad(self.diff_log_pdf_), in_axes=[0], out_axes=0))
                self.cdf = jit(vmap(self.cdf_, in_axes=[0], out_axes=0))
                self.log_cdf = jit(vmap(self.log_cdf_, in_axes=[0], out_axes=0))
                self.diff_cdf = jit(vmap(grad(self.diff_cdf_), in_axes=[0], out_axes=0))
                self.diff_log_cdf = jit(vmap(grad(self.diff_log_cdf_), in_axes=[0], out_axes=0))
                self.sample = self.sample_
            else:
                self.sample = self.sample_
                self.pdf = vmap(self.pdf_, in_axes=[0], out_axes=0)
                self.diff_pdf = vmap(grad(self.diff_pdf_), in_axes=[0], out_axes=0)
                self.log_pdf = vmap(self.log_pdf_, in_axes=[0], out_axes=0)
                self.diff_log_pdf = vmap(grad(self.diff_log_pdf_), in_axes=[0], out_axes=0)
                self.cdf = vmap(self.cdf_, in_axes=[0], out_axes=0)
                self.log_cdf = vmap(self.log_cdf_, in_axes=[0], out_axes=0)
                self.diff_cdf = vmap(grad(self.diff_cdf_), in_axes=[0], out_axes=0)
                self.diff_log_cdf = vmap(grad(self.diff_log_cdf_), in_axes=[0], out_axes=0)


        else:
            pass


class Uniform(ContinuousDistributions):
    def __init__(self, lower: float = None, upper: float = None, activate_jit: bool = False) -> None:
        """
        Continuous uniform distribution
        :param lower: The lower limit of uniform distribution
        :param upper: The upper limit of uniform distribution
        """
        super(Uniform, self).__init__(lower=lower, upper=upper, activate_jit=activate_jit)
        # check for the consistency of the input of the probability distribution
        if not isinstance(self.lower, type(self.upper)):
            raise Exception('The input parameters are not consistent (Uniform Distribution)!')

        if jnp.any(self.lower >= self.upper):
            raise Exception('The lower limit of the uniform distribution is greater than the upper limit!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Uniform distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        return jnp.where((x > self.lower) & (x < self.upper), 1 / (self.upper - self.lower), 0)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where((x > self.lower) & (x < self.upper), -jnp.log((self.upper - self.lower)), -jnp.inf)

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x < self.lower, 0, jnp.where(x < self.upper, (x - self.lower) / (self.upper - self.lower), 1))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(
            jnp.where(x < self.lower, 0, jnp.where(x < self.upper, (x - self.lower) / (self.upper - self.lower), 1)))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        return random.uniform(key=self.key, minval=self.lower, maxval=self.upper, shape=(size, 1))

    @property
    def statistics(self):
        """
        Statistics calculated for the Uniform distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        values = {'mean': 0.5 * (self.lower + self.upper),
                  'median': 0.5 * (self.lower + self.upper),
                  'variance': (1 / 12) * (self.lower - self.upper) ** 2,
                  'MAD': (1 / 4) * (self.lower + self.upper),
                  'skewness': 0,
                  'kurtosis': -6 / 5,
                  'Entropy': jnp.log(self.upper - self.lower)
                  }
        return values


class Normal(ContinuousDistributions):
    def __init__(self, sigma: float = None, variance: float = None, mu: float = None,
                 activate_jit: bool = False) -> None:
        """
        Continuous Normal distribution
        :param sigma: The standard deviation of the distribution
        :param variance: The variance of the distribution
        :param mu: The center of the distribution
        :param activate_jit: Activating just-in-time evaluation of the methods
        """
        super(Normal, self).__init__(sigma=sigma, variance=variance, mu=mu, activate_jit=activate_jit)
        # check for the consistency of the input of the probability distribution

        if self.mu is None or self.sigma is None:
            raise Exception('The value of either mean or standard deviation is not specified (Normal distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        return (1 / (self.sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return -jnp.log((self.sigma * jnp.sqrt(2 * jnp.pi))) - ((x - self.mu) ** 2) / (2 * self.sigma ** 2)

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        z = (x - self.mu) / (self.sigma * jnp.sqrt(2))
        return lax.erf(z)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Normal distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))
        return vmap(scipy.special.erfinv, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Normal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        values = {'mean': self.mu,
                  'median': self.mu,
                  'first_quentile': self.mu + self.sigma * jnp.sqrt(2) * scipy.special.erfinv(2 * 0.25 - 1),
                  'third_quentile': self.mu + self.sigma * jnp.sqrt(2) * scipy.special.erfinv(2 * 0.75 - 1),
                  'variance': self.variance,
                  'mode': self.mu,
                  'MAD': self.sigma * jnp.sqrt(2 / jnp.pi),
                  'skewness': 0,
                  'kurtosis': 0,
                  'Entropy': 0.5 * (1 + jnp.log(2 * jnp.pi * self.sigma ** 2))
                  }
        return values


class TruncatedNormal(ContinuousDistributions):
    def __init__(self, lower: float = None, upper: float = None, sigma: float = None,
                 variance: float = None, mu: float = None, activate_jit: bool = False) -> None:
        """
        Continuous Truncated Normal distribution
        :param lower: The lower bound of the distribution
        :param upper: The upper bound of the distribution
        :param sigma: The standard deviation of the distribution
        :param variance: The variance of the distribution
        :param mu: The center of the distribution
        :param activate_jit: Activating just-in-time evaluation of the methods
        """
        super(TruncatedNormal, self).__init__(lower=lower, upper=upper, sigma=sigma,
                                              variance=variance, mu=mu, activate_jit=activate_jit)
        # check for the consistency of the input of the probability distribution

        if self.mu is None or self.sigma is None:
            raise Exception('The value of either mean or standard deviation is not specified (Normal distribution)!')

        if self.lower >= self.upper:
            raise Exception('The lower bound of the distribution cannot be greater than the upper bound!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Truncated Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """

        def pdf_in_range(x: jnp.ndarray) -> jnp.ndarray:
            arg_r = (self.upper - self.mu) / self.sigma
            arg_l = (self.lower - self.mu) / self.sigma
            normal_fcn_value = (1 / (jnp.sqrt(2 * jnp.pi))) * jnp.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
            return (1 / self.sigma) * (normal_fcn_value /
                                       (0.5 * (1 + lax.erf(arg_r / jnp.sqrt(2))) - 0.5 * (
                                               1 + lax.erf(arg_l / jnp.sqrt(2)))))

        return jnp.where(x < self.lower, 0, jnp.where(x > self.upper, 0, pdf_in_range(x)))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return (self.log_pdf(x))[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        def middle_range(x: jnp.ndarray) -> jnp.ndarray:
            b = (self.upper - self.mu) / self.sigma
            a = (self.lower - self.mu) / self.sigma
            erf_r = 0.5 * (1 + lax.erf(b / jnp.sqrt(2)))
            ert_l = 0.5 * (1 + lax.erf(a / jnp.sqrt(2)))
            ert_xi = 0.5 * (1 + lax.erf(((x - self.mu) / self.sigma) / jnp.sqrt(2)))
            return (ert_xi - ert_l) / (erf_r - ert_l)

        return jnp.where(x < self.lower, 0, jnp.where(x > self.upper, 1, middle_range(x)))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Normal distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def reverse_cdf(y: jnp.ndarray) -> jnp.ndarray:
            b = (self.upper - self.mu) / self.sigma
            a = (self.lower - self.mu) / self.sigma
            erf_r = 0.5 * (1 + lax.erf(b / jnp.sqrt(2)))
            ert_l = 0.5 * (1 + lax.erf(a / jnp.sqrt(2)))
            z = (erf_r - ert_l) * y + ert_l
            return scipy.special.erfinv(2 * z - 1) * self.sigma * jnp.sqrt(2) + self.mu

        return vmap(reverse_cdf, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Normal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        alpha = (self.lower - self.mu) / self.sigma
        beta = (self.upper - self.mu) / self.sigma
        fi_alpha_ = 0.5 * (1 + lax.erf(alpha / jnp.sqrt(2)))
        fi_beta_ = 0.5 * (1 + lax.erf(beta / jnp.sqrt(2)))
        denominator = fi_beta_ - fi_alpha_
        fi_alpha = (1 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * alpha ** 2)
        fi_beta = (1 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * beta ** 2)
        mean_ = self.mu + ((fi_alpha - fi_beta) / denominator) * self.sigma
        median_ = self.mu + scipy.special.erfinv(0.5 * (fi_alpha_ + fi_beta_)) * self.sigma
        mode_ = jnp.where(self.mu < self.lower, self.lower, jnp.where(self.mu > self.upper, self.upper, self.mu))
        variance_ = (self.sigma ** 2) * (
                1 + ((alpha * fi_alpha - beta * fi_beta) / denominator) - ((fi_alpha - fi_beta) /
                                                                           denominator) ** 2)
        entropy_ = 0.5 * ((alpha * fi_alpha - beta * fi_beta) / denominator) + \
                   jnp.log(denominator * self.sigma * jnp.sqrt(2 * jnp.pi * jnp.exp(1)))
        values = {'mean': mean_,
                  'median': median_,
                  'variance': variance_,
                  'mode': mode_,
                  'Entropy': entropy_
                  }
        return values


class HalfNormal(ContinuousDistributions):
    def __init__(self, sigma: float = None, variance: float = None, activate_jit: bool = False) -> None:
        """
        Continuous Half Normal distribution
        :param sigma: The standard deviation of the distribution
        :param variance: The variance of the distribution
        :param activate_jit: Activating just-in-time evaluation of the methods
        """
        super(HalfNormal, self).__init__(sigma=sigma, variance=variance, activate_jit=activate_jit)
        # check for the consistency of the input of the probability distribution

        if self.variance is None or self.sigma is None:
            raise Exception(
                'The value of either variance or standard deviation is not specified (Normal distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Half Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        return jnp.where(x < 0, 0, (jnp.sqrt(2 / jnp.pi) / self.sigma) * jnp.exp(-(x ** 2) / (2 * self.sigma ** 2)))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        log_pdf = jnp.where(x >= 0, jnp.log(self.pdf_(x)), - jnp.inf)
        return log_pdf

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.where(x >= 0, lax.erf(x / (self.sigma * jnp.sqrt(2))), 0)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Normal distribution
        :param size:
        :return:
        """

        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y):
            return self.sigma * jnp.sqrt(2) * scipy.special.erfinv(y)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Half Normal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """

        values = {'mean': self.sigma * jnp.sqrt(2 / jnp.pi),
                  'median': self.sigma * jnp.sqrt(2) * scipy.special.erfinv(0.5),
                  'first_quantile': self.sigma * jnp.sqrt(2) * scipy.special.erfinv(0.25),
                  'third_quantile': self.sigma * jnp.sqrt(2) * scipy.special.erfinv(0.75),
                  'variance': (self.sigma ** 2) * (1 - 2 / jnp.pi),
                  'skewness': (jnp.sqrt(2) * (4 - jnp.pi)) / (jnp.pi - 2) ** 1.5,
                  'mode': 0,
                  'kurtosis': (8 * (jnp.pi - 3)) / (jnp.pi - 2) ** 2.0,
                  'entropy': 0.5 * jnp.log2(2 * jnp.pi * jnp.exp(1) * self.sigma ** 2) - 1
                  }
        return values

class HalfNormal(ContinuousDistributions):
    def __init__(self, sigma: float = None, variance: float = None, activate_jit: bool = False) -> None:
        """
        Continuous Half Normal distribution
        :param sigma: The standard deviation of the distribution
        :param variance: The variance of the distribution
        :param activate_jit: Activating just-in-time evaluation of the methods
        """
        super(HalfNormal, self).__init__(sigma=sigma, variance=variance, activate_jit=activate_jit)
        # check for the consistency of the input of the probability distribution

        if self.variance is None or self.sigma is None:
            raise Exception(
                'The value of either variance or standard deviation is not specified (Normal distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Half Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        return jnp.where(x < 0, 0, (jnp.sqrt(2 / jnp.pi) / self.sigma) * jnp.exp(-(x ** 2) / (2 * self.sigma ** 2)))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        log_pdf = jnp.where(x >= 0, jnp.log(self.pdf_(x)), - jnp.inf)
        return log_pdf

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.where(x >= 0, lax.erf(x / (self.sigma * jnp.sqrt(2))), 0)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Normal distribution
        :param size:
        :return:
        """

        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y):
            return self.sigma * jnp.sqrt(2) * scipy.special.erfinv(y)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Half Normal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """

        values = {'mean': self.sigma * jnp.sqrt(2 / jnp.pi),
                  'median': self.sigma * jnp.sqrt(2) * scipy.special.erfinv(0.5),
                  'first_quantile': self.sigma * jnp.sqrt(2) * scipy.special.erfinv(0.25),
                  'third_quantile': self.sigma * jnp.sqrt(2) * scipy.special.erfinv(0.75),
                  'variance': (self.sigma ** 2) * (1 - 2 / jnp.pi),
                  'skewness': (jnp.sqrt(2) * (4 - jnp.pi)) / (jnp.pi - 2) ** 1.5,
                  'mode': 0,
                  'kurtosis': (8 * (jnp.pi - 3)) / (jnp.pi - 2) ** 2.0,
                  'entropy': 0.5 * jnp.log2(2 * jnp.pi * jnp.exp(1) * self.sigma ** 2) - 1
                  }
        return values













x = random.uniform(key=random.PRNGKey(7), minval=-20, maxval=20, shape=(10000, 1))
activate_jit = False

KK = HalfNormal(sigma=4, activate_jit=activate_jit)
E1 = KK.pdf(x)
plt.figure(dpi=150)
plt.plot(x, E1, '*')
plt.title('PDF')
plt.show()

E6 = KK.diff_pdf(x)
plt.figure(dpi=150)
plt.plot(x, E6, '*')
plt.title('Diff PDF')
plt.show()

E2 = KK.log_pdf(x)
plt.figure(dpi=150)
plt.plot(x, E2, '*')
plt.title('LOG PDF')
plt.show()

E3 = KK.diff_log_pdf(x)
plt.figure(dpi=150)
plt.plot(x, E3, '*')
plt.title('DIFF LOG PDF')
plt.show()

E4 = KK.cdf(x)
plt.figure(dpi=150)
plt.plot(x, E4, '*')
plt.title('CDF')
plt.show()

E5 = KK.log_cdf(x)
plt.figure(dpi=150)
plt.plot(x, E5, '*')
plt.title('LOG CDF')
plt.show()

E8 = KK.diff_cdf(x)
plt.figure(dpi=150)
plt.plot(x, E8, '*')
plt.title('DIFF CDF')
plt.show()

E9 = KK.diff_log_cdf(x)
plt.figure(dpi=150)
plt.plot(x, E9, '*')
plt.title('DIFF LOG CDF')
plt.show()

E7 = KK.sample(size=20000)
plt.figure(dpi=150)
plt.hist(E7, 30)
plt.title('samples')
plt.show()

E


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
