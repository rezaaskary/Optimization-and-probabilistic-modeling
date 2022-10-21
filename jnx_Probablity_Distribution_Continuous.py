import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap, jit, grad, random, lax, scipy
from tensorflow_probability.substrates.jax.math.special import owens_t, betaincinv
from jax.lax import switch


class ContinuousDistributions:
    def __init__(self,
                 b: jnp.ndarray = None,
                 lambd: jnp.ndarray = None,
                 beta: jnp.ndarray = None,
                 mu: jnp.ndarray = None,
                 alpha: jnp.ndarray = None,
                 variance: jnp.ndarray = None,
                 sigma: jnp.ndarray = None,
                 lower: jnp.ndarray = None,
                 upper: jnp.ndarray = None,
                 variant_chains: bool = False,
                 activate_jit: bool = False,
                 nchains: int = 1,
                 rng: int = 1) -> None:

        self.key = random.PRNGKey(rng)

        if isinstance(lambd, (jnp.ndarray, float, int)):
            self.lambd = lambd
        elif lambd is None:
            self.lambd = None
        else:
            raise Exception('The value of lambda is not specified correctly!')

        if isinstance(b, (jnp.ndarray, float, int)):
            self.b = b
        elif b is None:
            self.b = None
        else:
            raise Exception('The value of b is not specified correctly!')

        if isinstance(beta, (jnp.ndarray, float, int)):
            self.beta = beta
        elif beta is None:
            self.beta = None
        else:
            raise Exception('The value of beta is not specified correctly!')

        if isinstance(mu, (jnp.ndarray, float, int)):
            self.mu = mu
        elif mu is None:
            self.mu = None
        else:
            raise Exception('The value of mu is not specified correctly!')

        if isinstance(alpha, (jnp.ndarray, float, int)):
            self.alpha = alpha
        elif alpha is None:
            self.alpha = None
        else:
            raise Exception('The value of alpha is not specified correctly!')

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
        Statistics calculated for the Truncated Normal distribution function given distribution parameters
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


class SkewedNormal(ContinuousDistributions):
    def __init__(self, mu: float = None, alpha: float = None, sigma: float = None, variance: float = None,
                 activate_jit: bool = False) -> None:
        """
        Continuous Skewed Normal distribution
        :param alpha:
        :param mu:
        :param sigma: The standard deviation of the distribution
        :param variance: The variance of the distribution
        :param activate_jit: Activating just-in-time evaluation of the methods
        """
        super(SkewedNormal, self).__init__(sigma=sigma, mu=mu, alpha=alpha, variance=variance,
                                           activate_jit=activate_jit)
        # check for the consistency of the input of the probability distribution

        if self.variance is None or self.sigma is None or self.mu is None:
            raise Exception(
                'The value of either mean or standard deviation is not specified (Normal distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Half Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        z = (x - self.mu) / self.sigma
        erf_part = 0.5 * (1 + lax.erf(z * (self.alpha / jnp.sqrt(2.0))))
        normal_part = (1 / (jnp.sqrt(2 * jnp.pi))) * jnp.exp(-0.5 * (z ** 2))
        return 2 * erf_part * normal_part

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
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        erf_part = 0.5 * (1 + lax.erf(x / jnp.sqrt(2)))
        return erf_part - 2 * owens_t((x - self.mu) / self.sigma, self.alpha)

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
            return None

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Half Normal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        delta_ = self.alpha / jnp.sqrt(1 + self.alpha ** 2)
        mean_ = self.mu + self.sigma * delta_ * jnp.sqrt(2 / jnp.pi)
        variance_ = (self.sigma ** 2) * (1 - 2 * (delta_ ** 2) / jnp.pi)
        gamma1_ = 0.5 * (4 - jnp.pi) * ((delta_ * jnp.sqrt(2 / jnp.pi)) ** 3) / (1 - 2 * (delta_ ** 2 / jnp.pi)) ** 1.5
        kurtosis_ = 2 * (jnp.pi - 3) * ((delta_ * jnp.sqrt(2 / jnp.pi)) ** 4) / (1 - 2 * (delta_ ** 2 / jnp.pi)) ** 2

        muz = jnp.sqrt(2 / jnp.pi)
        sigmaz_ = jnp.sqrt(1 - muz ** 2)
        m0 = muz - 0.5 * gamma1_ * sigmaz_ - 0.5 * jnp.sign(self.alpha) * jnp.exp(-(2 * jnp.pi) / jnp.abs(self.alpha))
        mode_ = self.mu + self.sigma * m0
        values = {'mean': mean_,
                  'variance': variance_,
                  'skewness': gamma1_,
                  'mode': mode_,
                  'kurtosis': kurtosis_,
                  }
        return values


class BetaPdf(ContinuousDistributions):

    def __init__(self, alpha: None, beta: None, activate_jit: bool = False) -> None:
        """
        Continuous Beta distribution
        :param alpha:
        :param beta:
        :param activate_jit:
        """
        super(BetaPdf, self).__init__(beta=beta, alpha=alpha, activate_jit=activate_jit)
        # check for the consistency of the input of the probability distribution

        if self.alpha <= 0:
            raise Exception('Parameter alpha (for calculating the beta distribution) should be positive')
        if self.beta <= 0:
            raise Exception('Parameter beta (for calculating the beta distribution) should be positive')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Half Normal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """

        def beta_(a, b):
            beta = (jnp.exp(scipy.special.gammaln(a)) * jnp.exp(scipy.special.gammaln(b))) / jnp.exp(
                scipy.special.gammaln(b + a))
            return beta

        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return ((x ** (self.alpha - 1)) * ((1 - x) ** (self.beta - 1))) / beta_(self.alpha, self.beta)

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

        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        log_prob = (self.alpha - 1) * jnp.log(x) + (self.beta - 1) * jnp.log(1 - x) + \
                   scipy.special.gammaln(self.alpha) + scipy.special.gammaln(self.beta) - \
                   scipy.special.gammaln(self.beta + self.alpha)
        return log_prob

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Beta probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return scipy.special.betainc(a=self.alpha, b=self.beta, x=x)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Normal probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Normal distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y):
            return betaincinv(a=self.alpha, b=self.beta, y=y)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Beta distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """

        variance_ = (self.alpha * self.beta) / ((self.alpha + self.beta + 1) * (self.alpha + self.beta) ** 2)
        skewmess_ = (2 * (self.beta - self.alpha) * jnp.sqrt(self.beta + self.alpha + 1)) / (
                (self.alpha + self.beta + 2) *
                jnp.sqrt(self.alpha * self.beta))

        values = {'mean': self.alpha / (self.alpha + self.beta),
                  'variance': variance_,
                  'skewness': skewmess_,
                  }
        return values


class Kumaraswamy(ContinuousDistributions):

    def __init__(self, alpha: None, beta: None, activate_jit: bool = False) -> None:
        """
        Kumaraswamy distribution
        :param alpha:
        :param beta:
        :param activate_jit:
        """
        super(Kumaraswamy, self).__init__(beta=beta, alpha=alpha, activate_jit=activate_jit)
        # check for the consistency of the input of the probability distribution

        if self.alpha <= 0:
            raise Exception('Parameter alpha (for calculating the beta distribution) should be positive')
        if self.beta <= 0:
            raise Exception('Parameter beta (for calculating the beta distribution) should be positive')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Kumaraswamy distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """

        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        term1 = (x ** (self.alpha - 1))
        term2 = (1 - x ** self.alpha)
        return self.beta * self.alpha * term1 * (term2 ** (self.beta - 1))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Kumaraswamy distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Kumaraswamy probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        log_prob = jnp.log(self.alpha * self.beta) + (self.alpha - 1) * jnp.log(x) + (self.beta - 1) * jnp.log(
            (1 - x ** self.alpha))
        return log_prob

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Kumaraswamy probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Kumaraswamy probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return 1 - (1 - x ** self.alpha) ** self.beta

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Kumaraswamy probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Kumaraswamy probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1.0)
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Kumaraswamy distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return (1 - (1 - y) ** (1 / self.beta)) ** (1 / self.alpha)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Kumaraswamy distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        median_ = (1 - 2 ** (-1 / self.beta)) ** (1 / self.alpha)

        values = {'median': median_}
        return values


class Exponential(ContinuousDistributions):

    def __init__(self, lambd: None, activate_jit: bool = False) -> None:
        """
        Exponential distribution
        :param lambda:
        :param activate_jit:
        """
        super(Exponential, self).__init__(lambd=lambd, activate_jit=activate_jit)
        # check for the consistency of the input of the probability distribution

        if self.lambd <= 0:
            raise Exception('Parameter lambda (for calculating the Exponential distribution) should be positive')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Exponential distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """

        return jnp.where(x < 0, 0, self.lambd * jnp.exp(-self.lambd * x))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Exponential distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Kumaraswamy probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return jnp.where(x < 0, -jnp.inf, jnp.log(self.lambd) - self.lambd * x)

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Kumaraswamy probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Kumaraswamy probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        return jnp.where(x < 0, 0, 1 - jnp.exp(- self.lambd * x))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Kumaraswamy probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Kumaraswamy probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Kumaraswamy distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return jnp.log(1 - y) / -self.lambd

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Kumaraswamy distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        first_quantile_ = jnp.log(0.75) / -self.lambd
        third_quantile_ = jnp.log(0.25) / -self.lambd
        mean_ = 1 / self.lambd
        mode_ = 0
        variance_ = (1 / self.lambd) ** 2
        skewness_ = 2
        kurtosis_ = 6
        entropy_ = 1 - jnp.log(self.lambd)
        median_ = jnp.log(0.5) / -self.lambd

        values = {'median': median_,
                  'first_quantile': first_quantile_,
                  'third_quantile': third_quantile_,
                  'mean': mean_,
                  'mode': mode_,
                  'variance': variance_,
                  'skewness': skewness_,
                  'kurtosis': kurtosis_,
                  'entropy': entropy_
                  }
        return values


class Laplace(ContinuousDistributions):

    def __init__(self, mu: None, b: None, activate_jit: bool = False) -> None:
        """
        Exponential distribution
        :param b:
        :param mu
        :param activate_jit:
        """
        super(Laplace, self).__init__(mu=mu, b=b, activate_jit=activate_jit)
        # check for the consistency of the input of the probability distribution

        if self.b <= 0:
            raise Exception('Parameter b (for calculating the Laplace distribution) should be positive')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Laplace distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        return (1 / (2 * self.b)) * jnp.exp((-1 / self.b) * jnp.abs(x - self.mu))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Laplace distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Laplace probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(2 * self.b) - (1 / self.b) * jnp.abs(x - self.mu)

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Laplace probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Laplace probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        return jnp.where(x >= self.mu, 1 - 0.5 * jnp.exp((-1 / self.b) * (x - self.mu)), 0.5 * jnp.exp((1 / self.b) *
                                                                                                       (x- self.mu)))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Laplace probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Laplace probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Laplace distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return self.mu - self.b * jnp.sign(y - 0.5) * jnp.log(1 - 2 * jnp.abs(y-0.5))

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Laplace distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        median_ = self.mu

        first_quantile_ = self.mu + self.b * jnp.log(2*0.25)
        third_quantile_ = self.mu + self.b * jnp.log(2-2*0.75)
        mean_ = self.mu
        mode_ = self.mu
        variance_ = 2 * self.b ** 2
        skewness_ = 0
        kurtosis_ = 3
        entropy_ = jnp.log(2 * self.b * jnp.exp(1))

        values = {'median': median_,
                  'first_quantile': first_quantile_,
                  'third_quantile': third_quantile_,
                  'mean': mean_,
                  'mode': mode_,
                  'variance': variance_,
                  'skewness': skewness_,
                  'kurtosis': kurtosis_,
                  'entropy': entropy_
                  }
        return values


x = random.uniform(key=random.PRNGKey(7), minval=-2, maxval=2, shape=(1000, 1))
activate_jit = False

KK = Laplace(mu=0, b=1, activate_jit=activate_jit)
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

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
