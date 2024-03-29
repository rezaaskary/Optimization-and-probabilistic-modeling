import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap, jit, grad, random, lax, scipy
from tensorflow_probability.substrates.jax.math.special import owens_t, betaincinv, igammainv
from tensorflow_probability.substrates.jax.math.hypergeometric import _hyp2f1_fraction, hyp2f1_small_argument
from jax.lax import switch


class ContinuousDistributions:
    def __init__(self,
                 kappa: jnp.ndarray = None,
                 b: jnp.ndarray = None,
                 a: jnp.ndarray = None,
                 c: jnp.ndarray = None,
                 lambd: jnp.ndarray = None,
                 beta: jnp.ndarray = None,
                 gamma: jnp.ndarray = None,
                 mu: jnp.ndarray = None,
                 nu: jnp.ndarray = None,
                 alpha: jnp.ndarray = None,
                 variance: jnp.ndarray = None,
                 sigma: jnp.ndarray = None,
                 xm: jnp.ndarray = None,
                 lower: jnp.ndarray = None,
                 upper: jnp.ndarray = None,
                 variant_chains: bool = False,
                 activate_jit: bool = False,
                 n_chains: int = 1,
                 random_seed: int = 1) -> None:

        if isinstance(random_seed, int):
            self.key = random.PRNGKey(random_seed)
        else:
            raise Exception('The random seed is not specified correctly!')

        if isinstance(nu, (jnp.ndarray, float, int)):
            self.nu = nu
        elif nu is None:
            self.nu = None
        else:
            raise Exception('The value of nu is not specified correctly!')

        if isinstance(xm, (jnp.ndarray, float, int)):
            self.xm = xm
        elif xm is None:
            self.xm = None
        else:
            raise Exception('The value of xm is not specified correctly!')

        if isinstance(gamma, (jnp.ndarray, float, int)):
            self.gamma = gamma
        elif gamma is None:
            self.gamma = None
        else:
            raise Exception('The value of gamma is not specified correctly!')

        if isinstance(kappa, (jnp.ndarray, float, int)):
            self.kappa = kappa
        elif kappa is None:
            self.kappa = None
        else:
            raise Exception('The value of kappa is not specified correctly!')

        if isinstance(lambd, (jnp.ndarray, float, int)):
            self.lambd = lambd
        elif lambd is None:
            self.lambd = None
        else:
            raise Exception('The value of lambda is not specified correctly!')

        if isinstance(b, (jnp.ndarray, float, int)):
            self.b = float(b)
        elif b is None:
            self.b = None
        else:
            raise Exception('The value of b is not specified correctly!')

        if isinstance(a, (jnp.ndarray, float, int)):
            self.a = float(a)
        elif a is None:
            self.a = None
        else:
            raise Exception('The value of a is not specified correctly!')

        if isinstance(c, (jnp.ndarray, float, int)):
            self.c = float(c)
        elif c is None:
            self.c = None
        else:
            raise Exception('The value of c is not specified correctly!')

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
                self.sigma = variance ** 0.5
                self.variance = variance
            else:
                raise Exception('The standard deviation should be a positive value!')

        if sigma is None and variance is None:
            self.sigma = None
            self.variance = None

        if isinstance(lower, (jnp.ndarray, float, int)):
            self.lower = lower
        elif lower is None:
            self.lower = None
        else:
            raise Exception('The value of lower is not specified correctly!')

        if isinstance(upper, (jnp.ndarray, float, int)):
            self.upper = upper
        elif upper is None:
            self.upper = None
        else:
            raise Exception('The value of upper is not specified correctly!')

        if isinstance(n_chains, int):
            self.n_chains = n_chains
        elif n_chains is None:
            self.n_chains = 1
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
                self.pdf = jit(vmap(self.pdf_, in_axes=[1], out_axes=1))
                self.diff_pdf = jit(vmap(grad(self.diff_pdf_), in_axes=[0], out_axes=0))
                self.log_pdf = jit(vmap(self.log_pdf_, in_axes=[1], out_axes=1))
                self.diff_log_pdf = jit(vmap(grad(self.diff_log_pdf_), in_axes=[0], out_axes=0))
                self.cdf = jit(vmap(self.cdf_, in_axes=[1], out_axes=1))
                self.log_cdf = jit(vmap(self.log_cdf_, in_axes=[1], out_axes=1))
                self.diff_cdf = jit(vmap(grad(self.diff_cdf_), in_axes=[0], out_axes=0))
                self.diff_log_cdf = jit(vmap(grad(self.diff_log_cdf_), in_axes=[0], out_axes=0))
                self.sample = self.sample_
            else:
                self.sample = self.sample_
                self.pdf = vmap(self.pdf_, in_axes=[1], out_axes=1)
                self.diff_pdf = vmap(grad(self.diff_pdf_), in_axes=[0], out_axes=0)
                self.log_pdf = vmap(self.log_pdf_, in_axes=[1], out_axes=1)
                self.diff_log_pdf = vmap(grad(self.diff_log_pdf_), in_axes=[0], out_axes=0)
                self.cdf = vmap(self.cdf_, in_axes=[1], out_axes=1)
                self.log_cdf = vmap(self.log_cdf_, in_axes=[1], out_axes=1)
                self.diff_cdf = vmap(grad(self.diff_cdf_), in_axes=[0], out_axes=0)
                self.diff_log_cdf = vmap(grad(self.diff_log_cdf_), in_axes=[0], out_axes=0)

        else:
            pass


class Uniform(ContinuousDistributions):
    def __init__(self, lower: float = None, upper: float = None, activate_jit: bool = False,
                 random_seed: int = 1) -> None:
        """
        In probability theory and statistics, the continuous uniform distribution or rectangular distribution is a
        family of symmetric probability distributions. The distribution describes an experiment where there is an
        arbitrary outcome that lies between certain bounds. The bounds are defined by the parameters, lower and upper,
        which are the minimum and maximum values.
        [1] Dekking, Michel (2005). A modern introduction to probability and statistics : understanding why and how.
        London, UK: Springer. pp. 60–61. ISBN 978-1-85233-896-1

        Continuous uniform distribution
        :param lower: The lower limit of uniform distribution
        :param upper: The upper limit of uniform distribution
        """
        super(Uniform, self).__init__(lower=lower, upper=upper, activate_jit=activate_jit, random_seed=random_seed)
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
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        In statistics, a normal distribution or Gaussian distribution is a type of continuous probability distribution
        for a real-valued random variable. The parameter mu  is the mean or expectation of the distribution (and also
        its median and mode), while the parameter sigma  is its standard deviation. The variance of the distribution is
        sigma^2. A random variable with a Gaussian distribution is said to be normally distributed, and is called a
        normal deviate.
        [1] Lyon, A. (2014). Why are Normal Distributions Normal?, The British Journal for the Philosophy of Science.

        Continuous Normal distribution
        :param sigma: The standard deviation of the distribution
        :param variance: The variance of the distribution
        :param mu: The center of the distribution
        :param activate_jit: Activating just-in-time evaluation of the methods
        """
        super(Normal, self).__init__(sigma=sigma, variance=variance, mu=mu,
                                     activate_jit=activate_jit, random_seed=random_seed)
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
        return 0.5 * (1 + lax.erf(z))

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
                 variance: float = None, mu: float = None, activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        In probability and statistics, the truncated normal distribution is the probability distribution derived from
        that of a normally distributed random variable by bounding the random variable from either below or above
        (or both). The truncated normal distribution has wide applications in statistics and econometrics.

        Continuous Truncated Normal distribution
        :param lower: The lower bound of the distribution
        :param upper: The upper bound of the distribution
        :param sigma: The standard deviation of the distribution
        :param variance: The variance of the distribution
        :param mu: The center of the distribution
        :param activate_jit: Activating just-in-time evaluation of the methods
        """
        super(TruncatedNormal, self).__init__(lower=lower, upper=upper, sigma=sigma,
                                              variance=variance, mu=mu, activate_jit=activate_jit,
                                              random_seed=random_seed)
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
        return (self.log_pdf_(x))[0]

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
    def __init__(self, sigma: float = None, variance: float = None, activate_jit: bool = False,
                 random_seed: int = 1) -> None:
        """
        Continuous Half Normal distribution
        :param sigma: The standard deviation of the distribution
        :param variance: The variance of the distribution
        :param activate_jit: Activating just-in-time evaluation of the methods
        """
        super(HalfNormal, self).__init__(sigma=sigma, variance=variance, activate_jit=activate_jit,
                                         random_seed=random_seed)
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
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Continuous Skewed Normal distribution
        :param alpha:
        :param mu:
        :param sigma: The standard deviation of the distribution
        :param variance: The variance of the distribution
        :param activate_jit: Activating just-in-time evaluation of the methods
        """
        super(SkewedNormal, self).__init__(sigma=sigma, mu=mu, alpha=alpha, variance=variance,
                                           activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.variance is None or self.sigma is None or self.mu is None:
            raise Exception(
                'The value of either mean or standard deviation is not specified (Skewed Normal distribution)!')

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

        return 0.5 * (1 + lax.erf((x - self.mu) / (self.sigma * jnp.sqrt(2)))) - \
               2 * owens_t((x - self.mu) / self.sigma, self.alpha)

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

    def sample_(self, size: int = 1):
        """
        Sampling form the Normal distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray = None) -> None:
            return N

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

    def __init__(self, alpha: None, beta: None, activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Continuous Beta distribution
        :param alpha:
        :param beta:
        :param activate_jit:
        """
        super(BetaPdf, self).__init__(beta=beta, alpha=alpha, activate_jit=activate_jit, random_seed=random_seed)
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

    def __init__(self, alpha: None, beta: None, activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Kumaraswamy distribution
        :param alpha:
        :param beta:
        :param activate_jit:
        """
        super(Kumaraswamy, self).__init__(beta=beta, alpha=alpha, activate_jit=activate_jit, random_seed=random_seed)
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

    def __init__(self, lambd: None, activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Exponential distribution
        :param lambda:
        :param activate_jit:
        """
        super(Exponential, self).__init__(lambd=lambd, activate_jit=activate_jit, random_seed=random_seed)
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

    def __init__(self, mu: None, b: None, activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Laplace distribution
        :param b:
        :param mu
        :param activate_jit:
        """
        super(Laplace, self).__init__(mu=mu, b=b, activate_jit=activate_jit, random_seed=random_seed)
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
                                                                                                       (x - self.mu)))

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
            return self.mu - self.b * jnp.sign(y - 0.5) * jnp.log(1 - 2 * jnp.abs(y - 0.5))

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Laplace distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        median_ = self.mu

        first_quantile_ = self.mu + self.b * jnp.log(2 * 0.25)
        third_quantile_ = self.mu + self.b * jnp.log(2 - 2 * 0.75)
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


class AsymmetricLaplace(ContinuousDistributions):

    def __init__(self, kappa: float = None, mu: float = None, b: float = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Asymmetric Laplace Distribution
        :param kappa:
        :param mu:
        :param b:
        :param activate_jit:
        :param random_seed:
        """
        super(AsymmetricLaplace, self).__init__(mu=mu, b=b, kappa=kappa,
                                                activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.kappa <= 0:
            raise Exception('The values of Symmetric parameter should be positive(Asymmetric Laplace distribution)!')
        if self.b <= 0:
            raise Exception(
                'The rate of the change of the exponential term should be positive(Asymmetric Laplace distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Asymmetric Laplace distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        coefficient = self.b / (self.kappa + 1 / self.kappa)
        return jnp.where(x >= self.mu, coefficient * jnp.exp(-self.b * self.kappa * (x - self.mu)), coefficient *
                         jnp.exp((self.b / self.kappa) * (x - self.mu)))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Asymmetric Laplace distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Asymmetric Laplace probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Asymmetric Laplace probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Asymmetric Laplace probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        return jnp.where(x >= self.mu, 1 - (1 / (1 + self.kappa ** 2)) * jnp.exp(-self.b * self.kappa * (x - self.mu)),
                         (self.kappa ** 2 / (1 + self.kappa ** 2)) * jnp.exp((self.b / self.kappa) * (x - self.mu)))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Asymmetric Laplace probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Asymmetric Laplace probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Asymmetric Laplace distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            threshold = (self.kappa ** 2) / (1 + self.kappa ** 2)
            return jnp.where(y <= threshold, jnp.log(y / threshold) * (self.kappa / self.b) + self.mu,
                             jnp.log((1 - y) * (1 + self.kappa ** 2)) * (-1 / (self.kappa * self.b)) + self.mu)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Asymmetric Laplace distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        mean_ = self.mu + (1 - self.kappa ** 2) / (self.b * self.kappa)
        median_ = jnp.where(self.kappa > 1,
                            self.mu + (self.kappa / self.b) * jnp.log(
                                (1 + self.kappa ** 2) / (2 * self.kappa ** 2)),
                            self.mu - (1 / (self.b * self.kappa)) * jnp.log((1 + self.kappa ** 2) / 2)
                            )

        variance_ = (1 + self.kappa ** 4) / ((self.kappa ** 2) * (self.b ** 2))
        skewness_ = 2 * (1 - self.kappa ** 6) / (self.kappa ** 4 + 1) ** 1.5
        kurtosis_ = 6 * (1 + self.kappa ** 8) / (self.kappa ** 4 + 1) ** 2.0
        entropy_ = 1 + jnp.log((1 + self.kappa ** 2) / (self.kappa * self.b))

        values = {'median': median_,
                  'mean': mean_,
                  'variance': variance_,
                  'skewness': skewness_,
                  'kurtosis': kurtosis_,
                  'entropy': entropy_
                  }
        return values


class StudentT(ContinuousDistributions):

    def __init__(self, nu: float = None, activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        T student Distribution
        :param nu:
        :param activate_jit:
        :param random_seed:
        """
        super(StudentT, self).__init__(nu=nu, activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.nu <= 0:
            raise Exception('The value of nu should be positive (Student-t distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Student-t distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        coefficient = (jnp.exp(scipy.special.gammaln((self.nu + 1) / 2)) /
                       jnp.exp(scipy.special.gammaln(self.nu / 2))) * \
                      jnp.sqrt(1 / (jnp.pi * self.nu))
        return coefficient * (1 + (1 / self.nu) * x ** 2) ** (-(self.nu + 1) / 2)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Student-t distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Student-t probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Student-t probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Student-t probability distribution. It  is recommended to use  float 64 variables to acquire a
        dedent accuracy.
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        coefficient = (jnp.exp(scipy.special.gammaln((self.nu + 1) / 2)) /
                       jnp.exp(scipy.special.gammaln(self.nu / 2))) * \
                      jnp.sqrt(1 / (jnp.pi * self.nu))
        return 0.5 + x * coefficient * _hyp2f1_fraction(a=0.5, b=(self.nu + 1) / 2, c=1.5, z=-(x ** 2) / self.nu)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative t-student probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative T student  probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the T student distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray = None):
            return

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the  T student distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """

        variance_ = jnp.where(self.nu > 2, self.nu / (self.nu - 2), jnp.where(self.nu <= 1, None, jnp.inf))
        skewness_ = jnp.where(self.nu > 3, 0, None)

        values = {'median': 0,
                  'mean': 0,
                  'mode': 0,
                  'variance': variance_,
                  'skewness': skewness_
                  }
        return values


class HalfStudentT(ContinuousDistributions):
    def __init__(self, nu: jnp.ndarray = None, activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Half Student-T distribution
        :param nu:
        :param activate_jit:
        :param random_seed:
        """
        super(HalfStudentT, self).__init__(nu=nu, activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.nu <= 0:
            raise Exception('Parameter nu (the degree of freedom) should be positive')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Half Student T distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """

        coefficient = 2 * (jnp.exp(scipy.special.gammaln((self.nu + 1) / 2)) /
                           jnp.exp(scipy.special.gammaln(self.nu / 2))) * \
                      jnp.sqrt(1 / (jnp.pi * self.nu))
        return jnp.where(x > 0, coefficient * (1 + (1 / self.nu) * x ** 2) ** (-(self.nu + 1) / 2), 0)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Half Student T distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of --------- probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of half student T probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative half student t probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        coefficient = (jnp.exp(scipy.special.gammaln((self.nu + 1) / 2)) /
                       jnp.exp(scipy.special.gammaln(self.nu / 2))) * \
                      jnp.sqrt(1 / (jnp.pi * self.nu))
        return jnp.where(x > 0, 1 + 2 * x * coefficient *
                         _hyp2f1_fraction(a=0.5, b=(self.nu + 1) / 2, c=1.5, z=-(x ** 2) / self.nu), 0)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative half student t probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative half student t probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the half student t distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray):
            return

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the  half student t distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """

        values = {'median': None,
                  'mean': None,
                  'variance': None,
                  'skewness': None,
                  'kurtosis': None,
                  'entropy': None
                  }
        return values


class Cauchy(ContinuousDistributions):

    def __init__(self, gamma: float = None, mu: float = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Cauchy distribution
        :param gamma:
        :param mu:
        :param activate_jit:
        :param random_seed:
        """
        super(Cauchy, self).__init__(gamma=gamma, mu=mu,
                                     activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.gamma <= 0:
            raise Exception('The value of the gamma should be positive!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Cauchy distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        denominator = (1 + ((x - self.mu) / self.gamma) ** 2)
        return (1 / (jnp.pi * self.gamma)) * (1 / denominator)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Cauchy distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Cauchy probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Cauchy probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Cauchy probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.arctan((x - self.mu) / self.gamma) * (1 / jnp.pi) + 0.5

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Cauchy probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Cauchy probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Cauchy distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return self.gamma * jnp.tan((y - 0.5) * jnp.pi) + self.mu

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Cauchy distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        first_quantile = self.gamma * jnp.tan((0.25 - 0.5) * jnp.pi) + self.mu
        third_quantile = self.gamma * jnp.tan((0.75 - 0.5) * jnp.pi) + self.mu
        values = {'median': self.mu,
                  'mode': self.mu,
                  'first_quantile': first_quantile,
                  'third_quantile': third_quantile,
                  'MAD': self.gamma,
                  'fisher_information': 1 / (2 * self.gamma ** 2),
                  'entropy': jnp.log(4 * jnp.pi * self.gamma)
                  }
        return values


class HalfCauchy(ContinuousDistributions):

    def __init__(self, beta: float = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        HalfCauchy distribution
        :param b:
        :param mu
        :param activate_jit:
        """
        super(HalfCauchy, self).__init__(beta=beta, activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.beta <= 0:
            raise Exception('The value of beta should be positive (Half Cauchy)!')
        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Half Cauchy distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        denominator = (1 + (x / self.beta) ** 2)
        return jnp.where(x >= 0, (2 / (self.beta * jnp.pi)) * (1 / denominator), 0)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Half Cauchy distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Half Cauchy probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Half Cauchy probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Half Cauchy probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.where(x >= 0, (2 / jnp.pi) * jnp.arctan(x / self.beta), 0)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Half Cauchy probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Half Cauchy probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Half Cauchy distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return self.beta * jnp.tan(0.5 * y * jnp.pi)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the  ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """

        values = {'median': None,
                  'mean': None,
                  'variance': None,
                  'skewness': None,
                  'kurtosis': None,
                  'entropy': None
                  }
        return values


class GammaDistribution(ContinuousDistributions):
    def __init__(self, alpha: float = None, beta: float = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Gamma Distribution
        :param alpha:
        :param beta:
        :param activate_jit:
        :param random_seed:
        """
        super(GammaDistribution, self).__init__(alpha=alpha, beta=beta,
                                                activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.alpha <= 0:
            raise Exception('Parameter alpha (for calculating the Gamma distribution) should be positive')

        if self.beta <= 0:
            raise Exception('Parameter beta (for calculating the Gamma distribution) should be positive')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Gamma distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)
        coefficient = ((self.beta ** self.alpha) / jnp.exp(scipy.special.gammaln(self.alpha)))
        return coefficient * (x ** (self.alpha - 1)) * (jnp.exp(-self.beta * x))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Gamma distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Gamma probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Gamma probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Gamma probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        return (1 / jnp.exp(scipy.special.gammaln(self.alpha))) * scipy.special.gammainc(a=self.alpha, x=self.beta * x)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Gamma probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Gamma probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Gamma distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return igammainv(a=self.alpha, p=y * jnp.exp(scipy.special.gammaln(self.alpha))) / self.beta

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Gamma distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        values = {'mode': jnp.where(self.alpha >= 1, (self.alpha - 1) / self.beta, 0),
                  'mean': self.alpha / self.beta,
                  'variance': self.alpha / self.beta ** 2,
                  'skewness': 2 / self.alpha ** 0.5,
                  'kurtosis': 6 / self.alpha,
                  }
        return values


class InverseGamma(ContinuousDistributions):

    def __init__(self, alpha: float = None, beta: float = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Inverse Gamma distribution
        :param alpha:
        :param beta:
        :param activate_jit:
        :param random_seed:
        """

        super(InverseGamma, self).__init__(alpha=alpha, beta=beta,
                                           activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.alpha <= 0:
            raise Exception('The value of alpha should be positive (InverseGamma)!')
        if self.beta <= 0:
            raise Exception('The value of beta should be positive (InverseGamma)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Inverse Gamma distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)
        coefficient = ((self.beta ** self.alpha) / jnp.exp(scipy.special.gammaln(self.alpha)))
        return coefficient * (x ** (-self.alpha - 1)) * (jnp.exp(-self.beta / x))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Inverse Gamma distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Inverse Gamma probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Inverse Gamma probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Inverse Gamma probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        return scipy.special.gammaincc(a=self.alpha, x=self.beta / x)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Inverse Gamma probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Inverse Gamma probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Inverse Gamma distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Inverse Gamma distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        variance_ = jnp.where(self.alpha > 2, (self.beta ** 2) / ((self.alpha - 2) * (self.alpha - 1) ** 2), jnp.nan)
        skewness_ = jnp.where(self.alpha > 3, (4 * jnp.sqrt(self.alpha - 2)) / (self.alpha - 3), jnp.nan)
        kurtosis_ = (6 * (5 * self.alpha - 11)) / ((self.alpha - 3) * (self.alpha - 4))
        kurtosis_ = jnp.where(self.alpha > 4, kurtosis_, jnp.nan)
        values = {'mean': jnp.where(self.alpha > 1, self.beta / (self.alpha - 1), jnp.nan),
                  'mode': self.beta / (self.alpha + 1),
                  'variance': variance_,
                  'skewness': skewness_,
                  'kurtosis': kurtosis_
                  }
        return values


class Weibull(ContinuousDistributions):

    def __init__(self, kappa: jnp.ndarray = None, lambd: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Weibull distribution
        :param kappa:
        :param lambd:
        :param activate_jit:
        :param random_seed:
        """

        super(Weibull, self).__init__(lambd=lambd, kappa=kappa,
                                      activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.kappa <= 0:
            raise Exception('The value of kappa should be positive (Weibull distribution)!')
        if self.lambd <= 0:
            raise Exception('The value of lambda should be positive (Weibull distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Weibull distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=0, a_max=jnp.inf)
        return jnp.where(x >= 0, (self.kappa / self.lambd) * \
                         ((x / self.lambd) ** (self.kappa - 1)) * \
                         jnp.exp(-(x / self.lambd) ** self.kappa), 0)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Weibull distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Weibull probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Weibull probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Weibull probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=0, a_max=jnp.inf)
        return jnp.where(x >= 0, 1 - jnp.exp(-(x / self.lambd) ** self.kappa), 0)
    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Weibull probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Weibull probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the --- distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return -self.lambd * (jnp.log(1 - y)) ** (1 / self.kappa)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the  Weibull distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        mean_ = jnp.exp(scipy.special.gammaln(1 + 1 / self.kappa)) * self.lambd
        mode_ = jnp.where(self.kappa > 1, self.lambd * ((self.kappa - 1) / self.kappa) ** (1 / self.kappa), 0)
        values = {'median': self.lambd * (jnp.log(2)) ** (1 / self.kappa),
                  'mode': mode_,
                  'mean': mean_,
                  'variance': None,
                  'skewness': None,
                  'kurtosis': None,
                  'entropy': None
                  }
        return values


class ChiSquared(ContinuousDistributions):

    def __init__(self, kappa: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        ChiSquared distribution
        :param b:
        :param mu
        :param activate_jit:
        """
        super(ChiSquared, self).__init__(kappa=kappa,
                                         activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.kappa <= 0:
            raise Exception('The degree of freedom should be positive integer (Chi Squared distribution)!')
        if not isinstance(self.kappa, int):
            raise Exception('The degree of freedom should be positive integer (Chi Squared distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the ChiSquared distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """

        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)
        return (1 / 2 ** (self.kappa / 2)) * (1 / jnp.exp(scipy.special.gammaln(self.kappa / 2))) * \
               (x ** (0.5 * (self.kappa - 2))) * jnp.exp(-x / 2)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of ChiSquared distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of ChiSquared probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of ChiSquared probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative ChiSquared probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        return scipy.special.gammainc(a=0.5 * self.kappa, x=x / 2)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative ChiSquared probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative ChiSquared probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the ChiSquared distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return 2 * igammainv(a=0.5 * self.kappa, p=y)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the ChiSquared distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """

        values = {'median': self.kappa * (1 - 2 / (9 * self.kappa)) ** 3,
                  'mean': self.kappa,
                  'mode': jnp.where(self.kappa - 2 >= 0, self.kappa - 2, 0),
                  'variance': 2 * self.kappa,
                  'skewness': (8 / self.kappa) ** 0.5,
                  'kurtosis': 12 / self.kappa
                  }
        return values


class LogNormal(ContinuousDistributions):

    def __init__(self, mu: float = None, sigma: float = None, variance: float = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        LogNormal distribution
        :param mu:
        :param sigma:
        :param variance:
        :param activate_jit:
        :param random_seed:
        """
        super(LogNormal, self).__init__(mu=mu, variance=variance, sigma=sigma,
                                        activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.sigma <= 0:
            raise Exception('The value of the standard deviation should be positive (Log Normal distribution)!')

        if self.variance <= 0:
            raise Exception('The value of variance should be positive (Log Normal distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the LogNormal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)
        return (1 / (x * self.sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(
            -0.5 * (1 / self.sigma ** 2) * (jnp.log(x) - self.mu) ** 2)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of LogNormal distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of LogNormal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of LogNormal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative LogNormal probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)
        return 0.5 * (1 + scipy.special.erf((jnp.log(x) - self.mu) / (self.sigma * jnp.sqrt(2))))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative LogNormal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative LogNormal probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the LogNormal distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return jnp.exp(scipy.special.erfinv(2 * y - 1) * jnp.sqrt(2) * self.sigma + self.mu)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the  ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        first_quantile_ = jnp.exp(scipy.special.erfinv(-0.5) * jnp.sqrt(2) * self.sigma + self.mu)
        third_quantile_ = jnp.exp(scipy.special.erfinv(0.5) * jnp.sqrt(2) * self.sigma + self.mu)
        median_ = jnp.exp(self.mu)
        mode_ = jnp.exp(self.mu - self.sigma ** 2)
        variance_ = jnp.exp(2 * self.mu + self.sigma ** 2) * (jnp.exp(self.sigma ** 2) - 1)
        mean_ = jnp.exp(self.mu + 0.5 * self.sigma ** 2)
        skewness_ = (jnp.exp(self.sigma ** 2) + 2) * jnp.sqrt(jnp.exp(self.sigma ** 2) - 1)

        values = {'median': median_,
                  'mean': mean_,
                  'first_quantile': first_quantile_,
                  'third_quantile': third_quantile_,
                  'mode': mode_,
                  'variance': variance_,
                  'skewness': skewness_
                  }
        return values


class Wald(ContinuousDistributions):

    def __init__(self, lambd: jnp.ndarray = None, mu: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Wald distribution
        :param lambd:
        :param mu:
        :param b:
        :param activate_jit:
        :param random_seed:
        """
        super(Wald, self).__init__(mu=mu, lambd=lambd,
                                   activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.lambd <= 0:
            raise Exception('The value of lambda should be positive (Wald distribution)!')

        if self.mu <= 0:
            raise Exception('The value of mean should be positive (Log Normal distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Wald distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)
        return jnp.sqrt(self.lambd / (2 * jnp.pi * x ** 3)) * jnp.exp(
            (-self.lambd / (2 * x * self.mu ** 2)) * (x - self.mu) ** 2)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Wald distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Wald probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Wald probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Wald probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        def normal_fcn_cdf(z: jnp.ndarray) -> jnp.ndarray:
            return 0.5 * (1 + lax.erf(z/jnp.sqrt(2)))

        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=jnp.inf)
        return normal_fcn_cdf(jnp.sqrt(self.lambd / x) * (x / self.mu - 1)) + jnp.exp((2 * self.lambd) / self.mu) * \
               normal_fcn_cdf(-jnp.sqrt(self.lambd / x) * (x / self.mu + 1))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Wald probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Wald probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Wald distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the  ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        mean_ = self.mu
        variance_ = (self.mu ** 3) / self.lambd
        mode_ = self.mu * (jnp.sqrt(1 + (9 * self.mu ** 2) / (4 * self.lambd ** 2)) - 1.5 * (self.mu / self.lambd))
        skewness_ = 3 * jnp.sqrt(self.mu / self.lambd)
        kurtosis_ = 15 * (self.mu / self.lambd)

        values = {'mean': mean_,
                  'mode': mode_,
                  'variance': variance_,
                  'skewness': skewness_,
                  'kurtosis': kurtosis_
                  }
        return values


class Pareto(ContinuousDistributions):
    def __init__(self, xm: jnp.ndarray = None, alpha: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Pareto
        :param xm:
        :param alpha:
        :param activate_jit:
        :param random_seed:
        """
        super(Pareto, self).__init__(xm=xm, alpha=alpha,
                                     activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.xm <= 0:
            raise Exception('The value of xm should be positive (Pareto distribution)!')

        if self.alpha <= 0:
            raise Exception('The value of alpha should be positive (Pareto distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Pareto distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        return jnp.where(x >= self.xm, (self.alpha / (x ** (self.alpha + 1))) * (self.xm ** self.alpha), 0)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Pareto distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Pareto probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Pareto probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Pareto probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        return jnp.where(x >= self.xm, 1 - (self.xm / x) ** self.alpha, 0)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Pareto probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Pareto probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Pareto distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return self.xm / (1 - y) ** (1 / self.alpha)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Pareto distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        mean_ = jnp.where(self.alpha <= 1, jnp.inf, (self.xm * self.alpha) / (self.alpha - 1))
        median_ = self.xm * 2 ** (1 / self.alpha)
        mode_ = self.xm
        variance_ = jnp.where(self.alpha > 2, (self.alpha * self.xm ** 2) / ((self.alpha - 2) * (self.alpha - 1) ** 2),
                              jnp.inf)
        entropy_ = jnp.log((self.xm / self.alpha) * jnp.exp(1 + 1 / self.alpha))
        values = {'median': median_,
                  'mode': mode_,
                  'mean': mean_,
                  'variance': variance_,
                  'entropy': entropy_
                  }
        return values


class ExModifiedGaussian(ContinuousDistributions):

    def __init__(self, lambd: jnp.ndarray = None, mu: jnp.ndarray = None, sigma: jnp.ndarray = None,
                 variance: jnp.ndarray = None, activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        ExModifiedGaussian distribution
        :param lambd:
        :param mu:
        :param sigma:
        :param variance:
        :param activate_jit:
        :param random_seed:
        """
        super(ExModifiedGaussian, self).__init__(mu=mu, sigma=sigma, lambd=lambd, variance=variance,
                                                 activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.lambd <= 0:
            raise Exception('The value of lambda should be positive (Exponentially modified Gaussian distribution)!')

        if self.sigma <= 0:
            raise Exception('The value of sigma should be positive (Exponentially modified Gaussian distribution)!')

        if self.variance <= 0:
            raise Exception('The value of variance should be positive (Exponentially modified Gaussian distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the ExModifiedGaussian distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """

        arg_exp = 2 * self.mu + self.lambd * (self.sigma ** 2) - 2 * x
        arg_erfc = (self.mu + self.lambd * (self.sigma ** 2) - x) / (self.sigma * jnp.sqrt(2))
        return 0.5 * self.lambd * jnp.exp(0.5 * self.lambd * arg_exp) * scipy.special.erfc(arg_erfc)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of ExModifiedGaussian distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of ExModifiedGaussian probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of ExModifiedGaussian probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative ExModifiedGaussian probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        arg_exp = 2 * self.mu + self.lambd * (self.sigma ** 2) - 2 * x
        arg_erfc = (self.mu + self.lambd * (self.sigma ** 2) - x) / (self.sigma * jnp.sqrt(2))
        z = (x - self.mu) / (self.sigma * jnp.sqrt(2))
        return 0.5 * (1 + scipy.special.erf(z)) - 0.5 * jnp.exp(0.5 * self.lambd * arg_exp) * scipy.special.erfc(
            arg_erfc)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative ExModifiedGaussian probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative ExModifiedGaussian probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the ExModifiedGaussian distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the  ExModifiedGaussian distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        mean_ = self.mu + 1 / self.lambd
        variance_ = self.sigma ** 2 + 1 / self.lambd ** 2
        skewness_ = (2 / ((self.sigma ** 3) * (self.lambd ** 3))) * (
                1 + 1 / (self.sigma ** 2 * self.lambd ** 2)) ** -1.5
        values = {'mean': mean_,
                  'variance': variance_,
                  'skewness': skewness_
                  }
        return values


class Triangular(ContinuousDistributions):

    def __init__(self, a: float = None, b: float = None, c: float = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Triangular distribution
        :param a: left hand side location
        :param b: upper limit of the triangle
        :param c: the mode of the triangle
        :param activate_jit:
        """
        super(Triangular, self).__init__(a=a, b=b, c=c, activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution
        print(self.a)
        print(self.b)
        print(self.c)
        if self.a >= self.b:
            raise Exception('The value of a should be lower than b (Triangular distribution)!')

        if self.a > self.c:
            raise Exception('The value of a should be lower/equal than c (Triangular distribution)!')

        if self.c > self.b:
            raise Exception('The value of b should be higher/equal than c (Triangular distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Triangular distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """

        return jnp.where((self.a <= x) & (x <= self.c), 2 * ((x - self.a) / ((self.b - self.a) * (self.c - self.a))),
                         jnp.where((self.c < x) & (x <= self.b),
                                   2 * ((self.b - x) / ((self.b - self.a) * (self.b - self.c))),
                                   0))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Triangular distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Triangular probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Triangular probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Triangular probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.where((self.a <= x) & (x <= self.c), ((x - self.a) ** 2) / ((self.b - self.a) * (self.c - self.a)),
                         jnp.where((self.c < x) & (x <= self.b),
                                   1 - ((self.b - x) ** 2) / ((self.b - self.a) * (self.b - self.c)),
                                   jnp.where(x > self.b, 1.0, 0)))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Triangular probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Triangular probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Triangular distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(y <= (self.c - self.a) / (self.b - self.a),
                             self.a + jnp.sqrt(y * (self.b - self.a) * (self.c - self.a)),
                             self.b - jnp.sqrt((1 - y) * (self.b - self.a) * (self.b - self.c)))

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the  ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        mean_ = (self.a + self.b + self.c) / 3
        median_ = jnp.where(2 * self.c >= self.a + self.b,
                            self.a + jnp.sqrt(0.5 * (self.b - self.a) * (self.c - self.a)),
                            self.b - jnp.sqrt(0.5 * (self.b - self.a) * (self.b - self.c)))
        mode_ = self.c
        kurtosis_ = -0.6
        entropy_ = .5 + jnp.log(0.5 * (self.b - self.a))
        variance_ = (1 / 18) * (self.a ** 2 + self.b ** 2 + self.c ** 2
                                - self.a * self.b - self.a * self.c - self.b * self.c)

        values = {'median': median_,
                  'mean': mean_,
                  'mode': mode_,
                  'variance': variance_,
                  'kurtosis': kurtosis_,
                  'entropy': entropy_
                  }
        return values


class Gumbel(ContinuousDistributions):

    def __init__(self, mu: float = None, beta: float = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        Gumbel distribution
        :param mu:
        :param beta:
        :param activate_jit:
        :param random_seed:
        """
        super(Gumbel, self).__init__(mu=mu, beta=beta,
                                     activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.beta <= 0:
            raise Exception('The value of beta should be positive (Gumbel function)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Gumbel distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        z = (x - self.mu) / self.beta
        return (1 / self.beta) * jnp.exp(-(z + jnp.exp(-z)))

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Gumbel distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Gumbel probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Gumbel probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Gumbel probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        return jnp.exp(-jnp.exp(-(x - self.mu) / self.beta))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Gumbel probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Gumbel probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Gumbel distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return self.mu - self.beta * jnp.log(-jnp.log(y))

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Gumbel distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        mean_ = self.mu + 0.5772 * self.beta
        median_ = self.mu - self.beta * jnp.log(jnp.log(2))
        mode_ = self.mu
        variance_ = (1 / 6) * (self.beta ** 2 * jnp.pi ** 2)
        skewness_ = 1.14
        kurtosis_ = 12 / 5
        entropy_ = jnp.log(self.beta) + 0.5772 + 1
        values = {'median': median_,
                  'mode': mode_,
                  'mean': mean_,
                  'variance': variance_,
                  'skewness': skewness_,
                  'kurtosis': kurtosis_,
                  'entropy': entropy_
                  }
        return values


class Logistic(ContinuousDistributions):

    def __init__(self, mu: float = None, sigma: float = None, variance: float = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:

        """
        Logistic distribution
        :param mu:
        :param sigma:
        :param variance:
        :param activate_jit:
        :param random_seed:
        """
        super(Logistic, self).__init__(mu=mu, variance=variance, sigma=sigma,
                                       activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.sigma <= 0:
            raise Exception('The value of sigma should be positive (Logistic distribution)!')

        if self.variance <= 0:
            raise Exception('The value of variance should be positive (Logistic distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Logistic distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        z = (x - self.mu) / self.sigma
        return jnp.exp(-z) / (self.sigma * (1 + jnp.exp(-z)) ** 2)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Logistic distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Logistic probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of Logistic probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Logistic probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        return 1 / (1 + jnp.exp(-(x - self.mu) / self.sigma))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative Logistic probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative Logistic probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the Logistic distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return self.mu - self.sigma * jnp.log(1 / y - 1)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the Logistic distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        first_quantile_ = self.mu - self.sigma * jnp.log(1 / 0.25 - 1)
        third_quantile_ = self.mu - self.sigma * jnp.log(1 / 0.75 - 1)
        mean_ = self.mu
        median_ = self.mu
        mode_ = self.mu
        skewness_ = 0
        kurtosis_ = 1.2
        variance_ = (jnp.pi ** 2 * self.sigma ** 2) / 3
        entropy_ = jnp.log(self.sigma) + 2

        values = {'median': median_,
                  'mean': mean_,
                  'mode': mode_,
                  'first_quantile': first_quantile_,
                  'third_quantile': third_quantile_,
                  'variance': variance_,
                  'skewness': skewness_,
                  'kurtosis': kurtosis_,
                  'entropy': entropy_
                  }
        return values


class LogitNormal(ContinuousDistributions):

    def __init__(self, mu: float = None, sigma: float = None, variance: float = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        LogitNormal distribution
        :param mu:
        :param sigma:
        :param variance:
        :param activate_jit:
        :param random_seed:
        """
        super(LogitNormal, self).__init__(mu=mu, variance=variance, sigma=sigma,
                                          activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.sigma <= 0:
            raise Exception('The value of sigma should be positive (Logistic distribution)!')

        if self.variance <= 0:
            raise Exception('The value of variance should be positive (Logistic distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the LogitNormal distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1 - jnp.finfo(float).eps)
        return (1 / (self.sigma * jnp.sqrt(2 * jnp.pi))) * (1 / (x * (1 - x))) * jnp.exp(
            (-0.5 / self.sigma ** 2) * (jnp.log(x / (1 - x)) - self.mu) ** 2)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of LogitNormal distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of LogitNormal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of LogitNormal probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative  LogitNormal probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        x = jnp.clip(a=x, a_min=jnp.finfo(float).eps, a_max=1 - jnp.finfo(float).eps)
        return 0.5 * (1 + scipy.special.erf((jnp.log(x / (1 - x)) - self.mu) / (self.sigma * jnp.sqrt(2))))

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative LogitNormal probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative LogitNormal probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:

        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the LogitNormal distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            inversion_term = scipy.special.erfinv(2 * y - 1) * self.sigma * jnp.sqrt(2) + self.mu
            return jnp.exp(inversion_term) / (1 + jnp.exp(inversion_term))

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the LogitNormal distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """

        values = {'median': None,
                  'mean': None,
                  'variance': None,
                  'skewness': None,
                  'kurtosis': None,
                  'entropy': None
                  }
        return values


class PDF(ContinuousDistributions):

    def __init__(self, kappa: jnp.ndarray = None, mu: jnp.ndarray = None, b: jnp.ndarray = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:
        """
        ------- distribution
        :param b:
        :param mu
        :param activate_jit:
        """
        super(PDF, self).__init__(mu=mu, b=b, kappa=kappa,
                                  activate_jit=activate_jit, random_seed=random_seed)
        # check for the consistency of the input of the probability distribution

        if self.b <= 0:
            raise Exception('Parameter b (for calculating the Laplace distribution) should be positive')

        if self.kappa <= 0:
            raise Exception('The values of Symmetric parameter should be positive(Asymmetric Laplace distribution)!')
        if self.b <= 0:
            raise Exception(
                'The rate of the change of the exponential term should be positive(Asymmetric Laplace distribution)!')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the -------- distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability of the occurrence of the given variable Cx1
        """
        return jnp.where(x >= self.mu, 1, 1)

    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of -------------- distribution
        :param x: The input variable (Cx1)
        :return: The derivatives of the probability of the occurrence of the given variable Cx1
        """
        return (self.pdf_(x))[0]

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of --------- probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """

        return -jnp.log(self.pdf_(x))

    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of ------------ probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return self.log_pdf_(x)[0]

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative --------- probability distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """

        return jnp.where(x >= self.mu, 1, 1)

    def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The derivatives of the cumulative ----- probability distribution
        :param x: The input variable (Cx1)
        :return: The derivatives cumulative probability of the occurrence of the given variable Cx1
        """
        return (self.cdf_(x))[0]

    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log values of the cumulative ----- probability distribution
        :param x: The input variable (Cx1)
        :return: The log values of cumulative probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.cdf_(x))

    def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return (self.log_cdf_(x))[0]

    def sample_(self, size: int = 1) -> jnp.ndarray:
        """
        Sampling form the --- distribution
        :param size:
        :return:
        """
        y = random.uniform(key=self.key, minval=0.0, maxval=1.0, shape=(size, 1))

        def inversion_of_cdf_(y: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(y <= threshold, 1, 1)

        return vmap(inversion_of_cdf_, in_axes=0, out_axes=0)(y)

    @property
    def statistics(self):
        """
        Statistics calculated for the  ---- distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """

        values = {'median': median_,
                  'mean': mean_,
                  'variance': variance_,
                  'skewness': skewness_,
                  'kurtosis': kurtosis_,
                  'entropy': entropy_
                  }
        return values


#
x = random.uniform(key=random.PRNGKey(7), minval=0.01, maxval=20, shape=(1000, 1), dtype=jnp.float64)
# KK = Normal(sigma=4, mu=7)
# KK = Uniform(lower=-2,upper=3)
# KK = TruncatedNormal(lower=-3,upper=4,variance=3,mu=1)
# KK = HalfNormal(sigma=4)
# KK = SkewedNormal(mu=0,alpha=2,sigma=3)
# KK = BetaPdf(alpha=2,beta=3)
# KK = Kumaraswamy(alpha=3,beta=5)
# KK = Exponential(lambd=3)
# KK = Laplace(mu=3,b=2)
# KK = AsymmetricLaplace(kappa=2.0,mu=3,b=4)
# KK = StudentT(nu=5)
# KK = HalfStudentT(nu = 2)
# KK = Cauchy(gamma=2,mu=4)
# KK = HalfCauchy(beta=2)
# KK = GammaDistribution(alpha=2,beta=4)
# KK = InverseGamma(alpha=2.5,beta=3)
# KK = Weibull(kappa=2,lambd=1)
# KK = ChiSquared(kappa=2)
# KK  = LogNormal(mu=2,sigma=1)
# KK = Wald(lambd=2,mu=1)
# KK = Pareto(xm=4,alpha=2.1)
# KK = ExModifiedGaussian(lambd=1.1, mu=3,sigma=2.6)
# KK=Triangular(a=1,b=3,c=2)
# KK  = Gumbel(mu=3,beta = 2)
# KK = Logistic(mu=2,sigma=4)
# KK = LogitNormal(mu=0,sigma=2)
# E1 = KK.pdf(x)
# E6 = KK.diff_pdf(x)
# E2 = KK.log_pdf(x)
#
# E3 = KK.diff_log_pdf(x)
# E4 = KK.cdf(x)
# E5 = KK.log_cdf(x)
# E8 = KK.diff_cdf(x)
# E9 = KK.diff_log_cdf(x)
#
# samples = KK.sample(size=10000)
#
# if any(jnp.isnan(E1)):
#     print('there is NAN in pdf')
#
# if any(jnp.isnan(E6)):
#     print('there is NAN in diff pdf')
#
# if any(jnp.isnan(E2)):
#     print('there is NAN in log pdf')
#
# if any(jnp.isnan(E3)):
#     print('there is NAN in diff log pdf')
#
# if any(jnp.isnan(E4)):
#     print('there is NAN in cdf')
#
# if any(jnp.isnan(E5)):
#     print('there is NAN in log CDF')
#
# if any(jnp.isnan(E8)):
#     print('there is NAN in diff CDF')
#
# if any(jnp.isnan(E9)):
#     print('there is NAN in diff log CDF')
#
# print(KK.statistics)
#
# fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
#
# ax1.plot(x, E1, '*')
# ax1.set_title('PDF values')
# ax5.plot(x, E6, '*')
# ax5.set_title('derivatives of PDF ')
#
# ax2.plot(x, E2, '*')
# ax2.set_title('LOG of PDF values')
# ax6.plot(x, E3, '*')
# ax6.set_title('derivatives of LOG of PDF ')
#
# ax3.plot(x, E4, '*')
# ax3.set_title('CDF ')
# ax7.plot(x, E8, '*')
# ax7.set_title('derivatives of CDF ')
#
# ax4.plot(x, E5, '*')
# ax4.set_title('log CDF ')
# ax8.plot(x, E9, '*')
# ax8.set_title('derivatives of log CDF ')

# plt.figure(dpi=100)
# plt.hist(samples, bins=20)
# plt.title('samples')
# if any(jnp.isnan(samples)):
#     print('there is NAN in samples')
#
# show()
####################################################################################
# x = random.uniform(key=random.PRNGKey(7), minval=-2, maxval=2, shape=(1000, 1))
# activate_jit = False
#
# KK = Laplace(mu=0, b=1, activate_jit=activate_jit)
# E1 = KK.pdf(x)
# plt.figure(dpi=150)
# plt.plot(x, E1, '*')
# plt.title('PDF')
# plt.show()
#
# E6 = KK.diff_pdf(x)
# plt.figure(dpi=150)
# plt.plot(x, E6, '*')
# plt.title('Diff PDF')
# plt.show()
#
# E2 = KK.log_pdf(x)
# plt.figure(dpi=150)
# plt.plot(x, E2, '*')
# plt.title('LOG PDF')
# plt.show()
#
# E3 = KK.diff_log_pdf(x)
# plt.figure(dpi=150)
# plt.plot(x, E3, '*')
# plt.title('DIFF LOG PDF')
# plt.show()
#
# E4 = KK.cdf(x)
# plt.figure(dpi=150)
# plt.plot(x, E4, '*')
# plt.title('CDF')
# plt.show()
#
# E5 = KK.log_cdf(x)
# plt.figure(dpi=150)
# plt.plot(x, E5, '*')
# plt.title('LOG CDF')
# plt.show()
#
# E8 = KK.diff_cdf(x)
# plt.figure(dpi=150)
# plt.plot(x, E8, '*')
# plt.title('DIFF CDF')
# plt.show()
#
# E9 = KK.diff_log_cdf(x)
# plt.figure(dpi=150)
# plt.plot(x, E9, '*')
# plt.title('DIFF LOG CDF')
# plt.show()
#
# E7 = KK.sample(size=20000)
# plt.figure(dpi=150)
# plt.hist(E7, 30)
# plt.title('samples')
# plt.show()
