from tensorflow_probability.substrates.jax import distributions
from jax import vmap, jit, grad, random, lax, scipy, jacfwd
import jax.numpy as jnp


class DiscreteDistributions:
    def __init__(self,
                 total_count: int = None,
                 probs: int = None,
                 random_seed: int = 1,
                 activate_jit: bool = False,
                 variant_chains: bool = False) -> None:

        if isinstance(random_seed, int):
            self.key = random.PRNGKey(random_seed)
        else:
            raise Exception('The random seed is not specified correctly!')

        if isinstance(total_count, int):
            self.total_count = total_count
        elif total_count is None:
            self.total_count = None
        else:
            raise Exception('The value of total_count is not specified correctly!')

        if isinstance(probs, (int, float)):
            self.probs = probs
        elif probs is None:
            self.probs = None
        else:
            raise Exception('The value of probs is not specified correctly!')

        if isinstance(activate_jit, bool):
            self.activate_jit = activate_jit
        else:
            raise Exception('Please specify the activation of the just-in-time evaluation!')

        if isinstance(variant_chains, bool):
            self.variant_chains = variant_chains
        else:
            raise Exception('Please specify whether the number of chains are fixed or variant during simulation !')

    def parallelization(self):
        if not self.variant_chains:
            # when the number of parallel evaluation is fixed. Useful for MCMC
            if self.activate_jit:
                self.pdf = jit(vmap(self.pdf_, in_axes=[1], out_axes=1))
                # self.diff_pdf = jit(vmap(grad(self.diff_pdf_), in_axes=[0], out_axes=0))
                self.log_pdf = jit(vmap(self.log_pdf_, in_axes=[1], out_axes=1))
                # self.diff_log_pdf = jit(vmap(grad(self.diff_log_pdf_), in_axes=[0], out_axes=0))
                # self.cdf = jit(vmap(self.cdf_, in_axes=[1], out_axes=1))
                # self.log_cdf = jit(vmap(self.log_cdf_, in_axes=[1], out_axes=1))
                # self.diff_cdf = jit(vmap(grad(self.diff_cdf_), in_axes=[0], out_axes=0))
                # self.diff_log_cdf = jit(vmap(grad(self.diff_log_cdf_), in_axes=[0], out_axes=0))
                # self.sample = self.sample_
            else:
                # self.sample = self.sample_
                self.pdf = vmap(self.pdf_, in_axes=[1], out_axes=1)
                # self.diff_pdf = vmap(grad(self.diff_pdf_), in_axes=[0], out_axes=0)
                self.log_pdf = vmap(self.log_pdf_, in_axes=[1], out_axes=1)
                # self.diff_log_pdf = vmap(grad(self.diff_log_pdf_), in_axes=[0], out_axes=0)
                # self.cdf = vmap(self.cdf_, in_axes=[1], out_axes=1)
                # self.log_cdf = vmap(self.log_cdf_, in_axes=[1], out_axes=1)
                # self.diff_cdf = vmap(grad(self.diff_cdf_), in_axes=[0], out_axes=0)
                # self.diff_log_cdf = vmap(grad(self.diff_log_cdf_), in_axes=[0], out_axes=0)

        else:
            pass

    def visualize(self, lower_lim: float = -10, upper_lim: float = -10):
        """
        Visualizing the probability distribution
        :param lower_lim: the lower limit used in plotting the probability distribution
        :param upper_lim: the upper limit used in plotting the probability distribution
        :return: a line plot from matplotlib library
        """
        x_m = np.linspace(start=lower_lim, stop=upper_lim, num=1000)
        y_m = list()
        for i in range(len(x_m)):
            y_m.append(self.pdf(x_m[i]))
        plot(list(x_m.ravel()), y_m)
        grid(which='both')


class DiscreteMethods:

    def probablity_distribution(self):
        return
    def log_probablity_distribution(self):
        return
    def cumulative_distribution(self):
        return
    def cumulative_distribution(self):
        return
    def log_probablity_distribution(self):
        return
    def experimental_fit(self):
        return
    def sample_from(self):
        return

    @property
    def statistics(self):
        return








class Binomial(DiscreteDistributions):
    def __init__(self, total_count: int = None, probs: int = None,
                 activate_jit: bool = False, random_seed: int = 1) -> None:

        super(Binomial, self).__init__(total_count=total_count, probs=probs,
                                       activate_jit=activate_jit, random_seed=random_seed)

        # check for the consistency of the input of the probability distribution
        if self.total_count < 0:
            raise Exception('The negative value for input parameters total_count is not accepted!')

        if self.probs > 1 or self.probs < 0:
            raise Exception('The of input parameters probs cannot be outside range [0,1] !')

        DiscreteDistributions.parallelization(self)
        self.distance_function = distributions.Binomial(total_count=self.total_count, probs=self.probs)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Binomial distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        return self.distance_function.prob(x)

    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The log of Binomial probability distribution
        :param x: The input variable (Cx1)
        :return: The log of the probability of the occurrence of the given variable Cx1
        """
        return jnp.log(self.distance_function.prob(value=x))

    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The cumulative Binomial distribution
        :param x: The input variable (Cx1)
        :return: The cumulative probability of the occurrence of the given variable Cx1
        """
        return self.distance_function.cdf(value=x)

    # def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
    #     return (self.pdf_(x))[0]
    #
    # def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
    #     return jnp.where((x > self.lower) & (x < self.upper), -jnp.log((self.upper - self.lower)), -jnp.inf)
    #
    # def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
    #     return self.log_pdf_(x)[0]
    #
    # def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
    #     return jnp.where(x < self.lower, 0, jnp.where(x < self.upper, (x - self.lower) / (self.upper - self.lower), 1))
    #
    # def diff_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
    #     return (self.cdf_(x))[0]
    #
    # def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
    #     return jnp.log(
    #         jnp.where(x < self.lower, 0, jnp.where(x < self.upper, (x - self.lower) / (self.upper - self.lower), 1)))
    #
    # def diff_log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
    #     return (self.log_cdf_(x))[0]
    #
    # def sample_(self, size: int = 1) -> jnp.ndarray:
    #     return random.uniform(key=self.key, minval=self.lower, maxval=self.upper, shape=(size, 1))
    #
    # @property
    # def statistics(self):
    #     """
    #     Statistics calculated for the Uniform distribution function given distribution parameters
    #     :return: A dictionary of calculated metrics
    #     """
    #     values = {'mean': 0.5 * (self.lower + self.upper),
    #               'median': 0.5 * (self.lower + self.upper),
    #               'variance': (1 / 12) * (self.lower - self.upper) ** 2,
    #               'MAD': (1 / 4) * (self.lower + self.upper),
    #               'skewness': 0,
    #               'kurtosis': -6 / 5,
    #               'Entropy': jnp.log(self.upper - self.lower)
    #               }
    #     return values


# x = random.uniform(key=random.PRNGKey(7), minval=0.01, maxval=20, shape=(1000, 1), dtype=jnp.float64)
x = random.randint(key=random.PRNGKey(7), shape=(1000, 1), minval=0, maxval=60, dtype=int)

KK = Binomial(total_count=20, probs=0.5)
FF = KK.pdf(x)
plt.plot(x, FF, '*')
plt
