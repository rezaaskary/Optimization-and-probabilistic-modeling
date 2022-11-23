import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap, jit, grad, random, lax, scipy
from jax.scipy.special import factorial
from matplotlib.pyplot import plot, grid, figure, show


class DiscreteDistributions:
    def __init__(self,
                 n: int = None,
                 p: int = None) -> None:

        if isinstance(n, int):
            self.n = n
        elif n is None:
            self.n = None
        else:
            raise Exception('The lower bound is not specified correctly!')

        if isinstance(p, int):
            self.p = p
        elif p is None:
            self.p = None
        else:
            raise Exception('The lower bound is not specified correctly!')

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
        show()


class Binomial(DiscreteDistributions):
    def __init__(self, n: int = None, p: int = None) -> None:
        super(Binomial, self).__init__(n=n, p=p)

        # check for the consistency of the input of the probability distribution

        if self.n < 0:
            raise Exception('The negative value for input parameters n is not accepted!')

        if not self.p in [0, 1]:
            raise Exception('The of input parameters p can be either 0 or 1 !')

        ContinuousDistributions.parallelization(self)

    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Uniform distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        return factorial(self.n) / (factorial(x) * factorial(self.n - x)) * (self.p ** x) * (1 - self.p) ** (self.n - x)

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
