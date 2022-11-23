from tensorflow_probability.substrates.jax import distributions
from jax import vmap, jit, grad, random, lax, scipy
import jax.numpy as jnp


class ContinuousDistributions:
    def __init__(self,
                 lower: jnp.ndarray = None,
                 upper: jnp.ndarray = None,
                 variant_chains: bool = False,
                 activate_jit: bool = False,
                 n_chains: int = 1,
                 random_seed: int = 1) -> None:

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

        if isinstance(random_seed, int):
            self.key = random.PRNGKey(random_seed)
        else:
            raise Exception('The random seed is not specified correctly!')
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






class ContinuousMethods:

    def probablity_distribution_(self, x: jnp.ndarray = None) -> jnp.ndarray:
        return self.distance_function.pdf(value=x, name='prob')

    def cumulative_distribution_(self) -> jnp.ndarray:
        return self.distance_function.cdf(value=x, name='cdf')

    def log_probablity_distribution_(self) -> jnp.ndarray:
        return self.distance_function.log_prob(value=x, name='log prob')

    def log_cumulative_distribution_(self) -> jnp.ndarray:
        return self.distance_function.log_cdf(value=x, name='log cdf')

    def diff_probablity_distribution_(self, x: jnp.ndarray = None) -> jnp.ndarray:
        return (self.distance_function.pdf(value=x, name='diff prob'))[0]

    def diff_cumulative_distribution_(self) -> jnp.ndarray:
        return (self.distance_function.cdf(value=x, name='diff cdf'))[0]

    def diff_log_probablity_distribution_(self) -> jnp.ndarray:
        return (self.distance_function.log_prob(value=x, name='diff log  prob'))[0]

    def diff_log_cumulative_distribution_(self) -> jnp.ndarray:
        return (self.distance_function.log_cdf(value=x, name='diff log  cdf'))[0]


    # def experimental_fit_(self):
    #     return
    #
    # def sample_from_(self, n_samples: int = None):
    #     return
    #
    # def sample_from_distribution_(self):
    #     return


class Uniform(ContinuousDistributions, ContinuousMethods):
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

        if not isinstance(self.lower, type(self.upper)):
            raise Exception('The input parameters are not consistent (Uniform Distribution)!')
        if jnp.any(self.lower >= self.upper):
            raise Exception('The lower limit of the uniform distribution is greater than the upper limit!')

        self.distance_function = distributions.Uniform(low=self.lower, high=self.upper, name='Uniform')
        ContinuousMethods.probablity_distribution_(self)
        ContinuousMethods.log_probablity_distribution_(self)

        ContinuousDistributions.parallelization(self)
