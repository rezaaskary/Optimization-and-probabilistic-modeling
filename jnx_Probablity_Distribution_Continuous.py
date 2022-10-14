import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap, jit, grad
import jax
from jax.lax import switch

RNG = jax.random.PRNGKey(60616)


class ContinuousDistributions:
    def __init__(self,
                 lower: jnp.ndarray = None,
                 upper: jnp.ndarray = None,
                 variant_chains: bool = False,
                 activate_jit: bool = False,
                 nchains: int = 1) -> None:

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



class Uniform(ContinuousDistributions):
    def __init__(self, lower: float = None, upper: float = None) -> None:
        """
        Continuous uniform distribution
        :param lower: The lower limit of uniform distribution
        :param upper: The upper limit of uniform distribution
        """
        super(Uniform, self).__init__(lower=lower, upper=upper)
        # check for the consistency of the input of the probability distribution
        if not isinstance(self.lower, type(self.upper)):
            raise Exception('The input parameters are not consistent (Uniform Distribution)!')

        if jnp.any(self.lower >= self.upper):
            raise Exception('The lower limit of the uniform distribution is greater than the upper limit!')

        if not self.variant_chains:
            # when the number of parallel evaluation is fixed. Useful for MCMC

            if self.activate_jit:
            else:
                self.pdf = vmap(self.pdf_, in_axes=[0], out_axes=0)
                self.diff_pdf = vmap(self.diff_pdf_, in_axes=[0], out_axes=0)
                self.log_pdf = vmap(self.log_pdf_, in_axes=[0], out_axes=0)
                self.diff_log_pdf = vmap(self.diff_log_pdf_, in_axes=[0], out_axes=0)
                self.cdf = vmap(self.cdf_, in_axes=[0], out_axes=0)
                self.log_cdf = vmap(self.log_cdf_, in_axes=[0], out_axes=0)
        else:



    @property
    def statistics(self):
        """
        Statistics calculated for the Uniform distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None


    def pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Uniform distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        return jnp.where((x > self.lower) & (x < self.upper), 1 / (self.upper - self.lower), 0)
    def diff_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.pdf_(x)[0,0]
    def log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where((x > self.lower) & (x < self.upper), -jnp.log((self.upper - self.lower)), -jnp.inf)
    def diff_log_pdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.pdf_(x)[0,0]
    def cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x<self.lower, jnp.where(x>self.upper, 1,(x-self.lower) / (self.upper - self.lower)))
    def log_cdf_(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(jnp.where(x < self.lower, jnp.where(x > self.upper, 1, (x - self.lower) / (self.upper - self.lower))))

# x = (jnx.random.uniform(low = -200, high=200, size=10000)).reshape((-1,1))
# ts = Uniform(a=4,b=7)
x = jax.random.uniform(key=RNG, minval=-20, maxval=20, shape=(10, 1))

ts = Uniform(a=4, b=10)

# R = (ts.pdf, in_axes=0, out_axes=0)
mm=ts.pdf(x)
R2 = DD(x)
plt.plot(x, R, '*')
R
