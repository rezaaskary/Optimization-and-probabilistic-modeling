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
                 true: bool = False,
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


class Uniform(ContinuousDistributions):
    def __init__(self, lower: float = None, upper: float = None) -> None:
        """
        Continuous uniform distribution
        :param a: The lower limit of uniform distribution
        :param b: The upper limit of uniform distribution
        """
        super(Uniform, self).__init__(lower=lower, upper=upper)
        # check for the consistency of the input of the probability distribution
        if not isinstance(self.lower, type(self.upper)):
            raise Exception('The input parameters are not consistent (Uniform Distribution)!')

        if jnp.any(self.lower >= self.upper):
            raise Exception('The lower limit of the uniform distribution is greater than the upper limit!')





    @property
    def statistics(self):
        """
        Statistics calculated for the Uniform distribution function given distribution parameters
        :return: A dictionary of calculated metrics
        """
        return None

    @jit
    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parallelized calculating the probability of the Uniform distribution
        :param x: An numpy array values determining the variable we are calculating its probability distribution (Cx1)
        :return: The probability (and the derivative) of the occurrence of the given variable (Cx1, Cx1)
        """
        return jnp.where((x > self.lower) & (x < self.upper), 1 / (self.upper - self.lower), 0)

        #
        # if x <= self.a:
        #     return 0
        # elif x >= self.b:
        #     return 0
        # else:
        #     return 1 / (self.b - self.a)


# x = (jnx.random.uniform(low = -200, high=200, size=10000)).reshape((-1,1))
# ts = Uniform(a=4,b=7)
x = jax.random.uniform(key=RNG, minval=-20, maxval=20, shape=(10, 1))

ts = Uniform(a=4, b=10)

# R = (ts.pdf, in_axes=0, out_axes=0)
mm=ts.pdf(x)
R2 = DD(x)
plt.plot(x, R, '*')
R
