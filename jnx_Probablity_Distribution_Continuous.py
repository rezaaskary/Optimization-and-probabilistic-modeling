import jax.numpy as jnx
from jax import vmap, jit, grad
import jax
RNG = jax.random.PRNGKey(60616)


class ContinuousDistributions:
    def __init__(self,
                 a: jnx.ndarray = None,
                 b: jnx.ndarray = None) -> None:
        if isinstance(a, (jnx.ndarray, float, int)):
            self.a = a
        elif a is None:
            self.a = None
        else:
            raise Exception('The value of a is not specified correctly!')

        if isinstance(b, (jnx.ndarray, float, int)):
            self.b = b
        elif b is None:
            self.b = None
        else:
            raise Exception('The value of b is not specified correctly!')


class Uniform(ContinuousDistributions):
    def __init__(self, a: float = None, b: float = None) -> None:
        """
        Continuous uniform distribution
        :param a: The lower limit of uniform distribution
        :param b: The upper limit of uniform distribution
        """
        super(Uniform, self).__init__(a=a, b=b)

        if not type(self.a) == type(self.b):
            raise Exception('The input parameters are not consistent (Uniform Distribution)!')


        if self.a >= self.b:
            raise Exception('The lower limit of the uniform distribution is greater than the upper limit!')



