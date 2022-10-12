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






