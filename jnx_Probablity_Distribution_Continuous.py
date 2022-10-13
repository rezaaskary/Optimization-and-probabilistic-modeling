import jax.numpy as jnx
from jax import vmap, jit, grad
import jax
RNG = jax.random.PRNGKey(60616)


class ContinuousDistributions:
    def __init__(self,
                 a: jnx.ndarray = None,
                 b: jnx.ndarray = None,
                 variant_chains:  bool = False,
                 chains: int = 1) -> None:

        if isinstance(variant_chains, bool):
            self.variant_chains = variant_chains
        else:
            raise Exception('Please specify whether the number of chains are fixed or not !')

        if isinstance(chains, int):
            self.chains = chains
        elif chains is None:
            self.chains = None
        else:
            raise Exception('Please specify the number of chains!')




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

        if not isinstance(self.a,  type(self.b)):
            raise Exception('The parameters of the Uniform distribution are not consistent!')

        if not self.variant_chains:
            if isinstance(self.a, int):
                self.n_chains = 1





