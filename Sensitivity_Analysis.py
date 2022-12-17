import jax.numpy as jnp
import jax.random
from jax import lax, vmap, jit

bounds = jnp.array([[-jnp.pi, jnp.pi], [1.0, 0.2], [3, 0.5]], dtype=jnp.float32)

problem = {
    'names': ['x1', 'x2', 'x3'],
    'num_vars': 3,
    'bounds': bounds,
    'groups': ['G1', 'G2', 'G1'],
    'dists': ['unif', 'lognorm', 'triang']
}


class FourierAmplitudeSensitivityTest:
    def __init__(self,
                 problem: dict = None,
                 n: int = None,
                 terms: int = None,
                 seed: int = None):
        if isinstance(problem, dict):
            self.problem = problem
            if 'names' in self.problem:
                self.names = self.problem['names']
            if 'num_vars' in self.problem:
                self.num_vars = self.problem['num_vars']
            if 'groups' in self.problem:
                self.groups = self.problem['groups']
            if 'dists' in self.problem['dists']:
                self.dist = self.problem['dists']
        else:
            raise Exception('The problem is not defined in the format of a dictionary!')

        if isinstance(n, int):
            self.n = n
        else:
            raise Exception('The number samples is not specified correctly!')

        if isinstance(terms, int):
            self.terms = terms
        else:
            raise Exception('The number of terms used for calculating the Fourier transformation')

        if isinstance(seed, int):
            self.key = jax.random.PRNGKey(seed)
        else:
            raise Exception('Please enter an integer value to fix the random number generator')

        if self.n <= 4*terms**2:
            raise Exception('Sample size n > 4terms^2 is required!')

        self.omega = jnp.zeros([self.num_vars])
        self.omega = self.omega.at[0].set(jnp.floor((self.n - 1) / (2 * self.terms)))








