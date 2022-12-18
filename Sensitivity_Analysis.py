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

        if self.n <= 4 * terms ** 2:
            raise Exception('Sample size n > 4terms^2 is required!')

        self.omega = jnp.zeros([self.num_vars])
        self.omega2 = self.omega.copy()
        self.omega = self.omega.at[0].set(jnp.floor((self.n - 1) / (2 * self.terms)))

        self.m = jnp.floor(self.omega[0] / (2 * self.terms))

        self.omega = self.omega.at[1:self.num_vars].set(jnp.where(self.m >= self.num_vars - 1,
                                                                  jnp.floor(jnp.linspace(1, self.m, self.num_vars - 1)),
                                                                  jnp.arange(self.num_vars - 1) % self.m + 1))
        self.s = (2 * jnp.pi / self.n) * jnp.arange(self.n)
        self.x = jnp.zeros([self.n * self.terms, self.terms])

        idx_new = jnp.arange(start=1, stop=self.num_vars, dtype=jnp.int32)
        idex_old = jnp.arange(start=0, stop=self.num_vars - 1, dtype=jnp.int32)
        z = jnp.arange(start=0, stop=self.n, dtype=jnp.int32)
        def _phase_shift(j, values_2):

            return

        def _phase_shift(i: int, values_1: tuple) -> tuple:
            omega2, omega, idx_new, idex_old, z_idx = values_1
            idx_new = idx_new.at[0:i].set(idex_old[0:i])
            omega2 = omega2.at[idx_new].set(omega[1:])
            z = z_idx.at[:].set(z_idx + 1)

            return omega2, omega, idx_new, idex_old, z_idx

