import jax.numpy as jnp
from jax import lax, vmap, jit, random

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
            self.key = random.PRNGKey(seed)
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
    def solve(self):
        idx_new = jnp.arange(start=1, stop=self.num_vars, dtype=jnp.int32)
        idex_old = jnp.arange(start=0, stop=self.num_vars - 1, dtype=jnp.int32)
        z = jnp.arange(start=0, stop=self.n, dtype=jnp.int32)
        phi_rng_uniform = random.uniform(key=self.key, shape=(self.num_vars,), dtype=jnp.float32, maxval=1.0, minval=0)

        def _phase_shift_inner(j: int, values_2: tuple) -> tuple:
            omega2, z_idx, x_arg, phi_arg = values_2
            g = 0.5 + (1 / jnp.pi) * jnp.arcsin(jnp.sin(omega2[j] * self.s + phi_arg))
            x_arg = x_arg.at[z_idx, j].set(g)
            return omega2, z_idx, x_arg, phi_arg

        def _phase_shift(i: int, values_1: tuple) -> tuple:
            omega2, omega, idx_new, idex_old, z_idx, x_arg = values_1
            idx_new = idx_new.at[0:i].set(idex_old[0:i])
            omega2 = omega2.at[idx_new].set(omega[1:])
            z_idx = z_idx.at[:].set(z_idx + 1)
            phi = 2 * jnp.pi * phi_rng_uniform[i]

            omega2, z_idx, x_arg, phi_arg = lax.fori_loop(lower=0, upper=self.num_vars, body_fun=_phase_shift_inner,
                                                          init_val=(omega2, z_idx, x_arg, phi))
            return omega2, omega, idx_new, idex_old, z_idx, x_arg

        self.omega2, self.omega2, idx_new, idex_old, z, \
            self.x = lax.fori_loop(lower=0, upper=self.num_vars, body_fun=_phase_shift,
                                   init_val=(self.omega2, self.omega, idx_new, idex_old, z, self.x))


problem = {
    'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6' ],
    'num_vars': 6,
    'bounds': [[-jnp.pi, jnp.pi], [1.0, 0.2], [3, 0.5], [3, 0.5], [3, 0.5], [3, 0.5]],
    'groups': ['G1', 'G2', 'G1', 'G1', 'G1', 'G1'],
    'dists': ['unif', 'lognorm', 'triang', 'triang', 'triang', 'triang']
}

MM =FourierAmplitudeSensitivityTest(problem=problem, n=2048, terms=5, seed=1)
RD = MM.solve()